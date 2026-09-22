from pathlib import Path
import hashlib
import json
import numpy as np
import torch


def _sha256(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _train_tinystories_bpe(tokenizer_path, vocab_size=10_000, stories=100_000):
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|unk|>"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def iterator():
        for i, row in enumerate(ds):
            if i >= stories:
                break
            yield row["text"]

    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=stories)
    actual = tokenizer.get_vocab_size()
    if actual != vocab_size:
        raise RuntimeError(f"Tokenizer vocab is {actual}, expected exactly {vocab_size}.")
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def prepare_tinystories_cache(
    out_dir="data",
    train_tokens=52_000_000,
    val_tokens=1_000_000,
    vocab_size=10_000,
    tokenizer_training_stories=100_000,
    batch_texts=256,
):
    """Train TinyStories BPE then stream/tokenize fixed-size train/val caches."""
    from datasets import load_dataset
    from tokenizers import Tokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out / f"tinystories_bpe{vocab_size}.json"
    train_path = out / f"tinystories_bpe{vocab_size}_train_52m.bin"
    val_path = out / f"tinystories_bpe{vocab_size}_val_1m.bin"
    meta_path = out / "dataset_meta.json"

    expected = [tokenizer_path, train_path, val_path, meta_path]
    if all(p.exists() for p in expected):
        meta = json.loads(meta_path.read_text())
        if meta.get("vocab_size") == vocab_size:
            return meta

    if not tokenizer_path.exists():
        tokenizer = _train_tinystories_bpe(
            tokenizer_path, vocab_size=vocab_size, stories=tokenizer_training_stories
        )
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None:
        raise RuntimeError("Tokenizer is missing <|endoftext|>.")

    def write_split(split, path, target):
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(target,))
        cursor = 0
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        buf = []

        def flush(texts, cursor):
            encodings = tokenizer.encode_batch(texts)
            for enc in encodings:
                seq = enc.ids + [eos]
                take = min(len(seq), target - cursor)
                if take <= 0:
                    break
                arr[cursor:cursor + take] = np.asarray(seq[:take], dtype=np.uint16)
                cursor += take
            return cursor

        for row in ds:
            buf.append(row["text"])
            if len(buf) >= batch_texts:
                cursor = flush(buf, cursor)
                buf = []
                if cursor >= target:
                    break
        if buf and cursor < target:
            cursor = flush(buf, cursor)
        arr.flush()
        if cursor != target:
            raise RuntimeError(f"Only collected {cursor:,}/{target:,} tokens for {split}.")
        return cursor

    write_split("train", train_path, train_tokens)
    write_split("validation", val_path, val_tokens)

    meta = {
        "dataset": "roneneldan/TinyStories",
        "tokenizer": "custom ByteLevel BPE trained on TinyStories",
        "vocab_size": vocab_size,
        "tokenizer_training_stories": tokenizer_training_stories,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
        "dtype": "uint16",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


class TokenMemmap:
    def __init__(self, path, seq_len, device, seed=1337):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = int(seq_len)
        self.device = device
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size, valid_target_tokens=None):
        max_start = len(self.data) - self.seq_len - 1
        starts = self.rng.integers(0, max_start, size=batch_size)
        x = np.stack([
            np.asarray(self.data[s:s+self.seq_len], dtype=np.int64)
            for s in starts
        ])
        y = np.stack([
            np.asarray(self.data[s+1:s+self.seq_len+1], dtype=np.int64)
            for s in starts
        ])
        x = torch.from_numpy(x).to(self.device, non_blocking=True)
        y = torch.from_numpy(y).to(self.device, non_blocking=True)

        if valid_target_tokens is not None:
            valid_target_tokens = int(valid_target_tokens)
            total = y.numel()
            if not 0 < valid_target_tokens <= total:
                raise ValueError(f"valid_target_tokens must be in [1,{total}], got {valid_target_tokens}")
            if valid_target_tokens < total:
                y.view(-1)[valid_target_tokens:] = -100
        return x, y


def count_valid_targets(y):
    return int((y != -100).sum().item())
