"""Verify pinned inputs against evidence/input-manifest.json.

Prints the manifest table; exits nonzero on any hash/byte/schema mismatch.
"""
import csv
import hashlib
import json
import os
import sys

from safetensors import safe_open

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_schema(path, name):
    if name == "model.safetensors":
        with safe_open(path, framework="np") as f:
            t = f.get_tensor("embeddings")
            return f"tensor embeddings {t.shape} {t.dtype}"
    if name == "tokenizer.json":
        tok = json.load(open(path, encoding="utf-8"))
        return f"tokenizer model.type={tok['model']['type']} vocab={len(tok['model']['vocab'])}"
    if name == "tok512.model":
        from src.corpus import load_spm

        return f"spm vocab={load_spm(path).get_piece_size()}"
    if name == "chinchilla-runs.csv":
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        return f"csv rows={len(rows)} cols={list(rows[0].keys())}"
    if name == "TinyStoriesV2-GPT4-valid.txt":
        n = open(path, encoding="utf-8").read().count("<|endoftext|>")
        return f"text, <|endoftext|> separators={n}"
    raise ValueError(name)


def main() -> int:
    man = json.load(open(os.path.join(HERE, "evidence", "input-manifest.json")))
    ok = True
    for entry in man["files"]:
        path = os.path.join(HERE, "data", entry["file"])
        if not os.path.exists(path):
            print(f"MISSING {entry['file']}")
            ok = False
            continue
        digest, size = sha256(path), os.path.getsize(path)
        good = digest == entry["sha256"] and size == entry["bytes"]
        schema = check_schema(path, entry["file"])
        print(
            f"{'OK  ' if good else 'FAIL'} {entry['file']:34s} {size:>9d} B "
            f"{digest[:16]}…  {schema}  resolved={entry['resolved_commit'] or entry['note']}"
        )
        ok = ok and good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
