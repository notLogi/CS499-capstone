#!/usr/bin/env python3
"""
Fully local RAG - no API key, no cost, nothing leaves your machine.

Chroma ships a built-in embedding model (all-MiniLM-L6-v2, run locally via
onnxruntime). If we hand it the film text instead of precomputed OpenAI
vectors, it embeds everything itself - so the whole pipeline runs offline and
every teammate can use it with zero keys.

    python local_rag.py build              # build the local collection (one time)
    python local_rag.py "something cozy for a rainy sunday"
    python local_rag.py "a heist movie" -k 8

Data source is movies_rag.jsonl (its "text" field already blends plot + mood).
The local model is smaller than OpenAI's, so retrieval is a touch coarser, but
it needs no key and is plenty good for a demo.
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent
JSONL_FILE = str(ROOT / "metadata" / "movies_rag.jsonl")
DB_DIR = str(ROOT / "chroma_local")   # separate from the OpenAI-vector chroma_db/
COLLECTION = "films_local"
BATCH_SIZE = 500


def flatten_metadata(md: dict) -> dict:
    out = {}
    for k, v in md.items():
        if isinstance(v, list):
            out[k] = ", ".join(str(x) for x in v) if v else ""
        elif v is None:
            out[k] = ""
        else:
            out[k] = v
    return out


def build() -> None:
    if not Path(JSONL_FILE).exists():
        sys.exit(f"{JSONL_FILE} not found. Run: python recommend.py export-rag")
    rows = [json.loads(l) for l in open(JSONL_FILE, encoding="utf-8")]
    print(f"Loaded {len(rows)} rows", file=sys.stderr)

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    # No embedding_function passed -> Chroma uses its default local model.
    coll = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        coll.add(
            ids=[str(r["id"]) for r in batch],
            documents=[r["text"] for r in batch],       # Chroma embeds these locally
            metadatas=[flatten_metadata(r["metadata"]) for r in batch],
        )
        print(f"  embedded {min(start + BATCH_SIZE, len(rows))}/{len(rows)}", file=sys.stderr)
    print(f"Collection '{COLLECTION}' has {coll.count()} films in {DB_DIR}/", file=sys.stderr)


def query(text: str, k: int) -> None:
    client = chromadb.PersistentClient(path=DB_DIR)
    coll = client.get_collection(COLLECTION)
    res = coll.query(query_texts=[text], n_results=k)   # query embedded locally too
    for i, (meta, dist) in enumerate(zip(res["metadatas"][0], res["distances"][0]), 1):
        sim = 1 - dist
        print(f"{i}. {meta['title']} ({meta['release_year']})  "
              f"[{meta['pool']}, {meta['vote_average']}★, {meta['pacing'] or '-'}]  sim {sim:.3f}")
        if meta.get("embedding_summary"):
            print(f"     {meta['embedding_summary']}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", help='"build" to build the local collection, else a free-text query')
    p.add_argument("-k", type=int, default=5, help="number of results")
    args = p.parse_args()

    if args.query == "build":
        build()
    else:
        query(args.query, args.k)


if __name__ == "__main__":
    main()
