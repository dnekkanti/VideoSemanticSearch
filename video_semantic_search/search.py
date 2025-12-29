# search.py
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import hnswlib
import numpy as np
import torch
import open_clip


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_meta(meta_jsonl: Path) -> Dict[int, dict]:
    meta = {}
    with meta_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta[int(rec["id"])] = rec
    return meta


def embed_text(model, tokenizer, device: str, query: str) -> np.ndarray:
    # support comma-separated multi-prompts: average them
    parts = [p.strip() for p in query.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty query")

    tokens = tokenizer(parts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        v = feats.mean(dim=0, keepdim=True)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.detach().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Folder with index.bin/meta.jsonl/config.json")
    ap.add_argument("--query", type=str, required=True, help='Text query, e.g. "greenery, trees"')
    ap.add_argument("--topk", type=int, default=25, help="How many matches to return")
    ap.add_argument("--per_video", type=int, default=3, help="How many matches to show per video in summary")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    index_path = out_dir / "index.bin"
    meta_path = out_dir / "meta.jsonl"
    config_path = out_dir / "config.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    meta = load_meta(meta_path)

    device = get_device()

    model, _, _ = open_clip.create_model_and_transforms(config["model"], pretrained=config["pretrained"])
    tokenizer = open_clip.get_tokenizer(config["model"])
    model = model.eval().to(device)

    index = hnswlib.Index(space="cosine", dim=int(config["embed_dim"]))
    index.load_index(str(index_path))
    index.set_ef(int(config["hnsw"]["ef_query"]))

    qv = embed_text(model, tokenizer, device, args.query)
    labels, distances = index.knn_query(qv, k=args.topk)

    # hnswlib cosine space returns distance, where smaller is better.
    # Convert to similarity-ish score: sim = 1 - dist
    results = []
    for idx, dist in zip(labels[0].tolist(), distances[0].tolist()):
        rec = meta.get(int(idx))
        if not rec:
            continue
        results.append({
            "score": float(1.0 - dist),
            "video_path": rec["video_path"],
            "t": float(rec["t"]),
            "id": int(idx),
        })

    # Print raw results
    print("\n=== Top matches ===")
    for r in results:
        print(f'{r["score"]:.3f} | {r["video_path"]} @ {r["t"]:.1f}s')

    # Summarize by video (best hits)
    by_video = defaultdict(list)
    for r in results:
        by_video[r["video_path"]].append(r)
    for v in by_video:
        by_video[v].sort(key=lambda x: x["score"], reverse=True)

    ranked_videos = sorted(by_video.items(), key=lambda kv: kv[1][0]["score"], reverse=True)

    print("\n=== Top videos ===")
    for v, hits in ranked_videos[:10]:
        top_hits = hits[:args.per_video]
        times = ", ".join([f'{h["t"]:.1f}s({h["score"]:.2f})' for h in top_hits])
        print(f"{v}\n  {times}")

    # Save JSON for export step
    out_json = out_dir / "last_search_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[info] wrote {out_json}")


if __name__ == "__main__":
    main()
