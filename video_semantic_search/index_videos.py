# index_videos.py
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import hnswlib
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def list_videos(root: Path) -> List[Path]:
    vids = []
    for p in root.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS and p.is_file():
            vids.append(p)
    return sorted(vids)


def get_duration_seconds(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps > 0 and frame_count > 0:
        return float(frame_count / fps)
    # Fallback: try CAP_PROP_POS_MSEC by seeking to end (not always reliable)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    return float(ms / 1000.0)


def sample_frames_seek(video_path: Path, stride_s: float, max_frames: int | None) -> List[Tuple[float, np.ndarray]]:
    """
    Samples one frame every stride_s seconds using timestamp seeking.
    Returns list of (timestamp_seconds, frame_rgb_uint8).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    duration = get_duration_seconds(cap)
    if duration <= 0:
        cap.release()
        return []

    samples: List[Tuple[float, np.ndarray]] = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            t += stride_s
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        samples.append((t, frame_rgb))

        if max_frames is not None and len(samples) >= max_frames:
            break

        t += stride_s

    cap.release()
    return samples


def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", type=str, required=True, help="Folder containing videos (recursively searched).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for index + metadata.")
    ap.add_argument("--stride_s", type=float, default=3.0, help="Sample 1 frame every N seconds.")
    ap.add_argument("--max_frames_per_video", type=int, default=0, help="0 means no cap; else limit frames per video.")
    ap.add_argument("--batch_size", type=int, default=32, help="Embedding batch size.")
    ap.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name.")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag.")
    ap.add_argument("--ef_construction", type=int, default=200, help="HNSW ef_construction.")
    ap.add_argument("--M", type=int, default=16, help="HNSW M parameter.")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    max_frames_per_video = None if args.max_frames_per_video == 0 else args.max_frames_per_video

    device = get_device()
    print(f"[info] device={device}")

    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.eval().to(device)
    embed_dim = model.text_projection.shape[1] if hasattr(model, "text_projection") else 512

    vids = list_videos(videos_dir)
    if not vids:
        raise SystemExit(f"No videos found under {videos_dir}")

    # First pass: estimate total samples for HNSW init (fast-ish: duration estimation)
    estimated_total = 0
    for vp in tqdm(vids, desc="Estimating total frames"):
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            continue
        duration = get_duration_seconds(cap)
        cap.release()
        if duration <= 0:
            continue
        est = int(np.ceil(duration / args.stride_s))
        if max_frames_per_video is not None:
            est = min(est, max_frames_per_video)
        estimated_total += est

    if estimated_total == 0:
        raise SystemExit("Could not estimate any frames. Are the videos readable by OpenCV?")

    print(f"[info] estimated_total_frames={estimated_total}")

    # Init HNSW index
    index = hnswlib.Index(space="cosine", dim=embed_dim)
    index.init_index(max_elements=estimated_total, ef_construction=args.ef_construction, M=args.M)
    index.set_ef(50)

    meta_path = out_dir / "meta.jsonl"
    index_path = out_dir / "index.bin"
    config_path = out_dir / "config.json"

    # Write metadata incrementally
    meta_f = meta_path.open("w", encoding="utf-8")

    next_id = 0

    for vp in tqdm(vids, desc="Indexing videos"):
        samples = sample_frames_seek(vp, args.stride_s, max_frames_per_video)
        if not samples:
            continue

        # Embed in batches
        for batch in batched(samples, args.batch_size):
            ts_list = [ts for ts, _ in batch]
            imgs = [Image.fromarray(arr) for _, arr in batch]
            img_tensor = torch.stack([preprocess(img) for img in imgs], dim=0).to(device)

            with torch.no_grad():
                feats = model.encode_image(img_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.detach().cpu().numpy().astype(np.float32)

            ids = np.arange(next_id, next_id + len(batch))
            index.add_items(feats_np, ids)

            for i, ts in zip(ids.tolist(), ts_list):
                rec = {
                    "id": i,
                    "video_path": str(vp),
                    "t": float(ts),
                    "stride_s": float(args.stride_s),
                }
                meta_f.write(json.dumps(rec) + "\n")

            next_id += len(batch)

    meta_f.close()

    print(f"[info] indexed_items={next_id}")

    # Shrink index capacity to actual count (optional; hnswlib supports resize)
    index.resize_index(next_id)

    index.save_index(str(index_path))

    config = {
        "model": args.model,
        "pretrained": args.pretrained,
        "embed_dim": embed_dim,
        "device_used": device,
        "stride_s": args.stride_s,
        "indexed_items": next_id,
        "hnsw": {"ef_construction": args.ef_construction, "M": args.M, "ef_query": 50},
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[done] wrote:\n  {index_path}\n  {meta_path}\n  {config_path}")


if __name__ == "__main__":
    main()
