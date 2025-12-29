# export_clips.py
import argparse
import json
import os
import subprocess
from pathlib import Path


def ffmpeg_export(input_path: str, t_center: float, seconds_before: float, seconds_after: float, out_path: str):
    t0 = max(0.0, t_center - seconds_before)
    dur = seconds_before + seconds_after

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{t0:.3f}",
        "-i", input_path,
        "-t", f"{dur:.3f}",
        "-c", "copy",
        out_path
    ]
    # If stream copy fails for some codecs/timecuts, fallback to re-encode:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        cmd2 = [
            "ffmpeg",
            "-y",
            "-ss", f"{t0:.3f}",
            "-i", input_path,
            "-t", f"{dur:.3f}",
            "-c:v", "libx264",
            "-c:a", "aac",
            out_path
        ]
        subprocess.run(cmd2, check=True)


def sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Same out_dir used earlier")
    ap.add_argument("--results_json", type=str, default="", help="Defaults to out_dir/last_search_results.json")
    ap.add_argument("--export_dir", type=str, required=True, help="Where to write exported clips")
    ap.add_argument("--topn", type=int, default=10, help="Export top N results")
    ap.add_argument("--before", type=float, default=3.0, help="Seconds before hit")
    ap.add_argument("--after", type=float, default=5.0, help="Seconds after hit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    results_path = Path(args.results_json).expanduser().resolve() if args.results_json else (out_dir / "last_search_results.json")
    export_dir = Path(args.export_dir).expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:args.topn]

    for i, r in enumerate(results, start=1):
        vp = r["video_path"]
        t = float(r["t"])
        score = float(r["score"])

        base = sanitize_filename(Path(vp).stem)
        out_name = f"{i:02d}_{base}_t{t:.1f}_s{score:.2f}.mp4"
        out_path = str(export_dir / out_name)

        ffmpeg_export(vp, t, args.before, args.after, out_path)
        print(f"[ok] {out_path}")

    print("[done]")


if __name__ == "__main__":
    main()
