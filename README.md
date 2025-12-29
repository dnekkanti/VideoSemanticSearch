# Local Video Semantic Search (v1)

Search your personal video archive with plain English.

Type something like **“greenery, forest, trees”** or **“sunset, beach, ocean”** and get back the best matching moments with timestamps. You can also export the top hits as real video clips.

This v1 focuses on **visual concepts** (scenes/objects/looks). Action recognition (e.g., “hiking”) can come later.

---

## What it does

- Scans a folder of videos (recursively)
- Samples frames every *N* seconds (configurable)
- Embeds frames using a CLIP-style model (OpenCLIP)
- Builds a local vector index (HNSW)
- Lets you search by text and returns:
  - video path
  - timestamp (seconds)
  - similarity score
- Exports matching segments via `ffmpeg`

---

## Requirements

- macOS (Apple Silicon supported), Linux, or Windows
- Python 3.10+ recommended
- `ffmpeg` for exporting clips

### Install ffmpeg (macOS)
```bash
brew install ffmpeg
