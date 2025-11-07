# Footfall Counter — YOLOv8n + ByteTrack + Supervision

**Author:** Thanan Cariappa K R

This project counts people crossing a virtual line in a video. It uses **Ultralytics YOLOv8n** for person detection, **ByteTrack** for multi‑object tracking, and a **custom crossing counter** so counting starts immediately (even near frame corners). The same counter drives the **top banner** and the **ROI line labels**, so the numbers always match.

> **Highlights**
> - People **detection** (YOLOv8n, class 0)
> - Multi‑object **tracking** (ByteTrack via `model.track`)
> - Robust **counting** from frame 1 (per‑ID center‑crossing + debounce)
> - Clean **visuals**: green rounded boxes, red ROI line (default at **65%** of frame height), top‑left banner (IN↑ / OUT↓ / TOTAL), and synced IN/OUT labels near the line
> - **Auto‑screenshots**: three PNG frames saved in `outputs/` during a run

---
## Repository

Clone this project from GitHub:

```bash
git clone https://github.com/thanan2002/footfall-counter-yolov8.git

## Video Source

- **File:** `mall_counting.mp4` (simulated mall corridor video with clear pedestrian flow)
- **Description:** Fixed camera facing a corridor; people walk in both directions across the ROI line.
- **Alternative:** You can use any public video (e.g., mall/corridor/entrance) or a webcam with `--source 0`.


## Screenshots (auto‑saved to `outputs/`)

After a run you’ll find:

- `outputs/snap_001.png`
- `outputs/snap_002.png`
- `outputs/snap_003.png`

Embedded preview:

![Snapshot 1](outputs/snap_001.png)
![Snapshot 2](outputs/snap_002.png)
![Snapshot 3](outputs/snap_003.png)

---

## Requirements

- Python **3.8+**
- Packages:
  - `ultralytics`
  - `supervision`
  - `opencv-python`
  - `numpy`

### Install with `uv` (recommended)
```bash
uv pip install -r requirements.txt
```
Or explicitly:
```bash
uv pip install ultralytics supervision opencv-python numpy
```

---

## Usage

Basic:
```bash
uv run python app.py --source mall_counting.mp4
```

Headless (no window):
```bash
uv run python app.py --source mall_counting.mp4 --no-show
```

Show tracker IDs + tune detection:
```bash
uv run python app.py --source mall_counting.mp4 --display-ids --imgsz 736 --conf 0.3
```

Move the ROI line:
```bash
# as a fraction of height (default 0.65)
uv run python app.py --source mall_counting.mp4 --line-y 0.60

# exact pixel position, overrides --line-y
uv run python app.py --source mall_counting.mp4 --line-px 420
```

Avoid double counts (debounce):
```bash
uv run python app.py --source mall_counting.mp4 --debounce-ms 600
```

Use GPU (if available):
```bash
uv run python app.py --source mall_counting.mp4 --device 0
```

Custom output path:
```bash
uv run python app.py --source mall_counting.mp4 --save outputs/run1.mp4
```

---

## How it works

1) **Detection** — YOLOv8n detects people (class 0).  
2) **Tracking** — ByteTrack assigns stable IDs.  
3) **Counting** — For each ID, compare previous & current bbox **center‑Y** to the **ROI Y**:  
   - `prev_y < line_y <= curr_y` → **IN++**  
   - `prev_y >= line_y > curr_y` → **OUT++**  
   A per‑ID **debounce** (default 400 ms) prevents double counts when a person jitters on the line.  
4) **Visualization** — Green rounded boxes, red ROI line, synced IN/OUT labels at the line, and a top‑left banner showing IN↑, OUT↓, TOTAL.  
5) **Auto‑Screenshots** — The script saves three frames (early/middle/later) to `outputs/` for reports and this README.

---

## requirements.txt

```txt
ultralytics>=8.2.0
supervision>=0.26.0
opencv-python>=4.8.0
numpy>=1.23.0
```

Install:
```bash
uv pip install -r requirements.txt
```

---

## Author

**Thanan Cariappa K R**
