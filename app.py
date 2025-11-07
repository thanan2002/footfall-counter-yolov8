# app.py — Footfall Counter (YOLOv8n + ByteTrack + Supervision)
# ------------------------------------------------------------------------------
# Summary:
#   - Detects people (YOLOv8n), tracks them (ByteTrack), and counts IN/OUT when
#     crossing a horizontal ROI line.
#   - Uses a robust custom per-ID center-crossing counter (with debounce) so
#     counting starts immediately and works near edges/corners.
#   - Visuals: green rounded boxes, red ROI line (default at 65% height),
#     top-left banner (IN↑ / OUT↓ / TOTAL), and synced labels near the line.
#   - Auto-screenshots: saves three PNG frames to outputs/ (snap_001/2/3.png).
#
# Tested with:
#   - supervision 0.26.1
#   - ultralytics 8.3.223
#
# Example:
#   uv run python app.py --source mall_counting.mp4
#   uv run python app.py --source mall_counting.mp4 --display-ids --conf 0.3 --debounce-ms 600
# ------------------------------------------------------------------------------

import argparse
import os
import time
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def parse_args():
    """CLI options for reproducible runs."""
    p = argparse.ArgumentParser(description="Footfall Counter — synced counts + ROI labels")
    p.add_argument("--source", type=str, default="mall_counting.mp4",
                   help="Path to input video OR webcam index like '0'")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default=None, help="Device: 'cpu' or '0' for CUDA:0")
    p.add_argument("--tracker", type=str, default=None, help="Optional tracker config (e.g., 'bytetrack.yaml')")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--line-y", type=float, default=0.65,
                   help="ROI line as fraction of height (0..1). Default 0.65 (65%).")
    g.add_argument("--line-px", type=int, default=None,
                   help="ROI line exact Y in pixels (overrides --line-y)")
    p.add_argument("--save", type=str, default="outputs/footfall_output.mp4",
                   help="Output MP4 path (directories auto-created)")
    p.add_argument("--no-show", action="store_true", help="Disable live preview window")
    p.add_argument("--display-ids", action="store_true", help="Render tracker IDs on boxes")
    p.add_argument("--debounce-ms", type=int, default=400,
                   help="Min ms between counts per ID to avoid bounce double-counts")
    return p.parse_args()


def probe_source(src_str):
    """Open source once to fetch (src, W, H, fps)."""
    src = int(src_str) if src_str.isdigit() else src_str
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {src_str}")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Failed to read a frame from: {src_str}")
    H, W = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return src, W, H, fps


# ---------- UI helpers: rounded banner tags (top-left) ----------

def draw_rounded_rect(img, x, y, w, h, color, radius=14, alpha=0.95):
    """Draw a filled rounded rectangle with alpha blending."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + w - radius, y + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + radius, y + h - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def draw_tag(img, x, y, label, value, bg_color, pad_x=16, pad_y=10, font_scale=0.9, thickness=2,
             icon=None, icon_color=(255, 255, 255)):
    """
    Draw a rounded 'pill' tag:  [ LABEL: value (↑/↓) ] and return (img, tag_width)
    icon: None | 'up' | 'down'
    """
    text = f"{label}: {value}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    icon_w = 24 if icon in ("up", "down") else 0
    gap = 10 if icon_w else 0
    w = tw + pad_x * 2 + gap + icon_w
    h = th + pad_y * 2

    draw_rounded_rect(img, x, y, w, h, bg_color, radius=14, alpha=0.95)

    tx = x + pad_x
    ty = y + h - pad_y
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    if icon in ("up", "down"):
        cx = x + w - pad_x - icon_w // 2
        cy = y + h // 2
        size = 8
        if icon == "up":
            pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]], dtype=np.int32)
        else:
            pts = np.array([[cx, cy + size], [cx - size, cy - size], [cx + size, cy - size]], dtype=np.int32)
        cv2.fillPoly(img, [pts], icon_color, lineType=cv2.LINE_AA)

    return img, w


def draw_banner_top_left(img, in_count, out_count):
    """Top-left banner with 3 tags: [IN ↑] [OUT ↓] [TOTAL]."""
    GREEN = (60, 170, 70)        # Apple green
    GREEN_DARK = (40, 120, 50)   # For TOTAL
    RED = (0, 0, 255)            # Red
    margin = 16
    spacing = 10
    x = margin
    y = margin

    img, w1 = draw_tag(img, x, y, "IN", in_count, GREEN, icon="up")
    x += w1 + spacing
    img, w2 = draw_tag(img, x, y, "OUT", out_count, RED, icon="down")
    x += w2 + spacing
    total = int(in_count) + int(out_count)
    img, _ = draw_tag(img, x, y, "TOTAL", total, GREEN_DARK)
    return img


# ---------- ROI helpers: manual line + synced labels ----------

def draw_roi_line_and_labels(img, y_line, W, in_count, out_count):
    """
    Draw a red ROI line and synced IN/OUT labels near it (same numbers as banner).
    """
    # Red line
    cv2.line(img, (0, y_line), (W, y_line), (0, 0, 255), 3, cv2.LINE_AA)

    # Labels
    left_x = 20
    right_x = max(20, W - 230)
    above = max(20, y_line - 10)
    below = min(img.shape[0] - 20, y_line + 25)

    cv2.putText(img, f"IN : {in_count}", (left_x, above),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (60, 170, 70), 2, cv2.LINE_AA)
    cv2.putText(img, f"OUT : {out_count}", (right_x, below),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return img


# ---------- Robust crossing counter (per-ID) ----------

class CrossingCounter:
    """
    Counts IN/OUT crossings per tracked ID using the bbox center Y with debounce.
      - If prev_y < line_y <= curr_y  -> IN++
      - If prev_y >= line_y > curr_y  -> OUT++
    """
    def __init__(self, line_y: int, debounce_ms: int = 400):
        self.line_y = line_y
        self.prev_y = {}     # id -> previous center y
        self.last_ts = {}    # id -> last count timestamp (ms)
        self.in_count = 0
        self.out_count = 0
        self.debounce_ms = debounce_ms

    def update(self, det: sv.Detections):
        now = int(time.time() * 1000)
        tids = det.tracker_id
        if tids is None:
            return
        for i, tid in enumerate(tids):
            if tid is None:
                continue

            x1, y1, x2, y2 = det.xyxy[i].astype(int)
            cy = (y1 + y2) // 2

            if tid not in self.prev_y:
                self.prev_y[tid] = cy
                continue

            prev = self.prev_y[tid]
            self.prev_y[tid] = cy

            # per-ID debounce
            last = self.last_ts.get(tid, 0)
            if now - last < self.debounce_ms:
                continue

            # crossing detection
            if prev < self.line_y <= cy:
                self.in_count += 1
                self.last_ts[tid] = now
            elif prev >= self.line_y > cy:
                self.out_count += 1
                self.last_ts[tid] = now


def main():
    args = parse_args()

    # 1) Load YOLO model
    model = YOLO("yolov8n.pt")

    # 2) Probe the source and compute ROI Y
    src, W, H, fps = probe_source(args.source)
    line_y = args.line_px if args.line_px is not None else int(H * args.line_y)

    # 3) Supervision annotators (boxes + optional labels)
    green = sv.Color.from_hex("#3CAA46")  # apple green for boxes
    bbox_annotator = sv.RoundBoxAnnotator(color=green, thickness=2)
    label_annotator = sv.LabelAnnotator()

    # 4) Output video + screenshot setup
    out_dir = os.path.dirname(args.save)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs("outputs", exist_ok=True)  # ensure default folder exists
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.save, fourcc, fps, (W, H))

    # Auto-screenshot config (save near start/middle/later)
    frame_idx = 0
    snap_targets = {1, 150, 300}
    snaps_done = []

    # 5) Tracking config (ByteTrack)
    track_cfg = dict(source=src, stream=True, classes=[0], imgsz=args.imgsz, conf=args.conf)
    if args.tracker: track_cfg["tracker"] = args.tracker
    if args.device:  track_cfg["device"] = args.device

    # 6) Authoritative counter
    counter = CrossingCounter(line_y=line_y, debounce_ms=args.debounce_ms)

    print(f"[INFO] Running... Output: {args.save} | ROI y={line_y} px")

    try:
        for result in model.track(**track_cfg):
            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)

            # Green rounded boxes
            frame = bbox_annotator.annotate(scene=frame, detections=detections)

            # Optional ID labels
            if args.display_ids and detections.tracker_id is not None:
                labels = [f"ID {int(t)}" if t is not None else "" for t in detections.tracker_id]
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # Update counts
            counter.update(detections)

            # Draw ROI line + synced labels
            frame = draw_roi_line_and_labels(frame, line_y, W, counter.in_count, counter.out_count)

            # Top-left banner (same numbers)
            frame = draw_banner_top_left(frame, counter.in_count, counter.out_count)

            # Save video frame
            writer.write(frame)

            # --- Auto-screenshots (3 images) ---
            frame_idx += 1
            if frame_idx in snap_targets and len(snaps_done) < 3:
                snap_dir = out_dir if out_dir else "outputs"
                snap_path = os.path.join(snap_dir, f"snap_{len(snaps_done)+1:03d}.png")
                cv2.imwrite(snap_path, frame)
                snaps_done.append(snap_path)

            # Show preview
            if not args.no_show:
                cv2.imshow("Footfall Counter — Press ESC to exit", frame)
                if cv2.waitKey(1) == 27:
                    break

    finally:
        # Ensure we have 3 screenshots (fill with last frame if needed)
        if len(snaps_done) < 3 and 'frame' in locals() and frame is not None:
            for i in range(len(snaps_done)+1, 4):
                snap_dir = out_dir if out_dir else "outputs"
                snap_path = os.path.join(snap_dir, f"snap_{i:03d}.png")
                cv2.imwrite(snap_path, frame)
                snaps_done.append(snap_path)

        writer.release()
        cv2.destroyAllWindows()

    print(f"\n✅ DONE\nFINAL → IN: {counter.in_count} | OUT: {counter.out_count}\n"
          f"Saved video: {args.save}\n"
          f"Screenshots: {', '.join(snaps_done)}")


if __name__ == "__main__":
    main()
