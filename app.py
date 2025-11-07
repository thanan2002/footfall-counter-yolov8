# app.py — Footfall Counter (YOLOv8n + ByteTrack + Supervision)
# ------------------------------------------------------------------------------
# Summary:
#   Detects people (YOLOv8n), tracks them (ByteTrack), and counts IN/OUT when
#   crossing a horizontal ROI line. Adds per‑ID trajectory trails and a live,
#   decaying heatmap overlay. Also exposes a FastAPI service for file uploads
#   and a live MJPEG stream endpoint.
#
# Why this works well:
#   - Robust per‑ID center‑crossing counter with debounce (prevents double counts)
#   - ByteTrack keeps identities stable during short occlusions
#   - UI (banner + ROI labels) is drawn AFTER overlays so it remains readable
#
# Webcam :
#   python app.py --source 0
#
# CLI Examples:
#   python app.py --source mall_counting.mp4
#   python app.py --source 0 --alpha-heatmap 0.25 --traj --traj-len 48
#   python app.py --source "rtsp://user:pass@ip:554/stream"
#
# API Examples:
#   python app.py --api --host 0.0.0.0 --port 8000
#   curl -F "file=@mall_counting.mp4" "http://localhost:8000/upload-video"
#   http://localhost:8000/stream?source=0
#
# Folders created on demand:
#   - outputs/   (rendered video & snapshots)
#   - uploads/   (API uploads)
# ------------------------------------------------------------------------------


import argparse
import os
import time
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque, defaultdict

# --- Optional FastAPI imports (lazy) ---
try:
    from fastapi import FastAPI, File, UploadFile, Query
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False


# ========================= Args =========================

def parse_args():
    p = argparse.ArgumentParser(description="Footfall Counter — trajectories + heatmap + API")
    p.add_argument("--source", type=str, default="mall_counting.mp4",
                   help="Video path, webcam index like '0', or RTSP URL")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--device", type=str, default=None, help="'cpu' or CUDA device like '0'")
    p.add_argument("--tracker", type=str, default=None, help="Tracker config (e.g., 'bytetrack.yaml')")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--line-y", type=float, default=0.65,
                   help="ROI line as fraction of height (0..1). Default 0.65")
    g.add_argument("--line-px", type=int, default=None, help="ROI line Y (px)")
    p.add_argument("--save", type=str, default="outputs/footfall_output.mp4",
                   help="Output MP4 path (directories auto-created)")
    p.add_argument("--no-show", action="store_true", help="Disable preview window")
    p.add_argument("--display-ids", action="store_true", help="Render tracker IDs on boxes")
    p.add_argument("--debounce-ms", type=int, default=400,
                   help="Min ms between counts per ID")

    # New visualization toggles
    p.add_argument("--traj", action="store_true", help="Draw per-ID trajectories")
    p.add_argument("--traj-len", type=int, default=32, help="Max points stored per ID path")
    p.add_argument("--heatmap", action="store_true", help="Overlay live heatmap")
    p.add_argument("--heatmap-decay", type=float, default=0.98,
                   help="Decay factor (0..1) per frame for heatmap (1=no decay)")
    p.add_argument("--heatmap-radius", type=int, default=10, help="Radius of hit per detection center")
    p.add_argument("--alpha-heatmap", type=float, default=0.35, help="Heatmap overlay alpha (0..1)")

    # API mode
    p.add_argument("--api", action="store_true", help="Run FastAPI server instead of CLI processing")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


# ========================= Helpers =========================

def ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def probe_source(src_str):
    src = int(src_str) if str(src_str).isdigit() else src_str
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


def draw_rounded_rect(img, x, y, w, h, color, radius=14, alpha=0.95):
    overlay = img.copy()
    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1, cv2.LINE_AA)
    for cx, cy in ((x+radius, y+radius), (x+w-radius, y+radius), (x+radius, y+h-radius), (x+w-radius, y+h-radius)):
        cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def draw_tag(img, x, y, label, value, bg_color, pad_x=16, pad_y=10, font_scale=0.9, thickness=2,
             icon=None, icon_color=(255, 255, 255)):
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
        pts = (np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]], dtype=np.int32)
               if icon == "up" else
               np.array([[cx, cy + size], [cx - size, cy - size], [cx + size, cy - size]], dtype=np.int32))
        cv2.fillPoly(img, [pts], icon_color, lineType=cv2.LINE_AA)

    return img, w


def draw_banner_top_left(img, in_count, out_count):
    GREEN = (60, 170, 70)
    GREEN_DARK = (40, 120, 50)
    RED = (0, 0, 255)
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


def draw_roi_line_and_labels(img, y_line, W, in_count, out_count):
    cv2.line(img, (0, y_line), (W, y_line), (0, 0, 255), 3, cv2.LINE_AA)
    left_x = 20
    right_x = max(20, W - 230)
    above = max(20, y_line - 10)
    below = min(img.shape[0] - 20, y_line + 25)

    cv2.putText(img, f"IN : {in_count}", (left_x, above),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (60, 170, 70), 2, cv2.LINE_AA)
    cv2.putText(img, f"OUT : {out_count}", (right_x, below),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return img


# ========================= Counting =========================

class CrossingCounter:
    def __init__(self, line_y: int, debounce_ms: int = 400):
        self.line_y = line_y
        self.prev_y = {}
        self.last_ts = {}
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
            last = self.last_ts.get(tid, 0)
            if now - last < self.debounce_ms:
                continue
            if prev < self.line_y <= cy:
                self.in_count += 1
                self.last_ts[tid] = now
            elif prev >= self.line_y > cy:
                self.out_count += 1
                self.last_ts[tid] = now


# ========================= Trajectories & Heatmap =========================

class VisualMemory:
    """Keeps per-ID trajectories and a running heatmap."""
    def __init__(self, H, W, traj_enabled=True, traj_len=32,
                 heatmap_enabled=True, decay=0.98, radius=10, alpha=0.35):
        self.H, self.W = H, W
        self.traj_enabled = traj_enabled
        self.traj_len = traj_len
        self.heatmap_enabled = heatmap_enabled
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.radius = int(max(1, radius))
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.paths = defaultdict(lambda: deque(maxlen=self.traj_len))  # id -> deque[(x,y)]
        self.hm = np.zeros((H, W), dtype=np.float32)

    def update(self, det: sv.Detections):
        if det.tracker_id is None:
            return
        for i, tid in enumerate(det.tracker_id):
            if tid is None:
                continue
            x1, y1, x2, y2 = det.xyxy[i].astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if self.traj_enabled:
                self.paths[tid].append((cx, cy))
            if self.heatmap_enabled:
                # Decay old heat gently
                if self.decay < 1.0:
                    self.hm *= self.decay
                cv2.circle(self.hm, (cx, cy), self.radius, (1.0,), thickness=-1)

    def draw(self, frame):
        # Draw trajectories
        if self.traj_enabled:
            for tid, q in self.paths.items():
                if len(q) >= 2:
                    pts = np.array(q, dtype=np.int32)
                    # thickness tapers with age
                    for i in range(1, len(pts)):
                        t = int(3 * (i / len(pts)) + 1)  # 1..4
                        cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), (255, 255, 255), t, cv2.LINE_AA)
        # Draw heatmap overlay
        if self.heatmap_enabled:
            hm_norm = self.hm.copy()
            mx = float(hm_norm.max()) if hm_norm.size else 0.0
            if mx > 0:
                hm_norm = (hm_norm / mx * 255.0).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                cv2.addWeighted(hm_color, self.alpha, frame, 1 - self.alpha, 0, frame)
        return frame


# ========================= Core Processing =========================

def process_video(source: str, save_path: str, imgsz=640, conf=0.25, device=None, tracker=None,
                  line_y_frac=0.65, line_px=None, display_ids=False, debounce_ms=400,
                  traj=True, traj_len=32, heatmap=True, heatmap_decay=0.98, heatmap_radius=10,
                  alpha_heatmap=0.35, show=False):
    """Run detection/tracking/visualization and write an output video. Returns counts & snapshots."""
    model = YOLO("yolov8n.pt")
    src, W, H, fps = probe_source(source)
    y_line = line_px if line_px is not None else int(H * line_y_frac)

    green = sv.Color.from_hex("#3CAA46")
    bbox_annotator = sv.RoundBoxAnnotator(color=green, thickness=2)
    label_annotator = sv.LabelAnnotator()

    ensure_dir(save_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    # Screenshots
    frame_idx = 0
    snap_targets = {1, 150, 300}
    snaps_done = []

    # Tracking config
    track_cfg = dict(source=src, stream=True, classes=[0], imgsz=imgsz, conf=conf)
    if tracker: track_cfg["tracker"] = tracker
    if device:  track_cfg["device"] = device

    counter = CrossingCounter(line_y=y_line, debounce_ms=debounce_ms)
    vismem = VisualMemory(H, W, traj_enabled=traj, traj_len=traj_len,
                          heatmap_enabled=heatmap, decay=heatmap_decay,
                          radius=heatmap_radius, alpha=alpha_heatmap)

    print(f"[INFO] Running... Output: {save_path} | ROI y={y_line} px")
    try:
        for result in model.track(**track_cfg):
            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)

            # Draw boxes & optional ids
            frame = bbox_annotator.annotate(scene=frame, detections=detections)
            if display_ids and detections.tracker_id is not None:
                labels = [f"ID {int(t)}" if t is not None else "" for t in detections.tracker_id]
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

            # Update state
            counter.update(detections)
            vismem.update(detections)

            # ✅ Draw heatmap + trajectories FIRST so UI stays on top
            frame = vismem.draw(frame)

            # ✅ ROI & banner on top of overlays
            frame = draw_roi_line_and_labels(frame, y_line, W, counter.in_count, counter.out_count)
            frame = draw_banner_top_left(frame, counter.in_count, counter.out_count)

            # Write & show
            writer.write(frame)

            frame_idx += 1
            if frame_idx in snap_targets:
                snap_dir = os.path.dirname(save_path) or "."
                snap_path = os.path.join(snap_dir, f"snap_{len(snaps_done)+1:03d}.png")
                cv2.imwrite(snap_path, frame)
                snaps_done.append(snap_path)

            if show:
                cv2.imshow("Footfall Counter — Press ESC to exit", frame)
                if cv2.waitKey(1) == 27:
                    break
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        writer.release()
        cv2.destroyAllWindows()

    return {
        "in": int(counter.in_count),
        "out": int(counter.out_count),
        "total": int(counter.in_count + counter.out_count),
        "save_path": save_path,
        "snaps": snaps_done,
        "roi_y": y_line,
        "width": W,
        "height": H,
    }


# ========================= API (FastAPI) =========================

app = None
if FASTAPI_AVAILABLE:
    app = FastAPI(title="Footfall Counter API", version="1.0.0")

    @app.post("/upload-video")
    async def upload_video(file: UploadFile = File(...),
                           conf: float = Query(0.25),
                           imgsz: int = Query(640)):
        ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
        in_dir = "uploads"; os.makedirs(in_dir, exist_ok=True)
        out_dir = "outputs"; os.makedirs(out_dir, exist_ok=True)
        in_path = os.path.join(in_dir, f"in_{int(time.time())}{ext}")
        out_path = os.path.join(out_dir, f"out_{int(time.time())}.mp4")
        with open(in_path, "wb") as f:
            f.write(await file.read())
        try:
            res = process_video(
                source=in_path,
                save_path=out_path,
                imgsz=imgsz,
                conf=conf,
                show=False,
                traj=True,
                heatmap=True,
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
        res["input_path"] = in_path
        return res

    def mjpeg_generator(source: str, imgsz=640, conf=0.25):
        model = YOLO("yolov8n.pt")
        src, W, H, _ = probe_source(source)
        y_line = int(H * 0.65)
        green = sv.Color.from_hex("#3CAA46")
        bbox_annotator = sv.RoundBoxAnnotator(color=green, thickness=2)
        counter = CrossingCounter(line_y=y_line, debounce_ms=400)
        vismem = VisualMemory(H, W, traj_enabled=True, traj_len=32,
                              heatmap_enabled=True, decay=0.98, radius=10, alpha=0.35)
        track_cfg = dict(source=src, stream=True, classes=[0], imgsz=imgsz, conf=conf)
        try:
            for result in model.track(**track_cfg):
                frame = result.orig_img
                detections = sv.Detections.from_ultralytics(result)
                frame = bbox_annotator.annotate(scene=frame, detections=detections)
                counter.update(detections)
                vismem.update(detections)
                # Draw heatmap/trajectories first
                frame = vismem.draw(frame)
                # UI on top
                frame = draw_roi_line_and_labels(frame, y_line, W, counter.in_count, counter.out_count)
                frame = draw_banner_top_left(frame, counter.in_count, counter.out_count)
                ok, buf = cv2.imencode('.jpg', frame)
                if not ok:
                    continue
                chunk = buf.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n")
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[MJPEG ERROR] {e}")

    @app.get("/stream")
    async def stream(source: str = Query(..., description="webcam index, path, or RTSP URL")):
        return StreamingResponse(mjpeg_generator(source), media_type='multipart/x-mixed-replace; boundary=frame')


# ========================= Main =========================

def main():
    args = parse_args()

    if args.api:
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI/uvicorn not installed. Install: pip install fastapi uvicorn")
        uvicorn.run(app, host=args.host, port=args.port)
        return

    # CLI run
    src = args.source
    y_px = args.line_px
    res = process_video(
        source=src,
        save_path=args.save,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        tracker=args.tracker,
        line_y_frac=args.line_y,
        line_px=y_px,
        display_ids=args.display_ids,
        debounce_ms=args.debounce_ms,
        traj=args.traj or True,            # enable by default for this build
        traj_len=args.traj_len,
        heatmap=args.heatmap or True,      # enable by default for this build
        heatmap_decay=args.heatmap_decay,
        heatmap_radius=args.heatmap_radius,
        alpha_heatmap=args.alpha_heatmap,
        show=not args.no_show,
    )

    print(f"\n✅ DONE\nFINAL → IN: {res['in']} | OUT: {res['out']} | TOTAL: {res['total']}\n"
          f"Saved video: {res['save_path']}\nScreenshots: {', '.join(res['snaps']) if res['snaps'] else 'None'}\n")


if __name__ == "__main__":
    main()
