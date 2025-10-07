# src/quick_preview_short.py
"""
Quick preview + short clip pipeline (Windows-friendly).

What it does:
1) Cuts the first N seconds (default 10s) of input video to data/traffic_short.mp4 (if not present).
2) Runs YOLO vehicle detection + optional local license plate model + EasyOCR on that short clip.
3) Writes annotated video to outputs/annotated_preview.mp4 (or .avi fallback),
   and writes outputs/detections.csv and outputs/counts_summary.txt.

Designed for CPU / limited GPUs (MX330). Defaults: yolo11n.pt (nano) and skip=2.
Usage (PowerShell):
    .\venv\Scripts\Activate.ps1
    python .\src\quick_preview_short.py --source data\traffic.mp4 --seconds 10 --veh_model yolo11n.pt --plate_model models\license_plate.pt --out outputs --skip 2 --conf 0.35
"""

import os
import sys
import cv2
import time
import argparse
import re
from collections import defaultdict
from datetime import datetime

import pandas as pd
from ultralytics import YOLO
import easyocr

# import Tracker from tracking.py (assumes src/tracking.py exists)
this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.append(this_dir)
try:
    from tracking import Tracker
except Exception:
    Tracker = None
    print("[WARN] tracking.py not found — the preview will still run but without tracking/double-count avoidance.")

# helper utilities
def clamp_box(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w-1, int(x1))))
    y1 = int(max(0, min(h-1, int(y1))))
    x2 = int(max(0, min(w-1, int(x2))))
    y2 = int(max(0, min(h-1, int(y2))))
    return [x1, y1, x2, y2]

def parse_yolo_result(result):
    boxes = []
    try:
        b = result.boxes.xyxy.cpu().numpy()
        c = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for bbox, clsid, conf in zip(b, c, confs):
            boxes.append((bbox.tolist(), int(clsid), float(conf)))
    except Exception:
        try:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                clsid = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                boxes.append((xyxy.tolist(), clsid, conf))
        except Exception:
            return []
    return boxes

# COCO vehicle mapping (we only use vehicles)
VEHICLE_CLASS_IDS = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

def cut_first_n_seconds(src_path, dst_path, seconds):
    """Cut first N seconds of src_path into dst_path using OpenCV (Windows-safe)."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_write = min(total_frames, int(round(fps * seconds)))
    print(f"[INFO] Cutting first {seconds}s -> {frames_to_write} frames (fps={fps}, size={width}x{height})")

    # try mp4 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("[WARN] mp4 VideoWriter failed for cut. Trying XVID .avi fallback.")
        dst_path_avi = dst_path.rsplit('.',1)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(dst_path_avi, fourcc, fps, (width, height))
        if writer.isOpened():
            print("[INFO] Using fallback:", dst_path_avi)
            dst_path = dst_path_avi
        else:
            writer = None
            print("[ERROR] Cannot create cut video writer. Will attempt to process frames in-memory instead.")
    count = 0
    while count < frames_to_write:
        ret, frame = cap.read()
        if not ret:
            break
        if writer:
            writer.write(frame)
        count += 1
    cap.release()
    if writer:
        writer.release()
    return dst_path

def run_preview(args):
    os.makedirs(args.out, exist_ok=True)
    short_path = os.path.join("data", "traffic_short.mp4")

    # if user provided explicit short exists, use it otherwise cut
    if os.path.exists(short_path):
        print("[INFO] Found existing short clip:", short_path)
    else:
        print("[INFO] Creating short clip:", short_path)
        short_path = cut_first_n_seconds(args.source, short_path, args.seconds)

    print("[INFO] Loading vehicle model:", args.veh_model)
    veh_model = YOLO(args.veh_model)

    plate_model = None
    if args.plate_model and os.path.exists(args.plate_model):
        print("[INFO] Loading plate model:", args.plate_model)
        plate_model = YOLO(args.plate_model)
    else:
        print("[INFO] No local plate model found. Plate detection will be skipped.")

    print("[INFO] Initializing EasyOCR")
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(short_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open short video: {short_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Short clip resolution: {width}x{height}, fps: {fps}")

    # Prepare writer
    out_mp4 = os.path.join(args.out, "annotated_preview.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("[WARN] mp4 writer failed; trying .avi XVID fallback")
        out_avi = os.path.join(args.out, "annotated_preview.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(out_avi, fourcc, fps, (width, height))
        if writer.isOpened():
            print("[INFO] Using AVI:", out_avi)
            out_path_used = out_avi
        else:
            print("[WARN] VideoWriter failed; frames will be processed but not saved to a single video")
            writer = None
            out_path_used = None
    else:
        out_path_used = out_mp4
        print("[INFO] Writing annotated preview to:", out_mp4)

    # tracking (optional)
    tracker = Tracker(iou_threshold=0.3, max_age=30) if Tracker is not None else None
    counts = defaultdict(int)
    csv_rows = []

    frame_id = 0
    max_frames = int(args.frames) if args.frames > 0 else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if max_frames and frame_id > max_frames:
            break

        # optionally skip frames to reduce compute
        if args.skip > 1 and (frame_id % args.skip != 0):
            # still write unchanged frame to output to keep video length consistent
            if writer:
                writer.write(frame)
            continue

        # optional resize to reduce CPU (we will run detection on resized frame)
        if args.resize_width and args.resize_width > 0:
            h, w = frame.shape[:2]
            if w > args.resize_width:
                scale = args.resize_width / float(w)
                frame_proc = cv2.resize(frame, (args.resize_width, int(h * scale)))
            else:
                frame_proc = frame.copy()
        else:
            frame_proc = frame.copy()

        # detection (suppress verbose)
        try:
            res = veh_model(frame_proc, verbose=False)
            parsed = parse_yolo_result(res[0]) if len(res) > 0 else []
        except Exception as e:
            print("[WARN] vehicle inference failed at frame", frame_id, e)
            parsed = []

        vehicles = []
        for bbox, clsid, conf in parsed:
            if conf < args.conf:
                continue
            if clsid in VEHICLE_CLASS_IDS:
                bb = clamp_box(bbox, frame_proc.shape[1], frame_proc.shape[0])
                # map bbox back to original frame size if resized
                if frame_proc.shape[1] != frame.shape[1]:
                    sx = frame.shape[1] / frame_proc.shape[1]
                    sy = frame.shape[0] / frame_proc.shape[0]
                    x1, y1, x2, y2 = bb
                    bb = [int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)]
                vehicles.append({'bbox': bb, 'cls': VEHICLE_CLASS_IDS[clsid], 'conf': conf, 'plate': None})

        # plate detection on original frame (if model present)
        plates = []
        if plate_model:
            try:
                pres = plate_model(frame, verbose=False)
                parsed_p = parse_yolo_result(pres[0]) if len(pres) > 0 else []
                for bbox, clsid, conf in parsed_p:
                    if conf < args.plate_conf:
                        continue
                    bb = clamp_box(bbox, frame.shape[1], frame.shape[0])
                    plates.append({'bbox': bb, 'conf': conf, 'text': ""})
            except Exception as e:
                print("[WARN] plate inference failed on frame", frame_id, e)
                plates = []

        # OCR and attach plates to nearest vehicle by IoU
        for p in plates:
            x1,y1,x2,y2 = p['bbox']
            pad = 6
            px1 = max(0, x1-pad); py1 = max(0, y1-pad)
            px2 = min(frame.shape[1]-1, x2+pad); py2 = min(frame.shape[0]-1, y2+pad)
            crop = frame[py1:py2, px1:px2]
            text = ""
            try:
                ocr_res = reader.readtext(crop)
                if ocr_res:
                    best = max(ocr_res, key=lambda x: x[2])
                    raw = best[1]
                    text = re.sub(r'[^A-Z0-9]', '', raw.upper())
            except Exception:
                text = ""
            p['text'] = text
            # attach
            best_i = 0.0; best_idx = None
            for vi, v in enumerate(vehicles):
                ax1,ay1,ax2,ay2 = v['bbox']; bx1,by1,bx2,by2 = p['bbox']
                inter_x1 = max(ax1,bx1); inter_y1 = max(ay1,by1)
                inter_x2 = min(ax2,bx2); inter_y2 = min(ay2,by2)
                inter_w = max(0, inter_x2 - inter_x1); inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                area_a = max(0,(ax2-ax1)) * max(0,(ay2-ay1))
                area_b = max(0,(bx2-bx1)) * max(0,(by2-by1))
                denom = area_a + area_b - inter_area
                iou_val = (inter_area / denom) if denom > 0 else 0.0
                if iou_val > best_i:
                    best_i = iou_val; best_idx = vi
            if best_idx is not None and best_i > 0.05:
                vehicles[best_idx]['plate'] = p['text']

        # update tracker & perform line counting if tracker available
        if tracker:
            tracks = tracker.update(vehicles, frame_id)
            # counting line at center (you can change)
            line_y = int(frame.shape[0] * 0.5)
            for t in tracks:
                if t.counted:
                    continue
                prev_c = t.prev_center()
                last_c = t.last_center()
                if prev_c and last_c and prev_c[1] < line_y <= last_c[1]:
                    counts[t.cls] += 1
                    t.counted = True
                    csv_rows.append({
                        "timestamp": datetime.now().isoformat(),
                        "frame_id": frame_id,
                        "track_id": t.id,
                        "vehicle_type": t.cls,
                        "plate": t.plate or ""
                    })
        else:
            # no persistent tracking: increment simple per-frame counts (approx)
            for v in vehicles:
                counts[v['cls']] += 0  # keep counts 0 to avoid miscounting without a tracker

        # annotate original frame (boxes, labels, counts overlay)
        ann = frame.copy()
        for v in vehicles:
            x1,y1,x2,y2 = v['bbox']
            cv2.rectangle(ann, (x1,y1),(x2,y2),(0,255,0),2)
            label = f"{v['cls']}"
            if v.get('plate'):
                label += f" | {v['plate']}"
            cv2.putText(ann, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        for p in plates:
            x1,y1,x2,y2 = p['bbox']
            cv2.rectangle(ann, (x1,y1),(x2,y2),(0,165,255),2)
            if p.get('text'):
                cv2.putText(ann, p['text'], (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # draw a counting line and overlay counts (if tracker used)
        line_y = int(frame.shape[0] * 0.5)
        cv2.line(ann, (0,line_y), (frame.shape[1], line_y), (0,255,0), 2)
        y0 = 20
        for i, (k, v) in enumerate(counts.items()):
            cv2.putText(ann, f"{k}: {v}", (10, y0 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if writer:
            try:
                writer.write(ann)
            except Exception as e:
                print("[WARN] writer.write failed:", e)
                try:
                    writer.release()
                except Exception:
                    pass
                writer = None

    cap.release()
    if writer:
        writer.release()

    # save CSVs
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_out = os.path.join(args.out, "detections.csv")
        df.to_csv(csv_out, index=False)
        print("[INFO] Saved detections CSV:", csv_out)
    summary_out = os.path.join(args.out, "counts_summary.txt")
    with open(summary_out, "w") as f:
        for k,v in counts.items():
            f.write(f"{k}: {v}\n")
    print("[INFO] Saved counts summary:", summary_out)
    if out_path_used:
        print("[INFO] Annotated preview video written to:", out_path_used)
    else:
        print("[INFO] No single video file produced; annotated frames processed.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/traffic.mp4", help="Original full-length video path")
    parser.add_argument("--seconds", type=int, default=10, help="Seconds to cut from start for short clip")
    parser.add_argument("--veh_model", type=str, default="yolo11n.pt", help="Vehicle model (yolo11n.pt recommended for MX330)")
    parser.add_argument("--plate_model", type=str, default="models/license_plate.pt", help="Local plate model (optional)")
    parser.add_argument("--out", type=str, default="outputs", help="Output folder")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process (0 = all frames in short clip)")
    parser.add_argument("--skip", type=int, default=2, help="Process every Nth frame (use 2 or 3 for MX330 to speed up)")
    parser.add_argument("--conf", type=float, default=0.35, help="Vehicle confidence")
    parser.add_argument("--plate_conf", type=float, default=0.35, help="Plate detection confidence")
    parser.add_argument("--resize_width", type=int, default=960, help="Resize width for processing (0=nothing)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_preview(args)
