# src/detection.py
"""
ANPR + ATCC pipeline using:
 - YOLOv11 (vehicles) via ultralytics
 - YOLO (local) license plate model at models/license_plate.pt
 - EasyOCR for plate text
 - Tracker (tracking.py) to avoid double counting

Usage:
    python src/detection.py --source data/traffic.mp4 --veh_model yolo11s.pt --plate_model models/license_plate.pt --out outputs --skip 1 --conf 0.35
"""

import os
import sys
import cv2
import time
import re
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from ultralytics import YOLO
import easyocr

# local tracker
this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.append(this_dir)
from tracking import Tracker

# COCO mapping snippet (we care about vehicle ids)
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    # rest omitted for brevity
]

VEHICLE_CLASS_IDS = {
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

def clamp_box(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w-1, int(x1))))
    y1 = int(max(0, min(h-1, int(y1))))
    x2 = int(max(0, min(w-1, int(x2))))
    y2 = int(max(0, min(h-1, int(y2))))
    return [x1, y1, x2, y2]

def parse_yolo_result(result):
    """
    Convert ultralytics result to a list of (bbox, clsid, conf)
    """
    boxes = []
    try:
        b = result.boxes.xyxy.cpu().numpy()
        c = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for bbox, clsid, conf in zip(b, c, confs):
            boxes.append((bbox.tolist(), int(clsid), float(conf)))
    except Exception:
        # safe fallback if API shape differs
        try:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                clsid = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                boxes.append((xyxy.tolist(), clsid, conf))
        except Exception:
            return []
    return boxes

class VideoProcessor:
    def __init__(self,
                 source,
                 veh_model_path="yolo11s.pt",
                 plate_model_path="models/license_plate.pt",
                 out_dir="outputs",
                 process_every_n_frames=1,
                 vehicle_conf_thresh=0.35,
                 plate_conf_thresh=0.35,
                 counting_line_ratio=0.5,max_frames=0):
        self.source = source
        self.veh_model_path = veh_model_path
        self.plate_model_path = plate_model_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.process_every = max(1, int(process_every_n_frames))
        self.vehicle_conf_thresh = vehicle_conf_thresh
        self.plate_conf_thresh = plate_conf_thresh
        self.counting_line_ratio = counting_line_ratio
        self.max_frames = max_frames

        # load vehicle model (ultralytics will auto-download if needed)
        print("[INFO] Loading vehicle model:", veh_model_path)
        self.veh_model = YOLO(veh_model_path)

        # plate model must exist locally (we require it)
        if not os.path.exists(plate_model_path):
            raise FileNotFoundError(f"License plate model not found at {plate_model_path}. Please download and place it there.")
        print("[INFO] Loading plate model:", plate_model_path)
        self.plate_model = YOLO(plate_model_path)

        # OCR
        print("[INFO] Initializing EasyOCR (CPU). First run may take a bit for model load.")
        self.reader = easyocr.Reader(['en'], gpu=False)

        # tracker
        self.tracker = Tracker(iou_threshold=0.3, max_age=30)

        # outputs
        self.csv_rows = []
        self.counts = defaultdict(int)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Video resolution: {width}x{height}, FPS: {fps}")
        line_y = int(height * self.counting_line_ratio)

        out_video_path = os.path.join(self.out_dir, "annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        frame_id = 0
        start_ts = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            if self.max_frames and frame_id >= self.max_frames:
                print(f"[INFO] Reached max_frames={self.max_frames}, stopping early.")
                break
            if frame_id % self.process_every != 0:
                writer.write(frame)
                continue

            # 1) vehicle detection
            veh_results = self.veh_model(frame,verbose=False)
            parsed = []
            if len(veh_results) > 0:
                parsed = parse_yolo_result(veh_results[0])
            vehicles = []
            for bbox, clsid, conf in parsed:
                if conf < self.vehicle_conf_thresh:
                    continue
                if clsid in VEHICLE_CLASS_IDS:
                    bb = clamp_box(bbox, width, height)
                    vehicles.append({'bbox': bb, 'cls': VEHICLE_CLASS_IDS[clsid], 'conf': conf, 'plate': None})

            # 2) plate detection (local model)
            plate_results = self.plate_model(frame)
            plates_parsed = []
            if len(plate_results) > 0:
                plates_parsed = parse_yolo_result(plate_results[0])
            plates = []
            for bbox, clsid, conf in plates_parsed:
                if conf < self.plate_conf_thresh:
                    continue
                bb = clamp_box(bbox, width, height)
                plates.append({'bbox': bb, 'conf': conf, 'text': ""})

            # 3) OCR each plate crop and attach to nearest vehicle using IoU
            for p in plates:
                x1, y1, x2, y2 = p['bbox']
                pad = 6
                px1 = max(0, x1 - pad)
                py1 = max(0, y1 - pad)
                px2 = min(width - 1, x2 + pad)
                py2 = min(height - 1, y2 + pad)
                crop = frame[py1:py2, px1:px2]
                plate_text = ""
                try:
                    ocr_res = self.reader.readtext(crop)
                    if ocr_res:
                        best = max(ocr_res, key=lambda x: x[2])
                        raw = best[1]
                        # basic cleanup: uppercase alnum
                        plate_text = re.sub(r'[^A-Z0-9]', '', raw.upper())
                except Exception:
                    plate_text = ""
                p['text'] = plate_text

                # match to vehicle by IoU
                best_i = 0.0
                best_vi = None
                for vi, v in enumerate(vehicles):
                    # compute IoU
                    ax1, ay1, ax2, ay2 = v['bbox']
                    bx1, by1, bx2, by2 = p['bbox']
                    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
                    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
                    area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
                    denom = area_a + area_b - inter_area
                    iou_val = (inter_area / denom) if denom > 0 else 0.0
                    if iou_val > best_i:
                        best_i = iou_val
                        best_vi = vi
                if best_vi is not None and best_i > 0.05:
                    vehicles[best_vi]['plate'] = p['text']

            # 4) update tracker & counting
            tracks = self.tracker.update(vehicles, frame_id)
            for t in tracks:
                if t.counted:
                    continue
                prev_c = t.prev_center()
                last_c = t.last_center()
                if prev_c is None or last_c is None:
                    continue
                # count when moving from above the line to below the line
                if (prev_c[1] < line_y) and (last_c[1] >= line_y):
                    self.counts[t.cls] += 1
                    t.counted = True
                    self.csv_rows.append({
                        'timestamp': datetime.now().isoformat(),
                        'frame_id': frame_id,
                        'track_id': t.id,
                        'vehicle_type': t.cls,
                        'plate': t.plate or "",
                        'conf': float(t.conf)
                    })

            # 5) annotate and write frame
            annotated = frame.copy()
            cv2.line(annotated, (0, line_y), (width, line_y), (0, 255, 0), 2)
            # overlay counts
            y0 = 30
            for i, (clsn, cnt) in enumerate(self.counts.items()):
                text = f"{clsn}: {cnt}"
                cv2.putText(annotated, text, (10, y0 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            # draw tracks
            for t in tracks:
                x1, y1, x2, y2 = clamp_box(t.bbox, width, height)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 100, 0), 2)
                label = f"ID:{t.id} {t.cls}"
                if t.plate:
                    label += f" | {t.plate}"
                cv2.putText(annotated, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            writer.write(annotated)

        cap.release()
        writer.release()
        duration = time.time() - start_ts
        print(f"[INFO] Done: frames processed: {frame_id}, time: {duration:.1f}s, out: {out_video_path}")

        # save CSV
        if self.csv_rows:
            df = pd.DataFrame(self.csv_rows)
            csv_out = os.path.join(self.out_dir, "detections.csv")
            df.to_csv(csv_out, index=False)
            print("[INFO] saved CSV:", csv_out)
        # save counts summary
        summary_out = os.path.join(self.out_dir, "counts_summary.txt")
        with open(summary_out, "w") as f:
            for k, v in self.counts.items():
                f.write(f"{k}: {v}\n")
        print("[INFO] saved counts summary:", summary_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0",
                        help="video file path or camera index (0)")
    parser.add_argument("--veh_model", type=str, default="yolo11s.pt",
                        help="vehicle model path/name (ultralytics will auto-download if name used)")
    parser.add_argument("--plate_model", type=str, default="models/license_plate.pt",
                        help="local license plate model (.pt) - REQUIRED")
    parser.add_argument("--out", type=str, default="outputs",
                        help="output directory")
    parser.add_argument("--skip", type=int, default=1,
                        help="process every Nth frame (1=every frame)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="vehicle confidence threshold")
    parser.add_argument("--max_frames", type=int, default=0,
                    help="Stop after this many frames (0 = process entire video)")

    args = parser.parse_args()

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    vp = VideoProcessor(
        source=source,
        veh_model_path=args.veh_model,
        plate_model_path=args.plate_model,
        out_dir=args.out,
        process_every_n_frames=args.skip,
        vehicle_conf_thresh=args.conf
    )
    vp.run()
