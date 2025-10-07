# src/detection.py
"""
Robust ANPR + ATCC pipeline for demo (Windows-friendly).
- vehicle detection: YOLOv11 (ultralytics)
- plate detection: local YOLO .pt (models/license_plate.pt) REQUIRED
- OCR: EasyOCR
- outputs: outputs/annotated_preview.mp4 (or avi), outputs/detections.csv, outputs/counts_summary.txt

Run example (PowerShell):
    .\venv\Scripts\Activate.ps1
    python .\src\detection.py --source .\data\traffic_short.mp4 --veh_model yolo11n.pt --plate_model .\models\license_plate.pt --out .\outputs --skip 3 --resize_width 1280 --conf 0.35
"""
import os, sys, cv2, time, argparse, re
from collections import defaultdict
from datetime import datetime

import pandas as pd
from ultralytics import YOLO
import easyocr

# Import tracker if present
this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
    sys.path.append(this_dir)
try:
    from tracking import Tracker
except Exception:
    Tracker = None

# Simple COCO vehicle mapping (we expect these ids from COCO-like weights)
VEHICLE_CLASS_IDS = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

def clamp_box(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return [int(max(0, min(w-1, int(x1)))),
            int(max(0, min(h-1, int(y1)))),
            int(max(0, min(w-1, int(x2)))),
            int(max(0, min(h-1, int(y2))))]

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

class Detector:
    def __init__(self, source, veh_model="yolo11n.pt", plate_model="models/license_plate.pt",
                 out_dir="outputs", skip=3, resize_width=1280, conf=0.35, plate_conf=0.35):
        self.source = source
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.skip = max(1, int(skip))
        self.resize_width = int(resize_width) if resize_width else 0
        self.conf = float(conf)
        self.plate_conf = float(plate_conf)

        print("[INFO] Loading vehicle model:", veh_model)
        self.veh_model = YOLO(veh_model)

        if not os.path.exists(plate_model):
            raise FileNotFoundError(f"License plate model not found at {plate_model}. Please download and place it at that path.")
        print("[INFO] Loading plate model:", plate_model)
        self.plate_model = YOLO(plate_model)

        print("[INFO] Initializing EasyOCR")
        self.reader = easyocr.Reader(['en'], gpu=False)

        self.tracker = Tracker(iou_threshold=0.3, max_age=30) if Tracker is not None else None
        self.csv_rows = []
        self.counts = defaultdict(int)

    def _open_writer(self, path, fps, width, height):
        # try mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            print("[INFO] VideoWriter opened (mp4):", path)
            return writer, path
        # try avi fallback
        avi_path = os.path.splitext(path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print("[INFO] VideoWriter opened (avi):", avi_path)
            return writer, avi_path
        print("[ERROR] Failed to open VideoWriter for mp4 and avi.")
        return None, None

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Input resolution: {orig_w}x{orig_h} @ {fps}fps")

        out_path = os.path.join(self.out_dir, "annotated_preview.mp4")
        writer = None
        out_used = None

        frame_id = 0
        start_ts = time.time()
        try:
            # open writer with the target size = original frame size (we annotate on original)
            writer, out_used = self._open_writer(out_path, fps, orig_w, orig_h)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                # optional frame skipping
                if self.skip > 1 and (frame_id % self.skip != 0):
                    # keep timeline consistent: write original frame un-annotated if writer available
                    if writer:
                        writer.write(frame)
                    continue

                # possibly resize a copy for faster inference
                frame_proc = frame
                sx = sy = 1.0
                if self.resize_width and frame.shape[1] > self.resize_width:
                    h, w = frame.shape[:2]
                    sx = self.resize_width / float(w)
                    new_h = int(h * sx)
                    frame_proc = cv2.resize(frame, (self.resize_width, new_h))
                # vehicle detection (suppress verbose)
                res = self.veh_model(frame_proc, verbose=False)
                parsed = parse_yolo_result(res[0]) if len(res) > 0 else []
                vehicles = []
                for bbox, clsid, conf in parsed:
                    if conf < self.conf:
                        continue
                    if clsid in VEHICLE_CLASS_IDS:
                        bb_proc = clamp_box(bbox, frame_proc.shape[1], frame_proc.shape[0])
                        # map to original coordinates if resized
                        if frame_proc.shape[1] != frame.shape[1]:
                            x1, y1, x2, y2 = bb_proc
                            bb = [int(x1 / sx), int(y1 / sx), int(x2 / sx), int(y2 / sx)]
                        else:
                            bb = bb_proc
                        vehicles.append({'bbox': bb, 'cls': VEHICLE_CLASS_IDS[clsid], 'conf': conf, 'plate': None})

                # plate detection on original frame
                pres = self.plate_model(frame, verbose=False)
                parsed_p = parse_yolo_result(pres[0]) if len(pres) > 0 else []
                plates = []
                for bbox, clsid, conf in parsed_p:
                    if conf < self.plate_conf:
                        continue
                    bb = clamp_box(bbox, frame.shape[1], frame.shape[0])
                    plates.append({'bbox': bb, 'conf': conf, 'text': ""})

                # OCR plates and attach to nearest vehicle (IoU)
                for p in plates:
                    x1,y1,x2,y2 = p['bbox']
                    pad = 6
                    px1,py1 = max(0,x1-pad), max(0,y1-pad)
                    px2,py2 = min(frame.shape[1]-1, x2+pad), min(frame.shape[0]-1, y2+pad)
                    crop = frame[py1:py2, px1:px2]
                    text = ""
                    try:
                        ocr_res = self.reader.readtext(crop)
                        if ocr_res:
                            best = max(ocr_res, key=lambda x: x[2])
                            raw = best[1]
                            text = re.sub(r'[^A-Z0-9]', '', raw.upper())
                    except Exception:
                        text = ""
                    p['text'] = text
                    # attach to closest vehicle
                    best_i = 0.0; best_vi = None
                    for vi, v in enumerate(vehicles):
                        ax1,ay1,ax2,ay2 = v['bbox']; bx1,by1,bx2,by2 = p['bbox']
                        inter_x1 = max(ax1,bx1); inter_y1 = max(ay1,by1)
                        inter_x2 = min(ax2,bx2); inter_y2 = min(ay2,by2)
                        inter_w = max(0, inter_x2 - inter_x1); inter_h = max(0, inter_y2 - inter_y1)
                        inter_area = inter_w * inter_h
                        area_a = max(0,(ax2-ax1))*max(0,(ay2-ay1))
                        area_b = max(0,(bx2-bx1))*max(0,(by2-by1))
                        denom = area_a + area_b - inter_area
                        iou_val = (inter_area / denom) if denom>0 else 0.0
                        if iou_val > best_i:
                            best_i = iou_val; best_vi = vi
                    if best_vi is not None and best_i > 0.02:
                        vehicles[best_vi]['plate'] = p['text']

                # tracking & counting
                if self.tracker:
                    tracks = self.tracker.update(vehicles, frame_id)
                    line_y = int(frame.shape[0]*0.5)
                    for t in tracks:
                        if t.counted:
                            continue
                        prev_c = t.prev_center(); last_c = t.last_center()
                        if prev_c and last_c and prev_c[1] < line_y <= last_c[1]:
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
                # annotate
                ann = frame.copy()
                # draw vehicles
                for v in vehicles:
                    x1,y1,x2,y2 = v['bbox']
                    cv2.rectangle(ann, (x1,y1),(x2,y2),(0,255,0),2)
                    label = v['cls']
                    if v.get('plate'):
                        label += f" | {v['plate']}"
                    cv2.putText(ann, label, (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                # draw plates (if any)
                for p in plates:
                    x1,y1,x2,y2 = p['bbox']
                    cv2.rectangle(ann, (x1,y1),(x2,y2),(0,165,255),2)
                    if p.get('text'):
                        cv2.putText(ann, p['text'], (x1, max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                # overlay counts
                y0 = 20
                for i,(k,v) in enumerate(self.counts.items()):
                    cv2.putText(ann, f"{k}: {v}", (10, y0 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # write
                if writer:
                    writer.write(ann)

            # end loop
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user (Ctrl+C). Finalizing and saving outputs...")
        except Exception as e:
            print("[ERROR] Exception during processing:", e)
        finally:
            cap.release()
            if writer:
                writer.release()
            # save CSV
            if self.csv_rows:
                df = pd.DataFrame(self.csv_rows)
                csv_out = os.path.join(self.out_dir, "detections.csv")
                df.to_csv(csv_out, index=False)
                print("[INFO] Saved CSV:", csv_out)
            # save counts
            summary_path = os.path.join(self.out_dir, "counts_summary.txt")
            with open(summary_path, "w") as f:
                for k,v in self.counts.items():
                    f.write(f"{k}: {v}\n")
            print("[INFO] Saved counts summary:", summary_path)
            print("[INFO] Annotated video (if created) at:", out_used)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="data/traffic_short.mp4")
    p.add_argument("--veh_model", type=str, default="yolo11n.pt")
    p.add_argument("--plate_model", type=str, default="models/license_plate.pt")
    p.add_argument("--out", type=str, default="outputs")
    p.add_argument("--skip", type=int, default=3)
    p.add_argument("--resize_width", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--plate_conf", type=float, default=0.35)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    det = Detector(source=args.source, veh_model=args.veh_model, plate_model=args.plate_model,
                   out_dir=args.out, skip=args.skip, resize_width=args.resize_width,
                   conf=args.conf, plate_conf=args.plate_conf)
    det.run()
