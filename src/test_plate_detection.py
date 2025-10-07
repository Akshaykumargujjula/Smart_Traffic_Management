# src/test_plate_detection.py
"""
Test plate detection + OCR on a single frame.
Saves crops to outputs/plates/ and prints OCR results.

Usage (PowerShell):
    .\venv\Scripts\Activate.ps1
    python .\src\test_plate_detection.py --source .\data\traffic_short.mp4 --frame 20 --plate_model .\models\license_plate.pt --out .\outputs
"""

import os, sys, argparse, cv2, re
from ultralytics import YOLO
import easyocr
import pandas as pd

def clamp_box(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return [int(max(0, min(w-1, int(x1)))),
            int(max(0, min(h-1, int(y1)))),
            int(max(0, min(w-1, int(x2)))),
            int(max(0, min(h-1, int(y2))))]

def preprocess_for_ocr(crop):
    # crop: BGR image
    # Convert to grayscale, upscale, apply CLAHE and adaptive threshold
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # upscale to make chars larger for OCR
    h, w = gray.shape[:2]
    scale = max(1.0, 320.0 / float(max(w, h)))  # enlarge up to 320 width/height
    if scale != 1.0:
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # optional adaptive threshold (works well for plates)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    return gray, th

def run(args):
    os.makedirs(args.out, exist_ok=True)
    plates_out = os.path.join(args.out, "plates")
    os.makedirs(plates_out, exist_ok=True)

    print("[INFO] Loading plate model:", args.plate_model)
    model = YOLO(args.plate_model)

    print("[INFO] Initializing EasyOCR reader (CPU)")
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open source: " + args.source)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Source opened, total frames = {total}")

    frame_id = 0
    saved_rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id < args.frame:
            continue
        # process the requested frame only
        h, w = frame.shape[:2]
        print(f"[INFO] Processing frame {frame_id} (size {w}x{h})")
        # run plate model (suppress verbose)
        res = model(frame,conf=0.15, verbose=False)
        detections = []
        if len(res) > 0:
            try:
                b = res[0].boxes.xyxy.cpu().numpy()
                confs = res[0].boxes.conf.cpu().numpy()
                # class ids may be 0 for single-class models; we don't care about cls id here
                for i, box in enumerate(b):
                    conf = float(confs[i])
                    x1,y1,x2,y2 = box.tolist()
                    bb = clamp_box([x1,y1,x2,y2], w, h)
                    detections.append({'bbox': bb, 'conf': conf})
            except Exception as e:
                print("[WARN] parse exception:", e)

        print(f"[INFO] Plate detections found: {len(detections)}")
        for i, det in enumerate(detections):
            x1,y1,x2,y2 = det['bbox']
            conf = det['conf']
            crop = frame[y1:y2, x1:x2]
            crop_path = os.path.join(plates_out, f"frame{frame_id:04d}_plate{i+1}_conf{conf:.2f}.jpg")
            cv2.imwrite(crop_path, crop)
            # preprocess
            gray, th = preprocess_for_ocr(crop)
            pre_path = os.path.join(plates_out, f"frame{frame_id:04d}_plate{i+1}_pre.jpg")
            cv2.imwrite(pre_path, gray)
            th_path = os.path.join(plates_out, f"frame{frame_id:04d}_plate{i+1}_th.jpg")
            cv2.imwrite(th_path, th)
            # run OCR on both raw crop and preprocessed (threshold) to compare
            try:
                res_raw = reader.readtext(crop)
            except Exception as e:
                res_raw = []
                print("[WARN] OCR raw failed:", e)
            try:
                res_pre = reader.readtext(th)
            except Exception as e:
                res_pre = []
                print("[WARN] OCR preprocessed failed:", e)

            def best_text(ocr_res):
                if not ocr_res:
                    return "", 0.0
                best = max(ocr_res, key=lambda x: x[2])
                txt = re.sub(r'[^A-Z0-9]', '', best[1].upper())
                conf = float(best[2])
                return txt, conf

            txt_raw, conf_raw = best_text(res_raw)
            txt_pre, conf_pre = best_text(res_pre)

            print(f"  - Det#{i+1} conf={conf:.2f}, OCR raw='{txt_raw}' ({conf_raw:.2f}), OCR pre='{txt_pre}' ({conf_pre:.2f})")
            saved_rows.append({
                'frame': frame_id,
                'plate_index': i+1,
                'bbox': f"{x1},{y1},{x2},{y2}",
                'det_conf': conf,
                'ocr_raw': txt_raw,
                'ocr_raw_conf': conf_raw,
                'ocr_pre': txt_pre,
                'ocr_pre_conf': conf_pre,
                'crop_path': crop_path,
                'pre_path': pre_path,
                'th_path': th_path
            })

        # done with the single frame
        break

    cap.release()
    if saved_rows:
        df = pd.DataFrame(saved_rows)
        df.to_csv(os.path.join(args.out, f"plate_debug_frame{args.frame}.csv"), index=False)
        print("[INFO] Saved debug CSV to outputs")
    else:
        print("[INFO] No plate detections on that frame; check different frames or lower plate_conf and re-run detection pipeline.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="data/traffic_short.mp4")
    p.add_argument("--frame", type=int, default=20, help="frame number to test")
    p.add_argument("--plate_model", type=str, default="models/license_plate.pt")
    p.add_argument("--out", type=str, default="outputs")
    args = p.parse_args()
    run(args)
