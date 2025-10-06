# src/tracking.py
"""
Lightweight IoU-based tracker to give persistent IDs and avoid double counting.
Tracks store bbox, class, plate (optional) and a 'counted' flag.
"""

from typing import List, Dict

def iou(boxA, boxB):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - interArea
    return (interArea / denom) if denom > 0 else 0.0

class Track:
    def __init__(self, track_id: int, bbox: List[int], cls: str, frame_id: int, conf: float = 0.0):
        self.id = track_id
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.cls = cls
        self.conf = conf
        self.last_seen = frame_id
        self.age = 0
        self.hits = 1
        self.counted = False
        self.history = []
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.history.append((cx, cy))
        self.plate = None

    def update(self, bbox: List[int], frame_id: int, conf: float = 0.0):
        self.bbox = bbox
        self.last_seen = frame_id
        self.conf = conf
        self.hits += 1
        self.age = 0
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        self.history.append((cx, cy))

    def mark_missed(self):
        self.age += 1

    def last_center(self):
        return self.history[-1] if len(self.history) > 0 else None

    def prev_center(self):
        return self.history[-2] if len(self.history) > 1 else None

class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """
        :param iou_threshold: minimum IoU to match detection to track
        :param max_age: frames after which a missing track is removed
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[Track] = []
        self._next_id = 1

    def _create_track(self, det: Dict, frame_id: int) -> Track:
        t = Track(self._next_id, det['bbox'], det['cls'], frame_id, det.get('conf', 0.0))
        if det.get('plate'):
            t.plate = det['plate']
        self._next_id += 1
        return t

    def update(self, detections: List[Dict], frame_id: int) -> List[Track]:
        """
        Update tracks with current detections.
        detections: list of {'bbox':[x1,y1,x2,y2], 'cls':str, 'conf':float, 'plate':optional}
        """
        if not self.tracks:
            for d in detections:
                self.tracks.append(self._create_track(d, frame_id))
            return self.tracks

        matched_ids = set()
        # greedy matching by detection confidence
        dets_sorted = sorted(detections, key=lambda x: x.get('conf', 0.0), reverse=True)
        for det in dets_sorted:
            best_t = None
            best_i = 0.0
            for t in self.tracks:
                if t.id in matched_ids:
                    continue
                if t.cls != det['cls']:
                    continue
                i = iou(t.bbox, det['bbox'])
                if i > best_i:
                    best_i = i
                    best_t = t
            if best_t is not None and best_i >= self.iou_threshold:
                best_t.update(det['bbox'], frame_id, det.get('conf', 0.0))
                if det.get('plate'):
                    best_t.plate = det['plate']
                matched_ids.add(best_t.id)
            else:
                new_t = self._create_track(det, frame_id)
                self.tracks.append(new_t)
                matched_ids.add(new_t.id)

        # mark missed tracks and remove old ones
        for t in self.tracks:
            if t.last_seen != frame_id:
                t.mark_missed()
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return self.tracks

    def get_active_tracks(self):
        return self.tracks
