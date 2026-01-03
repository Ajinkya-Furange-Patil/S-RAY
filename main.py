import time
import uuid
import base64
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


# =============================
# APP INITIALIZATION
# =============================
app = FastAPI(title="Real-Time People Tracking + Heatmap")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# For best detection, you can change to yolov8m.pt if your GPU/CPU is strong.
# model = YOLO("yolov8m.pt")
model = YOLO("yolov8n.pt")  # fast + decent accuracy


# =============================
# DATA MODELS
# =============================
@dataclass
class Person:
    uuid: str
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    confidence: float
    last_seen: float


# =============================
# TRACKER
# =============================
class PeopleTracker:
    def __init__(self, max_distance: float = 60.0, timeout: float = 1.0):
        self.people: Dict[str, Person] = {}
        self.max_distance = max_distance
        self.timeout = timeout

    def _centroid(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> Dict[str, Person]:
        now = time.time()

        if not detections:
            self._cleanup(now)
            return self.people

        det_boxes = [d[:4] for d in detections]
        det_centroids = [self._centroid(b) for b in det_boxes]
        det_conf = [d[4] for d in detections]

        if not self.people:
            for box, cen, conf in zip(det_boxes, det_centroids, det_conf):
                self._register(box, cen, conf, now)
            return self.people

        existing_ids = list(self.people.keys())
        existing_centroids = [self.people[i].centroid for i in existing_ids]

        distances = cdist(existing_centroids, det_centroids)

        matched_det = set()
        matched_people = set()

        while distances.size > 0:
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            if distances[i, j] > self.max_distance:
                break

            pid = existing_ids[i]
            self.people[pid].bbox = det_boxes[j]
            self.people[pid].centroid = det_centroids[j]
            self.people[pid].confidence = det_conf[j]
            self.people[pid].last_seen = now

            matched_people.add(i)
            matched_det.add(j)
            distances[i, :] = np.inf
            distances[:, j] = np.inf

        for j in range(len(det_boxes)):
            if j not in matched_det:
                self._register(det_boxes[j], det_centroids[j], det_conf[j], now)

        self._cleanup(now)
        return self.people

    def _register(self, box, cen, conf, now):
        pid = str(uuid.uuid4())
        self.people[pid] = Person(pid, box, cen, conf, now)

    def _cleanup(self, now):
        remove = [pid for pid, p in self.people.items() if now - p.last_seen > self.timeout]
        for pid in remove:
            del self.people[pid]

    def count(self) -> int:
        return len(self.people)


tracker = PeopleTracker(max_distance=70, timeout=1.2)


# =============================
# HEATMAP STATE
# =============================
# These will be initialized when first frame size is known
heatmap_accumulator = None  # float32 array same size as frame, accumulates counts
heatmap_decay = 0.98        # < 1.0 for slow fading over time
heatmap_scale_per_hit = 1.0 # how much to add for each person per frame


def update_heatmap(frame: np.ndarray, people: Dict[str, Person]) -> np.ndarray:
    """
    Update global heatmap_accumulator based on current detected people.
    Returns a color heatmap image aligned with frame.
    """
    global heatmap_accumulator

    h, w = frame.shape[:2]

    if heatmap_accumulator is None or heatmap_accumulator.shape[:2] != (h, w):
        heatmap_accumulator = np.zeros((h, w), dtype=np.float32)

    # Decay old heat so the map slowly fades if area is empty
    heatmap_accumulator *= heatmap_decay

    # Add heat where people are detected: use bounding boxes
    for p in people.values():
        x1, y1, x2, y2 = p.bbox
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 > x1 and y2 > y1:
            # add heat inside the box
            heatmap_accumulator[y1:y2, x1:x2] += heatmap_scale_per_hit

    # Normalize heatmap to 0-255 for visualization
    hm_norm = heatmap_accumulator.copy()
    if hm_norm.max() > 0:
        hm_norm = hm_norm / hm_norm.max()
    hm_norm = (hm_norm * 255).astype(np.uint8)

    # Apply color map (e.g. JET gives blue->red heatmap)
    heatmap_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)

    return heatmap_color


def overlay_heatmap(frame: np.ndarray, heatmap_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay colored heatmap on top of the original frame.
    alpha: transparency of heatmap (0=off, 1=full heatmap).
    """
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# =============================
# WEBSOCKET
# =============================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    last_time = time.time()

    try:
        while True:
            data = await ws.receive_json()
            frame_b64 = data.get("frame")
            if frame_b64 is None:
                continue

            frame_bytes = base64.b64decode(frame_b64)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # YOLO inference: persons only
            results = model(frame, classes=[0], conf=0.5, verbose=False)

            detections = []
            for r in results:
                if not hasattr(r, "boxes"):
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf = float(b.conf[0])
                    detections.append((x1, y1, x2, y2, conf))

            tracked = tracker.update(detections)

            # FPS
            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            # Draw bounding boxes + IDs on frame
            for p in tracked.values():
                x1, y1, x2, y2 = p.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    p.uuid[:8],
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # ==== HEATMAP UPDATE & OVERLAY ====
            heatmap_color = update_heatmap(frame, tracked)
            frame_with_heat = overlay_heatmap(frame, heatmap_color, alpha=0.5)

            # Encode both plain frame and heatmap overlay (you can use either on frontend)
            _, buf_plain = cv2.imencode(".jpg", frame)
            _, buf_heat = cv2.imencode(".jpg", frame_with_heat)

            frame_b64_plain = base64.b64encode(buf_plain).decode()
            frame_b64_heat = base64.b64encode(buf_heat).decode()

            await ws.send_json(
                {
                    "count": tracker.count(),
                    "fps": round(fps, 1),
                    "people": [asdict(p) for p in tracked.values()],
                    "frame": frame_b64_plain,        # original with boxes
                    "heat_frame": frame_b64_heat,    # frame + heatmap
                }
            )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WebSocket error:", e)


# =============================
# FRONTEND
# =============================
@app.get("/")
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
