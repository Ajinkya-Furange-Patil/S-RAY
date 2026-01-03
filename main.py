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
app = FastAPI(title="Real-Time People Tracking System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model once at startup
# For BEST detection and still OK speed on decent GPU/CPU, use 'yolov8m.pt'
# model = YOLO("yolov8m.pt")
model = YOLO("yolov8n.pt")  # nano: fastest, still good for person detection


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
# TRACKER (Centroid-based)
# =============================
class PeopleTracker:
    def __init__(self, max_distance: float = 60.0, timeout: float = 1.0):
        """
        max_distance: max centroid distance (pixels) to associate detections
        timeout: seconds after which a lost person is removed
        """
        self.people: Dict[str, Person] = {}
        self.max_distance = max_distance
        self.timeout = timeout

    def _centroid(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> Dict[str, Person]:
        """
        detections: list of (x1, y1, x2, y2, conf)
        """
        now = time.time()

        if not detections:
            self._cleanup(now)
            return self.people

        det_boxes = [d[:4] for d in detections]
        det_centroids = [self._centroid(b) for b in det_boxes]
        det_conf = [d[4] for d in detections]

        # If no existing people, register all detections
        if not self.people:
            for box, cen, conf in zip(det_boxes, det_centroids, det_conf):
                self._register(box, cen, conf, now)
            return self.people

        existing_ids = list(self.people.keys())
        existing_centroids = [self.people[i].centroid for i in existing_ids]

        # Shape: [num_existing, num_detections]
        distances = cdist(existing_centroids, det_centroids)

        matched_det = set()
        matched_people = set()

        # Greedy association by nearest centroid
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

        # Register unmatched detections as new people
        for j in range(len(det_boxes)):
            if j not in matched_det:
                self._register(det_boxes[j], det_centroids[j], det_conf[j], now)

        # Cleanup lost people
        self._cleanup(now)
        return self.people

    def _register(self, box, cen, conf, now):
        pid = str(uuid.uuid4())
        self.people[pid] = Person(pid, box, cen, conf, now)

    def _cleanup(self, now):
        to_remove = [pid for pid, p in self.people.items() if now - p.last_seen > self.timeout]
        for pid in to_remove:
            del self.people[pid]

    def count(self) -> int:
        return len(self.people)


tracker = PeopleTracker(max_distance=70, timeout=1.2)


# =============================
# WEBSOCKET ENDPOINT
# =============================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    last_time = time.time()

    try:
        while True:
            # Receive frame from client
            data = await ws.receive_json()
            frame_b64 = data.get("frame")
            if frame_b64 is None:
                continue

            frame_bytes = base64.b64decode(frame_b64)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # YOLO inference: only 'person' class (0)
            results = model(
                frame,
                classes=[0],     # person only
                conf=0.5,        # min confidence for detection
                verbose=False
            )

            detections = []
            for r in results:
                if not hasattr(r, "boxes"):
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf = float(b.conf[0])
                    detections.append((x1, y1, x2, y2, conf))

            tracked = tracker.update(detections)

            # FPS calculation
            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            # Draw boxes + UUID on server-side frame (optional; not used on front-end here)
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

            # Optionally, you can send back the processed frame to overlay in browser
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64_out = base64.b64encode(buffer).decode()

            await ws.send_json(
                {
                    "count": tracker.count(),
                    "fps": round(fps, 1),
                    "people": [asdict(p) for p in tracked.values()],
                    "frame": frame_b64_out,
                }
            )
    except WebSocketDisconnect:
        # client disconnected
        pass
    except Exception as e:
        # avoid crash on unexpected errors
        print("WebSocket error:", e)


# =============================
# FRONTEND ROUTE
# =============================
@app.get("/")
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


# =============================
# RUN SERVER
# =============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
