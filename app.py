"""
Real-Time People Detection and Tracking System
Uses FastAPI + WebSockets + OpenCV + YOLO
Assigns UUID to each detected person and tracks count
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import asyncio
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from scipy.spatial import distance
import base64

app = FastAPI(title="People Tracking System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO('yolov8n.pt')  # Using YOLOv8 nano for speed

@dataclass
class Person:
    uuid: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: Tuple[int, int]
    confidence: float
    last_seen: int

class PeopleTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        self.people: Dict[str, Person] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.frame_count = 0
        
    def update(self, detections: List[Tuple]) -> Dict[str, Person]:
        """Update tracker with new detections"""
        self.frame_count += 1
        
        if len(detections) == 0:
            # Mark all people as disappeared
            to_remove = []
            for person_id, person in self.people.items():
                if self.frame_count - person.last_seen > self.max_disappeared:
                    to_remove.append(person_id)
            for pid in to_remove:
                del self.people[pid]
            return self.people
        
        # Calculate centroids of new detections
        new_centroids = []
        for (x1, y1, x2, y2, conf) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            new_centroids.append(((x1, y1, x2, y2), (cx, cy), conf))
        
        if len(self.people) == 0:
            # Register all new people
            for bbox, centroid, conf in new_centroids:
                person_id = str(uuid.uuid4())
                self.people[person_id] = Person(
                    uuid=person_id,
                    bbox=bbox,
                    centroid=centroid,
                    confidence=conf,
                    last_seen=self.frame_count
                )
        else:
            # Match existing people with new detections
            existing_centroids = [(pid, p.centroid) for pid, p in self.people.items()]
            
            # Calculate distance matrix
            if len(existing_centroids) > 0 and len(new_centroids) > 0:
                dist_matrix = np.zeros((len(existing_centroids), len(new_centroids)))
                for i, (_, ec) in enumerate(existing_centroids):
                    for j, (_, nc, _) in enumerate(new_centroids):
                        dist_matrix[i, j] = distance.euclidean(ec, nc)
                
                # Match using minimum distance
                matched_existing = set()
                matched_new = set()
                
                for _ in range(min(len(existing_centroids), len(new_centroids))):
                    min_idx = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
                    i, j = min_idx
                    
                    if dist_matrix[i, j] < self.max_distance:
                        person_id = existing_centroids[i][0]
                        bbox, centroid, conf = new_centroids[j]
                        
                        self.people[person_id].bbox = bbox
                        self.people[person_id].centroid = centroid
                        self.people[person_id].confidence = conf
                        self.people[person_id].last_seen = self.frame_count
                        
                        matched_existing.add(i)
                        matched_new.add(j)
                        dist_matrix[i, :] = np.inf
                        dist_matrix[:, j] = np.inf
                
                # Register new unmatched detections
                for j, (bbox, centroid, conf) in enumerate(new_centroids):
                    if j not in matched_new:
                        person_id = str(uuid.uuid4())
                        self.people[person_id] = Person(
                            uuid=person_id,
                            bbox=bbox,
                            centroid=centroid,
                            confidence=conf,
                            last_seen=self.frame_count
                        )
                
                # Remove disappeared people
                to_remove = []
                for i, (person_id, _) in enumerate(existing_centroids):
                    if i not in matched_existing:
                        if self.frame_count - self.people[person_id].last_seen > self.max_disappeared:
                            to_remove.append(person_id)
                
                for pid in to_remove:
                    del self.people[pid]
        
        return self.people
    
    def get_count(self) -> int:
        """Get current people count"""
        return len(self.people)

# Global tracker instance
tracker = PeopleTracker()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/")
async def get():
    """Serve the frontend HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>People Tracking System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: #1a1a1a;
                color: #fff;
            }
            h1 {
                text-align: center;
                color: #4CAF50;
            }
            .container {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
            }
            .video-section {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
            }
            .stats-section {
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
            }
            #videoCanvas {
                width: 100%;
                border: 2px solid #4CAF50;
                border-radius: 5px;
            }
            .stat-box {
                background: #3a3a3a;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }
            .stat-label {
                font-size: 14px;
                color: #aaa;
            }
            .stat-value {
                font-size: 32px;
                font-weight: bold;
                color: #4CAF50;
            }
            .person-list {
                max-height: 400px;
                overflow-y: auto;
                margin-top: 20px;
            }
            .person-item {
                background: #3a3a3a;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                font-size: 12px;
            }
            button {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px 5px;
            }
            button:hover {
                background: #45a049;
            }
            button:disabled {
                background: #666;
                cursor: not-allowed;
            }
            .status {
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                text-align: center;
            }
            .status.connected {
                background: #4CAF50;
            }
            .status.disconnected {
                background: #f44336;
            }
        </style>
    </head>
    <body>
        <h1>🎥 Real-Time People Detection & Tracking</h1>
        
        <div id="status" class="status disconnected">Disconnected</div>
        
        <div style="text-align: center;">
            <button id="startBtn" onclick="startCamera()">Start Camera</button>
            <button id="stopBtn" onclick="stopCamera()" disabled>Stop Camera</button>
        </div>
        
        <div class="container">
            <div class="video-section">
                <canvas id="videoCanvas"></canvas>
                <video id="video" style="display:none;" autoplay></video>
            </div>
            
            <div class="stats-section">
                <div class="stat-box">
                    <div class="stat-label">Total People Detected</div>
                    <div class="stat-value" id="peopleCount">0</div>
                </div>
                
                <div class="stat-box">
                    <div class="stat-label">FPS</div>
                    <div class="stat-value" id="fps">0</div>
                </div>
                
                <h3>Active People</h3>
                <div class="person-list" id="personList"></div>
            </div>
        </div>

        <script>
            let ws = null;
            let video = document.getElementById('video');
            let canvas = document.getElementById('videoCanvas');
            let ctx = canvas.getContext('2d');
            let stream = null;
            let animationId = null;
            
            // Connect to WebSocket
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = () => {
                    document.getElementById('status').className = 'status connected';
                    document.getElementById('status').textContent = 'Connected';
                };
                
                ws.onclose = () => {
                    document.getElementById('status').className = 'status disconnected';
                    document.getElementById('status').textContent = 'Disconnected';
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    // Update stats
                    document.getElementById('peopleCount').textContent = data.count;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    
                    // Update person list
                    const personList = document.getElementById('personList');
                    personList.innerHTML = '';
                    data.people.forEach(person => {
                        const div = document.createElement('div');
                        div.className = 'person-item';
                        div.innerHTML = `
                            <strong>ID:</strong> ${person.uuid.substring(0, 8)}...<br>
                            <strong>Confidence:</strong> ${(person.confidence * 100).toFixed(1)}%<br>
                            <strong>Position:</strong> (${person.centroid[0]}, ${person.centroid[1]})
                        `;
                        personList.appendChild(div);
                    });
                    
                    // Draw detections
                    if (data.frame) {
                        const img = new Image();
                        img.onload = () => {
                            canvas.width = img.width;
                            canvas.height = img.height;
                            ctx.drawImage(img, 0, 0);
                            
                            // Draw bounding boxes
                            data.people.forEach(person => {
                                const [x1, y1, x2, y2] = person.bbox;
                                ctx.strokeStyle = '#4CAF50';
                                ctx.lineWidth = 3;
                                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                                
                                // Draw label
                                ctx.fillStyle = '#4CAF50';
                                ctx.fillRect(x1, y1 - 25, 150, 25);
                                ctx.fillStyle = '#fff';
                                ctx.font = '12px Arial';
                                ctx.fillText(`ID: ${person.uuid.substring(0, 8)}`, x1 + 5, y1 - 8);
                            });
                        };
                        img.src = 'data:image/jpeg;base64,' + data.frame;
                    }
                };
            }
            
            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 640, height: 480 } 
                    });
                    video.srcObject = stream;
                    
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    
                    connectWebSocket();
                    
                    // Send frames
                    video.onloadedmetadata = () => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        sendFrame();
                    };
                } catch (err) {
                    alert('Error accessing camera: ' + err.message);
                }
            }
            
            function sendFrame() {
                if (!stream) return;
                
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const frameData = canvas.toDataURL('image/jpeg', 0.8);
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ frame: frameData.split(',')[1] }));
                }
                
                animationId = requestAnimationFrame(sendFrame);
            }
            
            function stopCamera() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                if (animationId) {
                    cancelAnimationFrame(animationId);
                }
                if (ws) {
                    ws.close();
                }
                
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    await manager.connect(websocket)
    
    try:
        last_time = asyncio.get_event_loop().time()
        
        while True:
            # Receive frame from client
            data = await websocket.receive_json()
            
            if 'frame' in data:
                # Decode frame
                frame_data = base64.b64decode(data['frame'])
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Run YOLO detection
                results = model(frame, classes=[0], verbose=False)  # class 0 is 'person'
                
                # Extract person detections
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        if conf > 0.5:  # Confidence threshold
                            detections.append((int(x1), int(y1), int(x2), int(y2), conf))
                
                # Update tracker
                tracked_people = tracker.update(detections)
                
                # Calculate FPS
                current_time = asyncio.get_event_loop().time()
                fps = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
                last_time = current_time
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Prepare response
                response = {
                    'count': tracker.get_count(),
                    'fps': fps,
                    'people': [asdict(p) for p in tracked_people.values()],
                    'frame': frame_base64
                }
                
                # Send to client
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)

@app.get("/api/stats")
async def get_stats():
    """REST API endpoint to get current stats"""
    return {
        'total_people': tracker.get_count(),
        'people': [asdict(p) for p in tracker.people.values()]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)