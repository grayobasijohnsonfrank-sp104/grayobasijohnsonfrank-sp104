import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'yolov5'))

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import tempfile

app = FastAPI()

# Allow GitHub Pages frontend to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup
model = torch.hub.load(
    os.path.join(os.path.dirname(__file__), '..'),
    'custom',
    path=os.path.join(os.path.dirname(__file__), 'best.pt'),
    source='local'
)
model.conf = 0.25


@app.get("/")
def health_check():
    return {"status": "running"}


# ── Image detection ──────────────────────────────────────────
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)
    results.render()

    annotated = Image.fromarray(results.ims[0])
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/jpeg")


# ── Video detection ──────────────────────────────────────────
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(contents)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".mp4", "_out.mp4")

    cap = cv2.VideoCapture(tmp_in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(tmp_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results.render()
        out.write(results.ims[0])

    cap.release()
    out.release()
    os.unlink(tmp_in_path)

    return StreamingResponse(open(tmp_out_path, "rb"), media_type="video/mp4")


# ── Webcam / live stream via WebSocket ───────────────────────
@app.websocket("/detect/stream")
async def detect_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive base64 encoded frame from frontend
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)

            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            results = model(frame)
            results.render()

            _, buffer = cv2.imencode('.jpg', results.ims[0])
            encoded = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(encoded)
    except Exception:
        await websocket.close()