# SP-104 — Real-Time Object Detection for Traffic Cameras

**Course:** 4850 Senior Project | Spring 2026  
**Team:** Zachary Gray, Bregan Frank, Jeyden Johnson, David Obasi  
**Project Site:** https://grayobasijohnsonfrank-sp104.github.io/grayobasijohnsonfrank-sp104/

---

## Overview

A YOLOv5-based object detection system trained to identify traffic-related objects from images. The trained model is served through a FastAPI backend hosted on Hugging Face Spaces and accessible via a GitHub Pages frontend.

**Detectable Classes:**

| ID | Class |
|----|-------|
| 0 | Human |
| 1 | Car |
| 2 | Truck |
| 3 | Regulatory Sign |
| 4 | Stop Sign |
| 5 | Warning Sign |

**Model Performance:** mAP@0.5 ≈ 0.85

---

## Repository Structure

```
grayobasijohnsonfrank-sp104/
├── index.html              # GitHub Pages frontend
├── yolov5/                 # YOLOv5 source (detached from upstream)
│   ├── backend/
│   │   ├── main.py         # FastAPI server
│   │   ├── best.pt         # Trained model weights
│   │   └── requirements.txt
│   ├── models/
│   ├── utils/
│   ├── train.py
│   ├── detect.py
│   └── data/
│       └── data.yaml
└── assets/
```

---

## Requirements

- Windows 10/11 (tested), macOS/Linux compatible
- NVIDIA GPU with CUDA support (RTX 3050 Ti or better recommended)
- Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Python 3.11

---

## Local Setup

**1. Clone the repository:**
```bash
git clone https://github.com/grayobasijohnsonfrank-sp104/grayobasijohnsonfrank-sp104
cd grayobasijohnsonfrank-sp104
```

**2. Create and activate the Conda environment:**
```bash
conda create --name trafficv5 python=3.11
conda activate trafficv5
```

**3. Install dependencies:**
```bash
cd yolov5
pip install -r requirements.txt
```

**4. Install PyTorch with CUDA support:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**5. Install full OpenCV (required for webcam/display):**
```bash
pip uninstall opencv-python-headless -y
pip install opencv-python
```

**6. Verify GPU is available:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Should return `True`. If not, repeat step 4.

---

## Running Detection (DISCLAIMER: VIDEO DETECTION HAS FLASHING LIGHTS AND COLORS. IF YOU HAVE COGNITIVE DISABILITY SUCH AS EPILEPSY PLEASE BE ADVISED)

All commands must be run from inside the `yolov5/` folder with the `trafficv5` environment activated.

**Single image:**
```bash
python detect.py --weights yolov5/backend/best.pt --img 640 --conf 0.25 --save-conf --source "path/to/image.jpg"
```

**Folder of images:**
```bash
python detect.py --weights yolov5/backend/best.pt --img 640 --conf 0.25 --save-conf --source "path/to/folder/"
```

**Video file:**
```bash
python detect.py --weights yolov5/backend/best.pt --img 640 --conf 0.25 --save-conf --source "path/to/video.mp4"
```

**Webcam:**
```bash
python detect.py --weights yolov5/backend/best.pt --img 640 --conf 0.25 --source 0
```

Results are saved to `runs/detect/expX/`.

To lower the confidence threshold and catch more detections:
```bash
--conf 0.1
```

---

## Training the Model

**Dataset configuration (`data/data.yaml`):**
```yaml
train: "path/to/data/train"
val: "path/to/data/validation"

nc: 6
names: ['human', 'car', 'truck', 'regulatory', 'stop', 'warning']
```

**Training command:**
```bash
python train.py --img 640 --batch 16 --epochs 10 --data data/data.yaml --weights yolov5s.pt --name my_run --device 0
```

Trained weights are saved to `runs/train/my_run/weights/best.pt`.

---

## Web Deployment

The detection endpoint is hosted on Hugging Face Spaces using Docker:

**Backend URL:** `https://sp104greentraff-traffic-detection-api.hf.space`

**Available endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/detect/image` | POST | Upload image, returns annotated result |
| `/docs` | GET | Interactive Swagger UI |

The GitHub Pages frontend sends images to the backend via `fetch()` and displays the annotated result with bounding boxes and confidence scores.

---

## Constraints

- Detection accuracy degrades in low-light, rain, or heavy occlusion
- Webcam detection requires full `opencv-python` (not headless)
- GPU strongly recommended for training — CPU training is significantly slower
- Hugging Face free tier uses CPU inference — image results may take a few seconds
- The Hugging Face Space may sleep after 15 minutes of inactivity — first request after sleep takes 30–60 seconds to wake up
- Model was trained on North American road conditions and signs

---

## Dataset Sources

- **COCO 2017** — human, car, truck classes (downloaded via FiftyOne)
- **US Road Signs v72** — regulatory, stop, warning sign classes (Roboflow)
- Total training images: ~1,200

---

## Links

- **Project Website:** https://grayobasijohnsonfrank-sp104.github.io/grayobasijohnsonfrank-sp104/
- **GitHub Repository:** https://github.com/grayobasijohnsonfrank-sp104/grayobasijohnsonfrank-sp104
- **Hugging Face Space:** https://huggingface.co/spaces/sp104greentraff/traffic-detection-api