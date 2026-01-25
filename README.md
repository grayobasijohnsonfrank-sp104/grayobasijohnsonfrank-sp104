.

🚀 Quick Start
bash
# Clone and setup
git clone https://github.com/your-org/sp104-traffic-detection.git
cd yolov5
pip install -r requirements.txt

# Train model
python train.py --img 640 --batch 16 --epochs 50 --data ../data/dataset.yaml --weights yolov5s.pt

# Run inference
python detect.py --weights runs/train/exp/weights/best.pt --source traffic_video.mp4
📊 Project Overview
Goal: Build a real-time object detection system for traffic cameras using YOLOv5.

Key Features:

Detect vehicles and pedestrians in real-time

Fine-tuned YOLOv5 on traffic data

Achieve high mAP (>0.75) at 30+ FPS

📂 Dataset
Source: Road Sign Detection Dataset

Classes: Vehicles, Pedestrians, Traffic Signs, Bicycles

🛠️ Implementation
1. Data Preparation
Annotate traffic images

Split into train/val/test sets

Configure dataset.yaml

2. Training
Start from pretrained YOLOv5 weights

Fine-tune on traffic data

Optimize hyperparameters

3. Evaluation
Calculate mAP@0.5

Test inference speed (FPS)

Analyze per-class performance

4. Deployment
Real-time stream processing

Video file inference

Webcam demonstration

📈 Results
Target Metrics:

mAP@0.5: > 0.75

Inference Speed: 30+ FPS (GPU)

Model Size: < 50MB (optimized)

📁 Project Structure
text
data/           # Dataset and annotations
models/         # Trained weights
src/            # Source code
├── training/   # Training scripts
├── inference/  # Detection scripts
└── utils/      # Helper functions
results/        # Evaluation metrics
stream_app/     # Real-time demo
🔗 References
YOLOv5 Docs

PyTorch Tutorials

Custom YOLOv5 Training Guide

