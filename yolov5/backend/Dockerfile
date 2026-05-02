FROM python:3.11-slim

WORKDIR /app

# Copy yolov5 core files
COPY ../models ./models
COPY ../utils ./utils
COPY ../hubconf.py .
COPY ../requirements.txt ./yolov5_requirements.txt

# Copy backend files
COPY main.py .
COPY best.pt .
COPY requirements .

# Install dependencies
RUN pip install --no-cache-dir -r requirements

# Expose port
EXPOSE 7860

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]