# CPU image: serves the studio in preview mode.
# For real generations, run on a CUDA host, install requirements-gpu.txt
# (or base this on a pytorch/pytorch CUDA image) and start with --gpus all.
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV YARDI_HOST=0.0.0.0 YARDI_PORT=7860 YARDI_OUTPUT_DIR=/data
VOLUME /data
EXPOSE 7860
CMD ["python", "app.py"]
