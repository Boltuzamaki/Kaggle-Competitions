FROM python:3.11-slim
WORKDIR /app
# system libs matplotlib/onnxruntime may need
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py graph_compiler.py .
COPY templates ./templates
ENV PROJECT_DIR=/project
EXPOSE 5000
CMD ["python", "app.py"]
