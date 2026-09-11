FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dr_pipeline.py app.py ./
# Place exported ONNX at model/resnet18_dr.onnx (optional)
RUN mkdir -p model
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
