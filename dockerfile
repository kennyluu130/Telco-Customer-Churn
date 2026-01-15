# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# Explicitly copy model (in case .dockerignore excluded mlruns)
# NOTE: destination changed to /app/src/serving/model to match inference.py's path
COPY src/serving/model /app/src/serving/model

# Copy MLflow run (artifacts + metadata) to the flat /app/model convenience path
COPY src/serving/model/m-9c85a6c242dd4d92816a9eac0fa568c7/model /app/model
COPY src/serving/model/m-9c85a6c242dd4d92816a9eac0fa568c7/artifacts/feature_columns.txt /app/model/feature_columns.txt
COPY src/serving/model/m-9c85a6c242dd4d92816a9eac0fa568c7/artifacts/preprocessing.pkl /app/model/preprocessing.pkl

# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).
# lets you import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1 \ 
    PYTHONPATH=/app/src

## 6. Expose FastAPI port only for local deployment
# EXPOSE 8000

# 7. Run the FastAPI app using uvicorn (change path if needed) can change $PORT to 8000 for local deployment
# CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD uvicorn src.app.main:app --host 0.0.0.0 --port $PORT