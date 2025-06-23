FROM python:3.10-slim

WORKDIR /app

# Copy application files
COPY model_loader.py inference.py serve.py requirements.txt /app/
COPY models /app/models/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# --- This line is correct for Gunicorn ---
CMD ["gunicorn", "--bind=0.0.0.0:8080", "serve:app"]