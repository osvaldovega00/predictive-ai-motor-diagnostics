# Use a slim Python image based on Debian Bullseye
FROM python:3.9-slim-bullseye

# Set environment variables to prevent Python from writing .pyc files and to flush output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy your app code into the container
COPY . /app

# Install dependencies without caching
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit runs on port 8080 for Cloud Run
EXPOSE 8080

# Launch the Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]