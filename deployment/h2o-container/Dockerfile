FROM python:3.10-slim

WORKDIR /app

# Install OpenJDK 17 (compatible with H2O)
RUN apt-get update && apt-get install -y openjdk-17-jdk && apt-get clean

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Copy application files
COPY model_loader.py inference.py serve.py requirements.txt /app/
COPY models /app/models/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "serve.py"]