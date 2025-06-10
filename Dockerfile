FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including FLAC
RUN apt-get update && apt-get install -y \
    flac \
    portaudio19-dev \
    python3-pyaudio \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .
RUN mkdir -p logs

EXPOSE 8501

CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]