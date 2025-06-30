FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

COPY src/ data/ .env /app/

EXPOSE 8501

CMD [ "streamlit", "run", "app.py" ]
