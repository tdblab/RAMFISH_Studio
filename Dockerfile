# RAMFISH Studio container (Java included for Bio-Formats).
#   docker build -t ramfish-studio .
#   docker run -p 8501:8501 ramfish-studio
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 JAVA_HOME=/usr/lib/jvm/default-java
RUN apt-get update && apt-get install -y --no-install-recommends default-jdk-headless maven libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt bioformats_jar
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
