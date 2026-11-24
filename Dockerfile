FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY main.py .
COPY data/raw/ data/raw/

# Bake cleaned universe, strict OOF shadow ratings, validation metrics, and fitted models at build time
RUN python -m src.train

EXPOSE 8080

CMD ["sh", "-c", "exec streamlit run main.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
