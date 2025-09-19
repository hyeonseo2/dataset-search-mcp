FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# copy metadata first (cache-friendly)
COPY pyproject.toml README.md ./
# copy package code
COPY src/ ./src/

# build & install
RUN pip install --no-cache-dir .

ENTRYPOINT ["dataset-search-mcp"]
