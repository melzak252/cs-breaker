FROM python:3.12-slim
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt \
    && useradd --uid 1000 --create-home research \
    && mkdir -p /app/data/app \
    && chown -R research:research /app/data
COPY cs_app ./cs_app
COPY scripts/collect_cs_oddspapi.py scripts/map_cs_oddspapi_history.py ./scripts/
USER research
EXPOSE 8017
CMD ["python", "-m", "cs_app", "serve", "--host", "0.0.0.0", "--port", "8017"]
