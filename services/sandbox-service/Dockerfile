FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
COPY src/ src/
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 appuser --ingroup appgroup && \
    chown -R appuser:appgroup /app
# sandbox MUST run as root to access Docker socket for sandboxing.
# This is an accepted risk — the service is isolated and network-disabled.
# TODO: Replace with sysbox/gVisor for production.
EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
