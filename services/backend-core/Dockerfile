FROM python:3.12-slim AS base
WORKDIR /app

# System dependencies for unstructured document parser (PDF, DOCX, PPT)
#
# `tesseract-ocr` and `poppler-utils` are what `unstructured[pdf]` actually
# needs: poppler provides pdftoppm/pdfinfo (page rasterisation, page count) and
# tesseract performs the OCR fallback for PDFs whose text layer is missing or
# image-only. Without them, ANY PDF upload fails at parse time with
# TesseractNotFoundError / PDFPageCountError — the dependency is declared in
# pyproject.toml, but the binaries are system-level and were simply absent, so
# PDF support never worked in this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    pandoc \
    libreoffice-writer \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
COPY services/backend-core/requirements.txt .
# Large CUDA wheels (torch 2.13 deps) need generous HTTP timeouts/retries
RUN UV_HTTP_TIMEOUT=300 UV_RETRIES=8 uv pip install --system -r requirements.txt
COPY services/backend-core/src/ src/
# Seed data for Neo4j demo course — loaded automatically at startup
COPY scripts/db/seed/ scripts/db/seed/
# One-off KP backfill utility
COPY scripts/db/backfill_resource_kp.py scripts/db/backfill_resource_kp.py
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 appuser --ingroup appgroup && \
    chown -R appuser:appgroup /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
