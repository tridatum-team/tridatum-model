# syntax=docker/dockerfile:1
# On-prem app image. Build context: repository root (deploy/onprem/wheels is
# prepared by build_bundle.sh). No torch/numpy anywhere in this image.
FROM python:3.12.8-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

COPY deploy/onprem/wheels /tmp/wheels
COPY deploy/onprem/requirements-app.txt /tmp/requirements-app.txt
RUN pip install --no-index --no-deps --require-hashes --find-links /tmp/wheels -r /tmp/requirements-app.txt \
    && rm -rf /tmp/wheels /tmp/requirements-app.txt

# The PDF worker runs as `python -I -m trid.pdf_extract_worker`, so the trimmed
# package must live in site-packages, not only in the working directory.
COPY trid/__init__.py trid/pdf_documents.py trid/pdf_isolated.py trid/pdf_extract_worker.py \
     trid/document_case.py trid/llama_client.py trid/prompts.py trid/case_prompts.py \
     /tmp/trid_src/trid/
COPY trid/onprem/ /tmp/trid_src/trid/onprem/
COPY trid/onprem_ui/ /tmp/trid_src/trid/onprem_ui/
RUN set -eu; \
    site="$(python -c 'import site;print(site.getsitepackages()[0])')"; \
    cp -r /tmp/trid_src/trid "$site/trid"; \
    rm -rf /tmp/trid_src; \
    find "$site/trid" -name '__pycache__' -type d -prune -exec rm -rf {} +; \
    python -I -c "import trid.pdf_extract_worker, trid.onprem.server, trid.onprem.__main__"; \
    if python -I -c "import torch" 2>/dev/null; then echo "torch must not be installed" >&2; exit 1; fi; \
    if python -I -c "import numpy" 2>/dev/null; then echo "numpy must not be installed" >&2; exit 1; fi

COPY runs/reader-factorial-01-cpt-v1-v2-v4/tokenizer.json /opt/trid/tokenizer.json
RUN echo "73551e65acfa80834c235602900f31dc353ce230bfb69939ffdf44ca26ebc58d  /opt/trid/tokenizer.json" | sha256sum -c -

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin trid \
    && mkdir /data && chown 10001:10001 /data
USER 10001:10001
WORKDIR /
VOLUME /data
EXPOSE 8080
CMD ["python", "-I", "-m", "trid.onprem", "serve", "--config", "/data/onprem.json"]
