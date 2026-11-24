FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code, trained models and the chat page. No secrets in the image:
# pass GROQ_API_KEY / GEMINI_API_KEY and LLM_PROVIDER at run time.
COPY health_agent/ health_agent/
COPY models/ models/
COPY web/ web/
# Merged training CSVs (small) so admin uploads can retrain inside the container.
COPY data/ data/

RUN useradd --create-home appuser && chown -R appuser /app/models /app/data
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"
CMD ["sh", "-c", "uvicorn health_agent.server:app --host 0.0.0.0 --port ${PORT}"]
