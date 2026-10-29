# StroboPhone web app (Hugging Face Docker Space; also runs anywhere Docker does).
FROM python:3.13-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user strobophone strobophone
COPY --chown=user app app
EXPOSE 7860
# MODEL_API_KEY comes from the Space's secrets; MAX_JOBS_PER_DAY caps API spend.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
