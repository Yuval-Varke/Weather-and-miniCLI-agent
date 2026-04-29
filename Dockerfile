FROM python:3.12-slim

WORKDIR /weather_agent

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only copy what you need
COPY app.py .
COPY main.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]