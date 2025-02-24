FROM python:3.10-slim

WORKDIR /app

# Copia e installa le dipendenze
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Cloniamo Lag-Llama e lo installiamo
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/time-series-foundation-models/lag-llama.git /app/lag-llama
RUN pip install -e /app/lag-llama  # Installazione locale in editable mode

# Copia il codice dell'app e il checkpoint
COPY . .
COPY /app/checkpoints/lag-llama.ckpt /app/checkpoints/lag-llama.ckpt

# Esponi la porta per Streamlit
EXPOSE 8504

# Avvio con uvicorn sulla porta 8504
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8504"]
