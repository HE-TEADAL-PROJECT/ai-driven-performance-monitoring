# Usa Python base image
FROM python:3.10

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file dell'applicazione
COPY app.py /app/
COPY requirements.txt /app/
COPY log.txt /app/
# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta di Streamlit
EXPOSE 8501

# Comando per avviare Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=debug"]

