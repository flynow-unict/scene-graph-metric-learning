FROM python:3.10-slim

WORKDIR /app

COPY cloud/workerAI/requirements.txt .

# Installazione di librerie di base necessarie per compilazioni (es. gcc) e dipendenze Python
RUN apt-get update && apt-get install -y openssh-client gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Installa torch, torchvision e torch_geometric
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch_geometric

# Copia prima la cartella src/ nella working directory /app/src/
COPY src/ ./src/

# Copia i file del worker (es. worker.py, cluster_orchestrator.py, .env)
COPY cloud/workerAI/ ./

CMD ["python", "worker.py"]