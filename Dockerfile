# SupportDesk AI — Customer Service Chatbot
# Build:  docker build -t supportdesk-ai .
# Run:    docker run -p 5000:5000 supportdesk-ai

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer across rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Train the intent classifier at build time so the image works standalone
# (models/intent_classifier.joblib is also tracked in git, but retraining
# here guarantees the model always matches the code that ships with it).
RUN python src/train_model.py

EXPOSE 5000

ENV PORT=5000
ENV FLASK_DEBUG=0

CMD ["python", "src/app.py"]
