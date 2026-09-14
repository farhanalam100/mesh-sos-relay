# Mesh-Net SOS Relay

An offline disaster-response mesh network with an AI triage layer, built for
Hyperbloom September (AI/ML hackathon).

## The problem
When disasters knock out cell towers and internet, people have no way to
call for help, and rescue teams have no way to tell who needs help most
urgently among the messages that do get through.

## The solution
- **Mesh routing** (`src/node.py`, `src/network.py`): nodes flood SOS
  messages hop-by-hop with TTL limits and simulated transmission loss,
  no central infrastructure required.
- **AI triage classifier** (`src/classifier.py`, `models/`): a TF-IDF +
  classifier-head model, trained on a labeled dataset of SOS messages
  (`data/sos_messages.csv`, `scripts/generate_dataset.py`), auto-scores
  every message that reaches the gateway by category and severity, and
  flags low-confidence reads for human review. Runs fully offline.
- **Live dashboard** (`frontend/`, `backend/app.py`): real-time mesh map
  and a triage feed sorted by urgency, streamed over WebSocket as the
  simulation runs.

## Running it
```
pip install -r requirements.txt
python3 scripts/generate_dataset.py     # regenerate the labeled dataset (optional, already included)
python3 scripts/train_classifier_tfidf.py   # retrain the classifier (optional, already included)
uvicorn backend.app:app --reload --port 8000
```
Open http://localhost:8000 and click "Run disaster simulation."

Notes and troubleshooting
- If you see scikit-learn unpickle warnings, install the pinned versions in `requirements.txt` to match the model used for training (scikit-learn==1.8.0).
- The backend supports a `stop` action from the UI; click the `Stop` button to interrupt a running simulation.
- To run in Docker:

```bash
docker build -t mesh-sos-relay .
docker run -p 8000:8000 mesh-sos-relay
```

