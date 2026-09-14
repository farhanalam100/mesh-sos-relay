# Mesh-Net SOS Relay

### **When the towers go down, we don't lose the network. We become it.**

An offline disaster-response prototype that combines **multi-hop mesh communication** with **local ML-based SOS triage**.

Built for **Hyperbloom September — AI/ML Hackathon**.

---

## The problem

In a disaster, communication infrastructure can become part of the problem.

A conventional SOS depends on a chain like:

```text
Person → Cell Tower → Internet → Emergency Service
```

If the tower or internet connection goes down, that chain breaks.

But even when emergency messages manage to get through, responders can face another problem: **too many messages and too little time**.

Mesh-Net explores both problems together.

---

## The idea

Instead of requiring every device to reach a central server directly, nearby devices can relay messages for one another.

```text
                     ┌─────────┐
                     │  Node A │
                     └────┬────┘
                          │
                     ┌────▼────┐
                     │  Node B │
                     └────┬────┘
                          │
                     ┌────▼────┐
                     │  Node C │
                     └────┬────┘
                          │
                          ▼
                     ┌─────────┐
                     │ Gateway │
                     └────┬────┘
                          │
                          ▼
                     ┌─────────┐
                     │   ML    │
                     │  Triage │
                     └────┬────┘
                          │
                          ▼
                     Responder
```

An SOS can therefore travel **hop-by-hop through the mesh** rather than depending on a direct connection.

Once it reaches the gateway, the message is passed to the local ML classifier.

---

# What happens to an SOS?

### 1. A person sends an SOS

For example:

```text
"My father is injured and we cannot leave the building."
```

### 2. The message enters the mesh

The nearest node attempts to forward it to neighbouring nodes.

The current simulation accounts for:

* multi-hop forwarding
* TTL limits
* transmission loss

### 3. The message reaches the gateway

The gateway receives messages that successfully make it through the simulated network.

### 4. Local ML classifies it

The classifier uses **TF-IDF + a trained classifier** to estimate:

```text
Category
Severity
Confidence
```

### 5. Responders see the priority

Messages are surfaced through the live dashboard so urgent cases can be dealt with first.

---

# Why add ML?

Getting the message through is only one part of emergency communication.

Imagine a responder receiving:

```text
"Need help."

"Water is entering our house."

"My father is injured and cannot walk."

"We are trapped on the second floor."

"Does anyone know when the road will reopen?"
```

The system shouldn't pretend that every message has the same urgency.

The ML layer provides an initial triage signal so responders can see **what needs attention first**.

It is deliberately designed as **decision support**, not as a replacement for human judgment.

Low-confidence predictions can be flagged for review.

---

# Why run the model locally?

The problem we're trying to solve is partly a connectivity problem.

Relying on a remote AI API would introduce another dependency:

```text
SOS
 ↓
Internet
 ↓
Cloud AI
```

Instead, the current classifier runs locally:

```text
SOS
 ↓
Local TF-IDF
 ↓
Local classifier
 ↓
Triage result
```

No external AI API is required during inference.

The current model is intentionally lightweight and serves as a baseline for future edge deployment.

---

# Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    MESH LAYER                      │
│                                                     │
│   Node ─── Node ─── Node                            │
│     │        │        │                             │
│     └────────┴────────┘                             │
│              │                                      │
│              ▼                                      │
│           Gateway                                   │
└──────────────┬──────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────┐
│       ML TRIAGE LAYER       │
│                             │
│  SOS Text                   │
│      ↓                      │
│  TF-IDF                     │
│      ↓                      │
│  Classifier                 │
│      ↓                      │
│  Category / Severity /      │
│  Confidence                 │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      RESPONDER DASHBOARD    │
│                             │
│  Mesh activity              │
│  Incoming SOS               │
│  Triage results             │
│  Severity                   │
└─────────────────────────────┘
```

---

# Technical implementation

### Mesh simulation

`src/node.py` and `src/network.py` handle the simulated mesh.

Messages are propagated between nodes using hop-by-hop forwarding.

The network includes:

**TTL**

Prevents messages from propagating indefinitely.

**Transmission loss**

Introduces unreliable links into the simulation instead of assuming perfect delivery.

This makes it possible to observe how messages behave under less-than-ideal network conditions.

---

### ML classifier

The current pipeline is:

```text
SOS message
      ↓
Text preprocessing
      ↓
TF-IDF vectorization
      ↓
Classifier
      ↓
Category
Severity
Confidence
```

Training data:

```text
data/sos_messages.csv
```

Dataset generation:

```bash
python3 scripts/generate_dataset.py
```

Model training:

```bash
python3 scripts/train_classifier_tfidf.py
```

The trained model is stored locally under `models/`.

---

### Real-time dashboard

The frontend is connected to the backend through **WebSockets**.

This allows the dashboard to receive simulation events as they happen rather than waiting for the simulation to finish.

The dashboard exposes the complete flow:

```text
Node
 ↓
Relay
 ↓
Gateway
 ↓
Classifier
 ↓
Triage feed
```

---

# Repository structure

```text
mesh-sos-relay/
│
├── backend/
│   └── app.py
│
├── data/
│   └── sos_messages.csv
│
├── frontend/
│
├── models/
│
├── scripts/
│   ├── generate_dataset.py
│   └── train_classifier_tfidf.py
│
├── src/
│   ├── node.py
│   ├── network.py
│   └── classifier.py
│
├── tests/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Run locally

### Clone

```bash
git clone https://github.com/farhanalam100/mesh-sos-relay.git
cd mesh-sos-relay
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the server

```bash
uvicorn backend.app:app --reload --port 8000
```

Open:

```text
http://localhost:8000
```

and start the disaster simulation.

---

# Docker

```bash
docker build -t mesh-sos-relay .
docker run -p 8000:8000 mesh-sos-relay
```

Then open:

```text
http://localhost:8000
```

---

# Project status

This repository is currently a **working simulation / proof of concept**.

The mesh network is simulated rather than running over physical radio hardware, and the ML model is a lightweight baseline rather than a production emergency-response model.

That distinction matters.

The purpose of this project is to demonstrate the architecture and explore how **resilient communication and local intelligence can work together under infrastructure failure**.

---

# What comes next

The next step is moving the simulation closer to physical deployment.

### Hardware mesh

Run nodes on low-power devices and replace simulated links with actual device-to-device communication.

### Better routing

Move beyond basic message flooding toward more efficient routing and store-and-forward mechanisms.

### Edge ML

Benchmark smaller models on constrained hardware and compare accuracy, latency and memory usage.

### Network metrics

Measure:

* delivery rate
* message latency
* hop count
* packet loss
* network recovery
* node availability

### Security

Add node authentication, message integrity and encrypted communication before considering real-world deployment.

---

# The bigger idea

Most emergency communication systems assume the network is already there.

Mesh-Net starts with the opposite assumption:

**What if it isn't?**

Instead of making the message depend on the network,

**the network is built around the message.**

And once the message arrives, local ML helps determine where human attention may be needed first.

---

## Built for Hyperbloom September

**Mesh-Net SOS Relay**

*When the towers go down, we don't lose the network. We become it.*

**Farhan Alam**

[GitHub](https://github.com/farhanalam100)
