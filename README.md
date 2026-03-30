# TicketAI — Ticket Classification and Auto-Response System

A lightweight, ML-powered support ticket classifier built with **sentence-transformers**, **scikit-learn**, and **Flask**.  
Submit any support ticket and receive an instant category, priority level, confidence score, and a professional draft response.

---

## Features

| Feature | Details |
|---|---|
| **Semantic classification** | `all-MiniLM-L6-v2` sentence-transformer embeddings + Logistic Regression |
| **6 categories** | Billing & Payments · Technical Support · Account Management · Feature Request · General Enquiry · Shipping & Delivery |
| **Auto-responses** | Per-category professional response templates |
| **Priority detection** | Keyword-based triage (High / Medium / Low) |
| **Confidence breakdown** | Probability scores for every category |
| **Premium UI** | Dark-mode SPA with animated bars, example pills, and copy-to-clipboard |
| **REST API** | `POST /classify` endpoint for programmatic use |

---

## Project Structure

```
ai-task/
├── app.py              # Flask backend & API routes
├── model.py            # ML logic (embedding, training, inference)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Single-page frontend
└── README.md
```

---

## Quick Start

### 1 — Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run.

### 3 — Run the application

```bash
python app.py
```

Open your browser at **http://localhost:5000**.

---

## API Reference

### `POST /classify`

Classify a support ticket and receive a full analysis.

**Request**

```json
{
  "ticket": "I was charged twice for my subscription this month."
}
```

**Response**

```json
{
  "category":       "Billing & Payments",
  "confidence":     0.9123,
  "confidence_pct": "91.2%",
  "priority":       "🟡 Low",
  "auto_response":  "Thank you for reaching out about a billing matter...",
  "all_scores": {
    "Billing & Payments":  0.9123,
    "Technical Support":   0.0312,
    "Account Management":  0.0201,
    "Feature Request":     0.0154,
    "General Enquiry":     0.0126,
    "Shipping & Delivery": 0.0084
  }
}
```

### `GET /health`

Returns `{ "status": "ok", "model_ready": true }`.

---

## How It Works

```
Ticket text
    │
    ▼
sentence-transformers (all-MiniLM-L6-v2)
    │  384-dimensional semantic embedding
    ▼
Logistic Regression classifier
    │  trained on 60 labelled examples (10 per category)
    ▼
Category + probability distribution
    │
    ▼
Priority (keyword heuristic) + Auto-response template
```

The classifier is trained **in-memory at startup** (< 2 seconds), so there are no external databases or model files to manage.

---

## Extending the System

- **Add categories** — extend `TRAINING_DATA` and `AUTO_RESPONSES` in `model.py`.  
- **More training data** — add more example strings per category to improve accuracy.  
- **Swap the classifier** — replace `LogisticRegression` with `SVC`, `RandomForestClassifier`, or a fine-tuned transformer.  
- **Persist the model** — use `joblib.dump` / `joblib.load` to save training time on restart.

---

## Requirements

- Python 3.10+
- Flask ≥ 3.0
- sentence-transformers ≥ 2.7
- scikit-learn ≥ 1.4
- numpy ≥ 1.26
