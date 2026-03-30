"""
model.py — Ticket Classification using SentenceTransformer + Logistic Regression.

The model is loaded and trained ONCE at module import time so every Flask
request reuses the same in-memory objects with zero startup overhead.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# 1. Load the sentence-transformer model (once, at import time)
# ---------------------------------------------------------------------------

_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# 2. Training data  — label : [example tickets]
# ---------------------------------------------------------------------------

_TRAINING_DATA: dict[str, list[str]] = {
    "Authentication": [
        "I forgot my password",
        "Unable to login",
        "Password incorrect",
        "My account is locked",
        "How do I reset my password?",
        "I cannot sign in to my account",
        "Login page keeps showing an error",
        "I entered the wrong password too many times",
        "Two-factor authentication is not working",
        "I did not receive the OTP on my phone",
        "My session keeps expiring immediately",
        "How do I change my current password?",
    ],
    "HR": [
        "Check leave balance",
        "How many leaves do I have",
        "I want to apply for annual leave",
        "Can I carry forward unused leaves?",
        "What is the maternity leave policy?",
        "How do I apply for sick leave?",
        "I need to check my payslip",
        "When will my salary be credited this month?",
        "How do I update my bank account details for payroll?",
        "What are the office working hours?",
        "I need a relieving letter from HR",
        "How do I submit my medical reimbursement claim?",
    ],
    "Technical Support": [
        "The application is crashing on startup",
        "I am getting a 500 internal server error",
        "The website is loading very slowly",
        "My data is not syncing across devices",
        "The export feature is not working",
        "I cannot upload files to the portal",
        "The dashboard is showing incorrect data",
        "My notifications are not appearing",
        "The mobile app freezes on the home screen",
        "I cannot connect to the VPN from home",
        "The report is not generating correctly",
        "Software update failed with an error code",
    ],
}

# ---------------------------------------------------------------------------
# 3. Auto-response templates per category
# ---------------------------------------------------------------------------

_RESPONSES: dict[str, str] = {
    "Authentication": (
        "Please use the 'Forgot Password' option on the login page to reset "
        "your credentials. If the issue persists, contact IT support with your "
        "employee ID."
    ),
    "HR": (
        "Please check the HR portal for leave balances, payslips, and policy "
        "details. For further assistance, raise a request through the HR helpdesk."
    ),
    "Technical Support": (
        "Our technical team has been notified. Please share your device type, "
        "OS version, and any error messages to help us resolve the issue faster. "
        "We aim to respond within 4 hours."
    ),
}

# ---------------------------------------------------------------------------
# 4. Prepare training data — texts + labels
# ---------------------------------------------------------------------------

_texts:  list[str] = []
_labels: list[str] = []

for category, tickets in _TRAINING_DATA.items():
    for ticket in tickets:
        _texts.append(ticket)
        _labels.append(category)

# ---------------------------------------------------------------------------
# 5. Encode texts into embeddings
# ---------------------------------------------------------------------------

_EMBEDDINGS: np.ndarray = _EMBED_MODEL.encode(_texts, show_progress_bar=False)

# ---------------------------------------------------------------------------
# 6. Train Logistic Regression classifier
# ---------------------------------------------------------------------------

_LABEL_ENC = LabelEncoder()
_ENCODED_LABELS = _LABEL_ENC.fit_transform(_labels)

_CLASSIFIER = LogisticRegression(
    max_iter=1000,
    C=5.0,
    solver="lbfgs",
    multi_class="multinomial",
    random_state=42,
)
_CLASSIFIER.fit(_EMBEDDINGS, _ENCODED_LABELS)

# ---------------------------------------------------------------------------
# 7. Public prediction function  (same interface as before)
# ---------------------------------------------------------------------------

def predict_ticket(text: str) -> dict[str, str]:
    """
    Classify a support ticket and return a category + auto-response.

    Parameters
    ----------
    text : str
        Raw ticket text submitted by the user.

    Returns
    -------
    dict with keys:
        "category" — one of: Authentication | HR | Technical Support
        "response" — suggested auto-response string
    """
    embedding: np.ndarray = _EMBED_MODEL.encode([text], show_progress_bar=False)
    predicted_index: int  = int(_CLASSIFIER.predict(embedding)[0])
    category: str         = _LABEL_ENC.inverse_transform([predicted_index])[0]
    response: str         = _RESPONSES[category]

    return {
        "category": category,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Quick self-test  (python model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_tickets = [
        # Authentication
        "I forgot my password",
        "My account is locked out",
        "OTP is not received on mobile",
        # HR
        "How many leaves are remaining?",
        "I need to apply for sick leave",
        "When will my salary be credited?",
        # Technical Support
        "The app keeps crashing",
        "I cannot connect to VPN",
        "Website is down with a 500 error",
        # Unseen / edge cases
        "Cannot sign in since yesterday",
        "Need my payslip for this month",
        "Dashboard data is incorrect",
    ]

    print(f"\n{'Ticket':<45}  {'Category':<22}  Response (truncated)")
    print("-" * 100)
    for ticket in test_tickets:
        result = predict_ticket(ticket)
        truncated = result["response"][:55] + "..."
        print(f"{ticket:<45}  {result['category']:<22}  {truncated}")
