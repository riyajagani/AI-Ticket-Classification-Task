"""
model.py — Lightweight Ticket Classifier using TF-IDF + Logistic Regression.

No PyTorch. No sentence-transformers. Total size: ~5 MB.
Same predict_ticket() interface — drop-in replacement.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Training data
# ---------------------------------------------------------------------------

_TRAINING_DATA: list[tuple[str, str]] = [
    # Authentication
    ("I forgot my password",                         "Authentication"),
    ("Unable to login",                              "Authentication"),
    ("Password incorrect",                           "Authentication"),
    ("My account is locked",                         "Authentication"),
    ("How do I reset my password?",                  "Authentication"),
    ("I cannot sign in to my account",               "Authentication"),
    ("Login page keeps showing an error",            "Authentication"),
    ("I entered wrong password too many times",      "Authentication"),
    ("Two-factor authentication is not working",     "Authentication"),
    ("I did not receive the OTP on my phone",        "Authentication"),
    ("My session keeps expiring immediately",        "Authentication"),
    ("How do I change my current password?",         "Authentication"),
    ("Account credentials are not working",          "Authentication"),
    ("Sign in failed with invalid credentials",      "Authentication"),
    ("I am locked out of the system",                "Authentication"),

    # HR
    ("Check leave balance",                          "HR"),
    ("How many leaves do I have",                    "HR"),
    ("I want to apply for annual leave",             "HR"),
    ("Can I carry forward unused leaves?",           "HR"),
    ("What is the maternity leave policy?",          "HR"),
    ("How do I apply for sick leave?",               "HR"),
    ("I need to check my payslip",                   "HR"),
    ("When will my salary be credited this month?",  "HR"),
    ("How do I update my bank details for payroll?", "HR"),
    ("What are the office working hours?",           "HR"),
    ("I need a relieving letter from HR",            "HR"),
    ("How do I submit my medical reimbursement?",    "HR"),
    ("What is the leave encashment policy?",         "HR"),
    ("I need to apply for work from home",           "HR"),
    ("How many sick leaves am I entitled to?",       "HR"),

    # Technical Support
    ("The application is crashing on startup",       "Technical Support"),
    ("I am getting a 500 internal server error",     "Technical Support"),
    ("The website is loading very slowly",           "Technical Support"),
    ("My data is not syncing across devices",        "Technical Support"),
    ("The export feature is not working",            "Technical Support"),
    ("I cannot upload files to the portal",          "Technical Support"),
    ("The dashboard is showing incorrect data",      "Technical Support"),
    ("My notifications are not appearing",           "Technical Support"),
    ("The mobile app freezes on the home screen",    "Technical Support"),
    ("I cannot connect to the VPN from home",        "Technical Support"),
    ("The report is not generating correctly",       "Technical Support"),
    ("Software update failed with an error code",    "Technical Support"),
    ("The system is down and I cannot access it",    "Technical Support"),
    ("Pages are not loading in the browser",         "Technical Support"),
    ("I am getting a network connection error",      "Technical Support"),
]

_texts  = [t for t, _ in _TRAINING_DATA]
_labels = [l for _, l in _TRAINING_DATA]

# ---------------------------------------------------------------------------
# 2. Auto-response templates
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
# 3. Build and train pipeline (TF-IDF → Logistic Regression)
#    Runs ONCE at import time — zero per-request overhead.
# ---------------------------------------------------------------------------

_PIPELINE = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams for better coverage
        sublinear_tf=True,    # log-scale TF to reduce impact of very common terms
        min_df=1,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )),
])

_PIPELINE.fit(_texts, _labels)

# ---------------------------------------------------------------------------
# 4. Public prediction function  (same interface as before)
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
    category: str = _PIPELINE.predict([text])[0]
    response: str = _RESPONSES[category]

    return {
        "category": category,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Quick self-test  (python model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_tickets = [
        ("I forgot my password",                    "Authentication"),
        ("My account is locked out",                "Authentication"),
        ("OTP is not received on mobile",           "Authentication"),
        ("How many leaves are remaining?",          "HR"),
        ("I need to apply for sick leave",          "HR"),
        ("When will my salary be credited?",        "HR"),
        ("The app keeps crashing",                  "Technical Support"),
        ("I cannot connect to VPN",                 "Technical Support"),
        ("Website is down with a 500 error",        "Technical Support"),
        # Unseen edge cases
        ("Cannot sign in since yesterday",          "Authentication"),
        ("Need my payslip for this month",          "HR"),
        ("Dashboard data looks wrong",              "Technical Support"),
    ]

    correct = 0
    print(f"\n{'Ticket':<45}  {'Expected':<22}  {'Got':<22}  OK?")
    print("-" * 100)
    for ticket, expected in test_tickets:
        result  = predict_ticket(ticket)
        matched = result["category"] == expected
        correct += matched
        mark    = "✅" if matched else "❌"
        print(f"{ticket:<45}  {expected:<22}  {result['category']:<22}  {mark}")

    print(f"\nAccuracy: {correct}/{len(test_tickets)}")
