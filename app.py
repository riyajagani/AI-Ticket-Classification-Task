"""
app.py — Flask backend for the Ticket Classification and Auto-Response System.
"""

from flask import Flask, request, jsonify, render_template
from model import predict_ticket

app = Flask(__name__)


# -----------------------------------------------------------
# Route: Home page
# -----------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------
# Route: Predict ticket category and response
# -----------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validate input
    if not data or "ticket" not in data:
        return jsonify({"error": "Missing 'ticket' field in request body."}), 400

    user_input = data["ticket"].strip()

    if not user_input:
        return jsonify({"error": "Ticket text cannot be empty."}), 400

    # Call model
    result = predict_ticket(user_input)

    # Return category + response
    return jsonify({
        "category": result["category"],
        "response": result["response"]
    })


# -----------------------------------------------------------
# Run the app
# -----------------------------------------------------------

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
