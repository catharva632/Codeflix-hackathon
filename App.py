from flask import Flask, render_template, request, redirect, url_for
import json
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load pre-trained model
try:
    model = pickle.load(open("risk_model.pkl", "rb"))
except FileNotFoundError:
    model = None

# Define risk categories and questions
risk_categories = {
    "Security": [
        "Is the blockchain implementation resistant to known vulnerabilities?",
        "Is private key management secure?",
        "Are smart contracts audited for vulnerabilities?"
    ],
    "Compliance": [
        "Does the blockchain comply with legal regulations in your region?",
        "Is data storage compliant with GDPR or other privacy laws?"
    ],
    "Performance": [
        "Does the blockchain handle the required transaction volume?",
        "Are there delays in transaction finality?"
    ],
    "Scalability": [
        "Can the system scale as users grow?",
        "Are there bottlenecks in the blockchain architecture?"
    ]
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect form data
        responses = request.form.to_dict()
        scores = {category: [] for category in risk_categories.keys()}

        # Organize responses by category
        for key, value in responses.items():
            category, idx = key.split("_")
            scores[category].append(int(value))

        # Calculate average scores per category
        avg_scores = {category: sum(scores[category]) / len(scores[category]) for category in scores}
        overall_risk = sum(avg_scores.values()) / len(avg_scores)

        # Predict risk level using the ML model (if available)
        prediction = None
        if model:
            prediction = model.predict([[score for score in avg_scores.values()]])[0]

        # Save results to JSON
        results = {
            "avg_scores": avg_scores,
            "overall_risk": overall_risk,
            "prediction": prediction
        }
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)

        return redirect(url_for("results"))
    
    return render_template("index.html", risk_categories=risk_categories)

@app.route("/results")
def results():
    # Load the latest results
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        return "No results found. Please complete an assessment first."

    return render_template("results.html", results=results)

@app.route("/train", methods=["POST"])
def train():
    # Train the model on historical data
    try:
        data = pd.read_csv("risk_data.csv")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X, y)

        # Save the trained model
        pickle.dump(model, open("risk_model.pkl", "wb"))
        return "Model trained successfully!"
    except FileNotFoundError:
        return "Training data not found. Please upload 'risk_data.csv' file."

if __name__ == "__main__":
    app.run(debug=True)
