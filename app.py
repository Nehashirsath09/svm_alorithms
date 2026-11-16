import os
import pickle

import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# ----------------------------------------------------------------------
# Model loading (file must sit next to app.py; override with MODEL_PATH)
# ----------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "svm.pkl")
)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = ["Age", "Gender", "Region", "Occupation", "Income"]

# ----------------------------------------------------------------------
# EDIT THESE so the dropdown labels match how you encoded the data
# during training: {numeric_code: "Label shown to user"}
# ----------------------------------------------------------------------
GENDER_OPTIONS = {0: "Female", 1: "Male"}
REGION_OPTIONS = {0: "Region 0", 1: "Region 1", 2: "Region 2", 3: "Region 3"}
OCCUPATION_OPTIONS = {
    0: "Occupation 0", 1: "Occupation 1", 2: "Occupation 2", 3: "Occupation 3",
}

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Purchase Predictor</title>
<style>
  :root{
    --bg1:#1e1b4b; --bg2:#0f766e;
    --card:#ffffff; --ink:#1e293b; --muted:#64748b;
    --primary:#4f46e5; --primary-dark:#3730a3;
    --accent:#14b8a6; --yes:#059669; --yes-bg:#d1fae5;
    --no:#dc2626; --no-bg:#fee2e2; --line:#e2e8f0;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;
    min-height:100vh; color:var(--ink);
    background:linear-gradient(135deg,var(--bg1),var(--bg2));
    display:flex; align-items:center; justify-content:center; padding:24px;
  }
  .card{
    background:var(--card); width:100%; max-width:520px;
    border-radius:20px; overflow:hidden;
    box-shadow:0 25px 60px rgba(0,0,0,.35);
  }
  header{
    padding:28px 32px 22px; color:#fff;
    background:linear-gradient(120deg,var(--primary),var(--accent));
  }
  header h1{font-size:1.6rem;font-weight:700}
  header p{margin-top:6px;font-size:.92rem;opacity:.9}
  form{padding:28px 32px 8px;display:grid;grid-template-columns:1fr 1fr;gap:18px}
  .field{display:flex;flex-direction:column;gap:6px}
  .field.full{grid-column:1/-1}
  label{font-size:.8rem;font-weight:600;color:var(--muted);
        text-transform:uppercase;letter-spacing:.05em}
  input,select{
    padding:12px 14px;border:1.5px solid var(--line);border-radius:10px;
    font-size:1rem;color:var(--ink);background:#f8fafc;transition:.15s;
  }
  input:focus,select:focus{
    outline:none;border-color:var(--primary);background:#fff;
    box-shadow:0 0 0 4px rgba(79,70,229,.15);
  }
  button{
    grid-column:1/-1;margin-top:6px;padding:14px;border:0;border-radius:12px;
    font-size:1.05rem;font-weight:600;color:#fff;cursor:pointer;
    background:linear-gradient(120deg,var(--primary),var(--primary-dark));
    transition:transform .1s, box-shadow .2s;
  }
  button:hover{transform:translateY(-1px);box-shadow:0 8px 20px rgba(79,70,229,.4)}
  .result,.error{margin:18px 32px 0;padding:18px;border-radius:12px;text-align:center}
  .result.yes{background:var(--yes-bg);color:var(--yes)}
  .result.no{background:var(--no-bg);color:var(--no)}
  .result small{display:block;font-size:.78rem;opacity:.8;letter-spacing:.06em;
                text-transform:uppercase}
  .result strong{font-size:2rem;display:block;margin:4px 0}
  .result span{font-size:.85rem;opacity:.85}
  .error{background:var(--no-bg);color:var(--no);font-size:.9rem}
  footer{padding:20px;text-align:center;font-size:.78rem;color:var(--muted)}
  @media (max-width:480px){
    form{grid-template-columns:1fr;padding:22px}
    .result,.error{margin:16px 22px 0}
  }
</style>
</head>
<body>
<div class="card">
  <header>
    <h1>Purchase Predictor</h1>
    <p>Enter customer details to get an instant prediction.</p>
  </header>

  <form method="post" action="/">
    <div class="field">
      <label for="age">Age</label>
      <input id="age" name="Age" type="number" min="0" max="120" step="1"
             required value="{{ values.get('Age','') }}">
    </div>
    <div class="field">
      <label for="gender">Gender</label>
      <select id="gender" name="Gender" required>
        {% for k,v in genders.items() %}
        <option value="{{k}}" {{ 'selected' if values.get('Gender')|string==k|string }}>{{v}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="field">
      <label for="region">Region</label>
      <select id="region" name="Region" required>
        {% for k,v in regions.items() %}
        <option value="{{k}}" {{ 'selected' if values.get('Region')|string==k|string }}>{{v}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="field">
      <label for="occ">Occupation</label>
      <select id="occ" name="Occupation" required>
        {% for k,v in occupations.items() %}
        <option value="{{k}}" {{ 'selected' if values.get('Occupation')|string==k|string }}>{{v}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="field full">
      <label for="income">Income</label>
      <input id="income" name="Income" type="number" min="0" step="any"
             required value="{{ values.get('Income','') }}">
    </div>
    <button type="submit">Predict</button>
  </form>

  {% if error %}
    <div class="error">{{ error }}</div>
  {% elif prediction %}
    <div class="result {{ prediction }}">
      <small>Prediction</small>
      <strong>{{ prediction|upper }}</strong>
      {% if score is not none %}<span>Decision score: {{ score }}</span>{% endif %}
    </div>
  {% endif %}

  <footer>Powered by an SVM model · Flask on Render</footer>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction, score, error, values = None, None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            row = {name: float(values[name]) for name in FEATURES}
            X = pd.DataFrame([row], columns=FEATURES)
            prediction = str(model.predict(X)[0])
            score = round(float(model.decision_function(X)[0]), 3)
        except (KeyError, ValueError):
            error = "Please fill in every field with valid numbers."
        except Exception as exc:  # keep the page alive on model errors
            error = f"Prediction failed: {exc}"
    return render_template_string(
        PAGE,
        prediction=prediction, score=score, error=error, values=values,
        genders=GENDER_OPTIONS, regions=REGION_OPTIONS,
        occupations=OCCUPATION_OPTIONS,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
