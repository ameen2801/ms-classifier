import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature order expected by the model
feature_order = [
    'Gender', 'Age', 'Schooling', 'Breastfeeding', 'Varicella', 'Initial_Symptom',
    'Mono_or_Polysymptomatic', 'Oligoclonal_Bands', 'LLSSEP', 'ULSSEP', 'VEP', 'BAEP',
    'Periventricular_MRI', 'Cortical_MRI', 'Infratentorial_MRI', 'Spinal_Cord_MRI',
    'Initial_EDSS', 'Final_EDSS'
]

# Case studies data
case_studies = [
    {
        "id": 1,
        "age": 34,
        "gender": "female",
        "prediction": "RRMS",
        "symptoms": "Optic neuritis, fatigue",
        "accuracy": "95.2%",
        "outcome": "Patient responded well to interferon therapy with reduced relapse frequency."
    },
    {
        "id": 2,
        "age": 52,
        "gender": "male",
        "prediction": "SPMS",
        "symptoms": "Progressive weakness, balance issues",
        "accuracy": "89.7%",
        "outcome": "Transitioned to SPMS after 15 years of RRMS. Siponimod therapy initiated."
    },
    {
        "id": 3,
        "age": 28,
        "gender": "female",
        "prediction": "RRMS",
        "symptoms": "Numbness in limbs, vision problems",
        "accuracy": "93.1%",
        "outcome": "Early intervention with glatiramer acetate showed good disease control."
    }
]

# Explanations and recommendations
explanations = {
    "RRMS": "Based on the clinical features provided, the model predicts Relapsing-Remitting MS (RRMS) with high confidence. Key indicators include the patient's younger age, pattern of EDSS scores, and symptom presentation. RRMS is characterized by clearly defined attacks of new or increasing neurologic symptoms followed by periods of partial or complete recovery.",
    "SPMS": "The prediction of Secondary-Progressive MS (SPMS) is indicated by the patient's age, progressive disability pattern, and EDSS scores. SPMS typically follows an initial relapsing-remitting course and is characterized by a steady progression of disability with or without occasional relapses."
}

recommendations = {
    "RRMS": [
        {"text": "Initiate disease-modifying therapy (DMT) immediately", "priority": "high"},
        {"text": "Consider interferon-beta or glatiramer acetate as first-line treatment", "priority": "medium"},
        {"text": "Schedule MRI follow-up in 6 months", "priority": "medium"},
        {"text": "Assess vitamin D levels and supplement if deficient", "priority": "low"},
        {"text": "Monitor for new neurological symptoms at each visit", "priority": "high"}
    ],
    "SPMS": [
        {"text": "Consider siponimod or ocrelizumab therapy", "priority": "high"},
        {"text": "Implement symptom management strategies", "priority": "medium"},
        {"text": "Physical therapy and rehabilitation program", "priority": "high"},
        {"text": "Regular EDSS assessments every 3 months", "priority": "medium"},
        {"text": "Monitor for disease progression with annual MRIs", "priority": "high"}
    ]
}

risk_factors = {
    "RRMS": [
        {"title": "Age < 30", "description": "Younger age correlates with higher relapse frequency",
         "priority": "medium"},
        {"title": "Recent relapse", "description": "Increased risk of additional relapses in first year",
         "priority": "high"}
    ],
    "SPMS": [
        {"title": "Age > 40", "description": "Increased risk of progressive disability", "priority": "high"},
        {"title": "High EDSS score", "description": "Baseline disability correlates with progression rate",
         "priority": "high"}
    ]
}


@app.route('/')
def home():
    return render_template('index.html', case_studies=case_studies)


@app.route('/predict', methods=['POST'])
def predict():
    # Create a dictionary with default values (0) for all features
    features = {key: 0 for key in feature_order}

    # Update with form values
    form_data = request.form
    for feature in feature_order:
        if feature in form_data:
            # Convert to appropriate type
            if feature in ['Initial_EDSS', 'Final_EDSS']:
                features[feature] = float(form_data[feature])
            else:
                features[feature] = int(form_data[feature])

    # Create feature array in correct order
    input_data = [features[key] for key in feature_order]

    # Scale features
    scaled_data = scaler.transform([input_data])

    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    confidence = round(max(probabilities) * 100, 1)

    # Get feature importances
    importances = model.feature_importances_
    importance_list = list(zip(feature_order, importances))
    importance_list.sort(key=lambda x: x[1], reverse=True)
    top_features = importance_list[:4]

    # Normalize top feature importances for visualization
    max_importance = max([imp for _, imp in top_features])
    normalized_features = [{"name": name, "width": round((imp / max_importance) * 100, 1)}
                           for name, imp in top_features]

    # Map prediction to subtype
    subtype = "RRMS" if "Relapsing" in prediction else "SPMS"

    # Prepare response
    response = {
        "subtype": subtype,
        "confidence": confidence,
        "feature_importances": normalized_features,
        "explanation": explanations.get(subtype, ""),
        "risk_factors": risk_factors.get(subtype, []),
        "recommendations": recommendations.get(subtype, [])
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)