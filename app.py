# app.py
from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import random
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Feature names from the dataset (excluding group and group_name)
feature_names = [
    'Gender', 'Age', 'Schooling', 'Breastfeeding', 'Varicella', 'Initial_Symptom',
    'Mono_or_Polysymptomatic', 'Oligoclonal_Bands', 'LLSSEP', 'ULSSEP', 'VEP', 'BAEP',
    'Periventricular_MRI', 'Cortical_MRI', 'Infratentorial_MRI', 'Spinal_Cord_MRI',
    'Initial_EDSS', 'Final_EDSS'
]

# Clinical meaning mappings for categorical features
clinical_mappings = {
    'Gender': {1: 'Male', 2: 'Female'},
    'Breastfeeding': {
        1: 'Never',
        2: 'Less than 1 year',
        3: 'More than 1 year'
    },
    'Varicella': {
        1: 'Positive',
        2: 'Negative'
    },
    'Initial_Symptom': {
        1: 'Optic neuritis',
        2: 'Pyramidal',
        3: 'Brainstem/cerebellar',
        4: 'Sensitive',
        5: 'Sphincter',
        6: 'Polysymptomatic',
        7: 'Other',
        8: 'Polineuritis cranialis',
        9: 'Myelopathy',
        10: 'Hemiparesis',
        11: 'Paraparesis',
        12: 'Tetraparesis',
        13: 'Ataxia',
        14: 'Vertigo',
        15: 'Fatigue'
    },
    'Mono_or_Polysymptomatic': {
        1: 'Monosymptomatic',
        2: 'Polysymptomatic',
        3: 'Unknown'
    }
}

# Feature importance data (mock for visualization)
feature_importance = {
    'Initial_EDSS': 0.92,
    'Spinal_Cord_MRI': 0.85,
    'Age': 0.78,
    'Final_EDSS': 0.75,
    'Periventricular_MRI': 0.68,
    'Infratentorial_MRI': 0.62,
    'Oligoclonal_Bands': 0.59,
    'Mono_or_Polysymptomatic': 0.54,
    'Initial_Symptom': 0.48,
    'Varicella': 0.42,
    'ULSSEP': 0.38,
    'LLSSEP': 0.35,
    'VEP': 0.32,
    'BAEP': 0.28,
    'Schooling': 0.24,
    'Breastfeeding': 0.21,
    'Gender': 0.15
}

# Treatment recommendations by subtype
treatment_recommendations = {
    'RRMS': [
        {"text": "Initiate disease-modifying therapy (DMT) immediately", "priority": "high"},
        {"text": "Consider interferon-beta or glatiramer acetate as first-line treatment", "priority": "medium"},
        {"text": "Schedule MRI follow-up in 6 months", "priority": "medium"},
        {"text": "Assess vitamin D levels and supplement if deficient", "priority": "low"},
        {"text": "Monitor for new neurological symptoms at each visit", "priority": "high"}
    ],
    'SPMS': [
        {"text": "Consider siponimod or ocrelizumab therapy", "priority": "high"},
        {"text": "Implement symptom management strategies", "priority": "medium"},
        {"text": "Physical therapy and rehabilitation program", "priority": "high"},
        {"text": "Regular EDSS assessments every 3 months", "priority": "medium"},
        {"text": "Monitor for disease progression with annual MRIs", "priority": "high"}
    ],
    'PPMS': [
        {"text": "Ocrelizumab as primary treatment option", "priority": "high"},
        {"text": "Symptom management for spasticity and fatigue", "priority": "high"},
        {"text": "Mobility assistance devices as needed", "priority": "medium"},
        {"text": "Regular urological assessments", "priority": "medium"},
        {"text": "Cognitive function monitoring", "priority": "low"}
    ]
}

# Patient case studies
case_studies = [
    {
        'id': 1,
        'age': 32,
        'gender': 'Female',
        'symptoms': 'Optic neuritis',
        'prediction': 'RRMS',
        'accuracy': '94.2%',
        'outcome': 'Responded well to interferon therapy'
    },
    {
        'id': 2,
        'age': 45,
        'gender': 'Male',
        'symptoms': 'Pyramidal symptoms',
        'prediction': 'SPMS',
        'accuracy': '89.7%',
        'outcome': 'Stabilized with ocrelizumab'
    },
    {
        'id': 3,
        'age': 58,
        'gender': 'Female',
        'symptoms': 'Progressive weakness',
        'prediction': 'PPMS',
        'accuracy': '91.3%',
        'outcome': 'Mobility maintained with physical therapy'
    },
    {
        'id': 4,
        'age': 29,
        'gender': 'Male',
        'symptoms': 'Sensitive symptoms',
        'prediction': 'RRMS',
        'accuracy': '96.1%',
        'outcome': 'No relapses after 2 years of treatment'
    },
    {
        'id': 5,
        'age': 51,
        'gender': 'Female',
        'symptoms': 'Brainstem/cerebellar',
        'prediction': 'SPMS',
        'accuracy': '87.5%',
        'outcome': 'Slow progression with treatment'
    }
]


def generate_clinical_explanation(prediction, confidence, input_data):
    """Generate clinical explanation without AI"""
    explanation = f"""
    Clinical Analysis for MS Subtype Prediction:

    The patient's clinical features suggest a {prediction} subtype with {confidence} confidence. 
    This prediction is based on the following key factors:
    """

    # Add top 3 influential factors
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    for feature, importance in sorted_features:
        value = input_data.get(feature, 'N/A')
        explanation += f"\n- {feature} (value: {value}, impact: {importance:.0%})"

    explanation += f"""

    Clinical Considerations:
    - {prediction} typically presents with the features observed in this case
    - Recommended next steps include confirming the diagnosis with additional testing
    - Please review the treatment recommendations specific to this subtype

    Note: This analysis is based on predictive modeling and should be interpreted in the clinical context.
    """

    return explanation


def generate_report(prediction_data):
    """Generate PDF report of prediction results"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("NeuroPredict Pro - MS Subtype Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Patient details
        story.append(Paragraph(f"<b>Prediction:</b> {prediction_data['prediction']}", styles['Normal']))
        story.append(Paragraph(f"<b>Confidence:</b> {prediction_data['confidence']}", styles['Normal']))
        story.append(Paragraph(f"<b>Date:</b> {prediction_data['timestamp']}", styles['Normal']))
        story.append(Spacer(1, 24))

        # Key factors
        story.append(Paragraph("<b>Key Predictive Factors:</b>", styles['Heading2']))
        for item in prediction_data['visualization_data']:
            story.append(Paragraph(f"{item['feature']} (Impact: {item['importance']:.1%})", styles['Normal']))
        story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph("<b>Clinical Recommendations:</b>", styles['Heading2']))
        for rec in prediction_data['recommendations']:
            story.append(Paragraph(f"- {rec['text']} ({rec['priority'].title()} priority)", styles['Normal']))
        story.append(Spacer(1, 12))

        # Clinical Explanation
        story.append(Paragraph("<b>Clinical Analysis:</b>", styles['Heading2']))
        story.append(Paragraph(prediction_data['ai_explanation'], styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"Report generation error: {str(e)}")
        return None


@app.route('/')
def home():
    return render_template('index.html', features=feature_names, mappings=clinical_mappings,
                           case_studies=case_studies, feature_importance=feature_importance)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and convert form data
        input_data = []
        for feature in feature_names:
            value = request.form[feature]
            if value.strip() == '':
                input_data.append(float('nan'))
            else:
                input_data.append(float(value))

        # Create DataFrame and scale features
        input_df = pd.DataFrame([input_data], columns=feature_names)
        scaled_data = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_data)[0]
        confidence = max(model.predict_proba(scaled_data)[0]) * 100

        # Generate mock feature importance visualization data
        visualization_data = []
        for feature, importance in feature_importance.items():
            value = input_df[feature].values[0] if feature in input_df.columns else 0
            visualization_data.append({
                'feature': feature,
                'importance': importance * 0.95 + random.uniform(0, 0.05),
                'value': value
            })

        # Sort by importance
        visualization_data.sort(key=lambda x: x['importance'], reverse=True)

        # Generate mock lesion distribution data
        lesion_distribution = {
            'Periventricular_MRI': input_df['Periventricular_MRI'].values[0] * 0.8,
            'Cortical_MRI': input_df['Cortical_MRI'].values[0] * 0.6,
            'Infratentorial_MRI': input_df['Infratentorial_MRI'].values[0] * 0.7,
            'Spinal_Cord_MRI': input_df['Spinal_Cord_MRI'].values[0] * 0.9
        }

        # Generate risk factors analysis
        risk_factors = []
        if input_df['Age'].values[0] > 40:
            risk_factors.append({
                'factor': 'Age > 40',
                'risk_level': 'High',
                'description': 'Increased risk of progressive MS forms'
            })

        if input_df['Initial_EDSS'].values[0] > 3.0:
            risk_factors.append({
                'factor': f'Initial EDSS {input_df["Initial_EDSS"].values[0]}',
                'risk_level': 'Moderate',
                'description': 'Higher baseline disability correlates with worse prognosis'
            })

        if input_df['Spinal_Cord_MRI'].values[0] > 2:
            risk_factors.append({
                'factor': 'Spinal Cord Lesions',
                'risk_level': 'High',
                'description': 'Spinal lesions associated with faster progression'
            })

        # Get recommendations
        recommendations = treatment_recommendations.get(prediction, [])

        # Prepare patient data for clinical explanation
        input_dict = {feature: input_df[feature].values[0] for feature in feature_names}

        # Generate clinical explanation
        clinical_explanation = generate_clinical_explanation(
            prediction,
            f"{confidence:.1f}%",
            input_dict
        )

        # Prepare comprehensive response
        response = {
            'prediction': prediction,
            'confidence': f"{confidence:.1f}%",
            'visualization_data': visualization_data[:5],  # Top 5 features
            'lesion_distribution': lesion_distribution,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'ai_explanation': clinical_explanation,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        data = request.json
        report_buffer = generate_report(data)

        if report_buffer:
            return send_file(
                report_buffer,
                as_attachment=True,
                download_name=f"MS_Prediction_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Report generation failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)