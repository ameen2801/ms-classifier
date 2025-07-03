// Example response handling
fetch('/predict', { method: 'POST', body: formData })
   .then(response => response.json())
   .then(data => {
       document.getElementById('resultValue').textContent = data.prediction;
       document.getElementById('aiExplanation').textContent = data.ai_explanation;
       // Updates risk factors, recommendations, etc.
   });