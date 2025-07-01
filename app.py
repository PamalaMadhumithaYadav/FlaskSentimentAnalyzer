from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model pipeline
print("Loading model pipeline...")
model = joblib.load('sentiment_model.pkl')
print("Model loaded.")

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives review text and returns prediction as JSON."""
    data = request.get_json(force=True)
    review_text = data['review']
    
    # The pipeline expects a list of documents
    prediction = model.predict([review_text])
    # Get probabilities for [negative, positive] classes
    probabilities = model.predict_proba([review_text])[0]
    
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    confidence = probabilities[1] if sentiment == 'Positive' else probabilities[0]

    return jsonify({
        'sentiment': sentiment,
        'confidence': f"{confidence*100:.2f}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)