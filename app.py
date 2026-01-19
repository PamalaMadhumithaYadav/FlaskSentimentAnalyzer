from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

# Load the trained model and tokenizer
print("Loading BERT model and tokenizer...")
output_dir = './model_bert/'
try:
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives review text and returns prediction as JSON."""
    if not model or not tokenizer:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(force=True)
    review_text = data['review']
    
    # Tokenize input
    encoded_dict = tokenizer.encode_plus(
        review_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs.logits
        
    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Get sentiment (0: Negative, 1: Positive)
    predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
    sentiment = 'Positive' if predicted_class == 1 else 'Negative'
    confidence = probabilities[predicted_class]

    return jsonify({
        'sentiment': sentiment,
        'confidence': f"{confidence*100:.2f}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)