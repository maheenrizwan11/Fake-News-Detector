from flask import Flask, request, jsonify, render_template
from newspaper import Article
from flask_cors import CORS

from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/bert_fake_news_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/check_fake', methods=['POST'])
def check_fake():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        # Extract article content
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        result = "Likely Fake" if prediction == 1 else "Likely Real"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
