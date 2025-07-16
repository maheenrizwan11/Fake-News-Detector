from flask import Flask, request, jsonify, render_template
from newspaper import Article
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

FAKE_MODEL_PATH = "models/bert_fake_news_model"
tokenizer = BertTokenizer.from_pretrained(FAKE_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(FAKE_MODEL_PATH)
model.eval()

HISTORY_FILE = "history.json"

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
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        result = "Likely Fake" if prediction == 1 else "Likely Real"

        entry = {
            "url": url,
            "result": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_history(entry)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        return jsonify(load_history())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"message": "History cleared."})
    
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(entry):
    history = load_history()
    history.insert(0, entry) 
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

if __name__ == '__main__':
    app.run(debug=True)
