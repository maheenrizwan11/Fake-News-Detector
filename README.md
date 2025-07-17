# Fake News Detector Web App
A web application that helps users detect fake news articles simply by pasting a URL. This project aims to make misinformation detection accessible to everyone through a clean UI and fast backend.

## Features
- **URL-based Fake News Checking** — enter a news article URL, and get an instant analysis.
- **BERT-based ML Model** — backend support for fine-tuned fake news detection using the LIAR dataset.
- **Search History** — previous searches are stored locally in the browser.
- **Responsive Design** — works on desktop.

## Tech Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Model**: BERT fine-tuned on LIAR dataset (`bert-base-uncased`)

## Project Structure
fake-news-detector/
│
├── models/
│ └── bert_fake_news_model/
│ ├── config.json
│ ├── model.safetensors
│ ├── vocab.txt
│ └── tokenizer_config.json
│
├── static/
│ ├── styles.css
│
├── templates/
│ └── index.html
│
├── app.py
├── requirements.txt
└── README.md

## 🧪 Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate on Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the app:
   ```bash
   python app.py
5. Open http://localhost:5000 in your browser.
   
Future Improvements"
Improve accuracy of the model
Store history in backend instead of localStorage
Integrate AI detection in the app

Screenshots
<img width="765" height="543" alt="image" src="https://github.com/user-attachments/assets/520dfa54-6079-4285-96ac-76de5cf30ecc" />

License
MIT License
