# Real-Time Fake News Detection System

This project combines BERT-based text classification with real-time web-based verification using Google Custom Search and NewsAPI.

## Features
- Trained on fake/real news datasets (English + Urdu)
- Uses BERT (`bert-base-uncased`)
- Real-time validation using Google and NewsAPI
- Hybrid confidence scoring
- Supports multilingual text (English, Urdu)

## Requirements
- Python 3.8+
- transformers
- torch
- newsapi-python
- requests
- sentence-transformers

## How to Run
```bash
python train_fake_news_bert.py
python predict_fake_news.py
