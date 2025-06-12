import time
from googlesearch import search
from newsapi import NewsApiClient
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='42330e9477354673bc284c55581931ea')

# Initialize BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

# Fake News Keywords
fake_keywords = ['fake', 'hoax', 'misleading', 'false', 'debunked', 'scam', 'fabricated']
real_keywords = ['confirmed', 'true', 'verified', 'proven', 'official', 'announced']

reputable_sources = ["cnn.com", "bbc.com", "reuters.com", "geo.tv", "nytimes.com", "aljazeera.com"]


def score_results(results):
    score = 0
    matching_sources = []
    for result in results:
        url = result.lower()
        for source in reputable_sources:
            if source in url:
                score += 2
                matching_sources.append(source)
        for keyword in real_keywords:
            if keyword in url:
                score += 1
        for keyword in fake_keywords:
            if keyword in url:
                score -= 2
    return score, list(set(matching_sources))


def predict_fake_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        fake_confidence = probs[0][0].item() * 100
        real_confidence = probs[0][1].item() * 100
        return fake_confidence, real_confidence


def google_search(query):
    try:
        return list(search(query, num_results=5))
    except Exception as e:
        print("Google Search Error:", e)
        return []


def newsapi_search(query):
    try:
        results = newsapi.get_everything(q=query, language='en', page_size=5)
        return [article['url'] for article in results['articles']]
    except Exception as e:
        print("NewsAPI Error:", e)
        return []


if __name__ == '__main__':
    while True:
        text = input("\nEnter a news headline or content (or type 'exit'): ")
        if text.lower() == 'exit':
            break

        print(f"\n🔎 Verifying: {text}\n")

        # ML Prediction
        fake_conf, real_conf = predict_fake_news(text)

        # Search-based verification
        g_results = google_search(text)
        n_results = newsapi_search(text)

        all_links = g_results + n_results
        verification_score, credible_sources = score_results(all_links)

        # Final decision logic
        confidence = real_conf if real_conf > fake_conf else fake_conf
        decision = "REAL" if verification_score >= 2 or real_conf > fake_conf else "FAKE"
        verdict_icon = "✅" if decision == "REAL" else "❌"

        # Display results
        print(f"{verdict_icon} Verdict: {decision} (Confidence: {confidence:.2f}%)")
        if credible_sources:
            print("Confirmed by:", ", ".join(credible_sources))
        elif verification_score <= -2:
            print("❗ Warning: Found possible signs of fake news from low-quality sources.")
        else:
            print("⚠️ No strong confirmation from reputable sources found.")

        time.sleep(1)
