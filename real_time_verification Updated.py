import time
from googlesearch import search
from newsapi import NewsApiClient
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='42330e9477354673bc284c55581931ea')

# Load your fine-tuned BERT model
model_path = "fake_news_bert_model"  # Make sure this path is correct
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Keywords and sources
fake_keywords = ['fake', 'hoax', 'misleading', 'false', 'debunked', 'scam', 'fabricated']
real_keywords = ['confirmed', 'true', 'verified', 'proven', 'official', 'announced']
reputable_sources = ["cnn.com", "bbc.com", "reuters.com", "geo.tv", "nytimes.com", "aljazeera.com"]
negation_keywords = ["not", "no", "never", "isn't", "wasn't", "aren't", "none", "nobody", "nothing", "don't", "doesn't", "didn't", "won't", "cannot"]

def contains_negation(text):
    words = text.lower().split()
    return any(neg in words for neg in negation_keywords)

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

        # 1. Prediction
        fake_conf, real_conf = predict_fake_news(text)

        # 2. Search Results
        g_results = google_search(text)
        n_results = newsapi_search(text)
        all_links = g_results + n_results
        verification_score, credible_sources = score_results(all_links)

        # 3. Negation detection
        has_negation = contains_negation(text)

        # 4. Decision logic
        confidence = real_conf if real_conf > fake_conf else fake_conf
        decision = "UNVERIFIED"
        verdict_icon = "⚠️"
        note = f"No strong BERT prediction or source confirmation (Confidence: {confidence:.2f}%)"

        if credible_sources and verification_score >= 4:
            if has_negation:
                decision = "UNVERIFIED"
                verdict_icon = "⚠️"
                note = f"Negation detected — sources may confirm opposite claim. BERT confidence: {real_conf:.2f}%"
            else:
                decision = "REAL"
                verdict_icon = "✅"
                note = f"Confirmed by: {', '.join(set(credible_sources))} — BERT confidence: {real_conf:.2f}%"
        elif has_negation and fake_conf > 60:
            decision = "FAKE"
            verdict_icon = "❌"
            note = f"BERT detected likely fake (Confidence: {fake_conf:.2f}%)"
        elif real_conf > 60:
            decision = "REAL"
            verdict_icon = "✅"
            note = f"BERT prediction only (Confidence: {real_conf:.2f}%)"

        # 5. Display
        print(f"{verdict_icon} Verdict: {decision}")
        print("📝", note)
        time.sleep(1)
