import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model_path = './fake_news_bert_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def predict_fake_news(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return "Real News" if prediction == 0 else "Fake News"

# Test
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a news headline or content (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        result = predict_fake_news(user_input)
        print(f"Prediction: {result}")
