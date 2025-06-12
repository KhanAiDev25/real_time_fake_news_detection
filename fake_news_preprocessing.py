import pandas as pd
import re
from langdetect import detect
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import urduhack

# Initialize urduhack for Urdu normalization (run once)
urduhack.download()

# Load BERT tokenizers
tokenizer_en = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer_ur = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Zا-ے0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def normalize_urdu(text):
    return " ".join(urduhack.normalization.normalize(text))

def preprocess_data(df, text_col='text', label_col='label'):
    lang_list, clean_list, tokens_list = [], [], []

    for text in df[text_col]:
        lang = detect_language(text)
        lang_list.append(lang)

        cleaned = clean_text(text)
        if lang == 'ur':
            cleaned = normalize_urdu(cleaned)

        clean_list.append(cleaned)

        # Tokenize
        if lang == 'ur':
            tokens = tokenizer_ur.tokenize(cleaned)
        else:
            tokens = tokenizer_en.tokenize(cleaned)

        tokens_list.append(tokens)

    df['language'] = lang_list
    df['clean_text'] = clean_list
    df['tokens'] = tokens_list

    if label_col in df.columns:
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df[label_col])

    return df

def main():
    print("Loading dataset...")
    df = pd.read_csv('merged_fake_news.csv')  # Change this filename to your dataset path

    print("Preprocessing dataset...")
    df_processed = preprocess_data(df)

    print("Saving processed dataset to 'processed_fake_news.csv'...")
    df_processed.to_csv('processed_fake_news.csv', index=False)

    print("Preprocessing complete! Processed file saved.")

if __name__ == "__main__":
    main()
