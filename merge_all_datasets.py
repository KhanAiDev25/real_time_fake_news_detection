import os
import pandas as pd

root_dir = "fake news datasets"

text_keywords = ['text', 'content', 'statement', 'headline', 'news', 'article', 'title']
label_keywords = ['label', 'class', 'verdict', 'target', 'subject', 'truthscore', 'truth']

merged_data = []

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            try:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False, on_bad_lines='skip')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1', low_memory=False, on_bad_lines='skip')

                print(f"Processing: {file_path} | Shape: {df.shape}")

                # Special case: FakeNewsNet datasets (no label column)
                if "FakeNewsNet-master" in subdir:
                    if "fake" in file.lower():
                        temp = df[['title']].copy()
                        temp.columns = ['text']
                        temp['label'] = 1
                        temp.dropna(subset=['text'], inplace=True)
                        merged_data.append(temp)
                        continue
                    elif "real" in file.lower():
                        temp = df[['title']].copy()
                        temp.columns = ['text']
                        temp['label'] = 0
                        temp.dropna(subset=['text'], inplace=True)
                        merged_data.append(temp)
                        continue
                    else:
                        print(f"Skipping unknown FakeNewsNet file: {file_path}")
                        continue

                cols_lower = [c.lower() for c in df.columns]

                if 'truthscore' in cols_lower:
                    label_col = df.columns[cols_lower.index('truthscore')]
                else:
                    label_col = next((col for col in df.columns if any(k in col.lower() for k in label_keywords)), None)

                if 'title' in cols_lower:
                    text_col = df.columns[cols_lower.index('title')]
                else:
                    text_col = next((col for col in df.columns if any(k in col.lower() for k in text_keywords)), None)

                if text_col and label_col:
                    temp = df[[text_col, label_col]].copy()
                    temp.columns = ['text', 'label']

                    if 'fake' in file.lower():
                        temp['label'] = 1
                    elif 'true' in file.lower() or 'real' in file.lower():
                        temp['label'] = 0
                    else:
                        temp['label'] = temp['label'].astype(str).str.lower().str.strip()
                        temp['label'] = temp['label'].replace({
                            'real': 0, 'true': 0, 'mostly-true': 0, 'half-true': 0, 'barely-true': 1,
                            'false': 1, 'fake': 1, 'pants-fire': 1, 'mostly-false': 1,
                            '0': 0, '1': 1, 'news': 0, 'politicsnews': 0
                        })

                    temp = temp[temp['label'].isin([0, 1])]
                    temp['label'] = temp['label'].astype(int)
                    temp.dropna(subset=['text', 'label'], inplace=True)
                    merged_data.append(temp)
                else:
                    print(f"Skipped (columns not found): {file_path} | Columns: {list(df.columns)}")

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

if merged_data:
    merged_df = pd.concat(merged_data, ignore_index=True)
    merged_df.to_csv("merged_fake_news.csv", index=False)
    print(f"Merged dataset saved: merged_fake_news.csv | Shape: {merged_df.shape}")
else:
    print("No valid datasets were found.")

print(merged_df['label'].value_counts())
