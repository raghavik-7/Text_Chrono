import csv
import requests
import os
import re
from collections import Counter
import numpy as np
import pandas as pd

def download_imdb_csv(save_path):
    url = 'https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset/master/train.csv'
    r = requests.get(url)
    if r.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded IMDB reviews to {save_path}")
    else:
        print(f"Failed to download IMDB reviews. Status code: {r.status_code}")

def ensure_imdb_csv_exists():
    path = 'TextChrono/data/reviews.csv'
    if not os.path.exists(path):
        print('IMDB reviews.csv not found. Downloading...')
        download_imdb_csv(path)
    else:
        print('IMDB reviews.csv already exists.')

def fetch_zenquotes_csv(n=1000, out_path='TextChrono/data/zen_quotes.csv'):
    quotes = []
    for _ in range(n):
        r = requests.get("https://zenquotes.io/api/random")
        if r.status_code == 200:
            data = r.json()[0]
            quotes.append([data['q'], data['a']])
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["quote", "author"])
        writer.writerows(quotes)
    print(f"Saved {len(quotes)} quotes to {out_path}")

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuations
    text = text.lower()
    return text

def load_imdb(filepath, max_samples=None):
    df = pd.read_csv(filepath)
    df['review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    texts = df['review'].tolist()
    labels = df['label'].tolist()
    if max_samples:
        texts, labels = texts[:max_samples], labels[:max_samples]
    return texts, labels

def load_tweets(path, max_samples=None):
    texts, labels = [], []
    with open(path, encoding='latin-1') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 6:
                continue
            labels.append(1 if row[0] == '4' else 0)
            texts.append(row[5])
            if max_samples and i+1 >= max_samples:
                break
    return texts, labels

def load_quotes(path, max_samples=None):
    texts = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            texts.append(line.strip())
            if max_samples and i+1 >= max_samples:
                break
    return texts

def fetch_quotes_from_api(n=1000, out_path=None):
    url = 'https://zenquotes.io/api/quotes'
    quotes = []
    for _ in range(n // 50):
        resp = requests.get(url)
        if resp.status_code == 200:
            quotes.extend([q['q'] for q in resp.json()])
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            for q in quotes:
                f.write(q + '\n')
    return quotes

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(texts, min_freq=2, max_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab['<UNK>']) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    return ids

def batch_generator(texts, labels, vocab, batch_size=32, max_len=50, shuffle=True):
    indices = np.arange(len(texts))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        batch_x = [encode(texts[i], vocab, max_len) for i in batch_idx]
        batch_y = [labels[i] for i in batch_idx]
        yield np.array(batch_x), np.array(batch_y) 