from wordcloud import WordCloud
import matplotlib.pyplot as plt

import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "cleaned", "cleaned_lyon_data.csv")

from src.algo.text_mining import build_text, basic_preprocess, DEFAULT_STOPWORDS, FRENCH_STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def plot_wordcloud(texts, stopwords=None, title=None):
    text = " ".join(texts)

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=stopwords
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def show_wordcloud(df, stopwords=None, title="Word cloud"):
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    texts = build_text(df).map(basic_preprocess)
    full_text = " ".join(texts.tolist())

    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        stopwords=set(stopwords)  # WordCloud attend un set
    ).generate(full_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()