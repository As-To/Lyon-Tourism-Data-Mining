# je fais du text mining pour extraire les termes les plus fréquents par cluster
# sachant que j'ai une colonne "cluster" dans mon dataframe
# j'utilise TF-IDF pour ça


import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def build_text(df):
    tags = df["tags"].fillna("")
    title = df["title"].fillna("")
    return (tags + " " + title).str.strip()

def basic_preprocess(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d+", " ", s)                 # remove digits
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s) # keep words + Arabic block
    s = re.sub(r"\s+", " ", s).strip()
    return s

def top_terms_by_cluster(df, cluster_col="cluster", top_k=10, stopwords=None):
    texts = build_text(df).map(basic_preprocess)

    vectorizer = TfidfVectorizer(
        stop_words=stopwords,      # ex: fr+en list
        min_df=3,                  # ignore very rare words
        max_df=0.8,                # ignore too common words
        ngram_range=(1, 1)
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    results = {}
    for cl in sorted(df[cluster_col].unique()):
        idx = np.where(df[cluster_col].values == cl)[0]
        if len(idx) == 0:
            continue
        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_k]
        results[cl] = list(terms[top_idx])

    return results
