# je fais du text mining pour extraire les termes les plus fréquents par cluster
# sachant que j'ai une colonne "cluster" dans mon dataframe
# j'utilise TF-IDF pour ça


# import re
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

# def build_text(df):
#     tags = df["tags"].fillna("")
#     title = df["title"].fillna("")
#     return (tags + " " + title).str.strip()

# def basic_preprocess(s: str) -> str:
#     s = s.lower()
#     s = re.sub(r"\d+", " ", s)                 # remove digits
#     s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s) # keep words + Arabic block
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def top_terms_by_cluster(df, cluster_col="cluster", top_k=10, stopwords=None):
#     texts = build_text(df).map(basic_preprocess)

#     vectorizer = TfidfVectorizer(
#         stop_words=stopwords,      # ex: fr+en list
#         min_df=3,                  # ignore very rare words
#         max_df=0.8,                # ignore too common words
#         ngram_range=(1, 1)
#     )
#     X = vectorizer.fit_transform(texts)
#     terms = np.array(vectorizer.get_feature_names_out())

#     results = {}
#     for cl in sorted(df[cluster_col].unique()):
#         idx = np.where(df[cluster_col].values == cl)[0]
#         if len(idx) == 0:
#             continue
#         mean_tfidf = X[idx].mean(axis=0).A1
#         top_idx = mean_tfidf.argsort()[::-1][:top_k]
#         results[cl] = list(terms[top_idx])

#     return results

# je fais du text mining pour extraire les termes les plus fréquents par cluster
# sachant que j'ai une colonne "cluster" dans mon dataframe
# j'utilise TF-IDF pour ça

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Stopwords FR/EN manuels pour éviter les mots outils
FRENCH_STOPWORDS = {
    "alors","au","aucun","aussi","autre","avant","avec","car","ce","cela","ces","cet","cette",
    "ceux","chaque","comme","comment","dans","de","des","du","donc","elle","elles","en","encore",
    "est","et","eu","fait","hormis","il","ils","je","jusqu","la","le","les","leur","leurs",
    "lui","ma","mais","mes","moi","mon","ne","nos","notre","nous","on","ou","par","pas","peu",
    "plus","pour","qu","que","qui","sa","se","ses","si","son","sont","sous","sur","ta","te",
    "tes","toi","ton","tous","tout","tres","tu","un","une","vos","votre","vous","y"
}
EXTRA_STOPWORDS = {"photo", "image", "tag", "titre", "title"}
DEFAULT_STOPWORDS = set(ENGLISH_STOP_WORDS) | FRENCH_STOPWORDS | EXTRA_STOPWORDS

def build_text(df):
    # s’assure que les colonnes existent
    for col in ("tags", "title"):
        if col not in df.columns:
            df[col] = ""
    tags = df["tags"].fillna("")
    title = df["title"].fillna("")
    return (tags + " " + title).str.strip()

def basic_preprocess(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d+", " ", s)
    # s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)  # garde lettres/chiffres + arabe
    s= re.sub(r"[^\w\s]", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

def lemmatize_optional(text: str):
    """
    Tente une lemmatisation légère si spaCy (fr/en) est dispo, sinon renvoie le texte brut.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("fr_core_news_sm")
        except Exception:
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return " ".join([t.lemma_ if t.lemma_ != "-PRON-" else t.text for t in doc])
    except Exception:
        return text

def top_terms_by_cluster(df, cluster_col="cluster", top_k=10, stopwords=None, use_lemmas=True, drop_noise=True):
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    # normalise pour scikit-learn
    if isinstance(stopwords, set):
        stopwords = sorted(stopwords)

    work = df.copy()
    if drop_noise and cluster_col in work.columns:
        work = work[work[cluster_col] != -1]

    texts = build_text(work).map(basic_preprocess)
    if use_lemmas:
        texts = texts.map(lemmatize_optional)

    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    results = {}
    for cl in sorted(work[cluster_col].unique()):
        idx = np.where(work[cluster_col].values == cl)[0]
        if len(idx) == 0:
            continue
        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_k]
        results[cl] = list(terms[top_idx])

    return results
