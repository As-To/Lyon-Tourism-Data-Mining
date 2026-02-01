import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


import math
import pandas as pd
from itertools import combinations
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import unicodedata


FRENCH_STOPWORDS = set(stopwords.words("french"))

EXTRA_STOPWORDS = {
    "photo", "image", "tag", "titre", "title",

    
    "uploaded", "upload", "flickrmobile", "flickriosapp",
    "instagram", "instagramapp", "iphoneography", "squareformat",
    "square", "filter", "nofilter", "iphone",

   
    "img", "img_", "dsc", "jpg", "jpeg", "png","bokeh"

    # mots que l'on juge peu informatifs dans notre contexte : rhone ça se discute
    "lyoncity", "lyonnais",
    "fr", "incity", "lyon", "france", "europe", "city", "ville", "street", "urban", "urbain",
    "day","europe", "europa",

    # après la visualisation des wordclouds
    "paper", "pasted", "pasteup", "wheatpaste", "wheatpaper",
    "collage",
    "nikon", "canon", "dsc",
    "streetart", 

    #artefact de plateforme
    "foursquare", "venue", "foursquare venue", "xproii", "proii", "vscocam", "vsco", "rhone", "rhonealpes", "moto", "bb", "sncf", 

    #ajouts 
    "photocmobile", "eos", "pentax", "pentaxk", "mm", "fb", "eb", "doctorwhoclassic", "jossarisfoto", 
    "lr", "fe", "sec", "mmf", "dslr", "xpro", "fujifilm", "fuji", "bij", "bfd", 
     "jossaris", "jossarisfoto", "patman", "choupinou", "architecture", "building", "tower", "villedelyon",
    "metropolisoflyon", "auvergnerhonealpes","bmx", "_dsc", "creditphotosjonathantmare", "iso", "oss"

}

DEFAULT_STOPWORDS = set(ENGLISH_STOP_WORDS) | FRENCH_STOPWORDS | EXTRA_STOPWORDS

# on commence par construire le texte combiné "tags + title"
def build_text(df):
    # s’assure que les colonnes existent
    for col in ("tags", "title"):
        if col not in df.columns:
            df[col] = ""
    tags = df["tags"].fillna("")
    title = df["title"].fillna("")
    return (tags + " " + title).str.strip()

# on fait un pré-traitement basique du texte 
def basic_preprocess(s: str) -> str:
    s = s.lower() # met en minuscules
    # supprime les tokens techniques type img_1234, dsc_5678
    s = re.sub(r"\b(img|dsc)\w+\b", " ", s) # supprime les mots commençant par img ou dsc
    s = re.sub(r"\d+", " ", s) # supprime les chiffres et remplace par un espace
    
    s= re.sub(r"[^\w\s]", " ", s) # enlève les caractères spéciaux et remplace par un espace
    
    s = re.sub(r"\s+", " ", s).strip()# remplace les espaces multiples par un seul espace
    return s

def normalize_stopwords(stopwords):
    return {basic_preprocess(w) for w in stopwords}

DEFAULT_STOPWORDS = normalize_stopwords(DEFAULT_STOPWORDS)


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

# fonction qui supprime les unigrams redondants avec des bigrams : 
# par exemple si "rhone" et "rhone alpes" sont tous les deux dans la liste, on supprime "rhone"
def remove_redundant_unigrams(terms):
    bigrams = {t for t in terms if " " in t}
    unigrams = set(terms)

    to_remove = set()
    for bg in bigrams:
        w1, w2 = bg.split()
        if w1 in unigrams:
            to_remove.add(w1)
        if w2 in unigrams:
            to_remove.add(w2)

    return [t for t in terms if t not in to_remove]


def top_terms_by_cluster(df, cluster_col="cluster", top_k=10, stopwords=None, use_lemmas=True, drop_noise=True):
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    # normalise pour scikit-learn
    if isinstance(stopwords, set): 
        stopwords = sorted(stopwords)

    work = df.copy()
    if drop_noise and cluster_col in work.columns: # si on veut supprimer le bruit (-1)
        work = work[work[cluster_col] != -1]

    texts = build_text(work).map(basic_preprocess) # construit le texte combiné et nettoyé
    if use_lemmas:
        texts = texts.map(lemmatize_optional)

    # crée la matrice TF-IDF
    # la matrice est de taille (n_samples, n_terms)
    # TfidVectorizer tokenize les "documents", c'est à dire chaque image
    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        strip_accents="unicode",
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2)
    )
    # vectorise les textes en TF-IDF ce qui permet de pondérer les termes
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    results = {}
    # pour chaque cluster, calcule les termes les plus représentatifs
    for cl in sorted(work[cluster_col].unique()):
        idx = np.where(work[cluster_col].values == cl)[0]
        if len(idx) == 0:
            continue
        mean_tfidf = X[idx].mean(axis=0).A1
        top_idx = mean_tfidf.argsort()[::-1][:top_k]
        
        # results[cl] = list(terms[top_idx]) à décommenter pour verison qui supprime pas les unigrams redondants
        
        # à commenter pour version qui supprime les unigrams redondants
        cluster_terms = list(terms[top_idx])
        cluster_terms = remove_redundant_unigrams(cluster_terms)
        results[cl] = cluster_terms


    return results




## 2ème ALGORITHME : Principe Apriori + Association Rules

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

# Reuse FRENCH_STOPWORDS, EXTRA_STOPWORDS and DEFAULT_STOPWORDS

# ----------------------------
# 1) Construction texte + preprocessing
# ----------------------------
def build_text(df: pd.DataFrame) -> pd.Series:
    """Concat tags + title (une photo = un document)."""
    for col in ("tags", "title"):
        if col not in df.columns:
            df[col] = ""
    tags = df["tags"].fillna("")
    title = df["title"].fillna("")
    return (tags + " " + title).str.strip()

def basic_preprocess(s: str) -> str:
    """Nettoyage basique: lowercase, suppression tokens img_/dsc_, chiffres, ponctuation -> espaces."""
    s = str(s).lower()
    s = re.sub(r"\b(img|dsc)\w+\b", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)     # caractères spéciaux -> espace
    s = re.sub(r"\s+", " ", s).strip() # espaces multiples -> 1
    s = strip_accents(s) # supprime accents

    return s

def tokenize(text: str, stopwords=DEFAULT_STOPWORDS, min_len=2):
    """
    Tokenisation simple: split sur espaces + filtres stopwords.
    On garde les tokens unicode (donc multilingue).
    """
    if not text:
        return []
    toks = text.split()
    toks = [t for t in toks if len(t) >= min_len and t not in stopwords]
    return toks

def make_transactions(df: pd.DataFrame, stopwords=DEFAULT_STOPWORDS) -> list[set[str]]:
    """
    Convertit le df en transactions Apriori:
    1 transaction = 1 photo = set(tokens uniques)
    """
    texts = build_text(df).map(basic_preprocess)
    transactions = []
    for t in texts:
        toks = tokenize(t, stopwords=stopwords)
        if toks:
            transactions.append(set(toks))
    return transactions


# ----------------------------
# 2) Apriori (itemsets fréquents)
# ----------------------------
def apriori_itemsets(transactions: list[set[str]], min_support=0.05, max_k=3):
    """
    Retourne un dict: {itemset(tuple trié): support(float)}
    """
    n = len(transactions)
    if n == 0:
        return {}

    min_count = math.ceil(min_support * n)

    # L1
    counts_1 = Counter()
    for t in transactions:
        for it in t:
            counts_1[(it,)] += 1

    L_prev = {it for it, c in counts_1.items() if c >= min_count}
    supports = {it: counts_1[it] / n for it in L_prev}

    k = 2
    while L_prev and k <= max_k:
        L_prev_sorted = sorted(L_prev)

        # Candidate generation (join)
        Ck = set()
        for i in range(len(L_prev_sorted)):
            for j in range(i + 1, len(L_prev_sorted)):
                a = L_prev_sorted[i]
                b = L_prev_sorted[j]
                # join si prefix commun (k-2 items)
                if a[:k-2] == b[:k-2]:
                    cand = tuple(sorted(set(a) | set(b)))
                    if len(cand) == k:
                        Ck.add(cand)
                else:
                    break

        # Prune: tous les sous-ensembles (k-1) doivent être fréquents
        L_prev_set = set(L_prev)
        Ck_pruned = set()
        for cand in Ck:
            ok = True
            for sub in combinations(cand, k - 1):
                if tuple(sorted(sub)) not in L_prev_set:
                    ok = False
                    break
            if ok:
                Ck_pruned.add(cand)

        # Count
        counts_k = Counter()
        for t in transactions:
            for cand in Ck_pruned:
                if set(cand).issubset(t):
                    counts_k[cand] += 1

        Lk = {cand for cand, c in counts_k.items() if c >= min_count}
        for cand in Lk:
            supports[cand] = counts_k[cand] / n

        L_prev = Lk
        k += 1

    return supports


# ----------------------------
# 3) Génération des règles d'association
# ----------------------------
def generate_rules_from_itemsets(itemset_supports: dict, min_confidence=0.4, min_lift=1.0):
    """
    Génère des règles A -> B à partir des itemsets fréquents.
    Retour: liste de dicts {antecedent, consequent, support, confidence, lift}
    """
    rules = []
    # supports des singletons nécessaires pour lift
    for itemset, sup_ab in itemset_supports.items():
        if len(itemset) < 2:
            continue

        items = tuple(itemset)
        # toutes les partitions non vides A,B
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = tuple(sorted(A))
                B = tuple(sorted(set(items) - set(A)))
                sup_a = itemset_supports.get(A)
                sup_b = itemset_supports.get(B)
                if sup_a is None or sup_b is None:
                    continue

                conf = sup_ab / sup_a if sup_a > 0 else 0.0
                lift = conf / sup_b if sup_b > 0 else 0.0

                if conf >= min_confidence and lift >= min_lift:
                    rules.append({
                        "antecedent": A,
                        "consequent": B,
                        "support": sup_ab,
                        "confidence": conf,
                        "lift": lift,
                    })

    # tri : lift puis confidence puis support
    rules.sort(key=lambda d: (d["lift"], d["confidence"], d["support"]), reverse=True)
    return rules


# ----------------------------
# 4) Apriori par cluster : c'est ce qui est appelé depuis K_means.py et dbscan_v2.py
# ----------------------------
def apriori_by_cluster(
    df: pd.DataFrame,
    cluster_col="cluster",
    min_support=0.05,
    max_k=3,
    min_confidence=0.4,
    min_lift=1.0,
    top_n_itemsets=5,
    top_n_rules=5,
    drop_noise=True,
    stopwords=None
):
    """
    Retourne deux dictionnaires:
      - itemsets_by_cluster: {cluster_id: [("a","b"), ("c","d","e"), ...]}
      - rules_by_cluster: {cluster_id: [rule_dict, ...]}
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    work = df.copy()
    if drop_noise and cluster_col in work.columns:
        work = work[work[cluster_col] != -1]

    itemsets_by_cluster = {}
    rules_by_cluster = {}

    for cl in sorted(work[cluster_col].unique()):
        sub = work[work[cluster_col] == cl]
        transactions = make_transactions(sub, stopwords=stopwords)

        # si cluster trop petit, on skip
        if len(transactions) < 10:
            itemsets_by_cluster[cl] = []
            rules_by_cluster[cl] = []
            continue

        supports = apriori_itemsets(transactions, min_support=min_support, max_k=max_k)

        # itemsets: on garde surtout taille 2-3 pour nommage (lisible)
        candidates = [(it, sup) for it, sup in supports.items() if 2 <= len(it) <= max_k]
        candidates.sort(key=lambda x: x[1], reverse=True)
        itemsets_by_cluster[cl] = [it for it, _ in candidates[:top_n_itemsets]]

        # rules : on génère à partir de tous les itemsets fréquents
        rules = generate_rules_from_itemsets(supports, min_confidence=min_confidence, min_lift=min_lift)
        rules_by_cluster[cl] = rules[:top_n_rules]

    return itemsets_by_cluster, rules_by_cluster

