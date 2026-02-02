import re
import numpy as np
import pandas as pd
import math
import unicodedata
from itertools import combinations
from collections import Counter

# Imports Scikit-Learn et NLTK
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import nltk
from nltk.corpus import stopwords

# Téléchargement silencieux des stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# --- 1. CONFIGURATION GLOBALE (STOPWORDS & SPACY) ---

# Chargement UNIQUE de Spacy pour la performance
try:
    import spacy
    # On essaie de charger le modèle français, sinon anglais, sinon rien.
    try:
        nlp = spacy.load("fr_core_news_sm")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None
except ImportError:
    nlp = None

# Définition des Stopwords
FRENCH_STOPWORDS = set(stopwords.words("french"))
EXTRA_STOPWORDS = {
    "photo", "image", "tag", "titre", "title",
    "uploaded", "upload", "flickrmobile", "flickriosapp",
    "instagram", "instagramapp", "iphoneography", "squareformat",
    "square", "filter", "nofilter", "iphone",
    "img", "img_", "dsc", "jpg", "jpeg", "png", "bokeh",
    "lyoncity", "lyonnais", "fr", "incity", "lyon", "france", "europe", 
    "city", "ville", "street", "urban", "urbain", "day", "europa",
    "paper", "pasted", "pasteup", "wheatpaste", "wheatpaper", "collage",
    "nikon", "canon", "streetart",
    "foursquare", "venue", "xproii", "proii", "vscocam", "vsco", "rhone", 
    "rhonealpes", "moto", "bb", "sncf",
    "photocmobile", "eos", "pentax", "pentaxk", "mm", "fb", "eb", 
    "doctorwhoclassic", "jossarisfoto", "lr", "fe", "sec", "mmf", "dslr", 
    "xpro", "fujifilm", "fuji", "bij", "bfd", "jossaris", "patman", 
    "choupinou", "architecture", "building", "tower", "villedelyon",
    "metropolisoflyon", "auvergnerhonealpes", "bmx", "_dsc", 
    "creditphotosjonathantmare", "iso", "oss"
}

# Fusion des listes
DEFAULT_STOPWORDS = set(ENGLISH_STOP_WORDS) | FRENCH_STOPWORDS | EXTRA_STOPWORDS


# --- 2. FONCTIONS DE NETTOYAGE (Unifiées) ---

def strip_accents(s: str) -> str:
    """Supprime les accents (ex: 'été' -> 'ete')."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def basic_preprocess(s: str) -> str:
    """Nettoyage complet : minuscules, suppression technique, accents, etc."""
    s = str(s).lower()
    # Supprime les mots techniques type img_1234, dsc_5678
    s = re.sub(r"\b(img|dsc)\w+\b", " ", s)
    s = re.sub(r"\d+", " ", s) # Supprime les chiffres
    s = re.sub(r"[^\w\s]", " ", s) # Supprime ponctuation/caractères spéciaux
    s = re.sub(r"\s+", " ", s).strip() # Nettoie les espaces
    s = strip_accents(s) # Supprime les accents
    return s

def build_text(df: pd.DataFrame) -> pd.Series:
    """Concatène tags + title."""
    # Sécurité si les colonnes manquent
    tags = df["tags"].fillna("") if "tags" in df.columns else pd.Series([""] * len(df))
    title = df["title"].fillna("") if "title" in df.columns else pd.Series([""] * len(df))
    return (tags + " " + title).str.strip()

def normalize_stopwords(stopwords_set):
    """Applique le même nettoyage aux stopwords pour qu'ils matchent le texte."""
    return {basic_preprocess(w) for w in stopwords_set}

# On normalise la liste globale une bonne fois pour toutes
DEFAULT_STOPWORDS = normalize_stopwords(DEFAULT_STOPWORDS)


def lemmatize_optional(text: str):
    """Lemmatisation optimisée (utilise l'objet nlp chargé globalement)."""
    if nlp is None:
        return text
    try:
        doc = nlp(text)
        return " ".join([t.lemma_ if t.lemma_ != "-PRON-" else t.text for t in doc])
    except Exception:
        return text


# --- 3. ALGORITHME 1 : TF-IDF (Top Terms) ---

def remove_redundant_unigrams(terms):
    """Supprime 'rhone' si 'rhone alpes' est présent."""
    bigrams = {t for t in terms if " " in t}
    unigrams = set(terms)
    to_remove = set()
    for bg in bigrams:
        parts = bg.split()
        for p in parts:
            if p in unigrams:
                to_remove.add(p)
    return [t for t in terms if t not in to_remove]

def top_terms_by_cluster(df, cluster_col="cluster", top_k=10, stopwords=None, use_lemmas=True, drop_noise=True):
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    if isinstance(stopwords, set):
        stopwords = sorted(list(stopwords))

    work = df.copy()
    if drop_noise and cluster_col in work.columns:
        work = work[work[cluster_col] != -1]

    # Optimisation : Utilise la colonne pré-calculée si dispo
    if 'clean_text' in work.columns:
        texts = work['clean_text']
    else:
        texts = build_text(work).map(basic_preprocess)
        if use_lemmas:
            texts = texts.map(lemmatize_optional)

    if texts.empty:
        return {}

    # TF-IDF
    try:
        vectorizer = TfidfVectorizer(
            stop_words=stopwords,
            strip_accents="unicode",
            min_df=2, # Au moins 2 occurrences pour être pertinent
            max_df=0.9,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(texts)
        terms = np.array(vectorizer.get_feature_names_out())
    except ValueError:
        # Cas où le vocabulaire est vide après filtrage
        return {}

    results = {}
    for cl in sorted(work[cluster_col].unique()):
        # On utilise l'index booléen pour récupérer les lignes du cluster
        idx = np.where(work[cluster_col].values == cl)[0]
        if len(idx) == 0:
            continue
        
        # Moyenne du TF-IDF pour ce cluster
        mean_tfidf = X[idx].mean(axis=0).A1
        # Indices des meilleurs mots
        top_idx = mean_tfidf.argsort()[::-1][:top_k]
        
        cluster_terms = list(terms[top_idx])
        cluster_terms = remove_redundant_unigrams(cluster_terms)
        results[cl] = cluster_terms

    return results


# --- 4. ALGORITHME 2 : APRIORI (Règles d'association) ---

def tokenize(text: str, stopwords=DEFAULT_STOPWORDS, min_len=2):
    if not text:
        return []
    toks = text.split()
    return [t for t in toks if len(t) >= min_len and t not in stopwords]

def make_transactions(df: pd.DataFrame, stopwords=DEFAULT_STOPWORDS) -> list[set[str]]:
    # Optimisation : Utilise la colonne pré-calculée
    if 'clean_text' in df.columns:
        texts = df['clean_text']
    else:
        texts = build_text(df).map(basic_preprocess)
        
    transactions = []
    for t in texts:
        toks = tokenize(t, stopwords=stopwords)
        if toks:
            transactions.append(set(toks))
    return transactions

def apriori_itemsets(transactions, min_support=0.05, max_k=3):
    n = len(transactions)
    if n == 0: return {}
    min_count = math.ceil(min_support * n)

    # L1
    counts = Counter()
    for t in transactions:
        for item in t:
            counts[(item,)] += 1
            
    L_curr = {it: c/n for it, c in counts.items() if c >= min_count}
    supports = L_curr.copy()
    
    k = 2
    while L_curr and k <= max_k:
        # Génération candidats (naïve mais suffisante pour k=2 ou 3)
        # Pour optimiser, on pourrait faire un join plus intelligent
        # mais vu la taille des descriptions courtes, c'est acceptable.
        items_in_L = set(item for it in L_curr for item in it)
        sorted_L = sorted(L_curr.keys())
        
        counts_k = Counter()
        # On scanne les transactions pour compter les k-itemsets
        # Optimisation : on ne garde que les items fréquents dans la transaction
        for t in transactions:
            relevant_items = sorted([x for x in t if x in items_in_L])
            for cand in combinations(relevant_items, k):
                counts_k[cand] += 1
        
        L_next = {it: c/n for it, c in counts_k.items() if c >= min_count}
        if not L_next:
            break
            
        supports.update(L_next)
        L_curr = L_next
        k += 1
        
    return supports

def generate_rules_from_itemsets(itemset_supports, min_confidence=0.4, min_lift=1.0):
    rules = []
    for itemset, sup_ab in itemset_supports.items():
        if len(itemset) < 2: continue
        
        # Pour chaque sous-ensemble A
        for r in range(1, len(itemset)):
            for A in combinations(itemset, r):
                A = tuple(sorted(A))
                B = tuple(sorted(set(itemset) - set(A)))
                
                sup_a = itemset_supports.get(A)
                sup_b = itemset_supports.get(B)
                
                if sup_a and sup_b:
                    conf = sup_ab / sup_a
                    lift = conf / sup_b
                    
                    if conf >= min_confidence and lift >= min_lift:
                        rules.append({
                            "antecedent": ", ".join(A), # Plus joli pour l'affichage
                            "consequent": ", ".join(B),
                            "support": sup_ab,
                            "confidence": conf,
                            "lift": lift
                        })
    
    rules.sort(key=lambda d: (d["lift"], d["confidence"]), reverse=True)
    return rules

def apriori_by_cluster(df, cluster_col="cluster", min_support=0.05, max_k=3, min_confidence=0.4, top_n=5, stopwords=DEFAULT_STOPWORDS):
    
    # Sécurisation des arguments
    if stopwords is None: stopwords = DEFAULT_STOPWORDS
    
    work = df[df[cluster_col] != -1] if cluster_col in df.columns else df.copy()
    
    itemsets_by_cl = {}
    rules_by_cl = {}
    
    for cl in sorted(work[cluster_col].unique()):
        sub = work[work[cluster_col] == cl]
        # Skip si trop petit
        if len(sub) < 10: 
            continue
            
        transactions = make_transactions(sub, stopwords)
        supports = apriori_itemsets(transactions, min_support, max_k)
        
        # Top Itemsets
        sorted_items = sorted(supports.items(), key=lambda x: x[1], reverse=True)
        # On filtre les singletons pour l'affichage, c'est souvent trivial
        itemsets_by_cl[cl] = [it for it, sup in sorted_items if len(it) > 1][:top_n]
        
        # Rules
        rules = generate_rules_from_itemsets(supports, min_confidence)
        rules_by_cl[cl] = rules[:top_n]
        
    return itemsets_by_cl, rules_by_cl