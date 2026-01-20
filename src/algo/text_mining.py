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


# la bonne version est celle la : 


import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


###


# FRENCH_STOPWORDS = {
#     "alors","au","aucun","aussi","autre","avant","avec","car","ce","cela","ces","cet","cette",
#     "ceux","chaque","comme","comment","dans","de","des","du","donc","elle","elles","en","encore",
#     "est","et","eu","fait","hormis","il","ils","je","jusqu","la","le","les","leur","leurs",
#     "lui","ma","mais","mes","moi","mon","ne","nos","notre","nous","on","ou","par","pas","peu",
#     "plus","pour","qu","que","qui","sa","se","ses","si","son","sont","sous","sur","ta","te",
#     "tes","toi","ton","tous","tout","tres","tu","un","une","vos","votre","vous","y"
# }
# EXTRA_STOPWORDS = {"photo", "image", "tag", "titre", "title"}

FRENCH_STOPWORDS = set(stopwords.words("french"))

EXTRA_STOPWORDS = {
    "photo", "image", "tag", "titre", "title",

    
    "uploaded", "upload", "flickrmobile", "flickriosapp",
    "instagram", "instagramapp", "iphoneography", "squareformat",
    "square", "filter", "nofilter", "iphone",

   
    "img", "img_", "dsc", "jpg", "jpeg", "png",

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
    "foursquare", "venue", "foursquare venue", "xproii", "proii", "vscocam", "vsco"

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
