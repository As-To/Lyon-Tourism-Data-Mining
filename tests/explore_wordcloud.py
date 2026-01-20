import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "cleaned", "cleaned_lyon_data.csv")
df = pd.read_csv(file_path)



from wordcloud_utils import plot_wordcloud
from src.algo.text_mining import build_text, basic_preprocess, DEFAULT_STOPWORDS, FRENCH_STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud_utils import show_wordcloud

# chargement des données
# df = pd.read_csv(file_path)

# # construction du texte
# texts = build_text(df).map(basic_preprocess)

# # word cloud globale
# plot_wordcloud(
#     texts,
#     stopwords=DEFAULT_STOPWORDS,
#     title="Word cloud globale - données nettoyées"
# )



# chargement du dataset nettoyé
# df = pd.read_csv("../../data/cleaned/cleaned_lyon_data.csv")

# STOPWORDS DE BASE (EN + FR seulement)
BASE_STOPWORDS = set(ENGLISH_STOP_WORDS) | FRENCH_STOPWORDS

# Word cloud AVANT
show_wordcloud(
    df,
    stopwords=BASE_STOPWORDS,
    title="Avant stopwords spécifiques Flickr"
)

# Word cloud APRÈS
show_wordcloud(
    df,
    stopwords=DEFAULT_STOPWORDS,
    title="Après stopwords spécifiques Flickr"
)
