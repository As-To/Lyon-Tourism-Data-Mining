# Data_mining

Partie 1 : Cleaning 

1) délimiter la zone de lyon
1) réfléchir à l'arborescence de fichiers  
2) retirer les lignes avec des dates incohérentes 
3) retirer les lignes de bug (les lignes avec des colonnes en plus) 
4) définir les rayons "de même zone"
5) retirer les duplicats : même lieu, même user, même date --> on choisit une précision du lieu au niveau du "jour"
--> si une personne fréquente un même lieu a deux jours différents on la comptabilise deux fois
6) retirer les colonnes que l'on ne va pas utiliser
7) garder une trace du nombre de lignes supprimées par filtre 
8) tester d'abord sur sample_data
9) absence de tag et description (non prioritaire) 


## Arborescence du projet

Voici l'organisation actuelle des fichiers :

- data/  
	- raw/  
		- flickr_data2.csv  (données brutes)  
	- cleaned/  
	- sample/  
		- sample_data.csv  (petit échantillon pour tests)
- src/  
	- cleaning/  
		- data_cleaning.py    (orchestrateur : lance les étapes d'analyse et de nettoyage)  
		- load_data.py        (chargement des fichiers CSV/XLSX et normalisation des noms de colonnes)  
		- clean_geo.py        (filtres géographiques — ex. zone de Lyon)  
		- clean_dates.py      (nettoyage lié aux dates, suppression des années incohérentes, colonnes d'upload, lignes buggées)  
		- clean_duplicates.py (détection/suppression des photos dupliquées)  
		- utils.py            (fonctions utilitaires — ex. sauvegarde)
	- clustering/  
	- visualization/  

## Explication rapide

- `data/raw/flickr_data2.csv` contient les données brutes importées. On travaille d'abord sur un échantillon (`data/sample/sample_data.csv`) pour les tests.  
- Les fonctions de nettoyage ont été séparées par responsabilité : chargement (`load_data`), géo (`clean_geo`), dates et colonnes (`clean_dates`), dédoublonnage (`clean_duplicates`).  
- `data_cleaning.py` orchestre l'enchaînement : analyse exploratoire, filtrage géographique (Lyon), suppression des doublons, nettoyage des dates/colonnes, puis sauvegarde optionnelle.  
- Les noms de colonnes sont normalisés (`.str.strip()`) au chargement pour éviter les erreurs dues aux espaces.  

