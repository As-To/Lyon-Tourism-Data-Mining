# Data_mining

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
