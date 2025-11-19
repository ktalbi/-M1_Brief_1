# -M1_Brief_1 : Réentraînement du Modèle & Suivi MLflow

##  Objectif

Ce projet met en place un pipeline complet d’entraînement, réentraînement et suivi de modèle grâce à MLflow.  
Deux opérations principales sont réalisées :

1. **Entraînement initial** d’un modèle sur les données historiques ('df_old.csv')  
2. **Réentraînement** d’un modèle existant sur les nouvelles données ('df_new.csv')

Les scripts principaux :

- 'mlflow_monitoring.py'
- 'mlflow_retrain.py'

---

## Pipeline de Réentraînement

### 1. Prétraitement des données
Les données sont :
- chargées via pandas,
- prétraitées avec un pipeline Scikit-Learn ('preprocessor.pkl'),
- séparées en jeu d’entraînement et test via la fonction 'split()'.

### 2. Entraînement initial
Le script 'mlflow_monitoring.py' :
- crée un modèle via 'create_nn_model()',
- l’entraîne sur 'df_old.csv',
- logue les paramètres et métriques dans MLflow.

### 3. Réentraînement
Le script 'mlflow_retrain.py' :
- recharge un modèle MLflow existant,
- l’entraîne sur les nouvelles données 'df_new.csv',
- logue les nouvelles métriques,
- sauve le modèle réentraîné sous 'retrained_mlflow_model'.

---

## Métriques Suivies

Les métriques suivies dans MLflow sont :

| Métrique | Description |
|---------|-------------|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **R²** | Coefficient de détermination |

Elles permettent de suivre l’évolution précise des performances. Screenshots disponibles dans le dossier "metriques".

---

## Comparaison des Résultats

Comparaison entre le modèle initial et le modèle réentraîné (60 epochs, batch 32) :

| Métrique | Initial | Réentraînement | Conclusion |
|---------|---------|----------------|------------|
| **MAE** | 3946.3 | **2334.5** | ✔️ Erreur réduite de 40% |
| **MSE** | 25 206 445 | **13 481 003** | ✔️ Amélioration significative |
| **R²** | 0.828 | **0.875** | ✔️ Meilleure qualité prédictive |

### Conclusion
Le réentraînement apporte un **gain net de performance** sur toutes les métriques.  
Le modèle réentraîné généralise mieux et explique davantage la variance des données.

---

## Tests 

Les tests sont implémentés dans le fichier 'test_1.py' et couvrent les points suivants :

### 1. Validation de la structure du modèle
Le test vérifie que :
- la dimension d’entrée du modèle est correcte,
- le modèle possède bien les méthodes essentielles ('fit', 'predict').

### 2. Fonctionnement de l’entraînement
Via un jeu de données synthétiques généré automatiquement :
- le test vérifie que la fonction 'train_model' retourne un historique cohérent,
- que le nombre d’epochs dans l’historique est correct.

### 3. Fonction de prédiction
Un test vérifie que :
- la prédiction fonctionne,
- le MSE calculé est bien un nombre fini,
- le modèle est capable de traiter des données prétraitées et cohérentes.

### Objectif des tests
Ces tests garantissent :
- la stabilité du pipeline,
- la validité du modèle,
- la non-régression en cas de modification du code.

Ils peuvent être exécutés avec :

'''bash
pytest -q
'''

---

## Structure du Projet

'''
.
├── data/
│   ├── df_old.csv
│   ├── df_new.csv
├── models/
│   ├── preprocessor.pkl
├── tests/
│   ├── test_1.py
├── README.md
├── mlflow_monitoring.py
├── mlflow_retrain.py
'''

---

## Conclusion

Ce projet met en place un pipeline complet MLOps comprenant :

- ingestion & preprocessing,  
- entraînement et réentraînement,  
- suivi MLflow,  
- tests assurant la robustesse.

Le modèle réentraîné montre une amélioration nette, confirmant l’intérêt d’intégrer régulièrement de nouvelles données.
