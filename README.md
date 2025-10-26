# load_iris

## Description

Ce petit projet montre comment charger et exploiter le jeu de données Iris (fichier intégré de scikit-learn). Il contient des étapes de :

* chargement des données,
* exploration et visualisation basique,
* préparation (train/test),
* entraînement d'un modèle simple (ex. : LogisticRegression, RandomForest),
* évaluation et sauvegarde du modèle.

L'objectif est pédagogique : servir de point de départ pour des TP ou prototypes ML.

---

## Prérequis

* Python 3.8+
* Bibliothèques Python :

  * scikit-learn
  * pandas
  * numpy
  * matplotlib
  * seaborn (optionnel)

Installation rapide :

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## Arborescence suggérée

```
load_iris/
├── README.md
├── requirements.txt
├── load_iris_example.py
├── notebooks/
│   └── EDA_iris.ipynb
└── models/
    └── iris_model.pkl
```

---

## Exemple de script (Python)

Fichier `load_iris_example.py` :

```python
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Charger les données
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

print('Aperçu des features :')
print(X.head())
print('\nClasses :', iris.target_names)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entraîner un modèle simple
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)
print('\nAccuracy:', accuracy_score(y_test, y_pred))
print('\nClassification report:\n', classification_report(y_test, y_pred, target_names=iris.target_names))

# Sauvegarder le modèle
joblib.dump(model, 'models/iris_model.pkl')
print('\nModèle sauvegardé dans models/iris_model.pkl')
```

---

## Notions d'exploration (EDA) recommandées

* affichage des 4 features (boxplots / histograms)
* scatter matrix (pd.plotting.scatter_matrix) ou pairplot seaborn
* vérifier la distribution par classe

Exemple rapide avec seaborn :

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = X.copy()
df['target'] = y
sns.pairplot(df, hue='target', vars=iris.feature_names)
plt.show()
```

---

## Conseils pour aller plus loin

* tester plusieurs modèles (SVM, KNN, LogisticRegression)
* normaliser / standardiser les features (StandardScaler)
* faire une recherche d'hyperparamètres (GridSearchCV / RandomizedSearchCV)
* pipeline scikit-learn pour enchaîner preprocessing + modèle
* sauvegarder et charger modèle avec `joblib` ou `pickle`

---

## Fichiers utiles

* `requirements.txt` : lister les dépendances (ex. : `scikit-learn==1.4.0`, `pandas`, `numpy`, ...)
* `notebooks/EDA_iris.ipynb` : notebook pour visualisations interactives

---

## Contact
## 👩‍💻 Auteur
**Nom :** Hiba Nadiri  
**École :** ENSA El Jadida  
**Projet :** Classification des vêtements avec CNN (Fashion MNIST)  
**Date :** Octobre 2025

Pour toute question ou amélioration proposée, ouvre une issue ou contacte l'auteur.

