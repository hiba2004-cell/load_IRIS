# Iris Classifier – Machine Learning Project

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

# Charger les données
iris = load_iris()
X, y = iris.data, iris.target
print(iris.feature_names)
print(iris.data)
print(iris.target)
print(iris.target_names)

# Define models
models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=200),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=3, random_state=42),
    'SVC': SVC(),
    'GaussianNB': GaussianNB(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'KMeans': KMeans(n_clusters=3, random_state=42),
    'Gaussian Mixture Model (GMM)': GaussianMixture(n_components=3, random_state=42)
    
}

# Split the data (for supervised models only)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate
for name, model in models.items():
    if name not in ['KMeans', 'Gaussian Mixture Model (GMM)']:
        # Supervised models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.2f}")
    else:
        # Unsupervised models
        model.fit(X)
        print(f"{name} fitted (unsupervised model)")

# Évaluer
prediction = model.predict([[5, 2, 4, 3]])
print(prediction)
print(iris.target_names[prediction])
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

