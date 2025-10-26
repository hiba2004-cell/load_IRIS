# 🌸 Iris Classifier – Machine Learning Project

## 🧠 Description du projet
Ce projet implémente un modèle de **Machine Learning supervisé** pour la **classification du dataset Iris** de Scikit-learn.  
L’objectif est de prédire l’espèce d’une fleur (*Iris setosa*, *Iris versicolor*, *Iris virginica*) à partir de ses caractéristiques :  
- longueur et largeur des sépales  
- longueur et largeur des pétales

Le projet inclut un **modèle d’apprentissage**, une **visualisation des données** et une **interface Streamlit** pour effectuer des prédictions interactives.

---

## 📂 Structure du projet

ris-Classifier/
│
├── data/
│ └── iris.csv
│
├── models/
│ └── iris_model.pkl
│
├── notebooks/
│ └── iris_classifier.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── evaluation.py
│ └── predict.py
│
├── requirements.txt
├── README.md
└── app.py


---

## 🔬 Dataset utilisé

Le dataset **Iris** est chargé directement depuis **Scikit-learn** :
```python
## Modèles testés

Plusieurs algorithmes ont été comparés :

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Le meilleur modèle retenu est la Random Forest, offrant la meilleure précision globale.

## Environnement & dépendances
Installation
git clone https://github.com/username/Iris-Classifier.git
cd Iris-Classifier
pip install -r requirements.txt

requirements.txt
scikit-learn
pandas
numpy
matplotlib
seaborn
streamlit
joblib

## Entraînement du modèle
python src/model_training.py


ou via le notebook :

jupyter notebook notebooks/iris_classifier.ipynb


Le modèle est sauvegardé dans le dossier models/iris_model.pkl.

## Résultats du modèle
Métrique	Valeur
Accuracy	0.97
F1-score	0.96

Le modèle présente une excellente capacité à distinguer les trois classes d’Iris.

##🖼️ Prédiction sur un nouvel échantillon
python src/predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2


Sortie :

## Prédiction : Iris-setosa

## Interface web (Streamlit)

Une interface Streamlit permet de saisir les valeurs des caractéristiques et d’obtenir instantanément la prédiction du modèle.

Lancer l’application :
streamlit run app.py

##Améliorations possibles

Intégration d’un modèle SVM optimisé par GridSearchCV

Visualisation 3D interactive des classes

Déploiement sur le web (Streamlit Cloud ou Hugging Face Spaces)

##👩‍💻 Auteur

Nom : Hiba Nadiri
École : ENSA El Jadida
Projet : Classification des fleurs Iris avec Scikit-learn
Date : Octobre 2025
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
