# 🔍 Détection d'Anomalies Réseau avec Clustering Non Supervisé

## 📘 Description du Projet

Ce projet consiste à concevoir une application web interactive pour la **détection d'activités anormales dans le trafic réseau** à partir de données issues du dataset **NSL-KDD**, un benchmark bien connu en cybersécurité.  
Il s'appuie sur des techniques de **clustering non supervisé**, notamment l'algorithme **KMeans**, pour identifier des comportements suspects sans supervision humaine.

L'application est développée avec **Streamlit**, permettant une interface simple, intuitive et interactive.

---

## 🧠 Objectifs

- Détecter les anomalies réseau à partir de données brutes.
- Utiliser des techniques de **machine learning non supervisé** pour analyser les connexions réseau.
- Fournir une interface visuelle pour l’analyse, l’interprétation et la visualisation des anomalies.

---

## 🎯 Fonctionnalités Principales

- 📤 Téléversement dynamique de fichiers CSV
- 🔧 Nettoyage et encodage des données (one-hot)
- 🔽 Réduction de dimension avec **PCA** (2D ou 3D)
- 🤖 Clustering avec **KMeans**
- 📊 Visualisation interactive des clusters (Seaborn, Plotly)
- 🔁 Rechargement et traitement automatique
- 📋 Analyse et affichage des statistiques descriptives

---
## 📂 Structure du Projet


├── appfinal.py # Code principal Streamlit'
├── data/'
│ ├── KDDTrain+.csv # Dataset complet
│ └── KDDTrainmoitier.csv # Sous-ensemble du dataset
├── requirements.txt # Fichier de dépendances Python
└── README.md # Documentation du projet

---

## Installer les dépendances :

pip install -r requirements.txt


---

## Lancer l'application :

streamlit run appfinal.py
