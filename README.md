# SetFit Fine-Tuning — Few-Shot Text Classification

## Description du projet

Ce projet a pour objectif de réaliser une tâche de **classification de texte en few-shot learning** avec **SetFit**.

Le few-shot learning consiste à entraîner un modèle avec très peu d'exemples annotés. Dans ce projet, plusieurs expériences ont été réalisées en faisant varier :

* le nombre d'échantillons d'entraînement ;
* le nombre d'epochs ;
* les performances obtenues sur un même jeu de test fixe.

Le projet respecte les consignes suivantes :

* utiliser un dataset de classification de texte ;
* diviser le dataset en 80 % pour l'entraînement / évaluation et 20 % pour le test ;
* entraîner plusieurs modèles SetFit avec différentes configurations ;
* évaluer chaque modèle avec les mêmes métriques ;
* comparer les résultats obtenus.

---

## Dataset utilisé

Le dataset utilisé est :

```text
SetFit/sst2
```

Il s'agit d'un dataset de classification de sentiment. Chaque texte est associé à une étiquette :

| Label | Signification     |
| ----: | ----------------- |
|     0 | Sentiment négatif |
|     1 | Sentiment positif |

Après fusion des splits du dataset, le nombre total d'exemples est :

```text
9613 exemples
```

La répartition des labels est :

| Label | Nombre d'exemples |
| ----: | ----------------: |
|     0 |              4650 |
|     1 |              4963 |

Le dataset a ensuite été divisé ainsi :

| Partie             | Nombre d'exemples |
| ------------------ | ----------------: |
| Train + évaluation |              7690 |
| Test fixe          |              1923 |

Le même test set a été utilisé pour toutes les expériences.

---

## Modèle utilisé

Le modèle utilisé pour les expériences est :

```text
sentence-transformers/paraphrase-MiniLM-L3-v2
```

Ce modèle a été choisi car il est léger et adapté à un entraînement sur CPU.

---

## Configuration technique

L'environnement utilisé :

```text
Python 3.12.10
SetFit
Transformers 4.44.2
Sentence-Transformers
Datasets
Scikit-learn
Pandas
```

L'entraînement a été réalisé sur CPU, car la carte graphique NVIDIA GT 730 détectée sur la machine est trop ancienne pour être correctement supportée par les versions modernes de PyTorch.

---

## Configurations testées

Les expériences ont été réalisées avec les nombres d'échantillons suivants :

```text
8, 10, 20, 50, 100
```

Et avec les nombres d'epochs suivants :

```text
1, 5, 10
```

Cela donne un total de :

```text
5 × 3 = 15 expériences
```

---

## Métriques utilisées

Chaque modèle a été évalué avec les métriques suivantes :

* Accuracy
* Precision
* Recall
* F1 score

Ces métriques permettent d'évaluer la qualité de la classification.

---

## Résultats obtenus

| Expérience | Training samples | Epochs | Accuracy | Precision | Recall | F1 score |
| ---------: | ---------------: | -----: | -------: | --------: | -----: | -------: |
|          1 |                8 |      1 |   0.5772 |    0.5896 | 0.5962 |   0.5929 |
|          2 |                8 |      5 |   0.5751 |    0.5873 | 0.5962 |   0.5917 |
|          3 |                8 |     10 |   0.5803 |    0.5973 | 0.5750 |   0.5859 |
|          4 |               10 |      1 |   0.5424 |    0.5673 | 0.4794 |   0.5197 |
|          5 |               10 |      5 |   0.5512 |    0.5754 | 0.4995 |   0.5348 |
|          6 |               10 |     10 |   0.5507 |    0.5756 | 0.4945 |   0.5320 |
|          7 |               20 |      1 |   0.6183 |    0.6400 | 0.5962 |   0.6173 |
|          8 |               20 |      5 |   0.6178 |    0.6364 | 0.6062 |   0.6209 |
|          9 |               20 |     10 |   0.6157 |    0.6326 | 0.6103 |   0.6212 |
|         10 |               50 |      1 |   0.7051 |    0.7341 | 0.6727 |   0.7020 |
|         11 |               50 |      5 |   0.7020 |    0.7147 | 0.7039 |   0.7093 |
|         12 |               50 |     10 |   0.7031 |    0.7171 | 0.7019 |   0.7094 |
|         13 |              100 |      1 |   0.7233 |    0.7536 | 0.6898 |   0.7203 |
|         14 |              100 |      5 |   0.7197 |    0.7506 | 0.6848 |   0.7162 |
|         15 |              100 |     10 |   0.7135 |    0.7500 | 0.6677 |   0.7064 |

---

## Meilleur résultat

Le meilleur résultat est obtenu avec la configuration suivante :

```text
Training samples : 100
Epochs : 1
Accuracy : 0.7233
Precision : 0.7536
Recall : 0.6898
F1 score : 0.7203
```

Cette expérience correspond à l'expérience numéro 13.

---

## Graphique des résultats

Le graphique suivant montre l'évolution du F1 score selon le nombre d'échantillons d'entraînement et le nombre d'epochs.

![Évolution du F1 score](results_setfit/setfit_f1_score.png)

---

## Analyse des résultats

Les résultats montrent que l'augmentation du nombre d'échantillons d'entraînement améliore globalement les performances du modèle.

Avec seulement 8 ou 10 exemples, le modèle obtient des résultats limités. À partir de 50 exemples, les performances deviennent nettement meilleures.

Le meilleur F1 score est obtenu avec 100 échantillons et 1 epoch.

On remarque également qu'augmenter le nombre d'epochs n'améliore pas toujours les performances. Par exemple, avec 100 échantillons, le modèle entraîné pendant 1 epoch obtient un meilleur F1 score que les modèles entraînés pendant 5 ou 10 epochs.

Cela peut indiquer un léger surapprentissage lorsque le modèle est entraîné trop longtemps sur un petit nombre de données.

---

## Conclusion

Ce projet montre que SetFit est une méthode intéressante pour réaliser de la classification de texte avec peu d'exemples annotés.

Les expériences montrent que :

* SetFit peut fonctionner avec très peu de données ;
* plus le nombre d'exemples augmente, meilleures sont les performances ;
* un nombre plus élevé d'epochs n'améliore pas toujours les résultats ;
* le meilleur compromis obtenu dans ce projet est 100 échantillons avec 1 epoch.

---

## Structure du projet

```text
nlp_ning/
│
├── setfit_experiments.py
├── plot_setfit_results.py
├── README.md
│
└── results_setfit/
    ├── setfit_results.csv
    ├── setfit_results.xlsx
    ├── setfit_results_progress.csv
    └── setfit_f1_score.png
```

---

## Lancer le projet

### 1. Activer l'environnement virtuel

```powershell
.\.venv-1\Scripts\Activate.ps1
```

### 2. Installer les dépendances principales

```powershell
python -m pip install setfit==1.1.0 transformers==4.44.2 sentence-transformers==3.0.1 datasets==2.21.0 evaluate scikit-learn pandas openpyxl matplotlib
```

### 3. Lancer les expériences

```powershell
python setfit_experiments.py
```

### 4. Générer le graphique

```powershell
python plot_setfit_results.py
```

---

## Fichiers de résultats

Les résultats sont disponibles dans le dossier :

```text
results_setfit/
```

Les fichiers générés sont :

```text
setfit_results.csv
setfit_results.xlsx
setfit_results_progress.csv
setfit_f1_score.png
```

---

## Auteur

Projet réalisé par Joachim Quérule dans le cadre du projet final NLP.
