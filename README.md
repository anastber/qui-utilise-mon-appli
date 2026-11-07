# Qui utilise mon appli ?

Projet réalisé dans le cadre du cours MOD 7.2 (Introduction à la science des données) de l'École Centrale de Lyon, encadré par Julien Velcin et Antoine Becquet.

Identifier l'utilisateur de Copilote (247 utilisateurs) à partir de la trace d'une session. Challenge Kaggle évalué au F1 macro.

**Solution finale : TF-IDF + SVM linéaire (`LinearSVC`) + contrainte « même navigateur », 0,988 sur Kaggle.**

## Installation

Python 3.14 est requis.

Avec [uv](https://docs.astral.sh/uv/) :

```bash
uv venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
uv pip install -r requirements.txt
```

Ou avec pip :

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Données

Les données ne sont pas versionnées. Les télécharger depuis l'onglet *Data* de la page Kaggle de la compétition, et les placer ainsi :

```
data/
├── train.csv
└── test.csv
```

## Lancer les notebooks

```bash
jupyter lab
```

Les notebooks se lisent et s'exécutent dans l'ordre :

| Notebook | Contenu | Durée |
|---|---|---|
| `0_consignes.ipynb` | consignes du projet | -- |
| `1_exploration.ipynb` | chargement, exploration des données, valeurs extrêmes | < 1 min |
| `2_features_manuelles_modeles.ipynb` | approche 1 : 756 variables construites à la main, un modèle par cellule, comparaison, soumission | ~ 10 min |
| `3_tfidf_modeles.ipynb` | approche 2 (finale) : sessions représentées en TF-IDF, choix des jetons, modèles testés, contrainte navigateur, soumission | ~ 10 min |

Ce qu'ils produisent :

- `images/` : les figures reprises dans le rapport ;
- `submissions/` : un fichier de soumission Kaggle. Dans les notebooks 2 et 3, la dernière cellule entraîne sur tout le train le modèle choisi (variable `MODELE_SOUMISSION`) et écrit ses prédictions. La soumission finale est `submissions/tfidf_linearsvc_c_30_navigateur.csv`.

## Document

`demarche.md` : synthèse de la démarche et de tous les résultats.

## Contributeurs

- [anastber](https://github.com/anastber)
- [AYR12](https://github.com/AYR12)
- [yangerr2](https://github.com/yangerr2)
