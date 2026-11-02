# Qui utilise mon appli ?

## Instructions pour le venv

### 1. Créer le virtual environment

```bash
uv venv
```

### 2. Activer le venv

Sur macOS / Linux :

```bash
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
uv pip install -r requirements.txt
```

### 4. Vérifier l'installation

```bash
python --version
```

Le projet est maintenant prêt à être utilisé.

## Organisation du projet

Les notebooks se lisent dans l'ordre :

| Notebook | Contenu |
|---|---|
| `0_consignes.ipynb` | consignes du projet |
| `1_exploration.ipynb` | chargement, première analyse des données, valeurs aberrantes |
| `2_features_manuelles_modeles.ipynb` | approche 1 : 143 features construites à la main, un modèle par cellule, comparaison, soumission |
| `3_tfidf_modeles.ipynb` | approche 2 (finale) : sessions représentées en TF-IDF, choix des jetons, modèles testés, contrainte navigateur, soumission |

Dans les notebooks 2 et 3, la dernière cellule génère une soumission dans `submissions/` pour le modèle choisi (variable `MODELE_SOUMISSION`).

- `demarche.md` : synthèse de la démarche et de tous les résultats.
- `explication_tfidf.md` : la TF-IDF et le SVM linéaire, suivis sur une vraie session.
- Les données (`data/train.csv`, `data/test.csv`) ne sont pas versionnées.
