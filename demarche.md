# Démarche : features, traitements et modèles

Objectif : identifier l'utilisateur de Copilote (247 trigrammes) à partir de la trace d'une session. Métrique du challenge : F1 moyen (on suit le F1 macro, plus l'accuracy).

**Résultat final : TF-IDF + SVM linéaire + contrainte « même navigateur », 0,988 sur Kaggle** (0,925 pour notre première soumission ExtraTrees). Notebooks : `2_features_manuelles_modeles.ipynb` (approche 1) et `3_tfidf_modeles.ipynb` (approche 2, finale).

## 1. Ce que contiennent les données

Une ligne = une session : `util, navigateur, action, action, …, tXX, …` (`util` absent du test).
Une action a la forme `Type(écran)<configuration>$chaîne$`, avec un `1` final si l'utilisateur est **en modification** dans la fiche. Les marqueurs `tXX` sont des repères de temps en secondes.

- **Classes déséquilibrées** : 13 sessions par utilisateur en moyenne (de 4 à 75). Toujours prédire le plus fréquent donne 2,3 % d'accuracy. D'où le F1 macro, où chaque utilisateur compte autant.
- **Longueurs très asymétriques** (médiane 340 actions, maximum 11 977). Durée et nombre d'actions sont redondants (≈ 0,9 action/s).
- **Chaque utilisateur n'a qu'un navigateur** (vérifié sur tout le train). Opera = un seul utilisateur.

## 2. Valeurs aberrantes

On retire du train les **3 sessions de moins de 10 actions** (dont 1 vide), où les fréquences n'ont pas de sens. Les sessions très longues (297 au-delà de 1817 actions) et les rafales (50 sessions à plus de 2 actions/s, ex. `avt`) sont gardées : ce sont de vraies sessions, souvent caractéristiques d'un utilisateur. Le test n'est jamais filtré.

## 3. Deux représentations des sessions


### 3.1 Features manuelles (143 colonnes) — `2_features_manuelles_modeles.ipynb`

Fréquences relatives (compte ÷ nombre d'actions) des 35 types d'action et des 50 écrans, 20 configurations et 30 chaînes les plus courants, plus 4 statistiques (`nb_actions_log`, `actions_par_s`, `part_modification`, `nb_types_distincts`) et le navigateur en one-hot.

- **Fréquences relatives plutôt que comptes** : un compte mélange l'utilisateur et la longueur de la session.
- **Limite du top-K** : 32 écrans et 20 configurations ne sont utilisés que par **un seul utilisateur** (des signatures), et aucun n'est dans le top-K.

### 3.2 Représentation textuelle TF-IDF (46 666 jetons) — `3_tfidf_modeles.ipynb`

Chaque session devient un document. Jetons : l'action complète et les bigrammes d'actions consécutives, le type seul et ses bigrammes, l'écran, la configuration, la chaîne et le navigateur. TF-IDF avec `sublinear_tf` (log des comptes) et `min_df=2`, ajustée dans chaque pli (pas de fuite).

Elle garde tout le vocabulaire, signatures comprises, et donne du poids aux jetons propres à un utilisateur.

## 4. Modèles testés

Validation croisée stratifiée à 4 plis (la classe la plus rare n'a que 4 sessions), même découpage pour tous les modèles. Les chiffres sont ceux des notebooks.

### 4.1 Sur les features manuelles

| Modèle | Accuracy | F1 macro |
|---|---|---|
| Classe majoritaire | 0,023 | 0,000 |
| LightGBM (réglage prudent) | 0,766 | 0,714 |
| XGBoost (`max_depth=3`) | 0,859 | 0,849 |
| Régression logistique (standardisée, C=1) | 0,867 | 0,858 |
| Random Forest (300 arbres) | 0,924 | 0,921 |
| ExtraTrees (500 arbres, défaut) | 0,931 | 0,930 |
| **ExtraTrees `max_features=0.3`** | **0,940** | **0,940** |

### 4.2 Sur la TF-IDF

| Modèle | F1 macro | F1 macro + contrainte navigateur |
|---|---|---|
| ExtraTrees (3000 meilleurs jetons, χ²) | à remplir | à remplir |
| SVM linéaire libsvm (`SVC(kernel="linear")`, C=3, un contre un) | à remplir | à remplir |
| LinearSVC C=1 | 0,943 | à remplir |
| **LinearSVC C=3** | **0,949** | **0,967** (soumission Kaggle : **0,988**) |
| LinearSVC C=10 | 0,950 | à remplir |
| LinearSVC C=30 | à remplir | à remplir |

Choix des jetons (LinearSVC, C=3) : actions + bigrammes, + trigrammes, vues multiples (retenue), vues multiples + trigrammes : à remplir.

**Contrainte « même navigateur »** : on prédit le meilleur utilisateur parmi ceux qui utilisent le navigateur de la session. Le SVM ne voit le navigateur que comme un jeton parmi des milliers : sans la contrainte, une partie de ses erreurs prédit un utilisateur d'un autre navigateur (66 évitées pour LinearSVC C=3).

### 4.3 Pourquoi ce classement

- **Le SVM linéaire est fait pour ce type de données** : des dizaines de milliers de colonnes creuses pour ~3300 sessions, où les classes sont presque séparables par un hyperplan.
- **Un contre tous plutôt qu'un contre un** : `LinearSVC` entraîne un classifieur par utilisateur sur toutes les sessions ; `SVC(kernel="linear")` en entraîne un par paire d'utilisateurs (30 381), chacun sur ~26 sessions seulement, et ses scores par utilisateur, reconstruits à partir de votes, se prêtent moins bien à la contrainte navigateur.
- **Les arbres se perdent dans les colonnes creuses** de la TF-IDF, mais restent le meilleur choix sur des features denses et construites à la main.
- **Le boosting sur-apprend** avec ~13 sessions par classe (un arbre par classe et par itération), et il est beaucoup plus lent.

## 5. Résultats Kaggle et analyse des erreurs

| Soumission | Validation croisée (F1 macro) | Kaggle |
|---|---|---|
| ExtraTrees (défaut), features manuelles | 0,930 | 0,925 |
| LinearSVC C=3 + contrainte navigateur, TF-IDF | 0,967 | **0,988** |

Le score Kaggle du SVM dépasse sa validation croisée. Explication probable (non vérifiée) : en validation, le modèle n'apprend que sur 3/4 des sessions de chaque utilisateur (≈ 10 au lieu de 13), un vrai handicap pour les petites classes ; le modèle soumis est entraîné sur tout le train.

Erreurs selon la longueur de la session (prédictions hors-pli) :

| Longueur (actions) | ExtraTrees | SVM + contrainte |
|---|---|---|
| ≤ 131 | 14,6 % | 7,8 % |
| 131 à 250 | 7,2 % | 2,9 % |
| 250 à 484 | 4,1 % | 2,3 % |
| 484 à 997 | 2,9 % | 0,8 % |
| > 997 | 1,4 % | 0,3 % |

- Les sessions courtes restent les plus difficiles, mais le SVM y divise les erreurs par deux : une partie du signal était présente, la représentation manuelle ne le captait pas.
- Avec l'ExtraTrees, les erreurs sont concentrées sur quelques utilisateurs Chrome aux sessions courtes et aux habitudes proches (`flj`, `skm`, `slq`).

## 6. Limites

- **La contrainte navigateur** suppose qu'aucun utilisateur ne change de navigateur entre train et test : vrai à 100 % sur le train, invérifiable sur le test (le score Kaggle suggère qu'elle tient).
- **Beaucoup de comparaisons sur les mêmes données** : `C`, les jetons et `max_features` ont été choisis sur la validation croisée qui les évalue ; les écarts inférieurs à ~0,005 ne sont pas fiables.
- **Non fait** : correction de la durée (dernière fenêtre `tXX` jamais fermée, impact jugé faible).
