# Démarche : features, traitements et modèles

Objectif : identifier l'utilisateur de Copilote (247 trigrammes) à partir de la trace d'une session. Métrique du challenge : F1 moyen (on suit le F1 macro, plus l'accuracy).

**Résultat final : TF-IDF + SVM linéaire + contrainte « même navigateur », 0,988 sur Kaggle** (0,944 pour l'ExtraTrees sur les features manuelles). Soumission finale : `C=30`, le meilleur en validation croisée ; `C=3` obtient exactement le même score sur Kaggle. Notebooks : `2_features_manuelles_modeles.ipynb` (approche 1) et `3_tfidf_modeles.ipynb` (approche 2, finale).

## 1. Ce que contiennent les données

Une ligne = une session : `util, navigateur, action, action, …, tXX, …` (`util` absent du test).
Une action a la forme `Type(écran)<configuration>$chaîne$`, avec un `1` final si l'utilisateur est **en modification** dans la fiche. Les marqueurs `tXX` sont des repères de temps en secondes.

- **Classes déséquilibrées** : 13 sessions par utilisateur en moyenne (de 4 à 75). Toujours prédire le plus fréquent donne 2,3 % d'accuracy. D'où le F1 macro, où chaque utilisateur compte autant.
- **Longueurs très asymétriques** (médiane 340 actions, maximum 11 977). Durée et nombre d'actions sont redondants (≈ 0,9 action/s).
- **Chaque utilisateur n'a qu'un navigateur** (vrai pour les 247 du train) : 114 utilisateurs sur Firefox, 93 sur Chrome, 39 sur Edge, 1 seul sur Opera.

## 2. Valeurs aberrantes

- **Règle de Tukey sur l'échelle log** : sur le nombre d'actions brut, la distribution est trop asymétrique (seuil bas négatif, 297 sessions « aberrantes »). Sur log(1 + nombre d'actions), la règle isole 3 sessions quasi vides (0, 1 et 2 actions) et 3 sessions de plus de 10 000 actions.
- **On retire du train les 3 sessions de moins de 10 actions** : elles n'apportent aucune information. Le test n'est jamais filtré (son minimum est de 9 actions).
- **On garde les sessions très longues et rapides**, et on l'a mesuré (LinearSVC C=3 + contrainte, retrait des plis d'entraînement seulement) :

| Choix | F1 macro | Accuracy sur les sessions longues | Accuracy sur les sessions rapides |
|---|---|---|---|
| **Tout garder (retenu)** | **0,967** | 0,997 | 0,98 |
| Sans les 297 sessions longues (> 1818 actions) | 0,951 | 0,963 | 0,98 |
| Sans les 50 sessions rapides (> 2 actions/s) | 0,966 | 0,997 | 0,96 |
| Sans les deux | 0,949 | 0,963 | 0,98 |

- Les sessions longues sont les plus riches en information : les retirer coûte 0,016 de F1.
- Les sessions rapides sont réparties sur 36 utilisateurs (au plus 4 chacun) : ce n'est pas une signature. Une partie vient de notre mesure de la durée, qui ne compte pas la dernière fenêtre `tXX`. Les retirer ne change rien.

## 3. Deux représentations des sessions


### 3.1 Features manuelles (756 colonnes) — `2_features_manuelles_modeles.ipynb`

Fréquences relatives (compte ÷ nombre d'actions) de chacun des 35 types d'action, 418 écrans, 219 configurations et 76 chaînes vus dans le train, plus 4 statistiques (`nb_actions_log`, `actions_par_s`, `part_modification`, `nb_types_distincts`) et le navigateur en one-hot.

- **Fréquences relatives plutôt que comptes** : un compte mélange l'utilisateur et la longueur de la session.
- **Tout le vocabulaire plutôt qu'une sélection** : pas de seuil arbitraire à justifier, et les valeurs rares sont gardées (32 écrans et 20 configurations ne sont utilisés que par **un seul utilisateur** : des signatures).
- **Limite** : chaque famille est comptée à part (un geste n'est pas relié à l'écran où il est fait, ni une action à la suivante), et pour un arbre une colonne rare est une colonne comme une autre.

### 3.2 Représentation textuelle TF-IDF (46 666 jetons) — `3_tfidf_modeles.ipynb`

Chaque session devient un document. Jetons : l'action complète et les bigrammes d'actions consécutives, le type seul et ses bigrammes, l'écran, la configuration, la chaîne et le navigateur. TF-IDF avec `sublinear_tf` (log des comptes) et `min_df=2`, ajustée dans chaque pli (pas de fuite).

Elle garde tout le vocabulaire, signatures comprises, et donne du poids aux jetons propres à un utilisateur.

## 4. Modèles testés

Validation croisée stratifiée à 4 plis (la classe la plus rare n'a que 4 sessions), même découpage pour tous les modèles. Les chiffres sont ceux des notebooks.

### 4.1 Sur les features manuelles

| Modèle | Accuracy | F1 macro |
|---|---|---|
| Classe majoritaire | 0,023 | 0,000 |
| LightGBM (réglage prudent) | 0,786 | 0,730 |
| Régression logistique (standardisée, C=1) | 0,802 | 0,767 |
| XGBoost (`max_depth=3`) | 0,870 | 0,850 |
| Random Forest (300 arbres) | 0,927 | 0,916 |
| ExtraTrees (500 arbres, défaut) | 0,938 | 0,926 |
| **ExtraTrees `max_features=0.3`** | **0,948** | **0,942** |

### 4.2 Sur la TF-IDF

| Modèle | F1 macro | F1 macro + contrainte navigateur |
|---|---|---|
| ExtraTrees (3000 meilleurs jetons, χ²) | 0,884 | 0,918 |
| SVM linéaire libsvm (`SVC(kernel="linear")`, C=3, un contre un) | 0,932 | 0,950 |
| LinearSVC C=1 | 0,943 | 0,963 |
| **LinearSVC C=3** | **0,949** | **0,967** (soumission Kaggle : **0,988**) |
| LinearSVC C=10 | 0,950 | 0,967 |
| LinearSVC C=30 | 0,951 | 0,968 (soumission finale) |

Choix des jetons (LinearSVC C=3, F1 macro sans / avec contrainte) : actions + bigrammes 0,946 / 0,965 ; + trigrammes 0,938 / 0,962 ; **vues multiples (retenue) 0,949 / 0,967** ; vues multiples + trigrammes 0,946 / 0,965.

**Contrainte « même navigateur »** : on prédit le meilleur utilisateur parmi ceux qui utilisent le navigateur de la session. Le SVM ne voit le navigateur que comme un jeton parmi des milliers : pour LinearSVC C=3, 66 de ses 132 erreurs prédisent un utilisateur d'un autre navigateur ; avec la contrainte, il reste 92 erreurs. Elle aide tous les modèles, l'ExtraTrees le plus (+0,034 de F1).

### 4.3 Pourquoi ce classement

- **Le SVM linéaire est fait pour ce type de données** : des dizaines de milliers de colonnes creuses pour ~3300 sessions, où les classes sont presque séparables par un hyperplan.
- **Un contre tous plutôt qu'un contre un** : `LinearSVC` entraîne un classifieur par utilisateur sur toutes les sessions ; `SVC(kernel="linear")` en entraîne un par paire d'utilisateurs (30 381), chacun sur ~26 sessions seulement, et ses scores par utilisateur, reconstruits à partir de votes, se prêtent moins bien à la contrainte navigateur.
- **Les arbres se perdent dans les colonnes creuses** de la TF-IDF, mais restent le meilleur choix sur les features manuelles.
- **Le boosting est en retrait** sur les features manuelles, et il est beaucoup plus lent (un arbre par classe et par itération).

## 5. Résultats Kaggle et analyse des erreurs

| Soumission | Validation croisée (F1 macro) | Kaggle |
|---|---|---|
| ExtraTrees `max_features=0.3`, features manuelles | 0,942 | 0,944 |
| LinearSVC C=3 + contrainte navigateur, TF-IDF | 0,967 | **0,988** |
| LinearSVC C=30 + contrainte navigateur, TF-IDF (finale) | 0,968 | **0,988** |

Le score Kaggle du SVM dépasse sa validation croisée. Explication probable (non vérifiée) : en validation, le modèle n'apprend que sur 3/4 des sessions de chaque utilisateur (≈ 10 au lieu de 13), un vrai handicap pour les petites classes ; le modèle soumis est entraîné sur tout le train.

Erreurs selon la longueur de la session (prédictions hors-pli) :

| Longueur (actions) | ExtraTrees | SVM + contrainte |
|---|---|---|
| ≤ 131 | 11,4 % | 7,8 % |
| 131 à 250 | 6,0 % | 2,9 % |
| 250 à 484 | 4,3 % | 2,3 % |
| 484 à 997 | 2,6 % | 0,8 % |
| > 997 | 1,5 % | 0,3 % |

- Les sessions courtes restent les plus difficiles, mais le SVM y fait nettement moins d'erreurs : une partie du signal était présente, la représentation manuelle ne le captait pas.
- Avec l'ExtraTrees, les utilisateurs les plus mal reconnus ont peu de sessions (`azx`, `hll`) ou des habitudes proches d'un autre (`flj` et `slq`, confondus avec `skm`).

## 6. Limites

- **La contrainte navigateur** suppose qu'aucun utilisateur ne change de navigateur entre train et test : vrai à 100 % sur le train, invérifiable sur le test (le score Kaggle suggère qu'elle tient).
- **Beaucoup de comparaisons sur les mêmes données** : `C`, les jetons et `max_features` ont été choisis sur la validation croisée qui les évalue ; les écarts inférieurs à ~0,005 ne sont pas fiables.
- **Non fait** : correction de la durée (dernière fenêtre `tXX` jamais fermée, impact jugé faible).
