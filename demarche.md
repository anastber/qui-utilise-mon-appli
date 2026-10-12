# Démarche : features, traitements et modèles

Objectif : identifier l'utilisateur de Copilote (247 trigrammes) à partir de la trace d'une session. Métrique du challenge : F1 moyen (on suit le F1 macro, plus l'accuracy).

**Résultat final : TF-IDF + SVM linéaire + contrainte « même navigateur », 0,988 sur Kaggle** (0,925 pour notre première soumission ExtraTrees). Notebook : `BE_CS1_tfidf_svm.ipynb`.

## 1. Ce que contiennent les données

Une ligne = une session : `util, navigateur, action, action, …, tXX, …` (`util` absent du test).
Une action a la forme `Type(écran)<configuration>$chaîne$`, avec un `1` final si l'utilisateur est **en modification** dans la fiche. Les marqueurs `tXX` sont des repères de temps en secondes.

- **Classes déséquilibrées** : 13 sessions par utilisateur en moyenne (de 4 à 75). Toujours prédire le plus fréquent donne 2,3 % d'accuracy. D'où le F1 macro, où chaque utilisateur compte autant.
- **Longueurs très asymétriques** (médiane 340 actions, maximum 11 977). Durée et nombre d'actions sont redondants (≈ 0,9 action/s).
- **Chaque utilisateur n'a qu'un navigateur** (vérifié sur tout le train). Opera = un seul utilisateur.
- **Le rythme ne distingue presque pas les gens** : longueur + rythme + navigateur seuls donnent ≈ 24 % d'accuracy. Le signal est dans *ce que* font les utilisateurs.
- **Pas de piège dans le découpage** : aucune session dupliquée entre train et test, ordre des lignes aléatoire.

## 2. Valeurs aberrantes

On retire du train les **3 sessions de moins de 10 actions** (dont 1 vide), où les fréquences n'ont pas de sens. Les sessions très longues (297 au-delà de 1817 actions) et les rafales (50 sessions à plus de 2 actions/s, ex. `avt`) sont gardées : ce sont de vraies sessions, souvent caractéristiques d'un utilisateur. Le test n'est jamais filtré.

## 3. Deux représentations des sessions

### 3.1 Features manuelles (143 colonnes)

Fréquences relatives (compte ÷ nombre d'actions) des 35 types d'action et des 50 écrans, 20 configurations et 30 chaînes les plus courants, plus 4 statistiques (`nb_actions_log`, `actions_par_s`, `part_modification`, `nb_types_distincts`) et le navigateur en one-hot.

Ce qu'on a appris en les construisant (ExtraTrees réglé, 3 découpages sauf mention) :
- **Les fréquences relatives sont indispensables** : en comptes bruts, le F1 macro tombe de 0,941 à 0,831 (0,874 en log), car un compte mélange l'utilisateur et la longueur de la session.
- **Limite du top-K** : 32 écrans et 20 configurations ne sont utilisés que par **un seul utilisateur** (des signatures), et aucun n'est dans le top-K. Le modèle se trompe d'ailleurs plus sur les sessions qui en contiennent (8,6 % contre 5,9 %).

### 3.2 Représentation textuelle TF-IDF (46 666 jetons)

Chaque session devient un document. Jetons : l'action complète et les bigrammes d'actions consécutives, le type seul et ses bigrammes, l'écran, la configuration, la chaîne et le navigateur. TF-IDF avec `sublinear_tf` (log des comptes) et `min_df=2`, ajustée dans chaque pli (pas de fuite).

Elle garde tout le vocabulaire, signatures comprises, et donne du poids aux jetons propres à un utilisateur.

## 4. Tous les modèles testés

Validation croisée stratifiée à 4 plis (la classe la plus rare n'a que 4 sessions), même découpage pour tous. « 3 déc. » = moyenne sur 3 découpages ; sinon un seul.

### 4.1 Sur les features manuelles

| Modèle | Accuracy | F1 macro | Remarque |
|---|---|---|---|
| Classe majoritaire | 0,023 | 0,000 | référence |
| LightGBM (baseline, 73 features plus simples) | 0,692 | 0,631 | premier essai |
| LightGBM (réglage prudent) | 0,767 | 0,720 | |
| SVM RBF (standardisé) | 0,824 | 0,812 | |
| LinearSVC (standardisé) | 0,851 | 0,826 | |
| XGBoost (meilleur de 3 réglages) | 0,860 | 0,847 | profondeur 6 : 0,783 |
| Régression logistique (standardisée, C=1) | 0,867 | 0,858 | C=0,1 : 0,837 |
| Random Forest (300 arbres) | 0,922 | 0,920 | |
| ExtraTrees (500 arbres, défaut) | 0,931 | 0,932 | **1re soumission Kaggle : 0,925** |
| **ExtraTrees `max_features=0.3`** (3 déc.) | **0,939** | **0,941** | meilleur modèle sur ces features |
| CatBoost | – | – | essai interrompu, non mesuré |

Modèles « de profils », sur les fréquences du top-K, avec contrainte navigateur :

| Modèle | Accuracy | F1 macro |
|---|---|---|
| Burrows' Delta | 0,765 | 0,789 |
| Centroïde cosinus | 0,855 | 0,876 |
| Naive Bayes multinomial | 0,878 | 0,886 |
| kNN cosinus (k=5) | 0,890 | 0,882 |
| Régression logistique (racine carrée) | 0,908 | 0,905 |
| Mélange ExtraTrees + régression logistique + Naive Bayes | 0,946 | 0,947 (optimiste : poids choisis sur ces mêmes prédictions) |

Réglage de l'ExtraTrees (3 déc.) : `max_features` de `sqrt` à 0,3 gagne +1,1 point de F1 ; 0,5 donne 0,937 ; `class_weight` équilibré (0,939), critère `entropy` (0,927) et 1000 arbres n'apportent rien.

### 4.2 Sur la TF-IDF

| Modèle | Accuracy | F1 macro | Remarque |
|---|---|---|---|
| LightGBM (3000 meilleurs jetons) | 0,816 | 0,789 | 10 min de calcul |
| ExtraTrees (3000 meilleurs jetons) | 0,895 | 0,884 | `max_features=0.3` : 0,857 |
| SVM RBF (C=3 ou 10) | 0,894 | 0,868 | |
| SVM linéaire, libsvm (`SVC(kernel="linear")`, C=3) | 0,941 | 0,931 | |
| LinearSVC C=1 | 0,954 | 0,943 | |
| **LinearSVC C=3** (3 déc.) | **0,959** | **0,949 ± 0,001** | C=10 : 0,950 ; C=30 : 0,951 |
| **LinearSVC C=3 + contrainte navigateur** (3 déc.) | **0,972** | **0,966** | **soumission Kaggle : 0,988** |

Variantes de jetons (LinearSVC, C=3) : actions + bigrammes seuls 0,946 ; + trigrammes 0,938 ; vues multiples (retenue) 0,949 ; vues multiples + trigrammes 0,946.

**Contrainte « même navigateur »** : on prédit le meilleur utilisateur parmi ceux qui utilisent le navigateur de la session. Le SVM ne voit le navigateur que comme un jeton parmi des milliers : sans la contrainte, ≈ 69 de ses ≈ 135 erreurs prédisent un utilisateur d'un autre navigateur. Elle apporte +1,8 point de F1, identique sur les 3 découpages. (L'ExtraTrees, lui, apprenait cette règle seul : contrainte sans effet.)

### 4.3 Pourquoi ce classement

- **Le SVM linéaire est fait pour ce type de données** : des dizaines de milliers de colonnes creuses pour ~3300 sessions, où les classes sont presque séparables par un hyperplan. Le score plafonne dès C=3 (peu de régularisation nécessaire).
- **Les arbres se perdent dans les colonnes creuses** de la TF-IDF, mais restent le meilleur choix sur des features denses et construites à la main.
- **Le boosting sur-apprend** avec ~13 sessions par classe (un arbre par classe et par itération), et il est 10 à 50 fois plus lent.
- **Les noyaux non linéaires (RBF) n'apportent rien**, que ce soit sur la TF-IDF ou sur les features manuelles.

## 5. Pistes de features testées et abandonnées

Sur les features manuelles, ExtraTrees, comparées à la base (0,939 / 0,941) :

| Piste | Accuracy / F1 macro |
|---|---|
| 400 bigrammes de types d'action | 0,937 / 0,931 (ExtraTrees par défaut, base 0,931 / 0,930) |
| 400 trigrammes de types | 0,925 / 0,909 |
| Écran, config, chaîne « propagés » (dernier écran vu) | 0,938 / 0,937 |
| 400 couples (type d'action, écran) | 0,935 / 0,936 |
| Fréquences de tout le vocabulaire (569 colonnes) | 0,943 / 0,936 |
| Présence binaire de tout le vocabulaire | 0,924 / 0,917 |
| Première action de la session | 0,939 / 0,937 |
| Statistiques de concentration (entropie, part dominante) | 0,918 / 0,911 (Random Forest) |
| Valeur dominante (configuration, chaîne) au lieu des fréquences | 0,891 / 0,874 (Random Forest) |

Leçon : pour des arbres, ajouter beaucoup de colonnes rares dilue le signal. Ces mêmes informations (bigrammes, vocabulaire complet, combinaisons type × écran) deviennent utiles dans la TF-IDF, avec un modèle linéaire.

## 6. Résultats Kaggle et analyse des erreurs

| Soumission | Validation croisée (F1 macro) | Kaggle |
|---|---|---|
| ExtraTrees, features manuelles | 0,932 | 0,925 |
| LinearSVC C=3 + contrainte navigateur, TF-IDF | 0,966 | **0,988** |

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
- Avec l'ExtraTrees, les erreurs étaient concentrées : `flj` (29 erreurs sur 42 sessions) était confondu avec `skm` et `slq`, trois utilisateurs Chrome aux sessions courtes et aux habitudes proches.

## 7. Limites

- **La contrainte navigateur** suppose qu'aucun utilisateur ne change de navigateur entre train et test : vrai à 100 % sur le train, invérifiable sur le test (le score Kaggle suggère qu'elle tient).
- **Beaucoup de comparaisons sur les mêmes données** : `C`, les jetons et `max_features` ont été choisis sur la validation croisée qui les évalue ; les écarts inférieurs à ~0,005 ne sont pas fiables.
- **Non fait** : correction de la durée (dernière fenêtre `tXX` jamais fermée, impact jugé faible) et mesure de CatBoost.
