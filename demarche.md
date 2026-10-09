# Démarche : features, traitements et modèle

Objectif : identifier l'utilisateur de Copilote (247 trigrammes) à partir de la trace d'une session. Métrique du challenge : F1 moyen (on suit le F1 macro, plus l'accuracy).

## 1. Ce que contiennent les données

Une ligne = une session : `util, navigateur, action, action, …, tXX, …` (`util` absent du test).
Une action a la forme `Type(écran)<configuration>$chaîne$`, avec un `1` final si l'utilisateur est **en modification** dans la fiche. Les marqueurs `tXX` sont des repères de temps en secondes.

Constats de l'exploration, qui guident tout le reste :
- **Classes déséquilibrées** : 13 sessions par utilisateur en moyenne (de 4 à 75). Toujours prédire le plus fréquent donne 2,3 % d'accuracy. D'où le F1 macro, où chaque utilisateur compte autant.
- **Longueurs très asymétriques** (médiane 340 actions, maximum 11 977). Durée et nombre d'actions sont redondants : le débit est presque constant (≈ 0,9 action/s).
- **Chaque utilisateur n'a qu'un navigateur** (vérifié sur le train). Opera = un seul utilisateur.
- **Le rythme ne distingue presque pas les gens** : longueur + rythme + navigateur seuls donnent ≈ 24 % d'accuracy. Le signal est dans *ce que* font les utilisateurs.

## 2. Traitement des valeurs aberrantes

| Cas | Décision | Pourquoi |
|---|---|---|
| 3 sessions de moins de 10 actions (dont 1 vide) | **supprimées du train** | les fréquences relatives n'ont plus de sens |
| 297 sessions très longues (> 1817 actions) | gardées | vraies sessions ; avec ~13 sessions par utilisateur, en perdre coûte cher |
| 50 sessions à plus de 2 actions/s | gardées | ce sont des rafales d'actions propres à certains utilisateurs (ex. `avt`) : un signal |

La suppression ne touche **que le train** : chaque session du test doit recevoir une prédiction.

## 3. Features (143 colonnes)

Principe commun : on utilise des **fréquences relatives** (comptes ÷ nombre d'actions de la session). Elles ne dépendent pas de la longueur de la session et rendent comparables une session de 20 actions et une de 5000. Tous les vocabulaires sont **fixés sur le train**, puis réutilisés tels quels sur le test (pas de fuite d'information).

| Bloc | Colonnes | Contenu | Pourquoi |
|---|---|---|---|
| Types d'action | 35 | fréquence de chaque type (« Exécution d'un bouton », « Raccourci », …) | décrit *comment* l'utilisateur agit |
| Statistiques de base | 4 | `nb_actions_log`, `actions_par_s`, `part_modification`, `nb_types_distincts` | longueur (en log pour écraser la queue lourde), rythme, mode édition/consultation, variété |
| Écrans / configurations / chaînes | 100 | fréquence des 50 écrans, 20 configurations et 30 chaînes les plus courants | décrit *où* l'utilisateur travaille : chacun a ses modules métier |
| Navigateur | 4 | one-hot | l'utilisateur en utilise un seul ; la modalité est fixée sur le train |

Détails de traitement :
- Le **type** d'une action est le texte avant `(`, `<`, `$`, sans le `1` final (qui devient `part_modification`).
- Une seule variable de longueur est gardée (en log), car durée et nombre d'actions sont redondants.
- Beaucoup de sessions n'ont aucune chaîne (la chaîne n'apparaît que sur une fiche) : ces colonnes valent alors 0, et ce 0 est lui-même informatif.

### Comment on a choisi ces features (mesures en validation croisée, Random Forest)

| Jeu de features | Accuracy | F1 macro |
|---|---|---|
| types + statistiques + navigateur | 0,859 | 0,845 |
| + configuration et chaîne **dominantes** (1 valeur chacune) | 0,891 | 0,874 |
| + **fréquences** de 20 configurations et 30 chaînes | 0,914 | 0,914 |
| + fréquences de 50 écrans (**jeu retenu**) | 0,926 | 0,925 |
| vocabulaire 4 à 8 fois plus grand | 0,937 | 0,927 |
| tout le vocabulaire (418 écrans, 219 configs, 76 chaînes) | 0,928 | 0,917 |
| + statistiques de concentration (entropie, part dominante…) | 0,918 | 0,911 |

Ce qu'on en retient :
- **Les fréquences battent la valeur dominante** : « la configuration la plus utilisée » prend 112 valeurs, il faut l'encoder (≈ 170 colonnes) pour un résultat moins bon.
- **Le score plafonne vers 50/20/30** : agrandir le vocabulaire ne gagne que 0,002 de F1, dans le bruit.
- **Tout garder dégrade un peu** (valeurs très rares = bruit), et les statistiques de concentration n'aident pas.

## 4. Modèle

Validation croisée stratifiée à **4 plis** (la classe la plus rare n'a que 4 sessions), identique pour tous les modèles pour que les scores soient comparables.

| Modèle | Accuracy | F1 macro |
|---|---|---|
| Classe majoritaire (référence) | 0,023 | 0,000 |
| Régression logistique (standardisée) | 0,867 | 0,858 |
| LinearSVC / SVM RBF (standardisés) | 0,851 / 0,824 | 0,826 / 0,812 |
| Random Forest (300 arbres) | 0,922 | 0,920 |
| **ExtraTrees (500 arbres)** | **0,931** | **0,932** |

**Modèle retenu : ExtraTrees.** C'est le meilleur, et il est stable (0,930 avec 1000 arbres, 0,932 avec une autre graine). Les arbres s'adaptent bien à des fréquences creuses et asymétriques, sans standardisation, et captent les interactions entre features (la régression logistique perd 6 points).

LightGBM n'a pas pu être retenu : avec 247 classes de ~10 exemples, il demande des réglages très prudents (69 % sur une version antérieure des features, contre 83 % pour un Random Forest sur les mêmes colonnes), et il ne se charge pas dans notre environnement (`libomp` manquant).

## 5. Limites et suite

- Les scores viennent d'**un seul tirage de plis** : des écarts de ~0,01 ne sont pas fiables. À répéter pour départager ExtraTrees et Random Forest.
- Le score est probablement **un peu optimiste** : les sessions d'un même utilisateur se retrouvent des deux côtés du découpage (comme entre train et test). Le score Kaggle fera foi.
- **Pas encore fait** :
  - correction de la durée (la dernière fenêtre `tXX` n'est jamais fermée, donc la durée est légèrement sous-estimée ; impact faible) ;
  - ordre des actions (n-grammes, transitions) ;
  - réglage des hyper-paramètres ;
  - analyse des erreurs : quels utilisateurs sont confondus.
