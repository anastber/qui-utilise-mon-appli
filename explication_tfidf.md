# La TF-IDF et le SVM linéaire, appliqués à notre problème

Ce document suit **une vraie session** du train, de la ligne du CSV jusqu'à la prédiction, avec les chiffres réellement calculés par le code de `3_tfidf_modeles.ipynb`.

```
ligne du CSV ──► liste de jetons ──► vecteur TF-IDF (46 666 colonnes) ──► 247 scores (SVM) ──► contrainte navigateur ──► utilisateur prédit
```

## 1. L'idée : traiter une session comme un texte

Un utilisateur se reconnaît à son « vocabulaire » : les écrans qu'il ouvre, les enchaînements qu'il répète, les configurations qu'il utilise. C'est le même problème que retrouver l'auteur d'un texte à partir des mots qu'il emploie. On réutilise donc l'outil classique de ce domaine : une session devient un **document**, ses actions deviennent des **mots** (on dira des **jetons**).

## 2. Étape 1 : de la ligne du CSV aux jetons

### La session d'exemple

Session de **`slq`** (Google Chrome, 32 actions). C'est un bon exemple : `slq` faisait partie du trio d'utilisateurs (`flj`, `skm`, `slq`) que l'ExtraTrees confondait. Ses premières actions :

```
Exécution d'un bouton
Raccourci(fr.infologic.infoc.client.forms.LaunchForm)
Lancement d'une action infocentre
Exécution d'un bouton
Saisie dans un champ(fr.infologic.achatsventes.client.modules.ficheartmat.FicheRefForm)
Raccourci(fr.infologic.ventes.client.modules.commandesfactures.avoir.AvoirForm)
Saisie dans un champ
...
```
Ensuite, surtout des « Exécution d'un bouton » (14 au total) et des « Saisie dans un champ » (11).

### Les règles de fabrication des jetons (`jetons_session`)

Pour chaque session, on produit plusieurs « vues » de chaque action :

| Famille | Exemple tiré de la session | Ce que ça capte |
|---|---|---|
| Action complète | `Saisie dans un champ(…ficheartmat.FicheRefForm)` | le geste précis, à l'endroit précis |
| Bigramme d'actions | `Exécution d'un bouton >> Saisie dans un champ(…FicheRefForm)` | une habitude d'enchaînement |
| Type seul (`T=`) | `T=Saisie dans un champ` | le geste, où qu'il soit fait |
| Bigramme de types (`T2=`) | `T2=Raccourci >> Lancement d'une action infocentre` | un enchaînement de gestes |
| Écran (`E=`), configuration (`C=`), chaîne (`H=`) | `E=fr.infologic.achatsventes.client.modules.ficheartmat.FicheRefForm` | l'endroit, quel que soit le geste |
| Navigateur (`NAV=`) | `NAV=Google Chrome` | le navigateur |

Les marqueurs de temps `tXX` sont ignorés. Le `1` final (mode modification) reste dans le texte de l'action complète : `Saisie dans un champ1` et `Saisie dans un champ` sont deux jetons différents.

**Pour notre session : 32 actions → 130 jetons, dont 49 distincts.**

## 3. Étape 2 : le vocabulaire

Le vocabulaire est la liste de tous les jetons distincts retenus sur le train. Il est appris **uniquement sur le train**.

| Famille | Jetons |
|---|---|
| Bigrammes d'actions complètes | 40 498 |
| Actions complètes | 4 638 |
| Bigrammes de types | 922 |
| Écrans | 303 |
| Configurations | 198 |
| Chaînes | 68 |
| Types | 35 |
| Navigateurs | 4 |
| **Total** | **46 666** |

- `min_df=2` : on ne garde que les jetons présents dans **au moins 2 sessions**. 53 166 jetons vus une seule fois sont écartés : ils ne peuvent rien apprendre au modèle, car ils ne réapparaîtront dans aucune autre session.
- Chaque session n'utilise qu'une petite partie du vocabulaire : **291 jetons non nuls en médiane** sur 46 666. Le vecteur d'une session est donc presque entièrement fait de zéros. On parle de vecteur **creux**.
- Notre session d'exemple a 48 jetons dans le vocabulaire : un seul de ses 49 jetons distincts n'a été vu dans aucune autre session.

## 4. Étape 3 : le calcul TF-IDF

Chaque jeton `j` d'une session reçoit un poids, en trois temps (formules exactes de scikit-learn avec nos réglages) :

1. **TF (fréquence dans la session)** : `tf = 1 + ln(compte)`, à cause de `sublinear_tf=True`. Un jeton vu 14 fois vaut 3,64, pas 14 : une action répétée ne doit pas tout écraser.
2. **IDF (rareté dans le train)** : `idf = ln((1 + n) / (1 + df)) + 1`, avec `n = 3276` sessions et `df` = nombre de sessions contenant le jeton. Un jeton présent partout vaut ≈ 1 ; un jeton présent dans 6 sessions vaut 7,15.
3. **Normalisation** : on divise tous les `tf × idf` de la session par leur norme (racine de la somme des carrés ; ici 28,6). Le vecteur a alors une longueur de 1, quelle que soit la longueur de la session. C'est ce qui rend comparables une session de 32 actions et une de 3000.

### Les poids de notre session

Jetons les plus lourds :

| Jeton | Compte | Sessions qui le contiennent (df) | idf | tf | Poids final |
|---|---|---|---|---|---|
| `Exécution d'un bouton >> Saisie dans un champ(…FicheRefForm)` | 1 | 4 | 7,49 | 1,00 | **0,262** |
| `Saisie dans un champ(…FicheRefForm)` | 1 | 6 | 7,15 | 1,00 | **0,250** |
| `E=…ficheartmat.FicheRefForm` | 1 | 6 | 7,15 | 1,00 | **0,250** |
| `Exécution d'un bouton >> Raccourci(…LaunchForm)` | 1 | 11 | 6,61 | 1,00 | 0,231 |
| `Raccourci(…AvoirForm)` | 1 | 12 | 6,53 | 1,00 | 0,228 |

Jetons les plus fréquents dans la session :

| Jeton | Compte | df | idf | tf | Poids final |
|---|---|---|---|---|---|
| `Saisie dans un champ` | 11 | 2 922 | 1,11 | 3,40 | 0,132 |
| `Exécution d'un bouton` | 14 | 3 232 | 1,01 | 3,64 | 0,129 |
| `T=Exécution d'un bouton` | 14 | 3 275 | 1,00 | 3,64 | 0,127 |
| `NAV=Google Chrome` | 1 | 1 337 | 1,90 | 1,00 | 0,066 |
| `T=Fermeture d'une dialogue` | 1 | 3 223 | 1,02 | 1,00 | 0,036 |

**C'est tout l'intérêt de la TF-IDF** : « Exécution d'un bouton » apparaît 14 fois, mais pèse **deux fois moins** que l'écran `FicheRefForm` vu **une seule fois**. Tout le monde clique sur des boutons ; très peu de gens passent par cet écran. D'ailleurs, sur tout le train, l'écran `FicheRefForm` n'est utilisé que par **3 utilisateurs** : `flj`, `slq` et `wjn`. Ce seul jeton réduit déjà les candidats de 247 à 3.

Dans nos features manuelles, cet écran n'est qu'une colonne parmi 756, nulle dans presque toutes les sessions, que l'ExtraTrees traite comme les autres. Ici, l'IDF le met en avant.

## 5. Étape 4 : le SVM linéaire

### Ce que le modèle apprend

`LinearSVC` entraîne **un classifieur par utilisateur** (« cet utilisateur contre tous les autres ») : 247 classifieurs. Chacun possède :
- **un poids par jeton** : 46 666 poids. Positif si le jeton est typique de l'utilisateur, négatif s'il est typique des autres ;
- **une constante** (l'*intercept*).

Pour une session, le **score** d'un utilisateur est :

```
score(utilisateur) = constante + somme sur les jetons de (poids du jeton × valeur TF-IDF du jeton dans la session)
```

On prédit l'utilisateur qui a le **plus haut score**. `C=3` règle la régularisation : un C élevé laisse les poids grandir pour coller aux données. Au-delà de 3, le score n'augmentait plus.

### La décision pour notre session

Pour que l'exemple soit honnête, le modèle a été entraîné **sans cette session**. Les 5 meilleurs scores :

| Utilisateur | Score | Navigateur |
|---|---|---|
| **`slq`** | **−0,462** | Chrome |
| `skm` | −0,518 | Chrome |
| `urs` | −0,524 | Chrome |
| `flj` | −0,848 | Chrome |
| `ngs` | −0,931 | Chrome |

La prédiction est **correcte** (`slq`), mais de peu devant `skm`. On retrouve la proximité de ces deux utilisateurs déjà vue dans l'analyse des erreurs. Tous les scores sont négatifs : chaque classifieur dit « plutôt pas moi », mais seul le classement compte.

Décomposition du score de `slq` : constante **−1,04** + contributions des jetons **+0,578** = **−0,462**.

Les jetons qui ont le plus poussé vers `slq` (contribution = poids × valeur TF-IDF) :

| Jeton | Valeur TF-IDF | Poids pour `slq` | Contribution |
|---|---|---|---|
| `Lancement d'une action infocentre >> Exécution d'un bouton` | 0,205 | 0,42 | +0,085 |
| `T2=Lancement d'une action infocentre >> Exécution d'un bouton` | 0,201 | 0,35 | +0,070 |
| `Exécution d'un bouton >> Saisie dans un champ(…FicheRefForm)` | 0,267 | 0,24 | +0,063 |
| `E=…infoc.client.forms.LaunchForm` | 0,177 | 0,32 | +0,058 |
| `E=…ficheartmat.FicheRefForm` | 0,253 | 0,22 | +0,057 |
| `NAV=Google Chrome` | 0,066 | 0,85 | +0,056 |

Et ceux qui ont tiré contre `slq` : `Raccourci(…AvoirForm)` (−0,072), le bigramme qui le suit (−0,072). Cet écran est plus typique d'autres utilisateurs.

Ce qu'il faut retenir :
- **La décision résulte de nombreux petits indices**, aucun n'est décisif seul. C'est pourquoi le modèle reste robuste quand un indice manque.
- **Les trois vues se complètent** : le bigramme d'actions, le bigramme de types et l'écran seul apportent chacun une contribution.

### Ce qui caractérise `slq` en général

Les jetons qui ont les plus gros poids positifs pour `slq` (ce que le modèle considère comme sa « signature ») : l'affichage du tableau d'accueil suivi d'un clic (`Affichage d'une dialogue(…HomeViewController) >> Exécution d'un bouton`, poids 1,11), la visionneuse PDF (`…PdfViewerForm`, 1,02), des saisies en comptabilité (`…SaisiEcritForm`, 0,95)… Ce sont des habitudes précises, à la fois « où » et « comment ».

## 6. Étape 5 : la contrainte « même navigateur »

Le SVM ne voit le navigateur que comme **un jeton parmi des milliers** (`NAV=Google Chrome`). Il lui arrive donc de prédire un utilisateur d'un autre navigateur : environ la moitié de ses erreurs en validation croisée. Or, sur le train, chaque utilisateur n'utilise **qu'un seul** navigateur.

On ajoute donc une règle après le SVM : on ne garde que les scores des utilisateurs du **même navigateur** que la session, puis on prend le maximum. Dans notre exemple, les 5 meilleurs sont déjà tous sur Chrome : la règle ne change rien. Sur l'ensemble du train, 66 des 132 erreurs prédisaient un utilisateur d'un autre navigateur ; la règle ramène les erreurs à 92 et fait passer le F1 macro de 0,949 à 0,967. Sur le test, elle a modifié 5 prédictions sur 324.

## 7. Pourquoi il n'y a pas de fuite d'information

- En validation croisée, le vectoriseur TF-IDF est **dans le pipeline** : dans chaque pli, le vocabulaire et les IDF sont calculés **sur les 3/4 d'entraînement seulement**, jamais sur les sessions évaluées.
- Pour la soumission, tout est appris sur le train complet ; un jeton du test absent du vocabulaire est simplement ignoré.
- La règle du navigateur ne s'appuie que sur le train (le navigateur de chaque utilisateur).

## 8. Pourquoi ça marche mieux que nos features manuelles

| | Features manuelles + ExtraTrees | TF-IDF + SVM linéaire |
|---|---|---|
| Vocabulaire | 418 écrans, 219 configurations, 76 chaînes, chacun en colonne séparée | 46 666 jetons, dont tous les écrans rares |
| Combinaisons | chaque famille comptée séparément | l'action complète combine type, écran, configuration, chaîne |
| Enchaînements | aucun | bigrammes d'actions et de types |
| Jetons rares | présents, mais une colonne comme une autre | mis en avant par l'IDF |
| F1 macro (validation croisée) | 0,942 | 0,967 (avec la contrainte) |
| Kaggle | 0,944 | **0,988** |

Le modèle est également **simple** : un poids par jeton et par utilisateur, et un score qui est une somme. On peut donc expliquer chaque décision, comme on l'a fait pour cette session.
