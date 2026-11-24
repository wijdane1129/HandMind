# GAME_DESIGN — HandMind (v1)

## 1. But
Un avatar 3D qui joue le mouvement imaginé par un sujet, décidé par le
classifieur à partir d'un dataset EEG rejoué en temps réel.

## 2. Les 5 mouvements
| Classe (nom exact dans le JSON) | Condition imaginée | Animation de l'avatar | Boucle |
|---|---|---|---|
| repos | Repos (T0) | Posture neutre | Oui |
| main_gauche | Main gauche (T1, runs 4/8/12) | Lève la main gauche | Non |
| main_droite | Main droite (T2, runs 4/8/12) | Lève la main droite | Non |
| deux_bras | Deux mains (T1, runs 6/10/14) | Lève les deux bras | Non |
| accroupi | Deux pieds (T2, runs 6/10/14) | S'accroupit | Non |

## 3. Règles
- **Repos** : état par défaut, animation en boucle.
- **Retour au repos** : à la fin d'un mouvement, l'avatar revient
  automatiquement au repos.
- **Même classe répétée** : si la classe reçue est celle en cours, on ne
  relance pas l'animation.
- **Transition** : fondu de 0,3 s entre deux animations.
- **Nouvelle commande pendant un mouvement** (réglage INTERROMPRE) :
  - `true` : le mouvement en cours est coupé, le nouveau démarre aussitôt
    (réactif, choix par défaut pour la démo en direct).
  - `false` : le mouvement en cours finit, la dernière commande reçue est
    mémorisée puis jouée (plus fluide visuellement).
  - Choix retenu : ... (à justifier après test)
- **Commande "repos" reçue** : elle passe toujours.
- **Animation manquante** : avertissement dans la console, aucun plantage.

## 4. Ce qu'on affiche
- Nom du mouvement en cours (grand texte).
- Classe prédite (vert si = vraie classe, rouge sinon).
- Classe vraie (issue des annotations du dataset).
- Barre de confiance (0–100 %).
- Boutons / touches 1–5 pour tester sans backend.
- À venir : état de la connexion (S5), tracé EEG (S6), historique et
  matrice de confusion (S7), vue « du signal au mouvement » (S8).

## 5. Format du message reçu (provisoire, figé en S4)
{"classe": "main_gauche", "confiance": 0.81,
 "vraie_classe": "main_gauche", "t": 12.75}

## 6. Questions ouvertes
- INTERROMPRE : true ou false ?
- Durée des animations vs cadence des décisions (toutes les 250 ms).