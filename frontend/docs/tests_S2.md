# Tests S2 : animations

Date : 05/11/2026 · Testeur : Étudiant B · Godot 4

| Test | Résultat | Remarque |
|---|---|---|
| repos → chaque mouvement, fondu propre | OK | Fondu de 0,3 s sans saut |
| mouvement → retour au repos fluide | OK | Retour automatique au repos |
| mouvement → autre mouvement (main_gauche puis accroupi) | OK | Pas de téléportation |
| spam 1→2→3→4→5 sans erreur console | OK | Aucun avertissement |
| spam de la même touche : pas de relance | OK | Le `return` de jouer() fonctionne |
| INTERROMPRE = true : observation | OK | Réactif : le nouveau mouvement démarre tout de suite |
| INTERROMPRE = false : observation | OK | Le geste finit, puis le suivant s'enchaîne |
| mode démo 2 min sans avertissement | OK | Aucun avertissement rouge |

## Choix INTERROMPRE

**true** (à confirmer avec l'Étudiant A).

Justification : le classifieur envoie une décision toutes les 250 ms, avec un lissage sur 3 fenêtres. Avec `true`, l'avatar suit la décision en direct, ce qui rend visible la prédiction du modèle (objectif 2 du projet). Avec `false`, l'avatar serait en retard de 2 à 3 s sur le signal et l'écart avec la « vraie classe » affichée serait déroutant.

À revoir en S6 avec le vrai classifieur : si l'avatar est saccadé quand le modèle hésite, passer à `false` ou augmenter le lissage.