# S6 — Intégration du vrai classifieur (Étudiant A)

Semaine 6 · Étudiant A (Signal & IA) · 24/11/2026

Objectif S6 (cahier, section 5) : remplacer le classifieur factice par le vrai,
corriger les problèmes de format, mesurer la précision en ligne (T15) et la
latence de traitement (T12). → **Démo 1 : avatar piloté par le dataset.**

## 1. Lancer la chaîne réelle (3 terminaux, depuis `backend/`)

```powershell
# 0. une seule fois : entraîner le modèle en ligne (Riemann, config B, 5 classes)
python -m scripts.train --online

# 1. lecteur (Étudiant B) : un sujet de DÉMO, runs mains + pieds
python reader/lecteur.py --sujet <sujet_demo> --runs 4 6 8 10 12 14 --attendre

# 2. pont réel (remplace handmind/server/pont_lsl.py)
python -m handmind.server.pont_classifieur

# 3. lancer la scène Godot (rien à changer côté Godot)
```

Options utiles du pont : `--threshold 0.35`, `--vote 5`, `--envoi changement`,
`--log docs/results/s6_live.csv`, `--modele models/online_riemann_B_3c.joblib`.
À la fin de la lecture (marqueur `FIN`), le pont affiche le bilan : précision des
décisions et T12.

`pont_lsl.py` (classifieur factice) reste utile à B pour tester l'interface sans modèle.

## 2. Tests T15 et T12

```powershell
python -m scripts.eval_online            # 3 premiers sujets de démo (S6)
python -m scripts.eval_online --all      # les 10 sujets de démo (S9)
```

→ `docs/results/s6_t15.md` / `.csv` et `docs/results/s6_t12.md`.

- **T15** : même modèle, deux façons de le faire tourner sur les mêmes zones de
  signal (0,5–4 s après la consigne). *Hors ligne* = fenêtres découpées dans
  chaque essai ; *en ligne* = enregistrement rejoué morceau par morceau (filtre
  causal avec mémoire, fenêtre glissante). Critère : écart ≤ 5 points.
- **T12** : du morceau de signal reçu à la décision prête (filtre + modèle + vote).
  Critère : médiane < 200 ms, 95e centile < 300 ms.

| Test | Résultat | Date |
|---|---|---|
| T15 (3 sujets de démo) | à remplir avec `s6_t15.md` | |
| T12 | à remplir avec `s6_t12.md` | |

## 3. Problèmes de format corrigés

| # | Problème | Effet avant S6 | Correction |
|---|---|---|---|
| 1 | Unité : le lecteur envoie des **microvolts**, le modèle est entraîné en **volts** (MNE) | Signal 10⁶ fois trop grand → prédictions fausses (Riemann et CSP ne sont pas invariants à l'échelle) | Lecture de `desc/unit` du flux, conversion automatique (`LSLSource.scale`) |
| 2 | Nom de classe : backend `pieds`, contrat/lecteur/Godot `deux_pieds` | Marqueurs « deux_pieds » ignorés, prédiction « pieds » rejetée par Godot | `config.CLASSES` utilise `deux_pieds` (code 4 inchangé, anciens modèles OK) ; `pieds` reste accepté |
| 3 | Champ `t` : temps relatif (A) vs heure Unix (contrat v1.1) | T13 non mesurable par Godot | `t` = heure Unix du **dernier échantillon** de la fenêtre classifiée |
| 4 | Champ `numero` absent | Tests de pertes/doublons de B impossibles | `numero` 0, 1, 2… |
| 5 | Message `{"type": "etat"}` absent du backend A | Interface « Flux EEG : inconnu » | Envoyé toutes les 0,5 s par le pont réel, comme `pont_lsl.py` |
| 6 | `NOUVEAU_RUN` : le lecteur colle les runs bout à bout | Fenêtres à cheval sur deux runs | Le filtre et la fenêtre repartent de zéro à chaque `NOUVEAU_RUN` |
| 7 | `FIN` non géré | Le pont tournait dans le vide | Bilan affiché, attente d'un nouveau lecteur |
| 8 | `vraie_classe` = `repos` avant le premier marqueur | Faux « juste/faux » au début | `""` (contrat : vraie classe inconnue) |
| 9 | `data/splits.json` et le modèle ignorés par Git | B ne connaît pas les sujets de démo et ne peut pas lancer le pont réel | `.gitignore` : exceptions `!data/splits.json` et `!models/online_*.joblib` |
| 10 | `CHOSEN_MODEL` / `CHOSEN_CONFIG` vides | Choix S4 non figé | `riemann` / `B` |

## 4. Réponses aux points ouverts du protocole (section 8 de `protocole_json.md`)

À reporter par l'Étudiant B dans le protocole **v1.2** :

1. **`t`** = heure Unix de la **fin** de la fenêtre classifiée (dernier échantillon).
   La latence mesurée par Godot (T13) exclut donc la durée de la fenêtre, comme
   le demande le cahier.
2. **Fréquence** : une prédiction toutes les 250 ms (défaut). Option
   `--envoi changement` pour n'envoyer que les changements.
3. **Sous le seuil de confiance** : le backend envoie `repos` (proposition de B
   retenue), avec la confiance calculée.

## 5. À faire côté Étudiant B

- Mettre `protocole_json.md` en v1.2 (points ci-dessus).
- Le mode `--synthetique` du lecteur nomme les canaux `E4…E64` : le vrai
  classifieur refuse ce flux (canaux FC3, CP1… manquants). Pour tester sans
  téléchargement, utiliser les 64 vrais noms (`config.CHANNELS_ALL`).
- Interface S6 : prédit vs vrai, confiance, EEG qui défile, compteur de précision.
- Vérifier T13 avec le pont réel (même commande Godot qu'en S5).

## 6. Limite connue

Le modèle en ligne est entraîné sur 20 sujets d'entraînement et testé sur des
sujets de démo jamais vus : c'est du **inter-sujets sans calibration**. En S4,
le LOSO 2 classes donnait ≈ 58 % ; en 5 classes, attendre une précision proche
du hasard (20 %). T15 compare en ligne / hors ligne et peut passer même si la
précision est faible. Pour une Démo 1 lisible :

```powershell
python -m scripts.train --online --classes 3
python -m scripts.eval_online --model-file ..\models\online_riemann_B_3c.joblib
python -m handmind.server.pont_classifieur --modele ..\models\online_riemann_B_3c.joblib
```

La calibration courte d'un nouveau sujet (S7, test T9) est la vraie solution.
