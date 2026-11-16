# Lecteur de dataset : contrat d'interface

Ce document est destiné à l'Étudiant A (et à toute personne qui branche du code sur le lecteur).
Le lecteur (`backend/reader/lecteur.py`) joue le rôle d'un casque EEG virtuel.

## Flux publiés (LSL)

### 1. `HandMind_EEG` (le signal)

| Propriété | Valeur |
|---|---|
| Type LSL | `EEG` |
| Canaux | 64 |
| Fréquence nominale | 160 Hz |
| Format | `float32` |
| **Unité** | **microvolts (µV)** (option `--volts` pour des volts) |
| Horodatage | instant théorique de l'échantillon (`début + i/160`) |
| Ordre des canaux | celui du fichier PhysioNet après `eegbci.standardize` |

Les noms de canaux sont dans les métadonnées du flux : `info.desc().child("channels")`, un `label` par canal.
Les métadonnées contiennent aussi `unit`, `sujet` et `runs`.

Lire les noms de canaux côté récepteur :

```python
from pylsl import StreamInlet, resolve_byprop
inlet = StreamInlet(resolve_byprop("name", "HandMind_EEG", timeout=10)[0])
ch = inlet.info().desc().child("channels").child("channel")
noms = []
while ch.name() == "channel":
    noms.append(ch.child_value("label"))
    ch = ch.next_sibling()
```

Aucun filtrage n'est appliqué par le lecteur : le signal est brut (le prétraitement 8–30 Hz + notch 60 Hz est fait par A).

### 2. `HandMind_Markers` (la vraie classe)

| Propriété | Valeur |
|---|---|
| Type LSL | `Markers` |
| Canaux | 1, format `string` |
| Fréquence | 0 (irrégulier) |

Un message est envoyé à chaque **changement de consigne**. La classe reste valable jusqu'au marqueur suivant.

| Message | Signification |
|---|---|
| `repos` | consigne de repos (T0) |
| `main_gauche` / `main_droite` | runs 4, 8, 12 (T1 / T2) |
| `deux_mains` / `deux_pieds` | runs 6, 10, 14 (T1 / T2) |
| `NOUVEAU_RUN:<n>` | début du run n. **Le signal est discontinu à cet instant** (changement de run ou retour au début en mode boucle) : réinitialiser les filtres et vider la fenêtre glissante. |
| `FIN:<n>` | le lecteur s'arrête, `n` échantillons envoyés au total |

Un marqueur de classe est envoyé juste **avant** l'échantillon où la consigne commence.

## Utilisation

```bash
cd backend/reader
python lecteur.py --sujet 3 --runs 4 8 12 --boucle     # démo sans fin
python lecteur.py --sujet 1 --runs 4 --vitesse 5       # développement, 5x plus vite
python lecteur.py --synthetique --boucle               # bruit, sans téléchargement
python lecteur.py --attendre --sujet 3                 # attend un récepteur avant de démarrer
```

| Option | Défaut | Rôle |
|---|---|---|
| `--sujet` | 1 | sujet de 1 à 109 |
| `--runs` | 4 8 12 | runs enchaînés (parmi 4 6 8 10 12 14) |
| `--vitesse` | 1 | 1 = temps réel. Avec 5, les horodatages sont compressés : ne pas utiliser pour mesurer. |
| `--boucle` | non | rejoue sans fin |
| `--duree` | 0 | arrêt après N secondes de signal (0 = illimité) |
| `--volts` | non | envoi en V au lieu de µV |
| `--attendre` | non | attend qu'un récepteur se connecte |
| `--synthetique` | non | bruit aléatoire (64 canaux, marqueurs réalistes) |

Le lecteur doit être lancé **avant** le récepteur.

## Test T11 (10 minutes)

```bash
# terminal 1
python lecteur.py --sujet 1 --runs 4 8 12 --boucle --duree 620
# terminal 2 (lancé juste après)
python recepteur_t11.py --duree 600
```

Critères : dérive < 1 %, 0 échantillon perdu, 0 horodatage désordonné.

## Points d'attention pour A

- Les 5 classes arrivent dans le flux de marqueurs **sous forme de noms**, pas de codes T0/T1/T2.
- Les runs 4/8/12 ne contiennent que gauche/droite/repos ; les runs 6/10/14 que deux mains/deux pieds/repos. Pour une démo à 5 classes, enchaîner des deux types : `--runs 4 6 8 10 12 14`.
- Chaque essai dure environ 4 s, séparé par du repos de même durée.