# HandMind : protocole de communication backend → avatar (v1)

Semaine 4 · Étudiant B (Avatar & Interface) · 16/11/2026

Ce document est le **contrat** entre le backend Python (Étudiant A, classifieur) et l'avatar Godot (Étudiant B). Tant que le format ci-dessous est respecté, chacun peut modifier son code sans casser celui de l'autre.

> Règle : ne pas modifier ce format sans prévenir l'autre étudiant et sans changer le numéro de version.

## 1. Transport

| Élément | Valeur |
| --- | --- |
| Protocole | WebSocket (texte) |
| Adresse par défaut | `ws://localhost:8765` |
| Sens | Serveur Python → client Godot (un seul sens) |
| Encodage | UTF-8, un message JSON par trame WebSocket |
| Bibliothèques | Python : `websockets` · Godot 4 : `WebSocketPeer` |
| Clients | Plusieurs clients possibles, chacun reçoit tous les messages |

Le serveur envoie uniquement quand au moins un client est connecté.
Le client Godot se reconnecte automatiquement toutes les 2 secondes si la connexion est perdue.

## 2. Format du message

```json
{
  "numero": 12,
  "classe": "main_gauche",
  "confiance": 0.81,
  "vraie_classe": "main_gauche",
  "t": 12.75
}
```

| Champ | Type | Obligatoire | Description |
| --- | --- | --- | --- |
| `classe` | texte | oui | Classe prédite par le modèle (voir section 3) |
| `confiance` | nombre entre 0 et 1 | oui | Probabilité de la classe prédite |
| `vraie_classe` | texte | oui | Classe réelle, issue des annotations du dataset (voir section 3) |
| `t` | nombre (secondes) | oui | Horodatage du message (voir section 5) |
| `numero` | entier ≥ 0 | non | Numéro du message, utilisé pour les tests (détection de pertes et de doublons) |

Règles de lecture côté Godot :
- Un champ manquant prend une valeur par défaut (`classe` vide, `confiance` 0, `numero` -1).
- Un champ inconnu est ignoré, ce qui permet d'ajouter des champs sans casser Godot.
- Un message qui n'est pas du JSON valide est ignoré et signalé dans la console.

## 3. Classes et animations

Valeurs autorisées pour `classe` et `vraie_classe` :

| Classe (JSON) | Code PhysioNet | Animation Godot | Mouvement de l'avatar |
| --- | --- | --- | --- |
| `repos` | T0 | `repos` | Posture neutre |
| `main_gauche` | T1 (runs 4, 8, 12) | `main_gauche` | Lève la main gauche |
| `main_droite` | T2 (runs 4, 8, 12) | `main_droite` | Lève la main droite |
| `deux_mains` | T1 (runs 6, 10, 14) | `deux_bras` | Lève les deux bras |
| `deux_pieds` | T2 (runs 6, 10, 14) | `accroupi` | S'accroupit |

Une classe qui n'est pas dans ce tableau ne déclenche aucune animation. Godot la compte comme une erreur.

Le champ `vraie_classe` peut aussi valoir une chaîne vide quand la vraie classe est inconnue. Dans ce cas, l'interface n'affiche ni vert ni rouge.

## 4. Exemples

Prédiction juste :

```json
{"numero": 3, "classe": "deux_mains", "confiance": 0.87, "vraie_classe": "deux_mains", "t": 1.5}
```

Prédiction fausse (texte rouge dans l'interface) :

```json
{"numero": 4, "classe": "main_gauche", "confiance": 0.55, "vraie_classe": "main_droite", "t": 2.0}
```

## 5. Points à valider avec l'Étudiant A

1. **Signification de `t`.** Le cahier des charges donne un exemple en secondes relatives (`12.75`). Le serveur de test envoie actuellement l'heure Unix. Il faut décider quel instant `t` représente : l'horodatage LSL de la fin de la fenêtre classifiée, ou le moment de l'envoi. Ce choix est nécessaire pour mesurer la latence de bout en bout (test T13).
2. **Fréquence des messages.** Une prédiction par pas de fenêtre glissante (250 ms), ou seulement quand la décision change ? L'avatar ne relance pas une animation déjà en cours, donc les deux sont acceptés.
3. **Classe de repos par défaut.** Quand le seuil de confiance n'est pas atteint, le backend envoie-t-il `repos`, ou un message avec `confiance` basse ? Proposition : envoyer `repos`.

## 6. Validation

| Test | Résultat |
| --- | --- |
| T14 : 100 messages envoyés → 100 animations correctes | Réussi le 16/11/2026 (détails dans `docs/resultats_T14.txt`) |

## 7. Historique des versions

| Version | Date | Changement |
| --- | --- | --- |
| v1 | 16/11/2026 | Première version : 4 champs du cahier des charges + `numero` optionnel 
# HandMind : protocole de communication backend → avatar (v1.1)

Semaine 5 · Étudiant B (Avatar & Interface) · 17/11/2026

Ce document est le **contrat** entre le backend Python (Étudiant A, classifieur) et l'avatar Godot (Étudiant B). Tant que le format ci-dessous est respecté, chacun peut modifier son code sans casser celui de l'autre.

> Règle : ne pas modifier ce format sans prévenir l'autre étudiant et sans changer le numéro de version.

## 1. Transport

| Élément | Valeur |
| --- | --- |
| Protocole | WebSocket (texte) |
| Adresse par défaut | `ws://localhost:8765` |
| Sens | Serveur Python → client Godot (un seul sens) |
| Encodage | UTF-8, un message JSON par trame WebSocket |
| Bibliothèques | Python : `websockets` · Godot 4 : `WebSocketPeer` |
| Clients | Plusieurs clients possibles, chacun reçoit tous les messages |

Le serveur envoie uniquement quand au moins un client est connecté.
Le client Godot se reconnecte automatiquement toutes les 2 secondes si la connexion est perdue.

## 2. Deux types de messages

| Type | Comment le reconnaître | Rôle |
| --- | --- | --- |
| Prédiction | Pas de champ `type` | Une décision du classifieur, qui fait bouger l'avatar |
| État | `"type": "etat"` | Santé du flux de données, affichée dans l'interface |

## 3. Message de prédiction

```json
{
  "numero": 12,
  "classe": "main_gauche",
  "confiance": 0.81,
  "vraie_classe": "main_gauche",
  "t": 1794865443.14
}
```

| Champ | Type | Obligatoire | Description |
| --- | --- | --- | --- |
| `classe` | texte | oui | Classe prédite par le modèle (voir section 4) |
| `confiance` | nombre entre 0 et 1 | oui | Probabilité de la classe prédite |
| `vraie_classe` | texte | oui | Classe réelle, issue des annotations du dataset (voir section 4) |
| `t` | nombre (secondes) | oui | **Heure Unix** de l'instant d'émission du marqueur par le lecteur (voir section 6) |
| `numero` | entier ≥ 0 | non | Numéro du message, utilisé pour les tests (détection de pertes et de doublons) |

Règles de lecture côté Godot :
- Un champ manquant prend une valeur par défaut (`classe` vide, `confiance` 0, `numero` -1).
- Un champ inconnu est ignoré, ce qui permet d'ajouter des champs sans casser Godot.
- Un message qui n'est pas du JSON valide est ignoré et signalé dans la console.

## 4. Classes et animations

Valeurs autorisées pour `classe` et `vraie_classe` :

| Classe (JSON) | Code PhysioNet | Animation Godot | Mouvement de l'avatar |
| --- | --- | --- | --- |
| `repos` | T0 | `repos` | Posture neutre |
| `main_gauche` | T1 (runs 4, 8, 12) | `main_gauche` | Lève la main gauche |
| `main_droite` | T2 (runs 4, 8, 12) | `main_droite` | Lève la main droite |
| `deux_mains` | T1 (runs 6, 10, 14) | `deux_bras` | Lève les deux bras |
| `deux_pieds` | T2 (runs 6, 10, 14) | `accroupi` | S'accroupit |

Une classe qui n'est pas dans ce tableau ne déclenche aucune animation. Godot la compte comme une erreur.

Le champ `vraie_classe` peut aussi valoir une chaîne vide quand la vraie classe est inconnue. Dans ce cas, l'interface n'affiche ni vert ni rouge.

## 5. Message d'état (nouveau en v1.1)

```json
{"type": "etat", "flux": true, "t": 1794865443.14}
```

| Champ | Type | Description |
| --- | --- | --- |
| `type` | texte | Toujours `"etat"` |
| `flux` | booléen | `true` si des échantillons EEG arrivent, `false` si aucun n'est arrivé depuis 1,5 s |
| `t` | nombre (secondes) | Heure Unix d'envoi du message d'état |

Le pont l'envoie toutes les 0,5 s. Godot affiche « Flux EEG : reçu » (vert) ou « Flux EEG : perdu » (rouge). Tant qu'aucun message d'état n'est arrivé, l'interface affiche « inconnu » (gris).

## 6. Définition du champ `t`

`t` est l'**heure Unix en secondes** (nombre décimal) de l'instant où le lecteur a émis le marqueur de classe. Godot calcule la latence en retranchant `t` de sa propre heure Unix au moment où il traite le message (test T13).

Conditions :
- La mesure n'est valable que si le lecteur, le backend et Godot tournent sur la **même machine** (même horloge).
- Si `t` est inférieur à 1 000 000 000 (temps relatif, comme `12.75`), Godot ne mesure pas la latence pour ce message.

## 7. Exemples

Prédiction juste :

```json
{"numero": 3, "classe": "deux_mains", "confiance": 0.87, "vraie_classe": "deux_mains", "t": 1794865443.14}
```

Prédiction fausse (texte rouge dans l'interface) :

```json
{"numero": 4, "classe": "main_gauche", "confiance": 0.55, "vraie_classe": "main_droite", "t": 1794865447.20}
```

Message d'état, flux perdu :

```json
{"type": "etat", "flux": false, "t": 1794865460.00}
```

## 8. Points à valider avec l'Étudiant A

1. **Définition de `t`.** La définition de la section 6 (heure Unix d'émission du marqueur) est la proposition de l'Étudiant B. Le vrai classifieur devra envoyer le même type d'horodatage, sinon le test T13 ne sera plus mesurable. Un classifieur en ligne produit une décision à partir d'une fenêtre de 1 s : il faut décider si `t` désigne le début ou la fin de la fenêtre classifiée.
2. **Fréquence des messages.** Une prédiction par pas de fenêtre glissante (250 ms), ou seulement quand la décision change ? L'avatar ne relance pas une animation déjà en cours, donc les deux sont acceptés.
3. **Classe de repos par défaut.** Quand le seuil de confiance n'est pas atteint, le backend envoie-t-il `repos`, ou un message avec `confiance` basse ? Proposition : envoyer `repos`.

## 9. Validation

| Test | Résultat |
| --- | --- |
| T14 : 100 messages envoyés → 100 animations correctes | Réussi le 16/11/2026 (`docs/resultats_T14.txt`) |
| T13 : latence de bout en bout < 500 ms | Réussi le 17/11/2026 : médiane 17,6 ms, 95e centile 29,6 ms (`docs/resultats_T13.txt`) |

## 10. Historique des versions

| Version | Date | Changement |
| --- | --- | --- |
| v1 | 16/11/2026 | Première version : 4 champs du cahier des charges + `numero` optionnel |
| v1.1 | 17/11/2026 | Ajout du message d'état du flux ; `t` défini comme heure Unix |