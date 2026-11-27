extends Node
# Autoload "Bus" : seul point de passage entre les sources de données
# (WebSocket ou source simulée) et l'interface. Aucune logique ici.

# true  = fausse source (source_simulee.gd), aucun backend nécessaire
# false = vrai WebSocket vers le pont de l'Étudiant A
const SIMULATION := false

signal prediction_recue(d: Dictionary)   # une décision du classifieur
signal eeg_recu(d: Dictionary)           # paquet de signal EEG (futur)
signal etat_recu(d: Dictionary)          # état du flux / de la lecture
signal stats_recues(d: Dictionary)       # précision en ligne, matrice (futur)
signal connexion_changee(ok: bool)       # WebSocket connecté ou non
signal commande(d: Dictionary)           # interface -> backend (futur)

const CLASSES := ["repos", "main_gauche", "main_droite", "deux_mains", "deux_pieds"]


# Met toutes les clés attendues avec des valeurs par défaut.
# Un message de l'ancien format (v1.1) reste donc valide.
func normaliser_prediction(d: Dictionary) -> Dictionary:
	return {
		"numero": int(d.get("numero", -1)),
		"classe": str(d.get("classe", "")),
		"confiance": float(d.get("confiance", 0.0)),
		"vraie_classe": str(d.get("vraie_classe", "")),
		"t": float(d.get("t", 0.0)),
		"probas": d.get("probas", {}),            # v2 : {} si absent
		"latence_ms": float(d.get("latence_ms", -1.0)),  # v2 : -1 si absent
		"seuil_ok": bool(d.get("seuil_ok", true)),
		"fenetres_ok": int(d.get("fenetres_ok", -1)),
	}


# Raccourci pour l'interface : Bus.envoyer({"cmd": "play"})
func envoyer(d: Dictionary) -> void:
	commande.emit(d)
