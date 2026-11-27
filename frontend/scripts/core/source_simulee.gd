extends Node
# Fausse source de données : imite le backend de l'Étudiant A (protocole v2).
# Active seulement si Bus.SIMULATION = true. Permet de développer l'interface
# sans lecteur, sans pont et sans modèle.

const CLASSES := ["repos", "main_gauche", "main_droite", "deux_mains", "deux_pieds"]
const PAS_PREDICTION := 0.25   # une prédiction toutes les 250 ms
const PAS_EEG := 0.05          # un paquet EEG toutes les 50 ms (8 échantillons)
const PAS_ETAT := 0.5
const PAS_STATS := 2.0
const PROBA_ERREUR := 0.25     # part de prédictions fausses simulées
const DUREE_TOTALE := 125.0

var numero := 0
var vraie := "repos"
var t_lecture := 0.0
var correctes := 0
var total := 0
var phase_eeg := 0.0
var acc_pred := 0.0
var acc_eeg := 0.0
var acc_etat := 0.0
var acc_stats := 0.0


func _ready() -> void:
	if not Bus.SIMULATION:
		set_process(false)
		return
	# on attend une image pour que l'interface ait eu le temps de se connecter au Bus
	await get_tree().process_frame
	Bus.connexion_changee.emit(true)


func _process(delta: float) -> void:
	t_lecture = fmod(t_lecture + delta, DUREE_TOTALE)
	acc_pred += delta
	acc_eeg += delta
	acc_etat += delta
	acc_stats += delta

	if acc_pred >= PAS_PREDICTION:
		acc_pred -= PAS_PREDICTION
		_emettre_prediction()
	if acc_eeg >= PAS_EEG:
		acc_eeg -= PAS_EEG
		_emettre_eeg()
	if acc_etat >= PAS_ETAT:
		acc_etat -= PAS_ETAT
		_emettre_etat()
	if acc_stats >= PAS_STATS:
		acc_stats -= PAS_STATS
		_emettre_stats()


func _emettre_prediction() -> void:
	# la vraie classe change toutes les 12 prédictions (3 s)
	if numero % 12 == 0:
		vraie = CLASSES[int(numero / 12.0) % CLASSES.size()]

	var predite: String = vraie
	if randf() < PROBA_ERREUR:
		predite = CLASSES[randi() % CLASSES.size()]

	# probabilités : la classe prédite domine, somme = 1
	var brut := {}
	var somme := 0.0
	for c in CLASSES:
		var v: float = randf() * 0.3
		if c == predite:
			v += 0.6
		brut[c] = v
		somme += v
	var probas := {}
	for c in CLASSES:
		probas[c] = snappedf(brut[c] / somme, 0.001)

	total += 1
	if predite == vraie:
		correctes += 1

	Bus.prediction_recue.emit({
		"numero": numero,
		"classe": predite,
		"confiance": probas[predite],
		"vraie_classe": vraie,
		"t": Time.get_unix_time_from_system(),
		"probas": probas,
		"latence_ms": randf_range(80.0, 160.0),
		"seuil_ok": probas[predite] >= 0.7,
		"fenetres_ok": 3 if probas[predite] >= 0.7 else 1,
	})
	numero += 1


func _emettre_eeg() -> void:
	var brut: Array = [[], [], []]
	var filtre: Array = [[], [], []]
	for i in 8:
		phase_eeg += 1.0 / 160.0
		for k in 3:
			var f: float = sin(TAU * 10.0 * phase_eeg + k) * 20.0 \
				+ sin(TAU * 22.0 * phase_eeg + 2.0 * k) * 8.0
			filtre[k].append(f)
			brut[k].append(f + sin(TAU * 0.3 * phase_eeg) * 40.0 + randf_range(-10.0, 10.0))
	Bus.eeg_recu.emit({
		"type": "eeg",
		"t": Time.get_unix_time_from_system(),
		"canaux": ["C3", "Cz", "C4"],
		"brut": brut,
		"filtre": filtre,
	})


func _emettre_etat() -> void:
	Bus.etat_recu.emit({
		"type": "etat",
		"flux": true,
		"t": Time.get_unix_time_from_system(),
		"t_lecture": t_lecture,
		"duree": DUREE_TOTALE,
		"sujet": "S001",
		"run": 4,
		"vitesse": 1,
		"pause": false,
		"config": "B",
		"classifieur": "Riemann",
	})


func _emettre_stats() -> void:
	var precision: float = 0.0
	if total > 0:
		precision = float(correctes) / float(total)
	Bus.stats_recues.emit({
		"type": "stats",
		"precision": precision,
		"correctes": correctes,
		"total": total,
	})
