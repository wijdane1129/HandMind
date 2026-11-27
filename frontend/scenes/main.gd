extends Node3D

@onready var anim: AnimationPlayer = $AnimationPlayer
@onready var label_etat: Label = $UI/LabelEtat
@onready var label_info: Label = $UI/LabelInfo

# ---------- Réglages ----------
const MODE_DEMO := false          # true = l'avatar enchaîne les mouvements tout seul
const INTERFACE_PAR_CODE := false # false = on utilise les Labels de la scène (UI/LabelEtat, UI/LabelInfo)
const FONDU := 0.3                # durée du fondu entre animations (secondes)
const INTERROMPRE := true         # true = une nouvelle commande coupe le mouvement en cours
const REPOS := "repos"
const MOUVEMENTS := ["main_gauche", "main_droite", "deux_bras", "accroupi"]

const TOUCHES := {
	KEY_1: "repos", KEY_2: "main_gauche", KEY_3: "main_droite",
	KEY_4: "deux_bras", KEY_5: "accroupi",
	KEY_Q: "repos", KEY_W: "main_gauche", KEY_E: "main_droite",
	KEY_R: "deux_bras", KEY_T: "accroupi",
}

# Classe envoyée par le serveur -> nom de l'animation dans Godot
const CORRESPONDANCE := {
	"repos": "repos",
	"main_gauche": "main_gauche",
	"main_droite": "main_droite",
	"deux_mains": "deux_bras",
	"deux_pieds": "accroupi",
}
const NB_MESSAGES_T14 := 100
const ACTIVER_T14 := false         # true uniquement pour rejouer T14 avec ws_server.py
const NB_MESSAGES_T13 := 50        # nombre de messages mesurés pour la latence (T13)
const SEUIL_LATENCE_MS := 500.0    # critère T13 : latence < 500 ms

# ---------- État ----------
var label_mouvement: Label
var label_predite: Label
var label_vraie: Label
var label_conf: Label
var barre_conf: ProgressBar
var en_attente := ""

var recus := 0
var animations_ok := 0
var erreurs := 0
var numeros_vus := []
var log_t14 := []
var log_ecrit := false

var label_flux: Label
var label_latence: Label
var latences: Array = []
var lignes_t13: Array = []
var t13_ecrit := false


func _ready() -> void:
	_configurer_boucles()
	anim.animation_finished.connect(_on_animation_finished)
	if INTERFACE_PAR_CODE:
		_creer_interface()
	anim.play(REPOS)
	label_etat.position = Vector2(20, 20)
	label_info.position = Vector2(20, 70)
	label_etat.add_theme_font_size_override("font_size", 24)
	label_info.add_theme_font_size_override("font_size", 24)
	label_etat.text = "Déconnecté"
	label_etat.add_theme_color_override("font_color", Color.RED)
	label_info.text = "En attente de messages..."
	_creer_labels_s5()
	# Les données arrivent via l'Autoload Bus (WebSocket ou source simulée)
	Bus.prediction_recue.connect(_on_prediction_bus)
	Bus.connexion_changee.connect(_on_etat_change)
	Bus.etat_recu.connect(_on_etat_bus)
	if MODE_DEMO:
		_demo()


# Force les modes de boucle : repos boucle, les mouvements non.
func _configurer_boucles() -> void:
	for nom in [REPOS] + MOUVEMENTS:
		if anim.has_animation(nom):
			var a := anim.get_animation(nom)
			a.loop_mode = Animation.LOOP_LINEAR if nom == REPOS else Animation.LOOP_NONE
		else:
			push_warning("Animation manquante : " + nom)


# ---------- Interface créée par code (optionnelle) ----------
func _creer_interface() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)

	var boite := HBoxContainer.new()
	boite.position = Vector2(20, 200)
	layer.add_child(boite)
	for nom in [REPOS] + MOUVEMENTS:
		var b := Button.new()
		b.text = nom
		b.pressed.connect(jouer.bind(nom))
		boite.add_child(b)

	label_mouvement = Label.new()
	label_mouvement.position = Vector2(20, 250)
	label_mouvement.add_theme_font_size_override("font_size", 36)
	layer.add_child(label_mouvement)
	_maj_label(REPOS)

	var panneau := VBoxContainer.new()
	panneau.position = Vector2(20, 310)
	layer.add_child(panneau)

	label_predite = Label.new()
	label_predite.text = "Prédite : —"
	label_predite.add_theme_font_size_override("font_size", 24)
	panneau.add_child(label_predite)

	label_vraie = Label.new()
	label_vraie.text = "Vraie : —"
	label_vraie.add_theme_font_size_override("font_size", 24)
	panneau.add_child(label_vraie)

	label_conf = Label.new()
	label_conf.text = "Confiance : —"
	panneau.add_child(label_conf)

	barre_conf = ProgressBar.new()
	barre_conf.min_value = 0
	barre_conf.max_value = 100
	barre_conf.value = 0
	barre_conf.show_percentage = false
	barre_conf.custom_minimum_size = Vector2(300, 20)
	panneau.add_child(barre_conf)


func _maj_label(nom: String) -> void:
	if label_mouvement:
		label_mouvement.text = nom.replace("_", " ").capitalize()


# Affiche prédiction, vraie classe et confiance (conf entre 0.0 et 1.0).
# Vert si la prédiction est juste, rouge sinon.
func maj_prediction(predite: String, vraie: String, conf: float) -> void:
	var couleur := Color.GREEN if predite == vraie else Color.RED
	if not INTERFACE_PAR_CODE:
		label_info.text = "Prédite : %s\nVraie : %s\nConfiance : %d %%" % [
			predite.replace("_", " "), vraie.replace("_", " "), int(conf * 100)]
		label_info.add_theme_color_override("font_color", couleur)
		return
	label_predite.text = "Prédite : " + predite.replace("_", " ")
	label_vraie.text = "Vraie : " + vraie.replace("_", " ")
	label_conf.text = "Confiance : %d %%" % int(conf * 100)
	barre_conf.value = conf * 100
	label_predite.add_theme_color_override("font_color", couleur)


# ---------- Entrées clavier ----------
func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if TOUCHES.has(event.keycode):
			jouer(TOUCHES[event.keycode])


# ---------- Point d'entrée unique : clavier, boutons, WebSocket ----------
func jouer(nom: String) -> void:
	if not anim.has_animation(nom):
		push_warning("Animation introuvable : " + nom)
		return

	# Même animation déjà en cours : on ne la relance pas
	if anim.is_playing() and anim.current_animation == nom:
		return

	var occupe := anim.is_playing() and anim.current_animation != REPOS

	if occupe and not INTERROMPRE and nom != REPOS:
		en_attente = nom
		return

	en_attente = ""
	anim.play(nom, FONDU)
	_maj_label(nom)


# ---------- WebSocket ----------
func _on_prediction_bus(d: Dictionary) -> void:
	_on_classe_recue(d["numero"], d["classe"], d["confiance"], d["vraie_classe"], d["t"])


func _on_etat_bus(d: Dictionary) -> void:
	if d.has("flux"):
		_on_etat_flux(bool(d["flux"]))


func _on_etat_change(connecte: bool) -> void:
	label_etat.text = "Connecté" if connecte else "Déconnecté"
	label_etat.add_theme_color_override("font_color", Color.GREEN if connecte else Color.RED)
	print("Connexion : ", connecte, " (reçus : ", recus, ")")
	if not connecte:
		_maj_flux_label("inconnu", Color.GRAY)
	# Nouvelle connexion = nouveau test : on repart de zéro
	if connecte:
		_reinitialiser()
		return
	# Filet de sécurité : si la connexion se coupe, on écrit quand même le bilan
	if ACTIVER_T14 and not connecte and recus > 0 and not log_ecrit:
		await get_tree().create_timer(0.5).timeout
		if not log_ecrit:
			_ecrire_log_t14()


func _on_classe_recue(numero: int, classe: String, confiance: float, vraie_classe: String, t: float) -> void:
	recus += 1
	numeros_vus.append(numero)
	print("Reçu n°", numero, " classe=", classe)

	var attendue: String = CORRESPONDANCE.get(classe, "")
	var ok := false
	if attendue != "":
		jouer(attendue)
		ok = (anim.current_animation == attendue)
	maj_prediction(classe, vraie_classe, confiance)

	if ok:
		animations_ok += 1
	else:
		erreurs += 1
	log_t14.append("n°%d classe=%s anim_attendue=%s anim_jouee=%s %s" % [
		numero, classe, attendue, anim.current_animation, "OK" if ok else "ERREUR"])

	_mesurer_latence(numero, classe, vraie_classe, t)

	if ACTIVER_T14 and recus == NB_MESSAGES_T14 and not log_ecrit:
		_ecrire_log_t14()


func _reinitialiser() -> void:
	recus = 0
	animations_ok = 0
	erreurs = 0
	numeros_vus.clear()
	log_t14.clear()
	log_ecrit = false
	latences.clear()
	lignes_t13.clear()
	t13_ecrit = false


func _ecrire_log_t14() -> void:
	log_ecrit = true
	var doublons := recus - _compter_uniques()
	# Le test n'est valide que si on a reçu exactement les numéros 0 à 99
	var sequence_complete: bool = numeros_vus.size() > 0 and numeros_vus.min() == 0 and numeros_vus.max() == NB_MESSAGES_T14 - 1
	var reussi: bool = recus == NB_MESSAGES_T14 and animations_ok == NB_MESSAGES_T14 and erreurs == 0 and doublons == 0 and sequence_complete
	var resume := "T14 : %d reçus, %d animations correctes, %d erreurs, %d doublons -> %s" % [
		recus, animations_ok, erreurs, doublons, "RÉUSSI" if reussi else "ÉCHEC"]
	print(resume)
	var f := FileAccess.open("user://resultats_T14.txt", FileAccess.WRITE)
	if f == null:
		push_warning("Impossible d'écrire le log T14")
		return
	f.store_line(resume)
	for ligne in log_t14:
		f.store_line(ligne)
	f.close()
	print("Log écrit dans : ", ProjectSettings.globalize_path("user://resultats_T14.txt"))


func _compter_uniques() -> int:
	var d := {}
	for n in numeros_vus:
		d[n] = true
	return d.size()


# ---------- Semaine 5 : état du flux et latence (T13) ----------
func _creer_labels_s5() -> void:
	label_flux = Label.new()
	label_flux.position = Vector2(20, 190)
	label_flux.add_theme_font_size_override("font_size", 24)
	$UI.add_child(label_flux)
	_maj_flux_label("inconnu", Color.GRAY)

	label_latence = Label.new()
	label_latence.position = Vector2(20, 230)
	label_latence.add_theme_font_size_override("font_size", 24)
	label_latence.text = "Latence : —"
	$UI.add_child(label_latence)


func _maj_flux_label(etat: String, couleur: Color) -> void:
	label_flux.text = "Flux EEG : " + etat
	label_flux.add_theme_color_override("font_color", couleur)


func _on_etat_flux(ok: bool) -> void:
	if ok:
		_maj_flux_label("reçu", Color.GREEN)
	else:
		_maj_flux_label("perdu", Color.RED)


func _mesurer_latence(numero: int, classe: String, vraie_classe: String, t: float) -> void:
	# t doit être une heure Unix (secondes). Sinon (temps relatif) on ne mesure pas.
	if t < 1.0e9:
		return
	var latence_ms: float = (Time.get_unix_time_from_system() - t) * 1000.0
	latences.append(latence_ms)
	lignes_t13.append("n°%d prédite=%s vraie=%s latence=%.1f ms" % [numero, classe, vraie_classe, latence_ms])
	label_latence.text = "Latence : %.0f ms (n=%d)" % [latence_ms, latences.size()]
	if latences.size() >= NB_MESSAGES_T13 and not t13_ecrit:
		_ecrire_log_t13()


func _ecrire_log_t13() -> void:
	t13_ecrit = true
	var tri: Array = latences.duplicate()
	tri.sort()
	var n: int = tri.size()
	var mediane: float = tri[int(n / 2.0)]
	var p95: float = tri[int(ceil(0.95 * n)) - 1]
	var maxi: float = tri[n - 1]
	var reussi: bool = p95 < SEUIL_LATENCE_MS
	var resume := "T13 : %d messages, latence médiane %.1f ms, 95e centile %.1f ms, max %.1f ms -> %s" % [
		n, mediane, p95, maxi, "RÉUSSI" if reussi else "ÉCHEC"]
	print(resume)
	var f := FileAccess.open("user://resultats_T13.txt", FileAccess.WRITE)
	if f == null:
		push_warning("Impossible d'écrire le log T13")
		return
	f.store_line(resume)
	for ligne in lignes_t13:
		f.store_line(ligne)
	f.close()
	print("Log écrit dans : ", ProjectSettings.globalize_path("user://resultats_T13.txt"))


# ---------- Retour automatique au repos ----------
func _on_animation_finished(nom: StringName) -> void:
	if nom == REPOS:
		return
	if en_attente != "":
		var suivante := en_attente
		en_attente = ""
		anim.play(suivante, FONDU)
		_maj_label(suivante)
	else:
		anim.play(REPOS, FONDU)
		_maj_label(REPOS)


# ---------- Mode démo ----------
func _demo() -> void:
	for nom in MOUVEMENTS:
		await get_tree().create_timer(3.0).timeout
		jouer(nom)
		maj_prediction(nom, nom, 0.85)
	await get_tree().create_timer(3.0).timeout
	jouer("main_gauche")
	maj_prediction("main_gauche", "main_droite", 0.55)
	await get_tree().create_timer(3.0).timeout
	jouer(REPOS)
