extends Node3D

@onready var anim: AnimationPlayer = $AnimationPlayer

# ---------- Réglages ----------
const MODE_DEMO := false      # true = l'avatar enchaîne les mouvements tout seul
								# (remets false après le test de la tâche 1)
const FONDU := 0.3              # durée du fondu entre animations (secondes)
const INTERROMPRE := true       # true = une nouvelle commande coupe le mouvement en cours
								# false = on finit le mouvement, puis on joue la commande en attente
const REPOS := "repos"
const MOUVEMENTS := ["main_gauche", "main_droite", "deux_bras", "accroupi"]

const TOUCHES := {
	KEY_1: "repos", KEY_2: "main_gauche", KEY_3: "main_droite",
	KEY_4: "deux_bras", KEY_5: "accroupi",
	KEY_Q: "repos", KEY_W: "main_gauche", KEY_E: "main_droite",
	KEY_R: "deux_bras", KEY_T: "accroupi",
}

# ---------- État ----------
var label_mouvement: Label
var label_predite: Label
var label_vraie: Label
var label_conf: Label
var barre_conf: ProgressBar
var en_attente := ""            # commande mémorisée quand INTERROMPRE = false


func _ready() -> void:
	_configurer_boucles()
	anim.animation_finished.connect(_on_animation_finished)
	_creer_interface()
	anim.play(REPOS)
	if MODE_DEMO:
		_demo()


# Force les modes de boucle par code : repos boucle, les mouvements non.
# (le signal animation_finished ne se déclenche jamais sur une animation en boucle)
func _configurer_boucles() -> void:
	for nom in [REPOS] + MOUVEMENTS:
		if anim.has_animation(nom):
			var a := anim.get_animation(nom)
			a.loop_mode = Animation.LOOP_LINEAR if nom == REPOS else Animation.LOOP_NONE
		else:
			push_warning("Animation manquante : " + nom)


# ---------- Interface ----------
func _creer_interface() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)

	# Boutons en haut
	var boite := HBoxContainer.new()
	boite.position = Vector2(20, 20)
	layer.add_child(boite)
	for nom in [REPOS] + MOUVEMENTS:
		var b := Button.new()
		b.text = nom
		b.pressed.connect(jouer.bind(nom))
		boite.add_child(b)

	# Nom du mouvement en grand
	label_mouvement = Label.new()
	label_mouvement.position = Vector2(20, 80)
	label_mouvement.add_theme_font_size_override("font_size", 36)
	layer.add_child(label_mouvement)
	_maj_label(REPOS)

	# Panneau "prédiction" (vide au départ, rempli plus tard par le WebSocket)
	var panneau := VBoxContainer.new()
	panneau.position = Vector2(20, 140)
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


# Appelée par le WebSocket en S4. Vert si la prédiction est juste, rouge sinon.
# conf est entre 0.0 et 1.0.
func maj_prediction(predite: String, vraie: String, conf: float) -> void:
	label_predite.text = "Prédite : " + predite.replace("_", " ")
	label_vraie.text = "Vraie : " + vraie.replace("_", " ")
	label_conf.text = "Confiance : %d %%" % int(conf * 100)
	barre_conf.value = conf * 100
	var couleur := Color.GREEN if predite == vraie else Color.RED
	label_predite.add_theme_color_override("font_color", couleur)


# ---------- Entrées clavier ----------
func _input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if TOUCHES.has(event.keycode):
			jouer(TOUCHES[event.keycode])


# ---------- Point d'entrée unique : clavier, boutons, et bientôt WebSocket ----------
func jouer(nom: String) -> void:
	if not anim.has_animation(nom):
		push_warning("Animation introuvable : " + nom)
		return

	# Même animation déjà en cours : on ne la relance pas
	# (important plus tard : le classifieur renverra la même classe plusieurs fois de suite)
	if anim.is_playing() and anim.current_animation == nom:
		return

	var occupe := anim.is_playing() and anim.current_animation != REPOS

	# Mode "finir le mouvement" : on mémorise la commande pour plus tard
	if occupe and not INTERROMPRE and nom != REPOS:
		en_attente = nom
		return

	en_attente = ""
	anim.play(nom, FONDU)
	_maj_label(nom)


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


# ---------- Mode démo (test de l'interface) ----------
func _demo() -> void:
	# 4 mouvements avec prédiction juste -> texte vert
	for nom in MOUVEMENTS:
		await get_tree().create_timer(3.0).timeout
		jouer(nom)
		maj_prediction(nom, nom, 0.85)
	# 1 prédiction fausse -> texte rouge
	await get_tree().create_timer(3.0).timeout
	jouer("main_gauche")
	maj_prediction("main_gauche", "main_droite", 0.55)
	# retour au repos
	await get_tree().create_timer(3.0).timeout
	jouer(REPOS)
