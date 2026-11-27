extends Control
# Interface HandMind (Étape 4) : squelette en 3 colonnes + avatar dans un SubViewport.
# Tout est construit par code. Les données viennent de l'Autoload Bus.

const ThemeHM := preload("res://scripts/ui/theme_handmind.gd")

# Cadrage de la caméra du panneau avatar (à ajuster selon ton avatar)
const CAM_POS := Vector3(0.0, 1.2, 3.6)
const CAM_CIBLE := Vector3(0.0, 1.1, 0.0)
const CAM_FOV := 40.0   # petit FOV = avatar plus gros, moins de déformation

const NOMS := {
	"repos": "Rest",
	"main_gauche": "Left Hand",
	"main_droite": "Right Hand",
	"deux_mains": "Both Hands",
	"deux_pieds": "Both Feet (Squat)",
}

var zone: VBoxContainer
var vues := {}

var label_classe: Label
var badge_resultat: Label
var label_verite: Label
var label_conf: Label
var label_mouvement: Label

var point_connexion: Label
var label_connexion: Label
var label_flux: Label
var label_latence: Label
var label_fps: Label
var label_temps: Label


func _ready() -> void:
	theme = ThemeHM.creer()
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	# Le rendu 3D se fait dans le SubViewport du panneau avatar, pas dans la fenêtre
	get_viewport().disable_3d = true
	_construire()

	Bus.prediction_recue.connect(_on_prediction)
	Bus.connexion_changee.connect(_on_connexion)
	Bus.etat_recu.connect(_on_etat)
	_on_connexion(false)

	var timer := Timer.new()
	timer.wait_time = 1.0
	timer.autostart = true
	timer.timeout.connect(_maj_fps)
	add_child(timer)


# ---------- Petits utilitaires ----------
func _label(texte: String, variation: String = "") -> Label:
	var l := Label.new()
	l.text = texte
	if variation != "":
		l.theme_type_variation = variation
	return l


func _etendre(c: Control, ratio: float = 1.0) -> Control:
	c.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	c.size_flags_vertical = Control.SIZE_EXPAND_FILL
	c.size_flags_stretch_ratio = ratio
	return c


func _espaceur() -> Control:
	var c := Control.new()
	c.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	return c


func _espaceur_v() -> Control:
	var c := Control.new()
	c.size_flags_vertical = Control.SIZE_EXPAND_FILL
	return c


# Panneau avec un titre ; le contenu s'ajoute avec _contenu(panneau)
func _panneau(titre: String) -> PanelContainer:
	var p := PanelContainer.new()
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 10)
	p.add_child(v)
	v.add_child(_label(titre, "Section"))
	return p


func _contenu(p: PanelContainer) -> VBoxContainer:
	return p.get_child(0) as VBoxContainer


func _ajouter_vue(cle: String, v: Control) -> void:
	vues[cle] = v
	_etendre(v)
	v.visible = (cle == "simulation")
	zone.add_child(v)


func _placeholder(titre: String, texte: String) -> PanelContainer:
	var p := _panneau(titre)
	_contenu(p).add_child(_label(texte, "Secondaire"))
	return p


# ---------- Construction ----------
func _construire() -> void:
	var fond := ColorRect.new()
	fond.color = ThemeHM.FOND
	fond.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	fond.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(fond)

	var marge := MarginContainer.new()
	marge.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for cote in ["left", "right", "top", "bottom"]:
		marge.add_theme_constant_override("margin_" + cote, 12)
	add_child(marge)

	var colonne := VBoxContainer.new()
	colonne.add_theme_constant_override("separation", 10)
	marge.add_child(colonne)

	colonne.add_child(_barre_haut())

	zone = VBoxContainer.new()
	_etendre(zone)
	colonne.add_child(zone)

	var simulation := HBoxContainer.new()
	simulation.add_theme_constant_override("separation", 10)
	simulation.add_child(_colonne_gauche())
	simulation.add_child(_colonne_centre())
	simulation.add_child(_colonne_droite())
	_ajouter_vue("simulation", simulation)
	_ajouter_vue("pipeline", _placeholder("Signal pipeline",
		"Vue « du signal au mouvement » : Étape 11 (semaine 8)."))
	_ajouter_vue("evaluation", _placeholder("Evaluation",
		"Matrice de confusion et résultats par configuration : Étape 11."))

	colonne.add_child(_barre_lecture())
	colonne.add_child(_barre_etat())


func _barre_haut() -> Control:
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 10)
	h.add_child(_label("HANDMIND", "Section"))
	h.add_child(_label("   "))

	var groupe := ButtonGroup.new()
	var titres := {"simulation": "Simulation", "pipeline": "Signal pipeline", "evaluation": "Evaluation"}
	for cle in titres:
		var b := Button.new()
		b.text = titres[cle]
		b.toggle_mode = true
		b.button_group = groupe
		b.button_pressed = (cle == "simulation")
		b.pressed.connect(_afficher_vue.bind(cle))
		h.add_child(b)

	h.add_child(_label("   Session 001 / Motor imagery", "Secondaire"))
	h.add_child(_espaceur())

	var badge := PanelContainer.new()
	badge.add_theme_stylebox_override("panel",
		ThemeHM.boite(Color(0.0, 0.83, 1.0, 0.12), ThemeHM.CYAN, 6, 10, 4))
	var txt := _label("DATASET REPLAY", "Secondaire")
	txt.add_theme_color_override("font_color", ThemeHM.CYAN)
	badge.add_child(txt)
	h.add_child(badge)

	var export := Button.new()
	export.text = "Export session"
	export.disabled = true   # Étape 12
	h.add_child(export)
	return h


func _afficher_vue(cle: String) -> void:
	for k in vues:
		vues[k].visible = (k == cle)


func _colonne_gauche() -> Control:
	var p := _panneau("Session setup")
	p.custom_minimum_size = Vector2(280, 0)
	var v := _contenu(p)
	# Valeurs fixes pour l'instant : les vrais contrôles arrivent à l'Étape 9
	var lignes := [
		["DATA SOURCE", "PhysioNet EEGMMIDB"],
		["SUBJECT / RUN", "S001 / R04"],
		["CHANNELS", "17 ch · Motor cortex"],
		["CLASSIFIER", "FBCSP + SVM"],
		["CONFIDENCE THRESHOLD", "70 %"],
	]
	for l in lignes:
		v.add_child(_label(l[0], "Secondaire"))
		v.add_child(_label(l[1]))
	v.add_child(_espaceur_v())
	v.add_child(_label("Dataset-only simulation\nNo physical EEG headset connected.", "Secondaire"))
	return _etendre(p, 0.22)


func _colonne_centre() -> Control:
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 10)
	v.add_child(_etendre(_panneau_avatar()))
	var eeg := _placeholder("EEG signal monitor", "Tracé C3 / Cz / C4 : Étape 7.")
	eeg.custom_minimum_size = Vector2(0, 180)
	v.add_child(eeg)
	return _etendre(v, 0.5)


func _panneau_avatar() -> PanelContainer:
	var p := _panneau("Avatar response")
	var v := _contenu(p)

	# Le SubViewport partage le monde 3D de la scène principale :
	# l'avatar et ses animations restent exactement où ils sont dans Main.tscn.
	var cadre := PanelContainer.new()
	cadre.add_theme_stylebox_override("panel",
		ThemeHM.boite(Color("0b1524"), ThemeHM.BORDURE, 10, 0, 0))
	_etendre(cadre)
	v.add_child(cadre)

	var conteneur := SubViewportContainer.new()
	conteneur.stretch = true
	_etendre(conteneur)
	cadre.add_child(conteneur)

	var sub := SubViewport.new()
	sub.transparent_bg = true
	conteneur.add_child(sub)

	var cam := Camera3D.new()
	sub.add_child(cam)
	cam.transform = Transform3D(Basis.looking_at(CAM_CIBLE - CAM_POS, Vector3.UP), CAM_POS)
	cam.fov = CAM_FOV
	cam.current = true

	var bas := HBoxContainer.new()
	bas.add_child(_label("CURRENT MOVEMENT", "Secondaire"))
	bas.add_child(_espaceur())
	label_mouvement = _label("Rest", "Section")
	bas.add_child(label_mouvement)
	v.add_child(bas)
	return p


func _colonne_droite() -> Control:
	var v := VBoxContainer.new()
	v.add_theme_constant_override("separation", 10)

	var p := _panneau("Movement classification")
	p.custom_minimum_size = Vector2(320, 0)
	var c := _contenu(p)
	c.add_child(_label("PREDICTED CLASS", "Secondaire"))
	var ligne := HBoxContainer.new()
	label_classe = _label("—", "TitreGrand")
	label_classe.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	badge_resultat = _label("")
	ligne.add_child(label_classe)
	ligne.add_child(badge_resultat)
	c.add_child(ligne)
	label_verite = _label("Dataset ground truth : —", "Secondaire")
	label_conf = _label("Confidence : —", "Secondaire")
	c.add_child(label_verite)
	c.add_child(label_conf)
	c.add_child(_label("Barres de probabilité : Étape 8.", "Secondaire"))
	v.add_child(p)

	v.add_child(_etendre(_placeholder("Prediction history", "Historique et précision : Étape 8.")))
	return _etendre(v, 0.28)


func _barre_lecture() -> Control:
	var p := PanelContainer.new()
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 12)
	p.add_child(h)
	for nom in ["Pause", "Stop"]:
		var b := Button.new()
		b.text = nom
		b.disabled = true   # Étape 10
		h.add_child(b)
	label_temps = _label("00:00.00 / 00:00.00", "Section")
	h.add_child(label_temps)
	var barre := ProgressBar.new()
	barre.show_percentage = false
	barre.custom_minimum_size = Vector2(0, 8)
	barre.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	barre.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	h.add_child(barre)
	h.add_child(_label("Contrôles de lecture : Étape 10", "Secondaire"))
	return p


func _barre_etat() -> Control:
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 18)
	point_connexion = _label("●")
	label_connexion = _label("", "Secondaire")
	label_flux = _label("Replay stream : waiting", "Secondaire")
	label_latence = _label("Latency : —", "Secondaire")
	label_fps = _label("Godot 4", "Secondaire")
	h.add_child(point_connexion)
	h.add_child(label_connexion)
	h.add_child(label_flux)
	h.add_child(label_latence)
	h.add_child(_espaceur())
	h.add_child(label_fps)
	return h


# ---------- Réaction aux données du Bus ----------
func _on_prediction(d: Dictionary) -> void:
	var classe: String = d["classe"]
	var vraie: String = d["vraie_classe"]
	var nom := str(NOMS.get(classe, classe))
	label_classe.text = nom
	label_mouvement.text = nom
	label_verite.text = "Dataset ground truth : " + str(NOMS.get(vraie, "—"))
	label_conf.text = "Confidence : %d %%" % int(d["confiance"] * 100.0)
	var juste := (classe == vraie)

	# Latence bout en bout : heure actuelle - heure Unix de la fenêtre (comme T13)
	var t: float = d["t"]
	if t > 1.0e9:
		var lat_ms: float = (Time.get_unix_time_from_system() - t) * 1000.0
		label_latence.text = "Latency : %d ms" % int(lat_ms)
	elif d["latence_ms"] >= 0.0:
		label_latence.text = "Latency : %d ms" % int(d["latence_ms"])
	badge_resultat.text = "✓ Correct" if juste else "✗ Error"
	badge_resultat.add_theme_color_override("font_color",
		ThemeHM.VERT if juste else ThemeHM.ROUGE)


func _on_connexion(ok: bool) -> void:
	point_connexion.add_theme_color_override("font_color", ThemeHM.VERT if ok else ThemeHM.ROUGE)
	if Bus.SIMULATION:
		label_connexion.text = "Simulated source"
	else:
		label_connexion.text = "Python backend connected" if ok else "Python backend disconnected"
	if not ok:
		label_flux.text = "Replay stream : waiting"


func _on_etat(d: Dictionary) -> void:
	if d.has("flux"):
		var ok := bool(d["flux"])
		label_flux.text = "Replay stream stable" if ok else "Replay stream lost"
		label_flux.add_theme_color_override("font_color", ThemeHM.TEXTE_2 if ok else ThemeHM.ORANGE)
	if d.has("t_lecture") and d.has("duree"):
		label_temps.text = "%s / %s" % [_fmt(float(d["t_lecture"])), _fmt(float(d["duree"]))]


func _fmt(s: float) -> String:
	return "%02d:%05.2f" % [int(s / 60.0), fmod(s, 60.0)]


func _maj_fps() -> void:
	label_fps.text = "Godot 4 · %d FPS" % Engine.get_frames_per_second()
