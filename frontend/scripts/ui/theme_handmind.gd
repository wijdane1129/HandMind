extends RefCounted
# Thème HandMind (couleurs de la maquette). Construit en code : pas de .tres à éditer.
# Usage : const ThemeHM := preload("res://scripts/ui/theme_handmind.gd")
#         theme = ThemeHM.creer()

const FOND := Color("0a1220")
const PANNEAU := Color("0f1a2b")
const BORDURE := Color("1e2d45")
const TEXTE := Color("c9d6ea")
const TEXTE_2 := Color("7f90ab")
const CYAN := Color("00d4ff")
const VERT := Color("34d399")
const ROUGE := Color("f87171")
const ORANGE := Color("fbbf24")


static func boite(fond: Color, bordure: Color, rayon: int, marge_h: int, marge_v: int) -> StyleBoxFlat:
	var s := StyleBoxFlat.new()
	s.bg_color = fond
	s.border_color = bordure
	s.set_border_width_all(1 if bordure.a > 0.0 else 0)
	s.set_corner_radius_all(rayon)
	s.content_margin_left = marge_h
	s.content_margin_right = marge_h
	s.content_margin_top = marge_v
	s.content_margin_bottom = marge_v
	return s


static func creer() -> Theme:
	var t := Theme.new()

	# Labels
	t.set_color("font_color", "Label", TEXTE)
	t.set_font_size("font_size", "Label", 14)

	# Variations de Label (label.theme_type_variation = "TitreGrand")
	t.set_type_variation("TitreGrand", "Label")
	t.set_font_size("font_size", "TitreGrand", 34)
	t.set_color("font_color", "TitreGrand", CYAN)

	t.set_type_variation("Section", "Label")
	t.set_font_size("font_size", "Section", 16)
	t.set_color("font_color", "Section", Color.WHITE)

	t.set_type_variation("Secondaire", "Label")
	t.set_font_size("font_size", "Secondaire", 12)
	t.set_color("font_color", "Secondaire", TEXTE_2)

	# Panneaux
	t.set_stylebox("panel", "PanelContainer", boite(PANNEAU, BORDURE, 12, 16, 14))

	# Boutons (le bouton "pressed" est cyan : sert aussi pour les onglets)
	t.set_stylebox("normal", "Button", boite(Color("14223a"), BORDURE, 8, 14, 8))
	t.set_stylebox("hover", "Button", boite(Color("1a2d4d"), CYAN, 8, 14, 8))
	t.set_stylebox("pressed", "Button", boite(CYAN, CYAN, 8, 14, 8))
	t.set_stylebox("hover_pressed", "Button", boite(CYAN, CYAN, 8, 14, 8))
	t.set_stylebox("disabled", "Button", boite(PANNEAU, BORDURE, 8, 14, 8))
	t.set_stylebox("focus", "Button", StyleBoxEmpty.new())
	t.set_color("font_color", "Button", TEXTE)
	t.set_color("font_hover_color", "Button", Color.WHITE)
	t.set_color("font_pressed_color", "Button", FOND)
	t.set_color("font_hover_pressed_color", "Button", FOND)
	t.set_color("font_disabled_color", "Button", TEXTE_2)

	# Barres de progression
	t.set_stylebox("background", "ProgressBar", boite(BORDURE, Color.TRANSPARENT, 4, 0, 0))
	t.set_stylebox("fill", "ProgressBar", boite(CYAN, Color.TRANSPARENT, 4, 0, 0))

	return t
