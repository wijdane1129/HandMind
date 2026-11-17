extends Node

signal classe_recue(numero, classe, confiance, vraie_classe, t)
signal etat_change(connecte)
signal etat_flux(ok)

const URL := "ws://localhost:8765"
const DELAI_RECONNEXION := 2.0

var socket := WebSocketPeer.new()
var connecte_precedent := false
var attente := 0.0
var compteur_recus := 0


func _ready():
	_connecter()


func _connecter():
	socket = WebSocketPeer.new()
	var err = socket.connect_to_url(URL)
	if err != OK:
		print("Erreur de connexion : ", err)


func _process(delta):
	socket.poll()
	var etat = socket.get_ready_state()

	# On n'émet le signal que si "connecté / déconnecté" change vraiment
	var connecte = (etat == WebSocketPeer.STATE_OPEN)
	if connecte != connecte_precedent:
		connecte_precedent = connecte
		etat_change.emit(connecte)

	# On vide la file des paquets même si la connexion vient de se fermer
	while socket.get_available_packet_count() > 0:
		var texte = socket.get_packet().get_string_from_utf8()
		var data = JSON.parse_string(texte)
		if not (data is Dictionary):
			print("JSON invalide : ", texte)
			continue

		# Message d'état du pont : {"type": "etat", "flux": true/false}
		if data.get("type", "") == "etat":
			etat_flux.emit(bool(data.get("flux", false)))
			continue

		compteur_recus += 1
		classe_recue.emit(
			int(data.get("numero", -1)),
			data.get("classe", ""),
			float(data.get("confiance", 0.0)),
			data.get("vraie_classe", ""),
			float(data.get("t", 0.0))
		)

	if etat == WebSocketPeer.STATE_CLOSED:
		attente += delta
		if attente >= DELAI_RECONNEXION:
			attente = 0.0
			print("Reconnexion...")
			_connecter()
