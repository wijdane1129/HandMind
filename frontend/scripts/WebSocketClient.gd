extends Node
# Client WebSocket vers le backend. Il ne connaît pas l'interface :
# il lit les messages JSON et les redistribue via l'Autoload Bus.
# Compatible avec le protocole v1.1 (sans "type" = prédiction) et la v2.

const URL := "ws://localhost:8765"
const DELAI_RECONNEXION := 2.0

var socket := WebSocketPeer.new()
var connecte_precedent := false
var attente := 0.0


func _ready() -> void:
	# En mode simulation, on ne se connecte à rien
	if Bus.SIMULATION:
		set_process(false)
		return
	Bus.commande.connect(envoyer)
	_connecter()


func _connecter() -> void:
	socket = WebSocketPeer.new()
	var err = socket.connect_to_url(URL)
	if err != OK:
		print("Erreur de connexion : ", err)


# Interface -> backend (play, pause, sujet, seuil...)
func envoyer(d: Dictionary) -> void:
	if socket.get_ready_state() == WebSocketPeer.STATE_OPEN:
		socket.send_text(JSON.stringify(d))
	else:
		push_warning("Commande ignorée (non connecté) : " + str(d))


func _process(delta: float) -> void:
	socket.poll()
	var etat = socket.get_ready_state()

	# On n'émet que si "connecté / déconnecté" change vraiment
	var connecte = (etat == WebSocketPeer.STATE_OPEN)
	if connecte != connecte_precedent:
		connecte_precedent = connecte
		Bus.connexion_changee.emit(connecte)

	# On vide la file même si la connexion vient de se fermer
	while socket.get_available_packet_count() > 0:
		var texte = socket.get_packet().get_string_from_utf8()
		var data = JSON.parse_string(texte)
		if not (data is Dictionary):
			print("JSON invalide : ", texte)
			continue

		var genre: String = str(data.get("type", ""))
		match genre:
			"etat":
				Bus.etat_recu.emit(data)
			"eeg":
				Bus.eeg_recu.emit(data)
			"stats":
				Bus.stats_recues.emit(data)
			_:
				# pas de "type" (ou "prediction") = message de prédiction
				Bus.prediction_recue.emit(Bus.normaliser_prediction(data))

	if etat == WebSocketPeer.STATE_CLOSED:
		attente += delta
		if attente >= DELAI_RECONNEXION:
			attente = 0.0
			print("Reconnexion...")
			_connecter()
