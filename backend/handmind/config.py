"""Central configuration for HandMind.

Every constant used by more than one file lives here, so that changing a
parameter (filter band, epoch window, channel list...) is done in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (backend/handmind/config.py -> project root is 2 levels up)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded EDF files (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"  # preprocessed epochs (.npz, gitignored)
SPLITS_FILE = DATA_DIR / "splits.json"  # train / demo subjects (committed)
FIGURES_DIR = ROOT_DIR / "docs" / "figures"
RESULTS_DIR = ROOT_DIR / "docs" / "results"  # CSV / Markdown result tables
MODELS_DIR = ROOT_DIR / "models"

# ---------------------------------------------------------------------------
# PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)
# ---------------------------------------------------------------------------
N_SUBJECTS = 109
SFREQ = 160.0 #Sampling frequency

# Imagery runs only (we never use the real-movement runs 3, 5, 7, 9, 11, 13)
RUNS_HANDS = [4, 8, 12]   # T1 = left hand,  T2 = right hand
RUNS_FEET = [6, 10, 14]   # T1 = both hands, T2 = both feet
ALL_RUNS = sorted(RUNS_HANDS + RUNS_FEET)

# Subjects with a known recording problem (wrong sampling rate or
# incomplete runs). Excluded from training AND from the demo draw.
# -> confirm during the S1 exploration and note it in the report.
EXCLUDED_SUBJECTS = [88, 92, 100, 104]

# Class names = the 5 avatar movements (section 2 of the cahier des charges)
# S6: "pieds" renamed "deux_pieds" to match the JSON contract of Etudiant B
# (docs/protocole_json.md). Codes are unchanged (deux_pieds = 4), so the
# epochs and models saved before S6 still work.
CLASSES = ["repos", "main_gauche", "main_droite", "deux_mains", "deux_pieds"]
EVENT_ID = {name: i for i, name in enumerate(CLASSES)}  # repos=0 ... deux_pieds=4
CLASS_ALIASES = {"pieds": "deux_pieds"}  # old name -> contract name

# PhysioNet code -> our class name, depending on the run type
ANNOTATION_MAP_HANDS = {"T0": "repos", "T1": "main_gauche", "T2": "main_droite"} #run 4/8/12
ANNOTATION_MAP_FEET = {"T0": "repos", "T1": "deux_mains", "T2": "deux_pieds"} #run 6/10/14

# ---------------------------------------------------------------------------
# Electrode configurations (section 4)
# Names are those produced by mne.datasets.eegbci.standardize()
# ---------------------------------------------------------------------------
CHANNELS_ALL = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
]

CHANNEL_CONFIGS = {
    "A": CHANNELS_ALL,  # 64 channels - reference
    "B": [               # 17 channels - motor cortex
        "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
    ],
    "C": ["C3", "Cz", "C4"],  # 3 channels - cheap headset
}

# ---------------------------------------------------------------------------
# Preprocessing (section 4)
# ---------------------------------------------------------------------------
L_FREQ = 8.0     # band-pass low edge (Hz)
H_FREQ = 30.0    # band-pass high edge (Hz)
NOTCH_FREQ = 60.0  # US mains (PhysioNet recorded in the USA)
TMIN = 0.5       # epoch start after cue (s)
TMAX = 4.0       # epoch end after cue (s)

# ---------------------------------------------------------------------------
# Demo subjects (section 4): drawn once in S1, never used for training
# ---------------------------------------------------------------------------
N_DEMO_SUBJECTS = 10
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# S3 - Offline baselines (2 classes, intra-subject)
# ---------------------------------------------------------------------------
BASELINE_CLASSES = ["main_gauche", "main_droite"]  # left vs right hand
N_BASELINE_SUBJECTS = 20   # first 20 TRAIN subjects of splits.json
N_FOLDS = 5                # stratified k-fold, shuffled with RANDOM_SEED
CSP_N_COMPONENTS = 6       # clipped to n_channels for config C (3 channels)  #C'est une méthode de traitement du signal EEG qui cherche à trouver des combinaisons spatiales d'électrodes permettant de mieux distinguer deux classes.

# ---------------------------------------------------------------------------
# S4 - 3 / 5 classes, FBCSP, model choice
# ---------------------------------------------------------------------------
CLASS_SETS = {
    2: ["main_gauche", "main_droite"],
    3: ["repos", "main_gauche", "main_droite"],
    5: CLASSES,
}
# "repos" has ~2x more trials than each movement: with 3 or 5 classes the
# classes are balanced (random under-sampling, RANDOM_SEED) before the CV.

# FBCSP filter bank. The saved epochs are already band-passed 8-30 Hz, so the
# bands stay inside 8-30 Hz (no need to rebuild data/processed/).
FBCSP_BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)] #Au lieu d'utiliser une seule bande de fréquences, on divise le signal en plusieurs sous-bandes.Chaque sous-bande est traitée séparément avec CSP.Parce que les informations discriminantes entre deux mouvements peuvent se trouver dans différentes fréquences.
FBCSP_N_COMPONENTS = 4   # CSP filters per band (clipped to n_channels)
FBCSP_K_BEST = 10        # features kept by mutual information

# Filled at the end of S4 (see docs/S4_RESULTS.md), read by the online
# classifier in S5.
# S4 choice: Riemann is the best model in every table (S3 2 classes, S4 3
# classes, LOSO, BCI IV 2a). Config B (17 ch) instead of A (64 ch): ~5
# points lower but 4x fewer channels, faster online, and the cahier's focus.
CHOSEN_MODEL = "riemann"#les méthodes riemanniennes exploitent la structure géométrique des matrices de covariance EEG pour effectuer la classification. Elles sont particulièrement adaptées aux signaux EEG, qui sont souvent bruités et non stationnaires. En utilisant des distances riemanniennes, ces méthodes peuvent capturer les variations subtiles dans les signaux EEG associées à différentes tâches ou états cognitifs.
CHOSEN_CONFIG = "B"

# BCI Competition IV 2a benchmark (test T8)
BCI2A_SUBJECTS = list(range(1, 10))
BCI2A_EVENT_ID = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
# In MOABB, each BCI IV 2a event marks the START of the trial (fixation
# cross); the cue appears 2 s later (MOABB: interval=[2, 6]). Windows are
# therefore given relative to the CUE and shifted by this offset.
BCI2A_CUE_OFFSET = 2.0   # s from the event to the cue
BCI2A_TMIN = 0.5         # s after the cue (standard window 0.5-2.5 s)
BCI2A_TMAX = 2.5
BCI2A_NOTCH = 50.0       # recorded in Austria (50 Hz mains)

# ---------------------------------------------------------------------------
# S5 - Online (real-time) classifier
# ---------------------------------------------------------------------------
ONLINE_WINDOW_S = 1.0    # sliding window length (s)
ONLINE_STEP_S = 0.25     # one decision every 250 ms (4 per second)
ONLINE_CLASSES = 5       # number of classes of the online model (2, 3 or 5)
ONLINE_THRESHOLD = 0.40  # below this probability the window counts as "repos"
                         # (chance = 0.20 with 5 classes) - tune in S6
ONLINE_N_VOTE = 3        # majority vote over the last 3 windows
# Training windows cut inside each trial, relative to the cue (s):
# 0.5-1.5, 1.0-2.0, ... 3.0-4.0  ->  6 windows per trial
ONLINE_TRAIN_STARTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# Causal filter (the zero-phase filter of S2 needs future samples)
ONLINE_FILTER_ORDER = 4  # Butterworth band-pass 8-30 Hz + notch 60 Hz

# LSL streams sent by the dataset player (Etudiant B, S3) - resolved by type
LSL_EEG_TYPE = "EEG"
LSL_MARKER_TYPE = "Markers"

# Stream names of the player (backend/reader/lecteur.py)
LSL_EEG_NAME = "HandMind_EEG"
LSL_MARKER_NAME = "HandMind_Markers"
# Unit of the EEG stream: the player sends microvolts by default (desc/unit),
# the models are trained in volts (MNE). Factor applied to the stream:
LSL_UNIT_SCALE = {"volts": 1.0, "v": 1.0, "microvolts": 1e-6, "uv": 1e-6,
                  "µv": 1e-6, "millivolts": 1e-3, "mv": 1e-3}
# Technical markers of the player: start of a new run (signal jump) and end
LSL_NEW_RUN_PREFIX = "NOUVEAU_RUN"
LSL_END_PREFIX = "FIN"

# WebSocket contract with the Godot front (docs/protocole_json.md, v1.2)
WS_HOST = "localhost"
WS_PORT = 8765
WS_STATE_PERIOD_S = 0.5    # {"type": "etat"} message every 0.5 s
WS_STREAM_LOST_S = 1.5     # no EEG sample for 1.5 s -> "flux perdu"

# ---------------------------------------------------------------------------
# S6 - Integration, T12 / T15
# ---------------------------------------------------------------------------
T12_MEDIAN_MS = 200.0      # le temps entre la réception d'une fenêtre EEG et la production de la décision soit évalué.
T12_P95_MS = 300.0         # 95 % des décisions doivent être produites en 300 ms ou moins.
T15_MAX_GAP = 0.05         # L'écart maximal toléré entre les performances offline et online.
N_T15_SUBJECTS = 3         # S6: 3 demo subjects (S9: all 10)


def online_model_name():
    """Model used online: the S4 choice, or Riemann until it is filled."""
    return CHOSEN_MODEL or "riemann"


def online_config_name():
    return CHOSEN_CONFIG or "B"
