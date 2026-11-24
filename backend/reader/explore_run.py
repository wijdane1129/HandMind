from mne.datasets import eegbci
from mne.io import read_raw_edf

fichiers = eegbci.load_data(1, [4])          # sujet 1, run 4
raw = read_raw_edf(fichiers[0], preload=True)

print("Fréquence :", raw.info["sfreq"])      # attendu : 160.0
print("Canaux :", len(raw.ch_names))         # attendu : 64
print("Durée (s) :", raw.times[-1])          # environ 125
print(raw.annotations)                       # T0, T1, T2
print(raw.ch_names[:10])                     # noms avec point : "Fc5."