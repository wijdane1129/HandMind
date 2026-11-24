"""Tests unitaires du lecteur (sans telechargement, sans reseau LSL)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reader"))
import lecteur  # noqa: E402


@pytest.mark.parametrize("run", [4, 8, 12])
def test_runs_gauche_droite(run):
    assert lecteur.code_vers_classe("T0", run) == "repos"
    assert lecteur.code_vers_classe("T1", run) == "main_gauche"
    assert lecteur.code_vers_classe("T2", run) == "main_droite"


@pytest.mark.parametrize("run", [6, 10, 14])
def test_runs_mains_pieds(run):
    assert lecteur.code_vers_classe("T0", run) == "repos"
    assert lecteur.code_vers_classe("T1", run) == "deux_mains"
    assert lecteur.code_vers_classe("T2", run) == "deux_pieds"


def test_run_invalide():
    with pytest.raises(ValueError):
        lecteur.code_vers_classe("T1", 3)


def test_timeline_synthetique():
    signal, noms, marqueurs = lecteur.construire_timeline(
        sujet=1, runs=[4, 6], synthetique=True)
    assert signal.shape[1] == 64
    assert len(noms) == 64 and noms[:3] == ["C3", "Cz", "C4"]
    # un marqueur NOUVEAU_RUN par run, au bon endroit
    assert "NOUVEAU_RUN:4" in marqueurs[0]
    debut_run6 = signal.shape[0] // 2
    assert "NOUVEAU_RUN:6" in marqueurs[debut_run6]
    # tous les autres marqueurs sont des noms de classes valides
    valides = {"repos", "main_gauche", "main_droite", "deux_mains", "deux_pieds"}
    for liste in marqueurs.values():
        for m in liste:
            assert m.startswith("NOUVEAU_RUN:") or m in valides