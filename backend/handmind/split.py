"""Train / demo subject split (section 4: "Sujets de démonstration").

The 10 demo subjects are drawn ONCE in week 1 with a fixed seed and written
to data/splits.json, which is committed to Git. Every later script reads
this file instead of re-drawing, so the demo subjects can never leak into
training or hyper-parameter tuning.
"""

import json

import numpy as np

from . import config


def draw_demo_subjects(n_demo=config.N_DEMO_SUBJECTS, seed=config.RANDOM_SEED):
    """Draw the demo subjects at random among the valid subjects."""
    valid = [s for s in range(1, config.N_SUBJECTS + 1)
             if s not in config.EXCLUDED_SUBJECTS]
    rng = np.random.default_rng(seed)#seed : fixer le hasard pour que le tirage soit reproductible.
    demo = sorted(int(s) for s in rng.choice(valid, size=n_demo, replace=False))#replace = False : ne pas tirer le même sujet plusieurs fois.
    train = [s for s in valid if s not in demo]
    return {"seed": seed, "excluded": config.EXCLUDED_SUBJECTS,
            "demo": demo, "train": train}


def save_split(split, path=config.SPLITS_FILE, overwrite=False):
    """Write the split to JSON. Refuses to overwrite an existing split."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists: the demo subjects are fixed. "
            "Use overwrite=True only if you really mean to redraw them."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))


def load_split(path=config.SPLITS_FILE):
    """Read the split and check that train and demo never overlap."""
    split = json.loads(path.read_text())
    overlap = set(split["train"]) & set(split["demo"])
    if overlap:
        raise ValueError(f"Subjects in both train and demo: {sorted(overlap)}")
    return split


def train_subjects():
    return load_split()["train"]


def demo_subjects():
    return load_split()["demo"]
