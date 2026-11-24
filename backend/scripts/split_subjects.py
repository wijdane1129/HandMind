"""S1 - Draw the 10 demo subjects once and save them to data/splits.json.

Usage (from backend/):
    python -m scripts.split_subjects

Run it ONCE, commit data/splits.json, never run it again.
"""

from handmind import config
from handmind.split import draw_demo_subjects, save_split


def main():
    split = draw_demo_subjects()
    save_split(split)
    print(f"Demo subjects (never train on these): {split['demo']}")
    print(f"Train subjects: {len(split['train'])}")
    print(f"Excluded: {split['excluded']}")
    print(f"Saved to {config.SPLITS_FILE} -> commit this file.")


if __name__ == "__main__":
    main()
