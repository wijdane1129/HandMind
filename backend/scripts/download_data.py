"""S1 - Download PhysioNet EEGMMIDB (imagery runs) and BCI IV 2a.

Usage (from backend/):
    python -m scripts.download_data                 # everything
    python -m scripts.download_data --subjects 1 2 3
    python -m scripts.download_data --skip-bci

Files go to data/raw/ (PhysioNet) and ~/mne_data (MOABB / BCI IV 2a).
Already-downloaded files are skipped, so the script can be re-run safely.
Expect a few GB and some time for the 109 subjects.
"""

import argparse

from handmind import config
from handmind.data import download_physionet, load_bci_iv_2a


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=list(range(1, config.N_SUBJECTS + 1)))
    parser.add_argument("--skip-bci", action="store_true")
    args = parser.parse_args()

    failed = []
    for s in args.subjects:
        try:
            download_physionet(s)
            print(f"[PhysioNet] S{s:03d} ok")
        except Exception as e:  # network hiccup: keep going, report at the end
            print(f"[PhysioNet] S{s:03d} FAILED: {e}")
            failed.append(s)

    if not args.skip_bci:
        for s in range(1, 10):
            load_bci_iv_2a(s)
            print(f"[BCI IV 2a] S{s} ok")

    if failed:
        print(f"\nRe-run with: --subjects {' '.join(map(str, failed))}")


if __name__ == "__main__":
    main()
