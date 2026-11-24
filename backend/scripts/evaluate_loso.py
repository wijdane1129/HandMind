"""S4 - Test T9 (part 1): inter-subject evaluation, leave-one-subject-out.

Question answered: can a model trained on OTHER people decode a new person
without any calibration? (In the demo, the avatar is driven by demo
subjects the model has never seen, so this is the realistic score.)

For each of the N subjects: train on the N-1 others (all their trials
pooled), test on all trials of the left-out subject.

Usage (from backend/):
    python -m scripts.evaluate_loso                         # 20 subjects, 2 classes, config B, 3 models
    python -m scripts.evaluate_loso --classes 5 --config C
    python -m scripts.evaluate_loso --models riemann --configs A B C

Outputs:
    docs/results/s4_loso_<k>classes.csv / .md
    docs/figures/s4_loso_<k>classes.png   intra-subject vs LOSO, if the
                                          intra-subject CSV of the same run exists

Note: pooling 19 subjects x 64 channels takes ~1 GB of RAM for 5 classes.
Start with --configs B C if your machine is short on memory.
"""

import argparse
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from handmind import config  # noqa: E402
from handmind.evaluation import loso  # noqa: E402 #LOSO : Effectuer l'évaluation Leave-One-Subject-Out.Can our model recognize motor imagery from a person it has never seen before?
from handmind.models import MODELS, make_model  # noqa: E402
from handmind.preprocessing import load_epochs  # noqa: E402
from scripts.evaluate import default_subjects, ensure_epochs  # noqa: E402


def to_markdown(df, n_classes):
    lines = [f"# Leave-one-subject-out - {n_classes} classes", "",
             f"Train on {df['n_train_subjects'].iloc[0]} subjects, test on the "
             "left-out one (no calibration on the test subject).", "",
             "| Model | Config | Accuracy | Kappa | > chance |",
             "|---|---|---|---|---|"]
    for (m, c), g in df.groupby(["model", "config"]):
        above = int((g["acc"] > g["chance_p05"]).sum())
        lines.append(f"| {m} | {c} | {g['acc'].mean():.3f} ± {g['acc'].std():.3f}"
                     f" | {g['kappa'].mean():.3f} ± {g['kappa'].std():.3f}"
                     f" | {above}/{len(g)} |")
    lines += ["", "## Accuracy per left-out subject", ""]
    piv = df.pivot_table(index="subject", columns=["model", "config"], values="acc")
    cols = list(piv.columns)
    lines.append("| Subject | " + " | ".join(f"{m} {c}" for m, c in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for s, row in piv.iterrows():
        lines.append(f"| S{s:03d} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    return "\n".join(lines) + "\n"


def plot_vs_intra(loso_df, intra_csv, path):
    intra = pd.read_csv(intra_csv)
    a = intra.groupby(["model", "config"])["acc_mean"].mean().rename("intra-subject")
    b = loso_df.groupby(["model", "config"])["acc"].mean().rename("LOSO")
    both = pd.concat([a, b], axis=1, join="inner")
    if both.empty:
        return False
    ax = both.plot.bar(figsize=(1.2 * len(both) + 3, 4), rot=30)
    ax.axhline(loso_df["chance_p05"].mean(), ls="--", c="grey",
               label="chance (p<0.05)")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("")
    ax.legend()
    ax.set_title("Intra-subject (5-fold) vs inter-subject (LOSO)")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=120)
    plt.close(ax.figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--classes", type=int, choices=sorted(config.CLASS_SETS),
                        default=2)
    parser.add_argument("--configs", nargs="+", default=["B"])
    parser.add_argument("--models", nargs="+", choices=list(MODELS),
                        default=list(MODELS))
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()

    classes = config.CLASS_SETS[args.classes]
    balance = args.classes > 2 and not args.no_balance
    subjects = args.subjects or default_subjects()
    if len(subjects) < 2:
        raise SystemExit("LOSO needs at least 2 subjects.")
    print(f"LOSO on {len(subjects)} subjects, classes {classes}, "
          f"configs {args.configs}, models {args.models}")

    rows = []
    for cfg in args.configs:
        datasets = []
        for s in subjects:
            ensure_epochs(s)
            datasets.append(load_epochs(s, cfg, classes=classes))
        for m in args.models:
            factory = partial(make_model, m)
            for r in loso(datasets, factory, balance=balance):
                r.update(model=m, config=cfg)
                rows.append(r)
                print(f"{cfg} {m:9s} test S{r['subject']:03d}: "
                      f"acc={r['acc']:.3f} kappa={r['kappa']:.3f} "
                      f"({r['seconds']:.1f}s)")

    df = pd.DataFrame(rows)
    name = f"s4_loso_{args.classes}classes"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = config.RESULTS_DIR / f"{name}.csv"
    md = config.RESULTS_DIR / f"{name}.md"
    df.to_csv(csv, index=False, float_format="%.4f")
    md.write_text(to_markdown(df, args.classes), encoding="utf-8")
    print("\n" + df.groupby(["model", "config"])[["acc", "kappa"]]
          .mean().to_string(float_format="%.3f"))
    print(f"\n-> {csv}\n-> {md}")

    # intra-subject results of the same class set (S4 run first, else S3)
    intra = config.RESULTS_DIR / f"s4_{args.classes}classes.csv"
    if args.classes == 2 and not intra.exists():
        intra = config.RESULTS_DIR / "s3_baselines.csv"
    if intra.exists():
        fig = config.FIGURES_DIR / f"{name}.png"
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if plot_vs_intra(df, intra, fig):
            print(f"-> {fig}")


if __name__ == "__main__":
    main()
