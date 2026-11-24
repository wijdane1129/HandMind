"""S3/S4 - Intra-subject comparison of the models on configs A/B/C.

For each subject x config x model: 5-fold intra-subject CV -> accuracy and
Cohen's kappa (+ a summed confusion matrix for 3 / 5 classes).

Usage (from backend/):
    python -m scripts.evaluate                                  # S3: 2 classes, csp_lda + riemann
    python -m scripts.evaluate --classes 2 --models csp_lda riemann fbcsp_svm
    python -m scripts.evaluate --classes 3                      # S4: + repos
    python -m scripts.evaluate --classes 5                      # S4: all 5 movements
    python -m scripts.evaluate --subjects 1 2 3 --configs C     # quick check

Outputs (commit them, they go in the report):
    docs/results/<name>.csv            one row per subject x config x model
    docs/results/<name>.md             summary table + per-subject table
    docs/figures/<name>.png            accuracy boxplot per model and config
    docs/figures/<name>_confusion.png  confusion matrices (3 / 5 classes only)
<name> defaults to s3_baselines (2 classes, S3 models) or s4_<k>classes.

Subjects come from data/splits.json (train only, the demo subjects are never
touched). With 3 / 5 classes, "repos" is under-sampled so that every class
has the same number of trials (--no-balance to disable).
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from handmind import config  # noqa: E402
from handmind.evaluation import evaluate_subject, summarize, to_markdown  # noqa: E402
from handmind.models import MODELS  # noqa: E402
from handmind.preprocessing import epochs_path, load_epochs  # noqa: E402
from handmind.split import train_subjects  # noqa: E402

S3_MODELS = ["csp_lda", "riemann"]


def default_subjects(n=config.N_BASELINE_SUBJECTS):
    if not config.SPLITS_FILE.exists():
        raise SystemExit(f"{config.SPLITS_FILE} not found: run "
                         "`python -m scripts.split_subjects` first (S1).")
    return train_subjects()[:n]


def ensure_epochs(subject):
    if not epochs_path(subject).exists():
        from scripts.build_epochs import process_subject
        print(f"  building epochs for S{subject:03d}: {process_subject(subject)}")


def plot_results(df, path, title):
    configs = sorted(df["config"].unique())
    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        data = [df[(df.model == m) & (df.config == c)]["acc_mean"] for c in configs]
        ax.boxplot(data)  # tick labels set below: works on any matplotlib
        ax.set_xticks(range(1, len(configs) + 1), configs)
        ax.axhline(df["chance_p05"].mean(), ls="--", c="grey",
                   label="chance (p<0.05)")
        ax.axhline(1 / df["n_classes"].iloc[0], ls=":", c="lightgrey")
        ax.set_title(m)
        ax.set_xlabel("config (A=64, B=17, C=3 ch.)")
    axes[0][0].set_ylabel("accuracy (5-fold CV)")
    axes[0][0].legend(loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusions(confusions, class_names, path):
    """confusions: {(model, config): summed matrix}. Rows normalised to 100 %."""
    keys = sorted(confusions)
    models = list(dict.fromkeys(m for m, _ in keys))
    configs = sorted({c for _, c in keys})
    fig, axes = plt.subplots(len(models), len(configs),
                             figsize=(3.6 * len(configs), 3.4 * len(models)),
                             squeeze=False)
    for i, m in enumerate(models):
        for j, c in enumerate(configs):
            ax = axes[i][j]
            cm = confusions.get((m, c))
            if cm is None:
                ax.axis("off")
                continue
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for (a, b), v in np.ndenumerate(norm):
                ax.text(b, a, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
            ax.set_xticks(range(len(class_names)), class_names, rotation=45,
                          ha="right", fontsize=7)
            ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
            ax.set_title(f"{m} - {c}", fontsize=9)
    fig.supxlabel("predicted")
    fig.supylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--classes", type=int, choices=sorted(config.CLASS_SETS),
                        default=2)
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=None,
                        help="default: S3 models for 2 classes, all 3 otherwise")
    parser.add_argument("--no-balance", action="store_true",
                        help="keep the extra 'repos' trials (3 / 5 classes)")
    parser.add_argument("--name", default=None,
                        help="basename of the output files")
    args = parser.parse_args()

    models = args.models or (S3_MODELS if args.classes == 2 else list(MODELS))
    classes = config.CLASS_SETS[args.classes]
    balance = args.classes > 2 and not args.no_balance
    name = args.name or ("s3_baselines" if args.classes == 2 and models == S3_MODELS
                         else f"s4_{args.classes}classes")

    subjects = args.subjects or default_subjects()
    print(f"Subjects: {subjects}\nClasses: {classes}  (balanced: {balance})\n"
          f"Configs: {args.configs}  Models: {models}")

    rows, confusions = [], {}
    for s in subjects:
        try:
            ensure_epochs(s)
            for cfg in args.configs:
                d = load_epochs(s, cfg, classes=classes)
                rows_s, conf_s = evaluate_subject(d, models, balance=balance)
                for r in rows_s:
                    r["config"] = cfg
                    rows.append(r)
                    print(f"S{s:03d} {cfg} {r['model']:9s} "
                          f"acc={r['acc_mean']:.3f}±{r['acc_std']:.3f} "
                          f"kappa={r['kappa_mean']:.3f}  ({r['seconds']:.1f}s)")
                for m, cm in conf_s.items():
                    key = (m, cfg)
                    confusions[key] = confusions.get(key, 0) + cm
        except Exception as e:  # one broken subject must not kill the run
            print(f"S{s:03d}: FAILED ({e})")

    if not rows:
        raise SystemExit("No result produced.")

    df = pd.DataFrame(rows)
    summary = summarize(df)
    title = f"{args.classes} classes, intra-subject ({', '.join(classes)})"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = config.RESULTS_DIR / f"{name}.csv"
    md = config.RESULTS_DIR / f"{name}.md"
    fig = config.FIGURES_DIR / f"{name}.png"
    df.to_csv(csv, index=False, float_format="%.4f")
    md.write_text(to_markdown(summary, df, title=title), encoding="utf-8")
    plot_results(df, fig, title)
    outputs = [csv, md, fig]
    if args.classes > 2:
        cfig = config.FIGURES_DIR / f"{name}_confusion.png"
        # matrices are indexed by sorted label code -> same order for names
        plot_confusions(confusions, sorted(classes, key=config.EVENT_ID.get), cfig)
        outputs.append(cfig)

    print("\n" + summary.to_string(index=False, float_format="%.3f"))
    print("\n" + "\n".join(f"-> {p}" for p in outputs))


if __name__ == "__main__":
    main()
