"""
TCGA-BRCA DNA methylation analysis pipeline
============================================
Classifies breast samples as Primary Tumor vs. Solid Tissue Normal from
Illumina 450K DNA-methylation array data, and produces all figures and
tables needed for the project report.

Input files (gzipped, as provided on the course site):
  * tumor matrix   : rows = CpG sites, columns = samples, values = 0..999
  * normal matrix  : same format
  * CpG annotation : IlmnID  chr  position  gene name(s)  gene region(s)

Usage (example, chr19 files):
  python methylation_pipeline.py ^
      --tumor  BRCA_Primary_Tumor.chr19.tsv.gz ^
      --normal BRCA_Solid_Tissue_Normal.chr19.tsv.gz ^
      --annot  Illumina_450k.chr19.txt.gz ^
      --outdir results_chr19

All outputs (Fig1..Fig5 PNG files, CSV tables, summary.txt) are written
into the --outdir folder.
"""

import argparse
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # no display needed; we only save PNG files
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Figure style (light surface, colorblind-safe categorical palette)
# ----------------------------------------------------------------------
C_NORMAL = "#2a78d6"   # blue   - healthy (Solid Tissue Normal)
C_TUMOR = "#eb6834"    # orange - cancer  (Primary Tumor)
C_SERIES3 = "#1baf7a"  # aqua   - third model line in ROC plot
C_HYPER = "#d03b3b"    # red pole   - hyper-methylated in tumor
C_HYPO = "#2a78d6"     # blue pole  - hypo-methylated in tumor
C_MUTED = "#898781"    # axis / non-significant points
C_GRID = "#e1e0d9"
C_INK = "#0b0b0b"
SURFACE = "#fcfcfb"

# one-hue sequential blue ramp (light -> dark) for the heat-map
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": C_INK,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.8,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "text.color": C_INK,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RANDOM_STATE = 42


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def load_matrix(path, chunksize=25000):
    """Read a gzipped CpG x sample matrix in chunks to keep memory low.

    Returns a float32 DataFrame of beta values in [0, 1]
    (raw integers 0..999 are divided by 999).
    """
    log("Loading %s ..." % os.path.basename(path))
    chunks = []
    reader = pd.read_csv(path, sep="\t", index_col=0, compression="gzip",
                         chunksize=chunksize, na_values=["NA", "NaN", ""])
    for chunk in reader:
        chunks.append((chunk.astype("float32") / 999.0))
    df = pd.concat(chunks)
    df.index = df.index.astype(str)
    log("  -> %d CpGs x %d samples" % df.shape)
    return df


def load_annotation(path):
    """Read the CpG annotation file (tab- or whitespace-separated)."""
    log("Loading annotation %s ..." % os.path.basename(path))
    with gzip.open(path, "rt") as fh:
        first = fh.readline()
    if "\t" in first:
        annot = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    else:
        annot = pd.read_csv(path, sep=r"\s+", engine="python", index_col=0,
                            compression="gzip", on_bad_lines="skip")
    annot.index = annot.index.astype(str)
    log("  -> annotations for %d CpGs" % annot.shape[0])
    return annot


def align_and_clean(tumor, normal, max_missing_frac=0.2):
    """Align the two matrices on their common CpGs (order in the files is
    NOT assumed to be identical), drop CpGs with too many missing values,
    and impute the remaining missing values with the per-CpG mean."""
    common = tumor.index.intersection(normal.index)
    log("Common CpGs in both files: %d" % len(common))
    common = sorted(common)         
    tumor = tumor.loc[common]
    normal = normal.loc[common]

    # drop CpGs that are missing in too many samples 
    frac_na = np.maximum(tumor.isna().mean(axis=1).values,
                         normal.isna().mean(axis=1).values)
    keep = frac_na <= max_missing_frac
    log("Dropping %d CpGs with >%d%% missing values; keeping %d"
        % ((~keep).sum(), int(max_missing_frac * 100), keep.sum()))
    tumor, normal = tumor.loc[keep], normal.loc[keep]

    # impute what is left with the per-CpG mean of each group
    def impute_row_mean(df):
        vals = df.values
        if np.isnan(vals).any():
            row_means = np.nanmean(vals, axis=1)
            r, c = np.where(np.isnan(vals))
            vals[r, c] = row_means[r]
        return pd.DataFrame(vals.astype("float32"), index=df.index,
                            columns=df.columns)

    return impute_row_mean(tumor), impute_row_mean(normal)


# ----------------------------------------------------------------------
# 2. Exploratory figures
# ----------------------------------------------------------------------
def fig1_beta_distribution(tumor, normal, outdir):
    """Fig 1 - distribution of beta values in tumor vs normal samples."""
    rng = np.random.default_rng(RANDOM_STATE)

    def sample_values(df, n=2_000_000):
        v = df.values.ravel()
        if v.size > n:
            v = rng.choice(v, size=n, replace=False)
        return v

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 101)
    ax.hist(sample_values(normal), bins=bins, density=True, histtype="step",
            lw=2, color=C_NORMAL, label="Normal (n=%d)" % normal.shape[1])
    ax.hist(sample_values(tumor), bins=bins, density=True, histtype="step",
            lw=2, color=C_TUMOR, label="Tumor (n=%d)" % tumor.shape[1])
    ax.set_xlabel("Methylation level (beta value)")
    ax.set_ylabel("Density")
    ax.set_title("Fig. 1  Distribution of DNA methylation levels")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Fig1_beta_distribution.png"), dpi=300)
    plt.close(fig)
    log("Saved Fig1_beta_distribution.png")


def fig2_pca(X, y, outdir, n_top_var=5000):
    """Fig 2 - PCA of all samples using the most variable CpGs."""
    var = X.var(axis=0)
    idx = np.argsort(var)[::-1][:min(n_top_var, X.shape[1])]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X[:, idx] - X[:, idx].mean(axis=0))

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for label, color, name in [(0, C_NORMAL, "Normal"), (1, C_TUMOR, "Tumor")]:
        m = y == label
        ax.scatter(pcs[m, 0], pcs[m, 1], s=22, alpha=0.75, lw=0,
                   color=color, label=name)
    ax.set_xlabel("PC1 (%.1f%% of variance)" % (100 * pca.explained_variance_ratio_[0]))
    ax.set_ylabel("PC2 (%.1f%% of variance)" % (100 * pca.explained_variance_ratio_[1]))
    ax.set_title("Fig. 2  PCA of samples (top %d variable CpGs)" % len(idx))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Fig2_pca.png"), dpi=300)
    plt.close(fig)
    log("Saved Fig2_pca.png")


def differential_methylation(tumor, normal):
    """Per-CpG Welch t-test tumor vs normal + mean beta difference."""
    log("Computing per-CpG differential methylation (Welch t-test) ...")
    t_vals, p_vals = stats.ttest_ind(tumor.values.astype("float64"),
                                     normal.values.astype("float64"),
                                     axis=1, equal_var=False)
    
    p_vals = np.clip(p_vals.astype("float64"), 1e-300, None)
    diff = pd.DataFrame({
        "mean_tumor": tumor.values.mean(axis=1),
        "mean_normal": normal.values.mean(axis=1),
        "t_stat": t_vals,
        "p_value": p_vals,
    }, index=tumor.index)
    diff["delta_beta"] = diff["mean_tumor"] - diff["mean_normal"]
    # Bonferroni-corrected significance threshold
    diff["significant"] = (diff["p_value"] < 0.05 / len(diff)) & \
                          (diff["delta_beta"].abs() > 0.2)
    return diff


def fig3_volcano(diff, outdir):
    """Fig 3 - volcano plot of differential methylation."""
    logp = -np.log10(diff["p_value"].values)
    delta = diff["delta_beta"].values
    sig = diff["significant"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(delta[~sig], logp[~sig], s=4, color=C_MUTED, alpha=0.35, lw=0,
               label="Not significant")
    hyper = sig & (delta > 0)
    hypo = sig & (delta < 0)
    ax.scatter(delta[hyper], logp[hyper], s=6, color=C_HYPER, alpha=0.6, lw=0,
               label="Hyper-methylated in tumor (n=%d)" % hyper.sum())
    ax.scatter(delta[hypo], logp[hypo], s=6, color=C_HYPO, alpha=0.6, lw=0,
               label="Hypo-methylated in tumor (n=%d)" % hypo.sum())
    ax.axvline(0.2, color=C_GRID, lw=1, ls="--")
    ax.axvline(-0.2, color=C_GRID, lw=1, ls="--")
    ax.set_xlabel(u"Δβ  (mean tumor − mean normal)")
    ax.set_ylabel(u"−log₁₀ p-value (Welch t-test)")
    ax.set_title("Fig. 3  Differential methylation, tumor vs normal")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Fig3_volcano.png"), dpi=300)
    plt.close(fig)
    log("Saved Fig3_volcano.png")


# ----------------------------------------------------------------------
# 3. Classification with cross-validation
# ----------------------------------------------------------------------
def build_models(k_features):
    """Three model pipelines. Feature selection (top-k CpGs by ANOVA
    F-score) is INSIDE each pipeline, so in every cross-validation fold
    it is fitted on training samples only -> no information leakage."""
    select = ("select", SelectKBest(f_classif, k=k_features))
    return {
        "Logistic regression": Pipeline([
            select,
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=1.0,
                                       class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "Random forest": Pipeline([
            select,
            ("clf", RandomForestClassifier(n_estimators=500,
                                           class_weight="balanced",
                                           n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
        "k-NN (k=5)": Pipeline([
            select,
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


def evaluate_models(X, y, outdir, k_features, n_folds):
    """Stratified k-fold cross-validation of the three models.
    Returns a metrics table; also draws Fig 4 (ROC curves)."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                         random_state=RANDOM_STATE)
    models = build_models(k_features)
    rows, roc_data = [], []

    for name, model in models.items():
        log("Cross-validating: %s ..." % name)
        scores = cross_validate(model, X, y, cv=cv, n_jobs=1,
                                scoring=["accuracy", "balanced_accuracy",
                                         "roc_auc"])
        # pooled out-of-fold probabilities -> ROC curve + confusion matrix
        proba = cross_val_predict(model, X, y, cv=cv,
                                  method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        fpr, tpr, _ = roc_curve(y, proba)
        roc_data.append((name, fpr, tpr, roc_auc_score(y, proba)))
        rows.append({
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
            "balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "sensitivity(recall tumor)": tp / (tp + fn),
            "specificity(recall normal)": tn / (tn + fp),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        })

    metrics = pd.DataFrame(rows).set_index("model").round(4)
    metrics.to_csv(os.path.join(outdir, "model_metrics.csv"))
    log("Saved model_metrics.csv")

    # ---- Fig 4: ROC curves ----
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for (name, fpr, tpr, aucv), color in zip(
            roc_data, [C_NORMAL, C_TUMOR, C_SERIES3]):
        ax.plot(fpr, tpr, lw=2, color=color,
                label="%s (AUC = %.3f)" % (name, aucv))
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Fig. 4  ROC curves (%d-fold cross-validation)" % n_folds)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Fig4_roc_curves.png"), dpi=300)
    plt.close(fig)
    log("Saved Fig4_roc_curves.png")
    return metrics


# ----------------------------------------------------------------------
# 4. Top discriminative CpGs: annotated table + heat-map
# ----------------------------------------------------------------------
def top_cpgs_report(diff, annot, tumor, normal, outdir, n_table=100,
                    n_heatmap=20):
    """Rank CpGs by t-statistic magnitude, annotate them with gene names,
    save a table (top n_table) and draw Fig 5, a heat-map of the top
    n_heatmap CpGs across all samples."""
    ranked = diff.reindex(diff["t_stat"].abs()
                          .sort_values(ascending=False).index)
    top = ranked.head(n_table).copy()

    # attach annotation columns that exist in the annotation file
    ann_cols = [c for c in annot.columns]
    top = top.join(annot[ann_cols], how="left")
    top.index.name = "IlmnID"
    top.to_csv(os.path.join(outdir, "top_cpgs_annotated.csv"))
    log("Saved top_cpgs_annotated.csv (top %d CpGs)" % n_table)

    # ---- Fig 5: heat-map ----
    cpgs = ranked.head(n_heatmap).index
    mat = pd.concat([normal.loc[cpgs], tumor.loc[cpgs]], axis=1).values
    n_norm, n_tum = normal.shape[1], tumor.shape[1]

    # row labels: CpG id + gene 
    def row_label(cpg):
        gene = ""
        if cpg in annot.index:
            g = annot.loc[cpg]
            for col in annot.columns:
                if "Name" in col and pd.notna(g[col]):
                    gene = str(g[col]).split(";")[0]
                    break
        return "%s  (%s)" % (cpg, gene) if gene else cpg

    labels = [row_label(c) for c in cpgs]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mat, aspect="auto", cmap=CMAP_SEQ, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.axvline(n_norm - 0.5, color=SURFACE, lw=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks([n_norm / 2, n_norm + n_tum / 2])
    ax.set_xticklabels(["Normal (n=%d)" % n_norm, "Tumor (n=%d)" % n_tum])
    ax.grid(False)
    ax.set_title("Fig. 5  Methylation of the top %d discriminative CpGs"
                 % n_heatmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Beta value")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Fig5_heatmap_top_cpgs.png"), dpi=300)
    plt.close(fig)
    log("Saved Fig5_heatmap_top_cpgs.png")
    return top


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="TCGA-BRCA methylation tumor/normal classifier")
    ap.add_argument("--tumor", required=True,
                    help="BRCA_Primary_Tumor[.chr19].tsv.gz")
    ap.add_argument("--normal", required=True,
                    help="BRCA_Solid_Tissue_Normal[.chr19].tsv.gz")
    ap.add_argument("--annot", required=True,
                    help="Illumina_450k[.chr19].txt.gz")
    ap.add_argument("--outdir", default="results",
                    help="output folder (default: results)")
    ap.add_argument("--topk", type=int, default=500,
                    help="number of CpG features selected inside each CV "
                         "fold (default: 500)")
    ap.add_argument("--folds", type=int, default=5,
                    help="cross-validation folds (default: 5)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    # --- load & clean -------------------------------------------------
    tumor = load_matrix(args.tumor)
    normal = load_matrix(args.normal)
    annot = load_annotation(args.annot)
    tumor, normal = align_and_clean(tumor, normal)
    n_cpgs = tumor.shape[0]

    # --- design matrix: samples x CpGs, y: 1=tumor, 0=normal ----------
    X = np.vstack([normal.values.T, tumor.values.T]).astype("float32")
    y = np.concatenate([np.zeros(normal.shape[1], dtype=int),
                        np.ones(tumor.shape[1], dtype=int)])
    log("Design matrix: %d samples x %d CpGs (tumor=%d, normal=%d)"
        % (X.shape[0], X.shape[1], (y == 1).sum(), (y == 0).sum()))

    # --- figures & statistics ------------------------------------------
    fig1_beta_distribution(tumor, normal, args.outdir)
    fig2_pca(X, y, args.outdir)
    diff = differential_methylation(tumor, normal)
    fig3_volcano(diff, args.outdir)

    # --- classification -------------------------------------------------
    k = min(args.topk, n_cpgs)
    metrics = evaluate_models(X, y, args.outdir, k_features=k,
                              n_folds=args.folds)

    # --- top CpGs --------------------------------------------------------
    top = top_cpgs_report(diff, annot, tumor, normal, args.outdir)

    # --- text summary (send this file back for the report!) -------------
    with open(os.path.join(args.outdir, "summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("TCGA-BRCA methylation pipeline - run summary\n")
        fh.write("=" * 60 + "\n")
        fh.write("Tumor file : %s\n" % args.tumor)
        fh.write("Normal file: %s\n" % args.normal)
        fh.write("Samples    : %d tumor, %d normal\n"
                 % ((y == 1).sum(), (y == 0).sum()))
        fh.write("CpGs after QC: %d\n" % n_cpgs)
        fh.write("Significant CpGs (Bonferroni p<0.05 and |delta beta|>0.2): "
                 "%d\n" % int(diff["significant"].sum()))
        fh.write("Features per CV fold (top-k by ANOVA F): %d\n" % k)
        fh.write("Cross-validation: stratified %d-fold\n\n" % args.folds)
        fh.write("Model performance:\n")
        fh.write(metrics.to_string() + "\n\n")
        fh.write("Top 10 discriminative CpGs:\n")
        fh.write(top.head(10).to_string() + "\n")
    log("Saved summary.txt")
    log("Done in %.1f minutes. All outputs are in '%s'."
        % ((time.time() - t0) / 60, args.outdir))


if __name__ == "__main__":
    main()
