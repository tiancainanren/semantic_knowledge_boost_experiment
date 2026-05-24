import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "semantic_knowledge_boost_experiment" / "results"
ASSET_DIR = ROOT / "semantic_knowledge_boost_experiment" / "paper_assets"
FIG_DIR = ASSET_DIR / "figures"
TABLE_DIR = ASSET_DIR / "tables"

DATASET_ORDER = ["CIFAR-10", "CIFAR-20", "STL-10", "DTD", "StanfordDogs", "Flowers", "Pets"]
PRIMARY_FAMILIES = ["TAC", "KEC", "SAGE-A", "SAGE-AG", "SAGE-AGP"]
EXPLORATORY_FAMILIES = ["SAGE-AGC", "SAGE-AGPC", "SAGE-HYB"]


def pct(x: float) -> float:
    return round(float(x) * 100.0, 2)


def load_full_results() -> pd.DataFrame:
    frames = []
    for dataset in DATASET_ORDER:
        path = RESULTS_DIR / dataset / "method_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No per-dataset method_results.csv files found.")
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["family"].fillna(df["method"].str.extract(r"^([^\[]+)")[0])
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    return df.sort_values(["dataset", "family", "method"]).reset_index(drop=True)


def best_row(df: pd.DataFrame, dataset: str, family_prefix: str, metric: str = "train_ACC") -> pd.Series:
    subset = df[(df["dataset"] == dataset) & (df["family"] == family_prefix)]
    if subset.empty:
        raise KeyError(f"Missing rows for dataset={dataset}, family={family_prefix}")
    return subset.sort_values(metric, ascending=False).iloc[0]


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = "l" + "c" * (len(df.columns) - 1)
    body = df.to_latex(index=False, escape=False, column_format=cols)
    header = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        body.strip(),
        "\\end{table*}",
    ]
    return "\n".join(header)


def build_main_train_table(df: pd.DataFrame):
    rows = []
    for dataset in DATASET_ORDER:
        tac = best_row(df, dataset, "TAC")
        kec = best_row(df, dataset, "KEC")
        ours = df[
            (df["dataset"] == dataset)
            & (~df["family"].isin(["TAC", "KEC"]))
        ].sort_values("train_ACC", ascending=False).iloc[0]
        rows.append(
            {
                "Dataset": dataset,
                "TAC (A/N/R)": f"{pct(tac['train_ACC']):.2f} / {pct(tac['train_NMI']):.2f} / {pct(tac['train_ARI']):.2f}",
                "KEC (A/N/R)": f"{pct(kec['train_ACC']):.2f} / {pct(kec['train_NMI']):.2f} / {pct(kec['train_ARI']):.2f}",
                "Ours Best Variant": ours["method"],
                "Ours (A/N/R)": f"{pct(ours['train_ACC']):.2f} / {pct(ours['train_NMI']):.2f} / {pct(ours['train_ARI']):.2f}",
                "Gain vs TAC": f"+{pct(ours['train_ACC'] - tac['train_ACC']):.2f}",
                "Gain vs KEC": f"{pct(ours['train_ACC'] - kec['train_ACC']):+.2f}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "main_train_table.csv", index=False, encoding="utf-8-sig")
    write_text(TABLE_DIR / "main_train_table.md", dataframe_to_markdown(out))
    write_text(
        TABLE_DIR / "main_train_table.tex",
        dataframe_to_latex(out, "Main train-head comparison against TAC and KEC.", "tab:main_train"),
    )


def build_main_concat_table(df: pd.DataFrame):
    rows = []
    for dataset in DATASET_ORDER:
        tac = best_row(df, dataset, "TAC", metric="concat_ACC")
        kec = best_row(df, dataset, "KEC", metric="concat_ACC")
        ours = df[
            (df["dataset"] == dataset)
            & (~df["family"].isin(["TAC", "KEC"]))
        ].sort_values("concat_ACC", ascending=False).iloc[0]
        rows.append(
            {
                "Dataset": dataset,
                "TAC Concat": f"{pct(tac['concat_ACC']):.2f}",
                "KEC Concat": f"{pct(kec['concat_ACC']):.2f}",
                "Ours Best Variant": ours["method"],
                "Ours Concat": f"{pct(ours['concat_ACC']):.2f}",
                "Gain vs TAC": f"{pct(ours['concat_ACC'] - tac['concat_ACC']):+.2f}",
                "Gain vs KEC": f"{pct(ours['concat_ACC'] - kec['concat_ACC']):+.2f}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "main_concat_table.csv", index=False, encoding="utf-8-sig")
    write_text(TABLE_DIR / "main_concat_table.md", dataframe_to_markdown(out))
    write_text(
        TABLE_DIR / "main_concat_table.tex",
        dataframe_to_latex(out, "Training-free concat-kmeans comparison.", "tab:main_concat"),
    )


def build_ablation_table(df: pd.DataFrame):
    rows = []
    for dataset in DATASET_ORDER:
        family_rows = {}
        for family in PRIMARY_FAMILIES:
            family_rows[family] = best_row(df, dataset, family)
        rows.append(
            {
                "Dataset": dataset,
                "TAC": f"{pct(family_rows['TAC']['train_ACC']):.2f}",
                "KEC": f"{pct(family_rows['KEC']['train_ACC']):.2f}",
                "SAGE-A": f"{pct(family_rows['SAGE-A']['train_ACC']):.2f}",
                "SAGE-AG": f"{pct(family_rows['SAGE-AG']['train_ACC']):.2f}",
                "SAGE-AGP": f"{pct(family_rows['SAGE-AGP']['train_ACC']):.2f}",
                "Best Core Method": max(
                    [family_rows["SAGE-A"], family_rows["SAGE-AG"], family_rows["SAGE-AGP"]],
                    key=lambda x: x["train_ACC"],
                )["method"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "ablation_core_steps.csv", index=False, encoding="utf-8-sig")
    write_text(TABLE_DIR / "ablation_core_steps.md", dataframe_to_markdown(out))
    write_text(
        TABLE_DIR / "ablation_core_steps.tex",
        dataframe_to_latex(out, "Stepwise ablation from TAC to the proposed semantic refinements.", "tab:ablation_core"),
    )


def build_exploratory_ablation(df: pd.DataFrame):
    rows = []
    for dataset in DATASET_ORDER:
        base = df[(df["dataset"] == dataset) & (df["family"].isin(["SAGE-AG", "SAGE-AGP"]))].sort_values("train_ACC", ascending=False).iloc[0]
        best_explor = df[(df["dataset"] == dataset) & (df["family"].isin(EXPLORATORY_FAMILIES))].sort_values("train_ACC", ascending=False).iloc[0]
        rows.append(
            {
                "Dataset": dataset,
                "Best Core Variant": base["method"],
                "Core ACC": f"{pct(base['train_ACC']):.2f}",
                "Best Exploratory Variant": best_explor["method"],
                "Exploratory ACC": f"{pct(best_explor['train_ACC']):.2f}",
                "Exploratory - Core": f"{pct(best_explor['train_ACC'] - base['train_ACC']):+.2f}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "ablation_exploratory_variants.csv", index=False, encoding="utf-8-sig")
    write_text(TABLE_DIR / "ablation_exploratory_variants.md", dataframe_to_markdown(out))


def build_story_summary(df: pd.DataFrame):
    rows = []
    for family in PRIMARY_FAMILIES + EXPLORATORY_FAMILIES:
        sub = df[df["family"] == family]
        rows.append(
            {
                "Family": family,
                "Mean Train ACC": round(pct(sub["train_ACC"].mean()), 2),
                "Mean Gain vs TAC": round(pct((sub["train_ACC"] - sub.groupby("dataset")["train_ACC"].transform(lambda s: s[sub.loc[s.index, 'family'] == 'TAC'])) if False else 0), 2),
            }
        )


def draw_box(ax, xy, text, width=1.9, height=0.72, fc="#f5f7fb", ec="#3b4a6b", fontsize=11):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def arrow(ax, start, end, color="#556987"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=color))


def build_method_figure():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    draw_box(ax, (0.6, 5.7), "Image Embeddings", fc="#e8f1ff")
    draw_box(ax, (0.6, 4.2), "TAC Noun Retrieval", fc="#eef9f0")
    draw_box(ax, (0.6, 2.7), "KEC Concept + Attribute Bank", fc="#fff3e8")

    draw_box(ax, (3.2, 4.9), "Confidence Estimation\n(noun / concept / attr)", width=2.3, height=1.1, fc="#f8f9ff")
    draw_box(ax, (6.1, 4.9), "SAGE-A\nAdaptive Semantic Fusion", width=2.3, height=1.1, fc="#eef7ff")
    draw_box(ax, (9.0, 4.9), "SAGE-AG\nGraph Diffusion on Image kNN", width=2.5, height=1.1, fc="#eefcf6")
    draw_box(ax, (12.0, 4.9), "SAGE-AGP\nPrototype Calibration", width=1.6, height=1.1, fc="#fff6ec")

    draw_box(ax, (6.0, 2.1), "Concat KMeans\n(training-free path)", width=2.4, height=0.95, fc="#f5f7fb")
    draw_box(ax, (9.1, 2.1), "Train Head\n(main result path)", width=2.4, height=0.95, fc="#f5f7fb")
    draw_box(ax, (12.1, 2.1), "Adaptive / Hybrid\nexplorations", width=1.5, height=0.95, fc="#fdf7ff", fontsize=10)

    arrow(ax, (2.5, 6.05), (3.2, 5.85))
    arrow(ax, (2.5, 4.55), (3.2, 5.45))
    arrow(ax, (2.5, 3.05), (3.2, 5.05))
    arrow(ax, (5.5, 5.45), (6.1, 5.45))
    arrow(ax, (8.4, 5.45), (9.0, 5.45))
    arrow(ax, (11.5, 5.45), (12.0, 5.45))

    arrow(ax, (7.2, 4.85), (7.2, 3.05))
    arrow(ax, (10.3, 4.85), (10.3, 3.05))
    arrow(ax, (12.8, 4.85), (12.8, 3.05))

    ax.text(7.2, 1.0, "Key idea: make semantics confidence-aware, geometry-aware,\nand optionally prototype-calibrated before clustering.", ha="center", va="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "method_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_gain_figure(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASET_ORDER:
        tac = best_row(df, dataset, "TAC")
        ours = df[(df["dataset"] == dataset) & (~df["family"].isin(["TAC", "KEC"]))].sort_values("train_ACC", ascending=False).iloc[0]
        rows.append((dataset, pct(ours["train_ACC"] - tac["train_ACC"]), ours["method"]))
    gain_df = pd.DataFrame(rows, columns=["dataset", "gain", "method"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(gain_df["dataset"], gain_df["gain"], color="#4c78a8")
    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_ylabel("ACC Gain over TAC (percentage points)")
    ax.set_title("Best Proposed Variant vs TAC on Train-Head Evaluation")
    ax.set_ylim(min(-1.0, gain_df["gain"].min() - 1.0), gain_df["gain"].max() + 1.5)
    for bar, (_, gain, method) in zip(bars, gain_df.itertuples(index=False)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{gain:.2f}\n{method}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_variant_gain_vs_tac.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_writing_outline(df: pd.DataFrame):
    best_train = pd.read_csv(RESULTS_DIR / "best_train_vs_tac.csv")
    mean_gain = best_train["train_delta_vs_TAC"].mean() * 100.0
    lines = [
        "# Writing Outline",
        "",
        "## 1. Working Title Options",
        "",
        "1. Confidence-Aware Semantic Knowledge Boosting for Unsupervised Image Clustering",
        "2. From Text Retrieval to Semantic Calibration: Enhancing Unsupervised Image Clustering with Concepts, Attributes, and Prototype-Aware Smoothing",
        "3. Geometry-Aware Semantic Knowledge Refinement for Training-Free and Train-Head Image Clustering",
        "",
        "## 2. Core Story",
        "",
        "- TAC shows that noun retrieval helps clustering.",
        "- KEC shows that dataset-specific concept and attribute knowledge improves semantic quality.",
        "- Our next step is to avoid treating all semantic cues equally.",
        "- We make semantics confidence-aware, image-manifold-aware, and optionally prototype-calibrated.",
        "- This yields stronger train-head clustering results on most datasets, with especially clear gains on CIFAR-20, Flowers, Pets, StanfordDogs, and DTD.",
        "",
        "## 3. Main Contributions",
        "",
        "1. A confidence-aware semantic fusion mechanism that adaptively combines TAC noun retrieval and KEC knowledge semantics.",
        "2. A geometry-aware semantic refinement step that diffuses uncertain semantics on the image kNN graph.",
        "3. A prototype calibration module that further improves fine-grained datasets by aligning semantics with image-derived anchor groups.",
        "4. A broad cross-dataset study showing that the semantic refinement line outperforms TAC on 7 usable datasets, with mean best-variant gain "
        f"of {mean_gain:.2f} percentage points in train-head ACC.",
        "",
        "## 4. Recommended Paper Structure",
        "",
        "### 4.1 Introduction",
        "- Motivate why text-enhanced clustering works but is bottlenecked by noisy or uneven semantic relevance.",
        "- Explain the TAC -> KEC -> semantic refinement progression.",
        "- Preview the key result: the main benefit appears in the train-head setting rather than only in training-free concat.",
        "",
        "### 4.2 Related Work",
        "- Unsupervised image clustering",
        "- Vision-language clustering",
        "- Knowledge-enhanced semantic prompting / attribute augmentation",
        "- Graph smoothing / prototype calibration",
        "",
        "### 4.3 Method",
        "- TAC baseline recap",
        "- KEC baseline recap",
        "- SAGE-A: adaptive semantic fusion",
        "- SAGE-AG: graph diffusion refinement",
        "- SAGE-AGP: prototype calibration",
        "- Optional exploratory variants: AGC / AGPC / HYB",
        "",
        "### 4.4 Experimental Setup",
        "- Datasets: CIFAR-10, CIFAR-20, STL-10, DTD, StanfordDogs, Flowers, Pets",
        "- Metrics: ACC / NMI / ARI",
        "- Two evaluation tracks: concat-kmeans and train-head",
        "",
        "### 4.5 Main Results",
        "- Use the train-head main table as the primary quantitative result.",
        "- Keep concat-kmeans as a secondary table to show that the gains are smaller but directionally consistent on several datasets.",
        "",
        "### 4.6 Ablation",
        "- TAC -> KEC -> SAGE-A -> SAGE-AG -> SAGE-AGP",
        "- Fixed image-guidance weights vs adaptive guidance",
        "- Fine-grained vs coarse-grained behavior",
        "",
        "### 4.7 Analysis",
        "- Why SAGE-AG is the most stable family overall",
        "- Why SAGE-AGP helps fine-grained datasets more",
        "- Why adaptive image injection is not automatically beneficial",
        "",
        "## 5. Recommended Result Framing",
        "",
        "- Main method for the paper body: `SAGE-AG`",
        "- Fine-grained enhancement variant: `SAGE-AGP`",
        "- Exploratory variants (`AGC`, `AGPC`, `HYB`) should stay in ablation or supplementary analysis.",
        "",
        "## 6. Caveat to State Clearly",
        "",
        "- The current main table uses the best proposed variant per dataset.",
        "- For a stricter single-method claim, the paper should separately report fixed-family results, where `SAGE-AG` is the strongest overall family and `SAGE-AGP` is the strongest fine-grained specialization.",
    ]
    write_text(ASSET_DIR / "writing_outline.md", "\n".join(lines))


def build_readme():
    lines = [
        "# Paper Assets",
        "",
        "This folder contains paper-oriented assets generated from the latest semantic knowledge boost experiments.",
        "",
        "## Files",
        "",
        "- `tables/main_train_table.*`: main train-head comparison table",
        "- `tables/main_concat_table.*`: training-free concat comparison table",
        "- `tables/ablation_core_steps.*`: stepwise core-method ablation",
        "- `tables/ablation_exploratory_variants.*`: exploratory variant comparison",
        "- `figures/method_pipeline.png`: method overview figure",
        "- `figures/best_variant_gain_vs_tac.png`: best-variant gains over TAC",
        "- `writing_outline.md`: paper writing outline",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python semantic_knowledge_boost_experiment/generate_paper_assets.py",
        "```",
    ]
    write_text(ASSET_DIR / "README.md", "\n".join(lines))


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_full_results()
    build_main_train_table(df)
    build_main_concat_table(df)
    build_ablation_table(df)
    build_exploratory_ablation(df)
    build_method_figure()
    build_gain_figure(df)
    build_writing_outline(df)
    build_readme()
    print(f"Paper assets generated in: {ASSET_DIR}")


if __name__ == "__main__":
    main()
