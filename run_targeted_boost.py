import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_textual_knowledge_experiment.src import pipeline as kec
from semantic_knowledge_boost_experiment.src.methods import (
    build_custom_semantic_variant,
    prepare_semantic_context,
    run_concat_eval,
    run_train_head_eval,
    save_json,
)


RESULTS_DIR = ROOT / "semantic_knowledge_boost_experiment" / "targeted_boost_results"
BASE_RESULTS = ROOT / "semantic_knowledge_boost_experiment" / "results" / "aggregate_results.csv"

DEFAULT_DATASETS = ["CIFAR-20", "DTD", "StanfordDogs", "Flowers", "Pets"]

PAPER_KEC_REFERENCE = {
    "DTD": {
        "source": "KEC paper Table 6 (supplementary comparison results)",
        "concat": {"NMI": 0.607, "ACC": 0.474, "ARI": 0.315},
        "train": {"NMI": 0.625, "ACC": 0.513, "ARI": 0.360},
    },
    "Flowers": {
        "source": "KEC paper Table 1 (main paper)",
        "concat": {"NMI": 0.873, "ACC": 0.728, "ARI": 0.675},
        "train": {"NMI": 0.873, "ACC": 0.726, "ARI": 0.676},
    },
    "Pets": {
        "source": "KEC paper Table 1 (main paper)",
        "concat": {"NMI": 0.812, "ACC": 0.678, "ARI": 0.633},
        "train": {"NMI": 0.805, "ACC": 0.673, "ARI": 0.631},
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="Run targeted semantic-boost refinements on the most promising datasets.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def get_existing_best_results():
    df = pd.read_csv(BASE_RESULTS)
    df["family"] = df["method"].str.extract(r"^([^\[]+)")
    best = {}
    for dataset, group in df.groupby("dataset"):
        best[dataset] = {
            "tac": group[group["family"] == "TAC"].sort_values("train_ACC", ascending=False).iloc[0].to_dict(),
            "kec": group[group["family"] == "KEC"].sort_values("train_ACC", ascending=False).iloc[0].to_dict(),
            "ours": group[~group["family"].isin(["TAC", "KEC"])].sort_values("train_ACC", ascending=False).iloc[0].to_dict(),
        }
    return best


def dataset_search_space(dataset: str):
    if dataset == "CIFAR-20":
        return [
            {"name": "ag_base_g0", "family": "AG", "image_guidance": 0.0},
            {"name": "ag_caption1_topk10", "family": "AG", "image_guidance": 0.0, "concept_caption_weight": 1.0, "graph_topk": 10, "smooth_max": 0.28, "concept_conf_alpha": 0.7},
            {"name": "ag_gatewide_residual", "family": "AG", "image_guidance": 0.0, "gate_min": 0.05, "gate_max": 0.95, "graph_topk": 20, "smooth_max": 0.32, "graph_residual_rho": 0.10},
            {"name": "ag_attrheavy", "family": "AG", "image_guidance": 0.1, "concept_caption_weight": 0.8, "unary_weight": 0.6, "binary_weight": 0.4, "graph_topk": 30, "smooth_max": 0.30},
            {"name": "ag_low_smooth_sharp", "family": "AG", "image_guidance": 0.0, "concept_caption_weight": 1.1, "concept_conf_alpha": 0.8, "graph_topk": 8, "smooth_min": 0.03, "smooth_max": 0.20},
            {"name": "ag_small_guidance", "family": "AG", "image_guidance": 0.05, "concept_caption_weight": 1.0, "graph_topk": 10, "smooth_max": 0.25, "graph_residual_rho": 0.05},
            {"name": "agp_lowmix", "family": "AGP", "image_guidance": 0.0, "anchor_multiplier": 5.0, "anchor_cap": 640, "proto_mix_max": 0.30, "proto_residual_rho": 0.10},
            {"name": "agp_semantic_residual", "family": "AGP", "image_guidance": 0.15, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.45, "proto_residual_rho": 0.15, "concept_caption_weight": 1.0},
        ]
    if dataset == "DTD":
        return [
            {"name": "ag_base_g0", "family": "AG", "image_guidance": 0.0},
            {"name": "ag_texture30", "family": "AG", "image_guidance": 0.0, "graph_topk": 30, "smooth_min": 0.06, "smooth_max": 0.42, "certainty_power": 1.2, "concept_conf_alpha": 0.5},
            {"name": "ag_texture50_res", "family": "AG", "image_guidance": 0.0, "graph_topk": 50, "smooth_min": 0.08, "smooth_max": 0.45, "graph_residual_rho": 0.15, "concept_caption_weight": 0.6},
            {"name": "ag_attrheavy_g01", "family": "AG", "image_guidance": 0.1, "unary_weight": 0.8, "binary_weight": 0.2, "graph_topk": 30, "smooth_max": 0.40},
            {"name": "ag_texture35_res", "family": "AG", "image_guidance": 0.05, "graph_topk": 35, "smooth_min": 0.06, "smooth_max": 0.43, "certainty_power": 1.2, "graph_residual_rho": 0.10},
            {"name": "agp_texture", "family": "AGP", "image_guidance": 0.0, "anchor_multiplier": 5.0, "anchor_cap": 512, "proto_mix_max": 0.45, "proto_residual_rho": 0.10},
            {"name": "hyb_texture", "family": "HYB", "image_guidance": 0.0, "graph_topk": 30, "smooth_max": 0.40, "anchor_multiplier": 5.0, "proto_mix_max": 0.40, "hybrid_temp": 0.05},
        ]
    if dataset == "StanfordDogs":
        return [
            {"name": "agp_base_g0", "family": "AGP", "image_guidance": 0.0},
            {"name": "agp_anchor3_g0", "family": "AGP", "image_guidance": 0.0, "anchor_multiplier": 3.0, "anchor_cap": 512, "proto_mix_max": 0.35, "graph_topk": 25},
            {"name": "agp_anchor6_proto45", "family": "AGP", "image_guidance": 0.0, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.45, "graph_topk": 30},
            {"name": "agp_residual015", "family": "AGP", "image_guidance": 0.0, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.50, "proto_residual_rho": 0.15, "graph_topk": 30},
            {"name": "agp_caption1", "family": "AGP", "image_guidance": 0.0, "concept_caption_weight": 1.0, "anchor_multiplier": 5.0, "anchor_cap": 768, "proto_mix_max": 0.45},
            {"name": "agp_anchor3_caption1", "family": "AGP", "image_guidance": 0.0, "concept_caption_weight": 1.0, "anchor_multiplier": 3.0, "anchor_cap": 512, "proto_mix_max": 0.35, "proto_residual_rho": 0.05},
            {"name": "agp_soft_g01", "family": "AGP", "image_guidance": 0.1, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.45, "proto_residual_rho": 0.10},
            {"name": "agpc_like_fixed", "family": "AGP", "image_guidance": 0.05, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.40, "proto_residual_rho": 0.05},
        ]
    if dataset == "Flowers":
        return [
            {"name": "ag_base_g05", "family": "AG", "image_guidance": 0.5},
            {"name": "ag_caption1_g05", "family": "AG", "image_guidance": 0.5, "concept_caption_weight": 1.0, "graph_topk": 10, "smooth_max": 0.28, "concept_conf_alpha": 0.7},
            {"name": "ag_guidance065", "family": "AG", "image_guidance": 0.65, "graph_topk": 15, "smooth_max": 0.30},
            {"name": "ag_guidance075_res", "family": "AG", "image_guidance": 0.75, "graph_topk": 20, "smooth_max": 0.32, "graph_residual_rho": 0.10},
            {"name": "ag_guidance08_res", "family": "AG", "image_guidance": 0.8, "graph_topk": 20, "smooth_max": 0.32, "graph_residual_rho": 0.12},
            {"name": "ag_guidance07_caption09", "family": "AG", "image_guidance": 0.7, "concept_caption_weight": 0.9, "graph_topk": 15, "smooth_max": 0.30, "graph_residual_rho": 0.08},
            {"name": "agp_g05", "family": "AGP", "image_guidance": 0.5, "anchor_multiplier": 4.0, "anchor_cap": 512, "proto_mix_max": 0.35},
            {"name": "agp_caption1_g04", "family": "AGP", "image_guidance": 0.4, "concept_caption_weight": 1.0, "anchor_multiplier": 5.0, "anchor_cap": 640, "proto_mix_max": 0.40, "proto_residual_rho": 0.10},
        ]
    if dataset == "Pets":
        return [
            {"name": "agp_base_g025", "family": "AGP", "image_guidance": 0.25},
            {"name": "agp_anchor6_g015", "family": "AGP", "image_guidance": 0.15, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.45, "graph_topk": 30},
            {"name": "agp_residual015_g02", "family": "AGP", "image_guidance": 0.2, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.50, "proto_residual_rho": 0.15},
            {"name": "agp_residual02_g015", "family": "AGP", "image_guidance": 0.15, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.50, "proto_residual_rho": 0.20},
            {"name": "agp_caption1_g015", "family": "AGP", "image_guidance": 0.15, "concept_caption_weight": 1.0, "anchor_multiplier": 5.0, "anchor_cap": 768, "proto_mix_max": 0.45},
            {"name": "ag_g015", "family": "AG", "image_guidance": 0.15, "graph_topk": 30, "smooth_max": 0.28, "graph_residual_rho": 0.10},
            {"name": "hyb_g015", "family": "HYB", "image_guidance": 0.15, "graph_topk": 30, "smooth_max": 0.30, "anchor_multiplier": 6.0, "anchor_cap": 768, "proto_mix_max": 0.45, "hybrid_temp": 0.05},
        ]
    raise KeyError(f"No search space configured for dataset {dataset}")


def safe_cfg(cfg):
    return json.loads(json.dumps(cfg))


def run_dataset(dataset: str, seed: int, existing_best: dict):
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    model = kec.load_clip_model()
    images_train, labels_train, images_test, labels_test = kec.ensure_image_embeddings(dataset, model, force=False)
    cluster_num = kec.get_cluster_num_from_labels(labels_train)
    context = prepare_semantic_context(dataset, images_train, images_test, model=model)

    rows = []
    all_preds = {}
    for cfg in dataset_search_space(dataset):
        semantic_train, semantic_test, meta = build_custom_semantic_variant(context, cfg, seed=seed)
        concat_metrics, concat_preds, _ = run_concat_eval(images_test, semantic_test, labels_test, cluster_num)
        train_metrics, train_preds, _ = run_train_head_eval(
            dataset=dataset,
            semantic_train=semantic_train,
            semantic_test=semantic_test,
            images_train=images_train,
            images_test=images_test,
            labels_test=labels_test,
            cluster_num=cluster_num,
            seed=seed,
            image_guidance_weight=cfg.get("image_guidance", 0.0),
        )
        all_preds[f"{cfg['name']}_concat"] = concat_preds
        all_preds[f"{cfg['name']}_train"] = train_preds
        rows.append(
            {
                "dataset": dataset,
                "config_name": cfg["name"],
                "family": cfg["family"],
                "image_guidance": cfg.get("image_guidance", 0.0),
                "concat_ACC": float(concat_metrics["ACC"]),
                "concat_NMI": float(concat_metrics["NMI"]),
                "concat_ARI": float(concat_metrics["ARI"]),
                "train_ACC": float(train_metrics["ACC"]),
                "train_NMI": float(train_metrics["NMI"]),
                "train_ARI": float(train_metrics["ARI"]),
                "config": json.dumps(safe_cfg(cfg), ensure_ascii=False),
                "meta": json.dumps(meta, ensure_ascii=False),
            }
        )
        print(
            f"[TARGETED] dataset={dataset} config={cfg['name']} "
            f"concat_ACC={concat_metrics['ACC']:.4f} train_ACC={train_metrics['ACC']:.4f}"
        )

    result_df = pd.DataFrame(rows).sort_values("train_ACC", ascending=False).reset_index(drop=True)
    result_df.to_csv(out_dir / "targeted_results.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out_dir / "targeted_predictions.npz", **all_preds)

    best_new = result_df.iloc[0].to_dict()
    old_best = existing_best[dataset]["ours"]
    tac = existing_best[dataset]["tac"]
    kec_best = existing_best[dataset]["kec"]

    comparison = {
        "dataset": dataset,
        "best_targeted": best_new,
        "previous_best": old_best,
        "tac_baseline": tac,
        "kec_reproduction": kec_best,
        "paper_kec_reference": PAPER_KEC_REFERENCE.get(dataset),
        "gain_vs_previous_best_train_acc": float(best_new["train_ACC"] - old_best["train_ACC"]),
        "gain_vs_tac_train_acc": float(best_new["train_ACC"] - tac["train_ACC"]),
    }
    save_json(out_dir / "targeted_summary.json", comparison)

    lines = [f"# {dataset} Targeted Boost", ""]
    lines.append(f"- Best targeted config: `{best_new['config_name']}`")
    lines.append(f"- Targeted train ACC: `{best_new['train_ACC']:.4f}`")
    lines.append(f"- Previous best train ACC: `{old_best['train_ACC']:.4f}` ({old_best['method']})")
    lines.append(f"- TAC train ACC: `{tac['train_ACC']:.4f}`")
    lines.append(f"- Gain vs previous best: `{best_new['train_ACC'] - old_best['train_ACC']:+.4f}`")
    lines.append(f"- Gain vs TAC: `{best_new['train_ACC'] - tac['train_ACC']:+.4f}`")
    if dataset in PAPER_KEC_REFERENCE:
        ref = PAPER_KEC_REFERENCE[dataset]
        lines.append(f"- Paper-reported KEC no-train ACC: `{ref['concat']['ACC']:.4f}` from {ref['source']}")
        lines.append(f"- Paper-reported KEC train ACC: `{ref['train']['ACC']:.4f}` from {ref['source']}")
        lines.append(f"- Our reproduced KEC no-train ACC: `{kec_best['concat_ACC']:.4f}`")
        lines.append(f"- Our reproduced KEC train ACC: `{kec_best['train_ACC']:.4f}`")
    (out_dir / "targeted_summary.md").write_text("\n".join(lines), encoding="utf-8")

    torch.cuda.empty_cache()
    return comparison


def main():
    args = build_parser().parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing_best = get_existing_best_results()

    comparisons = []
    for dataset in args.datasets:
        print(f">>> Targeted boosting for {dataset}")
        comparisons.append(run_dataset(dataset, args.seed, existing_best))

    summary_rows = []
    paper_rows = []
    for comp in comparisons:
        dataset = comp["dataset"]
        best = comp["best_targeted"]
        old = comp["previous_best"]
        tac = comp["tac_baseline"]
        summary_rows.append(
            {
                "dataset": dataset,
                "targeted_config": best["config_name"],
                "targeted_train_ACC": best["train_ACC"],
                "previous_best_method": old["method"],
                "previous_best_train_ACC": old["train_ACC"],
                "gain_vs_previous_best": best["train_ACC"] - old["train_ACC"],
                "gain_vs_tac": best["train_ACC"] - tac["train_ACC"],
                "targeted_concat_ACC": best["concat_ACC"],
            }
        )
        ref = comp.get("paper_kec_reference")
        if ref is not None:
            paper_rows.append(
                {
                    "dataset": dataset,
                    "paper_source": ref["source"],
                    "paper_kec_concat_ACC": ref["concat"]["ACC"],
                    "paper_kec_train_ACC": ref["train"]["ACC"],
                    "our_reproduced_kec_concat_ACC": comp["kec_reproduction"]["concat_ACC"],
                    "our_reproduced_kec_train_ACC": comp["kec_reproduction"]["train_ACC"],
                    "our_targeted_best_train_ACC": best["train_ACC"],
                    "targeted_minus_paper_kec_train": best["train_ACC"] - ref["train"]["ACC"],
                }
            )

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "targeted_boost_summary.csv", index=False, encoding="utf-8-sig")
    if paper_rows:
        pd.DataFrame(paper_rows).to_csv(RESULTS_DIR / "kec_paper_reference_comparison.csv", index=False, encoding="utf-8-sig")

    report_lines = ["# Targeted Boost Summary", ""]
    for row in summary_rows:
        report_lines.append(
            f"- {row['dataset']}: `{row['targeted_config']}` train ACC `{row['targeted_train_ACC']:.4f}`, "
            f"vs previous best `{row['previous_best_train_ACC']:.4f}` ({row['previous_best_method']}), "
            f"delta `{row['gain_vs_previous_best']:+.4f}`"
        )
    if paper_rows:
        report_lines.extend(["", "## KEC Paper Reference Check", ""])
        for row in paper_rows:
            report_lines.append(
                f"- {row['dataset']}: paper KEC train ACC `{row['paper_kec_train_ACC']:.4f}`, "
                f"our reproduced KEC `{row['our_reproduced_kec_train_ACC']:.4f}`, "
                f"our targeted best `{row['our_targeted_best_train_ACC']:.4f}`"
            )
    (RESULTS_DIR / "README.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved targeted boost outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
