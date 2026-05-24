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
    hybridize_semantics,
    prepare_semantic_context,
    preserve_semantic_core,
    run_concat_eval,
    run_train_head_eval,
    save_json,
)


RESULTS_DIR = ROOT / "semantic_knowledge_boost_experiment" / "second_wave_results"
BASE_RESULTS = ROOT / "semantic_knowledge_boost_experiment" / "results" / "aggregate_results.csv"
TARGETED_RESULTS = ROOT / "semantic_knowledge_boost_experiment" / "targeted_boost_results" / "targeted_boost_summary.csv"
SIMPLE_KEC_ROOT = ROOT / "hierarchical_textual_knowledge_experiment" / "results"
PROMPT_KEC_ROOT = ROOT / "hierarchical_textual_knowledge_experiment" / "results_prompted_local_llm_like"

DEFAULT_DATASETS = ["CIFAR-20", "DTD", "StanfordDogs", "Flowers", "Pets"]


def build_parser():
    parser = argparse.ArgumentParser(description="Second-wave semantic boosting with dual knowledge banks.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def get_existing_best_results():
    df = pd.read_csv(BASE_RESULTS)
    df["family"] = df["method"].str.extract(r"^([^\[]+)")
    targeted = pd.read_csv(TARGETED_RESULTS)
    targeted_map = {row["dataset"]: row for row in targeted.to_dict("records")}
    best = {}
    for dataset, group in df.groupby("dataset"):
        ours_best = group[~group["family"].isin(["TAC", "KEC"])].sort_values("train_ACC", ascending=False).iloc[0].to_dict()
        if dataset in targeted_map:
            ours_best = {
                "method": targeted_map[dataset]["targeted_config"],
                "train_ACC": float(targeted_map[dataset]["targeted_train_ACC"]),
                "concat_ACC": float(targeted_map[dataset]["targeted_concat_ACC"]),
            }
        best[dataset] = {
            "tac": group[group["family"] == "TAC"].sort_values("train_ACC", ascending=False).iloc[0].to_dict(),
            "kec": group[group["family"] == "KEC"].sort_values("train_ACC", ascending=False).iloc[0].to_dict(),
            "ours": ours_best,
        }
    return best


def dataset_recipe(dataset: str):
    recipes = {
        "CIFAR-20": {
            "base_cfg": {
                "family": "MOEP",
                "concept_caption_weight": 1.0,
                "concept_conf_alpha": 0.7,
                "graph_topk": 10,
                "smooth_min": 0.04,
                "smooth_max": 0.25,
                "anchor_multiplier": 5.0,
                "anchor_cap": 640,
                "proto_mix_max": 0.35,
                "proto_residual_rho": 0.10,
                "include_adaptive_expert": True,
                "moe_temp": 0.05,
                "preserve_min": 0.05,
                "preserve_scale": 0.15,
                "preserve_cap": 0.25,
                "image_guidance": 0.0,
            },
            "dual_temp": 0.05,
            "cross_guidance": 0.0,
            "preserve_weight": 0.08,
        },
        "DTD": {
            "base_cfg": {
                "family": "MOEP",
                "concept_caption_weight": 0.7,
                "graph_topk": 35,
                "smooth_min": 0.06,
                "smooth_max": 0.42,
                "certainty_power": 1.2,
                "anchor_multiplier": 5.0,
                "anchor_cap": 512,
                "proto_mix_max": 0.45,
                "proto_residual_rho": 0.12,
                "moe_temp": 0.06,
                "include_adaptive_expert": True,
                "image_guidance": 0.05,
            },
            "dual_temp": 0.06,
            "cross_guidance": 0.05,
            "preserve_weight": 0.12,
        },
        "StanfordDogs": {
            "base_cfg": {
                "family": "MOEP",
                "concept_caption_weight": 1.0,
                "graph_topk": 25,
                "smooth_min": 0.05,
                "smooth_max": 0.28,
                "anchor_multiplier": 6.0,
                "anchor_cap": 768,
                "proto_mix_max": 0.45,
                "proto_residual_rho": 0.12,
                "moe_temp": 0.045,
                "include_adaptive_expert": True,
                "preserve_min": 0.12,
                "preserve_scale": 0.22,
                "preserve_cap": 0.35,
                "image_guidance": 0.05,
            },
            "dual_temp": 0.045,
            "cross_guidance": 0.05,
            "preserve_weight": 0.18,
        },
        "Flowers": {
            "base_cfg": {
                "family": "MOEP",
                "concept_caption_weight": 0.95,
                "graph_topk": 15,
                "smooth_min": 0.04,
                "smooth_max": 0.30,
                "anchor_multiplier": 5.0,
                "anchor_cap": 640,
                "proto_mix_max": 0.40,
                "proto_residual_rho": 0.10,
                "moe_temp": 0.06,
                "include_adaptive_expert": True,
                "image_guidance": 0.65,
                "preserve_min": 0.08,
                "preserve_scale": 0.12,
                "preserve_cap": 0.22,
            },
            "dual_temp": 0.06,
            "cross_guidance": 0.65,
            "preserve_weight": 0.10,
        },
        "Pets": {
            "base_cfg": {
                "family": "MOEP",
                "concept_caption_weight": 1.0,
                "graph_topk": 30,
                "smooth_min": 0.05,
                "smooth_max": 0.28,
                "anchor_multiplier": 6.0,
                "anchor_cap": 768,
                "proto_mix_max": 0.45,
                "proto_residual_rho": 0.15,
                "moe_temp": 0.05,
                "include_adaptive_expert": True,
                "image_guidance": 0.2,
                "preserve_min": 0.10,
                "preserve_scale": 0.15,
                "preserve_cap": 0.25,
            },
            "dual_temp": 0.05,
            "cross_guidance": 0.2,
            "preserve_weight": 0.12,
        },
    }
    return recipes[dataset]


def run_candidates(dataset: str, seed: int, existing_best: dict):
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    model = kec.load_clip_model()
    images_train, labels_train, images_test, labels_test = kec.ensure_image_embeddings(dataset, model, force=False)
    cluster_num = kec.get_cluster_num_from_labels(labels_train)

    recipe = dataset_recipe(dataset)
    simple_ctx = prepare_semantic_context(dataset, images_train, images_test, model=model, knowledge_root=SIMPLE_KEC_ROOT)
    prompt_ctx = prepare_semantic_context(dataset, images_train, images_test, model=model, knowledge_root=PROMPT_KEC_ROOT)

    base_cfg = recipe["base_cfg"]
    agp_cfg = dict(base_cfg)
    agp_cfg["family"] = "AGP"

    simple_moep_train, simple_moep_test, simple_meta = build_custom_semantic_variant(simple_ctx, base_cfg, seed=seed)
    prompt_moep_train, prompt_moep_test, prompt_meta = build_custom_semantic_variant(prompt_ctx, base_cfg, seed=seed)
    simple_agp_train, simple_agp_test, simple_agp_meta = build_custom_semantic_variant(simple_ctx, agp_cfg, seed=seed)
    prompt_agp_train, prompt_agp_test, prompt_agp_meta = build_custom_semantic_variant(prompt_ctx, agp_cfg, seed=seed)

    dual_moep_train, dual_moep_alpha_train = hybridize_semantics(images_train, simple_moep_train, prompt_moep_train, temperature=recipe["dual_temp"])
    dual_moep_test, dual_moep_alpha_test = hybridize_semantics(images_test, simple_moep_test, prompt_moep_test, temperature=recipe["dual_temp"])

    dual_agp_train, dual_agp_alpha_train = hybridize_semantics(images_train, simple_agp_train, prompt_agp_train, temperature=recipe["dual_temp"])
    dual_agp_test, dual_agp_alpha_test = hybridize_semantics(images_test, simple_agp_test, prompt_agp_test, temperature=recipe["dual_temp"])

    cross_train, cross_alpha_train = hybridize_semantics(images_train, dual_moep_train, dual_agp_train, temperature=recipe["dual_temp"])
    cross_test, cross_alpha_test = hybridize_semantics(images_test, dual_moep_test, dual_agp_test, temperature=recipe["dual_temp"])

    preserve_train = preserve_semantic_core(cross_train, prompt_ctx["caches"]["knowledge_train"], recipe["preserve_weight"])
    preserve_test = preserve_semantic_core(cross_test, prompt_ctx["caches"]["knowledge_test"], recipe["preserve_weight"])

    candidates = [
        {
            "name": "prompt_moep",
            "semantic_train": prompt_moep_train,
            "semantic_test": prompt_moep_test,
            "image_guidance": base_cfg["image_guidance"],
            "meta": {"source": "prompt_root", "variant": "MOEP", "base_meta": prompt_meta},
        },
        {
            "name": "dual_moep",
            "semantic_train": dual_moep_train,
            "semantic_test": dual_moep_test,
            "image_guidance": base_cfg["image_guidance"],
            "meta": {
                "source": "simple+prompt",
                "variant": "dual_moep_hybrid",
                "mean_alpha_train_prompt": float(dual_moep_alpha_train.mean()),
                "mean_alpha_test_prompt": float(dual_moep_alpha_test.mean()),
            },
        },
        {
            "name": "dual_agp",
            "semantic_train": dual_agp_train,
            "semantic_test": dual_agp_test,
            "image_guidance": base_cfg["image_guidance"],
            "meta": {
                "source": "simple+prompt",
                "variant": "dual_agp_hybrid",
                "mean_alpha_train_prompt": float(dual_agp_alpha_train.mean()),
                "mean_alpha_test_prompt": float(dual_agp_alpha_test.mean()),
            },
        },
        {
            "name": "dual_cross_preserve",
            "semantic_train": preserve_train,
            "semantic_test": preserve_test,
            "image_guidance": recipe["cross_guidance"],
            "meta": {
                "source": "simple+prompt",
                "variant": "cross_hybrid_plus_prompt_preserve",
                "mean_cross_alpha_train_prompt_side": float(cross_alpha_train.mean()),
                "mean_cross_alpha_test_prompt_side": float(cross_alpha_test.mean()),
                "preserve_weight": float(recipe["preserve_weight"]),
            },
        },
    ]

    rows = []
    all_preds = {}
    for cand in candidates:
        concat_metrics, concat_preds, _ = run_concat_eval(images_test, cand["semantic_test"], labels_test, cluster_num)
        train_metrics, train_preds, _ = run_train_head_eval(
            dataset=dataset,
            semantic_train=cand["semantic_train"],
            semantic_test=cand["semantic_test"],
            images_train=images_train,
            images_test=images_test,
            labels_test=labels_test,
            cluster_num=cluster_num,
            seed=seed,
            image_guidance_weight=cand["image_guidance"],
        )
        all_preds[f"{cand['name']}_concat"] = concat_preds
        all_preds[f"{cand['name']}_train"] = train_preds
        rows.append(
            {
                "dataset": dataset,
                "config_name": cand["name"],
                "concat_ACC": float(concat_metrics["ACC"]),
                "concat_NMI": float(concat_metrics["NMI"]),
                "concat_ARI": float(concat_metrics["ARI"]),
                "train_ACC": float(train_metrics["ACC"]),
                "train_NMI": float(train_metrics["NMI"]),
                "train_ARI": float(train_metrics["ARI"]),
                "image_guidance": float(cand["image_guidance"]),
                "meta": json.dumps(cand["meta"], ensure_ascii=False),
            }
        )
        print(
            f"[SECOND-WAVE] dataset={dataset} config={cand['name']} "
            f"concat_ACC={concat_metrics['ACC']:.4f} train_ACC={train_metrics['ACC']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values("train_ACC", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "second_wave_results.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out_dir / "second_wave_predictions.npz", **all_preds)

    best_new = df.iloc[0].to_dict()
    prev_best = existing_best[dataset]["ours"]
    tac = existing_best[dataset]["tac"]
    kec_best = existing_best[dataset]["kec"]
    summary = {
        "dataset": dataset,
        "best_second_wave": best_new,
        "previous_best": prev_best,
        "tac_baseline": tac,
        "kec_reproduction": kec_best,
        "gain_vs_previous_best_train_acc": float(best_new["train_ACC"] - prev_best["train_ACC"]),
        "gain_vs_tac_train_acc": float(best_new["train_ACC"] - tac["train_ACC"]),
        "gain_vs_kec_train_acc": float(best_new["train_ACC"] - kec_best["train_ACC"]),
    }
    save_json(out_dir / "second_wave_summary.json", summary)
    torch.cuda.empty_cache()
    return summary


def main():
    args = build_parser().parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing_best = get_existing_best_results()

    rows = []
    failures = []
    for dataset in args.datasets:
        print(f">>> Second-wave boosting for {dataset}")
        try:
            summary = run_candidates(dataset, args.seed, existing_best)
        except Exception as exc:
            failures.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[SECOND-WAVE][WARN] dataset={dataset} failed: {type(exc).__name__}: {exc}")
            continue
        rows.append(
            {
                "dataset": dataset,
                "best_config": summary["best_second_wave"]["config_name"],
                "best_train_ACC": summary["best_second_wave"]["train_ACC"],
                "best_concat_ACC": summary["best_second_wave"]["concat_ACC"],
                "previous_best_method": summary["previous_best"]["method"],
                "previous_best_train_ACC": summary["previous_best"]["train_ACC"],
                "gain_vs_previous_best": summary["gain_vs_previous_best_train_acc"],
                "gain_vs_tac": summary["gain_vs_tac_train_acc"],
                "gain_vs_kec": summary["gain_vs_kec_train_acc"],
            }
        )
    if rows:
        summary_df = pd.DataFrame(rows).sort_values("dataset")
    else:
        summary_df = pd.DataFrame(
            columns=[
                "dataset",
                "best_config",
                "best_train_ACC",
                "best_concat_ACC",
                "previous_best_method",
                "previous_best_train_ACC",
                "gain_vs_previous_best",
                "gain_vs_tac",
                "gain_vs_kec",
            ]
        )
    summary_df.to_csv(RESULTS_DIR / "second_wave_summary.csv", index=False, encoding="utf-8-sig")
    save_json(RESULTS_DIR / "failures.json", failures)

    lines = ["# Second-Wave Dual-Knowledge Boost Summary", ""]
    for row in summary_df.to_dict("records"):
        lines.append(
            f"- {row['dataset']}: best `{row['best_config']}` train ACC `{row['best_train_ACC']:.4f}`, "
            f"vs previous best `{row['previous_best_train_ACC']:.4f}` ({row['previous_best_method']}), "
            f"delta `{row['gain_vs_previous_best']:+.4f}`"
        )
    if failures:
        lines.extend(["", "## Failed Datasets", ""])
        for item in failures:
            lines.append(f"- {item['dataset']}: {item['error']}")
    (RESULTS_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved second-wave outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
