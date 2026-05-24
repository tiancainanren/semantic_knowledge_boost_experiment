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
    build_semantic_methods,
    run_concat_eval,
    run_train_head_eval,
    save_json,
)


RESULTS_DIR = ROOT / "semantic_knowledge_boost_experiment" / "results"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run semantic knowledge boosting experiments on usable datasets."
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def get_tac_and_kec_baselines(dataset: str):
    base_dir = ROOT / "hierarchical_textual_knowledge_experiment" / "results" / dataset
    concat_metrics = json.load(open(base_dir / "metrics_concat_kmeans.json", "r", encoding="utf-8"))
    train_metrics = json.load(open(base_dir / "metrics_train_head.json", "r", encoding="utf-8"))
    return {
        "concat": {
            "TAC": concat_metrics["tac"],
            "KEC": concat_metrics["proposed"],
        },
        "train": {
            "TAC": train_metrics["tac"],
            "KEC": train_metrics["proposed"],
        },
    }


def _json_safe_meta(meta: dict):
    safe = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            safe[k] = {
                "type": "ndarray",
                "shape": list(v.shape),
                "mean": float(v.mean()),
                "std": float(v.std()),
            }
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                safe_list = []
                for item in v:
                    safe_item = {}
                    for ik, iv in item.items():
                        if isinstance(iv, np.ndarray):
                            safe_item[ik] = {
                                "type": "ndarray",
                                "shape": list(iv.shape),
                                "mean": float(iv.mean()),
                                "std": float(iv.std()),
                            }
                        else:
                            safe_item[ik] = iv
                    safe_list.append(safe_item)
                safe[k] = safe_list
            else:
                safe[k] = v
        else:
            safe[k] = v
    return safe


def run_dataset(dataset: str, seed: int):
    dataset_dir = RESULTS_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    model = kec.load_clip_model()
    images_train, labels_train, images_test, labels_test = kec.ensure_image_embeddings(dataset, model, force=False)
    cluster_num = kec.get_cluster_num_from_labels(labels_train)
    baselines = get_tac_and_kec_baselines(dataset)

    methods, extra = build_semantic_methods(
        dataset=dataset,
        images_train=images_train,
        images_test=images_test,
        seed=seed,
    )

    all_rows = []
    for method_name, payload in methods.items():
        concat_metrics, concat_preds, _ = run_concat_eval(
            images_test=images_test,
            semantic_test=payload["test"],
            labels_test=labels_test,
            cluster_num=cluster_num,
        )
        np.save(dataset_dir / f"{method_name}_concat_preds.npy", concat_preds)
        train_variants = payload["meta"].get("train_variants")
        if train_variants is None:
            guidance_options = payload["meta"].get("train_image_guidance_options", [0.5])
            train_variants = [
                {
                    "variant_suffix": f"g={guidance_weight}",
                    "image_guidance_train": float(guidance_weight),
                    "image_guidance_test": float(guidance_weight),
                }
                for guidance_weight in guidance_options
            ]

        variant_payloads = []
        for variant in train_variants:
            variant_name = f"{method_name}[{variant['variant_suffix']}]"
            train_metrics, train_preds, train_logits = run_train_head_eval(
                dataset=dataset,
                semantic_train=payload["train"],
                semantic_test=payload["test"],
                images_train=images_train,
                images_test=images_test,
                labels_test=labels_test,
                cluster_num=cluster_num,
                seed=seed,
                image_guidance_weight=variant["image_guidance_train"],
                image_guidance_weight_test=variant["image_guidance_test"],
            )
            np.save(dataset_dir / f"{variant_name}_train_preds.npy", train_preds)
            np.save(dataset_dir / f"{variant_name}_train_logits.npy", train_logits)
            variant_payloads.append(
                {
                    "variant_name": variant_name,
                    "variant_suffix": variant["variant_suffix"],
                    "train_metrics": train_metrics,
                }
            )
            all_rows.append(
                {
                    "dataset": dataset,
                    "method": variant_name,
                    "family": method_name,
                    "variant_suffix": variant["variant_suffix"],
                    "concat_ACC": float(concat_metrics["ACC"]),
                    "concat_NMI": float(concat_metrics["NMI"]),
                    "concat_ARI": float(concat_metrics["ARI"]),
                    "train_ACC": float(train_metrics["ACC"]),
                    "train_NMI": float(train_metrics["NMI"]),
                    "train_ARI": float(train_metrics["ARI"]),
                }
            )

        result_payload = {
            "dataset": dataset,
            "method_family": method_name,
            "concat_metrics": concat_metrics,
            "train_variants": variant_payloads,
            "meta": _json_safe_meta(payload["meta"]),
        }
        safe_name = method_name.replace("[", "_").replace("]", "_")
        save_json(dataset_dir / f"{safe_name}_metrics.json", result_payload)

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(dataset_dir / "method_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "dataset": dataset,
        "cluster_num": int(cluster_num),
        "train_samples": int(images_train.shape[0]),
        "test_samples": int(images_test.shape[0]),
        "knowledge_bank_size": int(extra["knowledge_bank_size"]),
        "tac_baseline": baselines["concat"]["TAC"],
        "kec_baseline_concat": baselines["concat"]["KEC"],
        "kec_baseline_train": baselines["train"]["KEC"],
    }
    save_json(dataset_dir / "dataset_summary.json", summary)

    lines = [f"# {dataset}", ""]
    lines.append("## Baselines")
    lines.append(
        f"- TAC concat ACC: {baselines['concat']['TAC']['ACC']:.4f}, TAC train ACC: {baselines['train']['TAC']['ACC']:.4f}"
    )
    lines.append(
        f"- KEC concat ACC: {baselines['concat']['KEC']['ACC']:.4f}, KEC train ACC: {baselines['train']['KEC']['ACC']:.4f}"
    )
    lines.append("")
    lines.append("## New Methods")
    for row in all_rows:
        lines.append(
            f"- {row['method']}: concat ACC {row['concat_ACC']:.4f}, train ACC {row['train_ACC']:.4f}"
        )
    best_concat = max(all_rows, key=lambda x: x["concat_ACC"])
    best_train = max(all_rows, key=lambda x: x["train_ACC"])
    lines.append("")
    lines.append(f"- Best concat method: {best_concat['method']} ({best_concat['concat_ACC']:.4f})")
    lines.append(f"- Best train method: {best_train['method']} ({best_train['train_ACC']:.4f})")
    with open(dataset_dir / "comparison_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    del model
    torch.cuda.empty_cache()
    return {
        "dataset": dataset,
        "baselines": baselines,
        "results": all_rows,
        "best_concat": best_concat,
        "best_train": best_train,
    }


def main():
    args = build_parser().parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = args.datasets
    else:
        candidates = ["CIFAR-10", "CIFAR-20", "STL-10", "DTD", "StanfordDogs", "Flowers", "Pets"]
        datasets = [d for d in candidates if (ROOT / "data" / f"{d}_image_embedding_train.npy").exists()]

    aggregate_rows = []
    manifest = []
    for dataset in datasets:
        print(f">>> Running semantic boost for {dataset}")
        output = run_dataset(dataset, seed=args.seed)
        manifest.append({"dataset": dataset, "best_concat": output["best_concat"], "best_train": output["best_train"]})

        tac_concat = output["baselines"]["concat"]["TAC"]["ACC"]
        tac_train = output["baselines"]["train"]["TAC"]["ACC"]
        kec_concat = output["baselines"]["concat"]["KEC"]["ACC"]
        kec_train = output["baselines"]["train"]["KEC"]["ACC"]

        for row in output["results"]:
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "method": row["method"],
                    "concat_ACC": row["concat_ACC"],
                    "concat_delta_vs_TAC": row["concat_ACC"] - tac_concat,
                    "concat_delta_vs_KEC": row["concat_ACC"] - kec_concat,
                    "train_ACC": row["train_ACC"],
                    "train_delta_vs_TAC": row["train_ACC"] - tac_train,
                    "train_delta_vs_KEC": row["train_ACC"] - kec_train,
                }
            )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(RESULTS_DIR / "aggregate_results.csv", index=False, encoding="utf-8-sig")
    save_json(RESULTS_DIR / "run_manifest.json", manifest)

    lines = ["# Semantic Knowledge Boost Aggregate Summary", ""]
    lines.append("| Dataset | Method | Concat ACC | Delta vs TAC | Delta vs KEC | Train ACC | Delta vs TAC | Delta vs KEC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in aggregate_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['concat_ACC']:.4f} | {row['concat_delta_vs_TAC']:+.4f} | {row['concat_delta_vs_KEC']:+.4f} | {row['train_ACC']:.4f} | {row['train_delta_vs_TAC']:+.4f} | {row['train_delta_vs_KEC']:+.4f} |"
        )
    with open(RESULTS_DIR / "aggregate_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
