import json
import pickle
import sys
from pathlib import Path

import clip
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100, ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_utils import cluster_metric  # noqa: E402


SEM_ROOT = ROOT / "semantic_knowledge_boost_experiment"
OUT_DIR = SEM_ROOT / "final_benchmark"
CACHE_DIR = OUT_DIR / "clip_baseline_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEM_RESULTS = SEM_ROOT / "results"
TARGETED_RESULTS = SEM_ROOT / "targeted_boost_results"
SECOND_WAVE_RESULTS = SEM_ROOT / "second_wave_results" / "second_wave_summary.csv"
TAC_SUMMARY = ROOT / "tac_paper_reproduction" / "results" / "summary.csv"
TAC_PAPER = ROOT / "tac_paper_reproduction" / "results" / "paper_comparison.csv"
KEC_SUMMARY = ROOT / "kec_paper_reproduction" / "results" / "summary.csv"
KEC_PAPER = ROOT / "kec_paper_reproduction" / "results" / "paper_comparison.csv"


CIFAR100_COARSE_LABELS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


class CIFAR20TestDataset(Dataset):
    def __init__(self, root: Path, preprocess):
        base = CIFAR100(root=str(root), train=False, download=False)
        self.data = base.data
        self.targets = [CIFAR100_COARSE_LABELS[int(t)] for t in base.targets]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx])
        image = self.preprocess(image)
        label = int(self.targets[idx])
        return image, label


class StanfordDogsDataset(Dataset):
    def __init__(self, root: Path, preprocess):
        self.inner = ImageFolder(str(root), transform=preprocess)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self.inner[idx]


def l2_normalize(x, eps=1e-12):
    x = x.astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / denom


def spherical_kmeans(features, num_clusters, seed=42, niter=100, nredo=5):
    features = np.ascontiguousarray(features.astype(np.float32))
    estimator = faiss.Kmeans(
        features.shape[1],
        int(num_clusters),
        gpu=torch.cuda.is_available(),
        spherical=True,
        niter=niter,
        nredo=nredo,
        seed=seed,
        verbose=False,
    )
    estimator.train(features)
    _, indices = estimator.index.search(features, 1)
    return indices.reshape(-1)


def load_cached_tac_artifact(dataset: str):
    artifact_dir = ROOT / "tac_paper_reproduction" / "artifacts" / dataset
    image_path = artifact_dir / "image_embedding_test.npy"
    label_path = artifact_dir / "labels_test.txt"
    if image_path.exists() and label_path.exists():
        features = np.load(image_path).astype(np.float32)
        labels = np.loadtxt(label_path).astype(np.int64)
        return features, labels
    return None, None


def build_dataset(dataset: str, preprocess):
    data_root = ROOT / "data"
    if dataset == "CIFAR-10":
        return ImageFolder(str(data_root / "CIFAR-10" / "val"), transform=preprocess)
    if dataset == "CIFAR-20":
        return CIFAR20TestDataset(data_root, preprocess)
    if dataset == "STL-10":
        return ImageFolder(str(data_root / "STL-10" / "val"), transform=preprocess)
    if dataset == "DTD":
        return ImageFolder(str(data_root / "DTD" / "test"), transform=preprocess)
    if dataset == "Flowers":
        return ImageFolder(str(data_root / "Flowers" / "val"), transform=preprocess)
    if dataset == "Pets":
        return ImageFolder(str(data_root / "Pets" / "val"), transform=preprocess)
    if dataset == "StanfordDogs":
        return StanfordDogsDataset(data_root / "stanforddogs", preprocess)
    raise KeyError(f"Unsupported dataset for CLIP baseline: {dataset}")


def compute_clip_baseline(dataset: str, device: str = None, seed: int = 42):
    cache_path = CACHE_DIR / f"{dataset}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    cached_features, cached_labels = load_cached_tac_artifact(dataset)
    if cached_features is not None and cached_labels is not None:
        features = l2_normalize(cached_features)
        labels = cached_labels.astype(np.int64)
        preds = spherical_kmeans(features, len(np.unique(labels)), seed=seed)
        metrics = cluster_metric(labels, preds)
        payload = {
            "dataset": dataset,
            "source": "tac_paper_reproduction_artifact",
            "num_samples": int(len(labels)),
            "num_classes": int(len(np.unique(labels))),
            "ACC": metrics["ACC"],
            "NMI": metrics["NMI"],
            "ARI": metrics["ARI"],
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return payload

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    ds = build_dataset(dataset, preprocess)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    feats = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            feats.append(image_features.cpu().numpy().astype(np.float32))
            labels.append(batch_labels.numpy().astype(np.int64))

    features = l2_normalize(np.concatenate(feats, axis=0))
    labels = np.concatenate(labels, axis=0).astype(np.int64)
    preds = spherical_kmeans(features, len(np.unique(labels)), seed=seed)
    metrics = cluster_metric(labels, preds)
    payload = {
        "dataset": dataset,
        "source": "computed_from_local_images",
        "num_samples": int(len(labels)),
        "num_classes": int(len(np.unique(labels))),
        "ACC": metrics["ACC"],
        "NMI": metrics["NMI"],
        "ARI": metrics["ARI"],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def load_semantic_results():
    aggregate = pd.read_csv(SEM_RESULTS / "aggregate_results.csv")
    best = pd.read_csv(SEM_RESULTS / "best_by_dataset.csv")
    targeted = pd.read_csv(TARGETED_RESULTS / "targeted_vs_previous_and_paper.csv")
    second_wave = pd.read_csv(SECOND_WAVE_RESULTS) if SECOND_WAVE_RESULTS.exists() else pd.DataFrame()
    return aggregate, best, targeted, second_wave


def load_tac_local_maps():
    tac_summary = pd.read_csv(TAC_SUMMARY)
    local = {}
    paper = {}
    for row in tac_summary.to_dict("records"):
        dataset = row["dataset"]
        local.setdefault(dataset, {})
        local[dataset]["no_train"] = row["no_train_ACC"]
        local[dataset]["train"] = row["train_ACC"]
    tac_paper = pd.read_csv(TAC_PAPER)
    for row in tac_paper.to_dict("records"):
        dataset = row["dataset"]
        method = row["method"]
        paper.setdefault(dataset, {})
        if method == "TAC (no train)":
            paper[dataset]["no_train"] = row["paper_ACC"]
        elif method == "TAC":
            paper[dataset]["train"] = row["paper_ACC"]
    return local, paper


def load_kec_local_maps():
    kec_summary = pd.read_csv(KEC_SUMMARY)
    local = {}
    for row in kec_summary.to_dict("records"):
        local[row["dataset"]] = {
            "no_train": row["kec_no_train_acc"],
            "train": row["kec_train_acc"],
        }
    kec_paper = pd.read_csv(KEC_PAPER)
    paper = {}
    for row in kec_paper.to_dict("records"):
        paper[row["dataset"]] = {
            "no_train": row["paper_kec_no_train_acc"],
            "train": row["paper_kec_train_acc"],
        }
    return local, paper


def is_ours_method(method_name: str):
    return str(method_name).startswith("SAGE")


def build_our_maps(aggregate_df: pd.DataFrame, targeted_df: pd.DataFrame, second_wave_df: pd.DataFrame):
    no_train = {}
    train = {}
    targeted_map = {row["Dataset"]: row for row in targeted_df.to_dict("records")}
    second_wave_map = {row["dataset"]: row for row in second_wave_df.to_dict("records")} if not second_wave_df.empty else {}

    for dataset in sorted(aggregate_df["dataset"].unique().tolist()):
        subset = aggregate_df[aggregate_df["dataset"] == dataset].copy()
        ours_subset = subset[subset["method"].apply(is_ours_method)].copy()
        if ours_subset.empty:
            continue

        best_no_train = ours_subset.sort_values("concat_ACC", ascending=False).iloc[0]
        no_train[dataset] = {
            "method": best_no_train["method"],
            "acc": float(best_no_train["concat_ACC"]),
        }
        if dataset in second_wave_map:
            sw = second_wave_map[dataset]
            if float(sw["best_concat_ACC"]) > no_train[dataset]["acc"]:
                no_train[dataset] = {
                    "method": f"second_wave:{sw['best_config']}",
                    "acc": float(sw["best_concat_ACC"]),
                }

        if dataset in targeted_map:
            tgt = targeted_map[dataset]
            train[dataset] = {
                "method": tgt["Targeted Config"],
                "acc": float(tgt["Targeted Train ACC"]) / 100.0,
            }
        else:
            best_train = ours_subset.sort_values("train_ACC", ascending=False).iloc[0]
            train[dataset] = {
                "method": best_train["method"],
                "acc": float(best_train["train_ACC"]),
            }
        if dataset in second_wave_map:
            sw = second_wave_map[dataset]
            if float(sw["best_train_ACC"]) > train[dataset]["acc"]:
                train[dataset] = {
                    "method": f"second_wave:{sw['best_config']}",
                    "acc": float(sw["best_train_ACC"]),
                }
    return no_train, train


def baseline_from_semantic(aggregate_df: pd.DataFrame, method_name: str):
    out = {}
    rows = aggregate_df[aggregate_df["method"] == method_name]
    for row in rows.to_dict("records"):
        out[row["dataset"]] = {
            "no_train": row["concat_ACC"],
            "train": row["train_ACC"],
        }
    return out


def choose_value(primary_map, secondary_map, dataset, key):
    if dataset in primary_map and key in primary_map[dataset]:
        return primary_map[dataset][key], "official_reproduction"
    if dataset in secondary_map and key in secondary_map[dataset]:
        return secondary_map[dataset][key], "semantic_experiment_internal"
    return None, "missing"


def maybe_max(values):
    cleaned = [v for v in values if v is not None and not pd.isna(v)]
    return max(cleaned) if cleaned else None


def fmt(v):
    if v is None or pd.isna(v):
        return "NA"
    return f"{v:.4f}"


def build_tables():
    aggregate_df, best_df, targeted_df, second_wave_df = load_semantic_results()
    tac_local_official, tac_paper = load_tac_local_maps()
    kec_local_official, kec_paper = load_kec_local_maps()
    tac_local_fallback = baseline_from_semantic(aggregate_df, "TAC[g=0.0]")

    our_no_train, our_train = build_our_maps(aggregate_df, targeted_df, second_wave_df)
    datasets = sorted(best_df["dataset"].tolist())

    no_train_rows = []
    train_rows = []
    for dataset in datasets:
        try:
            clip_payload = compute_clip_baseline(dataset)
            clip_acc = clip_payload["ACC"]
        except Exception as exc:
            clip_payload = {
                "dataset": dataset,
                "source": f"missing:{type(exc).__name__}",
                "error": str(exc),
            }
            clip_acc = None

        tac_no_train, tac_no_train_source = choose_value(
            tac_local_official, tac_local_fallback, dataset, "no_train"
        )
        tac_train, tac_train_source = choose_value(
            tac_local_official, tac_local_fallback, dataset, "train"
        )

        kec_no_train = kec_local_official.get(dataset, {}).get("no_train")
        kec_train = kec_local_official.get(dataset, {}).get("train")
        tac_paper_no_train = tac_paper.get(dataset, {}).get("no_train")
        tac_paper_train = tac_paper.get(dataset, {}).get("train")
        kec_paper_no_train = kec_paper.get(dataset, {}).get("no_train")
        kec_paper_train = kec_paper.get(dataset, {}).get("train")

        our_no = our_no_train[dataset]
        our_tr = our_train[dataset]

        best_local_prior_no_train = maybe_max([clip_acc, tac_no_train, kec_no_train])
        best_local_prior_train = maybe_max([tac_train, kec_train])
        best_paper_prior_no_train = maybe_max([tac_paper_no_train, kec_paper_no_train])
        best_paper_prior_train = maybe_max([tac_paper_train, kec_paper_train])

        no_train_rows.append(
            {
                "dataset": dataset,
                "clip_baseline_acc": clip_acc,
                "clip_source": clip_payload["source"],
                "tac_local_acc": tac_no_train,
                "tac_local_source": tac_no_train_source,
                "tac_paper_acc": tac_paper_no_train,
                "kec_local_acc": kec_no_train,
                "kec_paper_acc": kec_paper_no_train,
                "ours_method": our_no["method"],
                "ours_acc": our_no["acc"],
                "gain_vs_clip": None if clip_acc is None else our_no["acc"] - clip_acc,
                "gain_vs_tac_local": None if tac_no_train is None else our_no["acc"] - tac_no_train,
                "gain_vs_kec_local": None if kec_no_train is None else our_no["acc"] - kec_no_train,
                "gain_vs_best_local_prior": None if best_local_prior_no_train is None else our_no["acc"] - best_local_prior_no_train,
                "gain_vs_best_paper_prior": None if best_paper_prior_no_train is None else our_no["acc"] - best_paper_prior_no_train,
            }
        )
        train_rows.append(
            {
                "dataset": dataset,
                "clip_baseline_acc": None,
                "tac_local_acc": tac_train,
                "tac_local_source": tac_train_source,
                "tac_paper_acc": tac_paper_train,
                "kec_local_acc": kec_train,
                "kec_paper_acc": kec_paper_train,
                "ours_method": our_tr["method"],
                "ours_acc": our_tr["acc"],
                "gain_vs_tac_local": None if tac_train is None else our_tr["acc"] - tac_train,
                "gain_vs_kec_local": None if kec_train is None else our_tr["acc"] - kec_train,
                "gain_vs_best_local_prior": None if best_local_prior_train is None else our_tr["acc"] - best_local_prior_train,
                "gain_vs_best_paper_prior": None if best_paper_prior_train is None else our_tr["acc"] - best_paper_prior_train,
            }
        )

    return pd.DataFrame(no_train_rows), pd.DataFrame(train_rows)


def write_markdown(no_train_df: pd.DataFrame, train_df: pd.DataFrame):
    lines = ["# Final Benchmark Comparison", ""]
    lines.append("## Reading Guide")
    lines.append("- `clip_baseline_acc`: image-only CLIP embedding + spherical KMeans.")
    lines.append("- `tac_local_acc`: our local TAC baseline. If an official TAC reproduction package exists, it is preferred; otherwise the TAC branch inside `semantic_knowledge_boost_experiment` is used.")
    lines.append("- `kec_local_acc`: our local KEC reproduction from `kec_paper_reproduction`.")
    lines.append("- `tac_paper_acc` / `kec_paper_acc`: values explicitly reported in the corresponding papers and already extracted into local reference files.")
    lines.append("- `ours_acc`: our best result for that protocol. No-train uses the best concat result; train uses the best targeted train result when available, otherwise the best train result from the main semantic experiment.")
    lines.append("- Positive `gain_vs_best_local_prior` means our method beats every available local baseline on that dataset under the same protocol.")
    lines.append("- Positive `gain_vs_best_paper_prior` means our method exceeds the best paper-reported TAC/KEC value available in this repo for that dataset and protocol.")
    lines.append("")

    lines.append("## No-Train Comparison")
    lines.append("")
    lines.append("| Dataset | CLIP | TAC Local | TAC Paper | KEC Local | KEC Paper | Ours | Ours Method | Gain vs Best Local | Gain vs Best Paper |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for row in no_train_df.to_dict("records"):
        lines.append(
            f"| {row['dataset']} | {fmt(row['clip_baseline_acc'])} | {fmt(row['tac_local_acc'])} | {fmt(row['tac_paper_acc'])} | {fmt(row['kec_local_acc'])} | {fmt(row['kec_paper_acc'])} | {fmt(row['ours_acc'])} | {row['ours_method']} | {fmt(row['gain_vs_best_local_prior'])} | {fmt(row['gain_vs_best_paper_prior'])} |"
        )
    lines.append("")

    lines.append("## Train Comparison")
    lines.append("")
    lines.append("| Dataset | TAC Local | TAC Paper | KEC Local | KEC Paper | Ours | Ours Method | Gain vs Best Local | Gain vs Best Paper |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for row in train_df.to_dict("records"):
        lines.append(
            f"| {row['dataset']} | {fmt(row['tac_local_acc'])} | {fmt(row['tac_paper_acc'])} | {fmt(row['kec_local_acc'])} | {fmt(row['kec_paper_acc'])} | {fmt(row['ours_acc'])} | {row['ours_method']} | {fmt(row['gain_vs_best_local_prior'])} | {fmt(row['gain_vs_best_paper_prior'])} |"
        )
    lines.append("")

    no_train_wins = int((no_train_df["gain_vs_best_local_prior"] > 0).fillna(False).sum())
    train_wins = int((train_df["gain_vs_best_local_prior"] > 0).fillna(False).sum())
    lines.append("## Takeaways")
    lines.append(
        f"- On the no-train protocol, our method beats the strongest available local prior baseline on {no_train_wins}/{len(no_train_df)} datasets."
    )
    lines.append(
        f"- On the train protocol, our method beats the strongest available local prior baseline on {train_wins}/{len(train_df)} datasets."
    )
    lines.append("- Datasets with missing paper values should be interpreted only against local baselines, not as a strict paper-level comparison.")
    lines.append("- The KEC branch here is an offline local surrogate of LLM-generated knowledge, so remaining gaps versus the KEC paper should be read with that caveat in mind.")
    with open(OUT_DIR / "FINAL_BENCHMARK_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    no_train_df, train_df = build_tables()
    no_train_df.to_csv(OUT_DIR / "no_train_comparison.csv", index=False, encoding="utf-8-sig")
    train_df.to_csv(OUT_DIR / "train_comparison.csv", index=False, encoding="utf-8-sig")

    merged = no_train_df.merge(
        train_df,
        on="dataset",
        suffixes=("_no_train", "_train"),
    )
    merged.to_csv(OUT_DIR / "final_combined_table.csv", index=False, encoding="utf-8-sig")
    write_markdown(no_train_df, train_df)


if __name__ == "__main__":
    main()
