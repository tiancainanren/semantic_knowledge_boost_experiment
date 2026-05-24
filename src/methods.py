import json
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import faiss
import numpy as np
import torch
import torch.nn.functional as F

from hierarchical_textual_knowledge_experiment.src import pipeline as kec


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
KEC_RESULTS_DIR = ROOT / "hierarchical_textual_knowledge_experiment" / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_knowledge_bank(dataset: str, results_root: Path = None) -> List[Dict]:
    if results_root is None:
        results_root = KEC_RESULTS_DIR
    path = results_root / dataset / "knowledge_bank.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_kec_artifacts(dataset: str, results_root: Path = None):
    if results_root is None:
        results_root = KEC_RESULTS_DIR
    artifact_dir = results_root / dataset / "artifacts"
    return {
        "tac_retrieval_train": np.load(artifact_dir / "tac_retrieval_train.npy").astype(np.float32),
        "tac_retrieval_test": np.load(artifact_dir / "tac_retrieval_test.npy").astype(np.float32),
        "knowledge_train_raw": np.load(artifact_dir / "knowledge_train_raw.npy").astype(np.float32),
        "knowledge_test_raw": np.load(artifact_dir / "knowledge_test_raw.npy").astype(np.float32),
        "knowledge_train": np.load(artifact_dir / "knowledge_train.npy").astype(np.float32),
        "knowledge_test": np.load(artifact_dir / "knowledge_test.npy").astype(np.float32),
    }


def load_filtered_nouns_embedding(dataset: str, model=None, results_root: Path = None) -> np.ndarray:
    path = DATA_DIR / f"{dataset}_filtered_nouns_embedding.npy"
    if path.exists():
        return kec.normalize_np(np.load(path).astype(np.float32))

    tac_artifact_path = ROOT / "tac_paper_reproduction" / "artifacts" / dataset / "filtered_nouns_embedding.npy"
    if tac_artifact_path.exists():
        return kec.normalize_np(np.load(tac_artifact_path).astype(np.float32))

    if results_root is None:
        results_root = KEC_RESULTS_DIR
    candidate_jsons = [results_root / dataset / "filtered_nouns.json"]
    alt_prompt_root = ROOT / "hierarchical_textual_knowledge_experiment" / "results_prompted_local_llm_like"
    alt_simple_root = ROOT / "hierarchical_textual_knowledge_experiment" / "results"
    for alt_root in [alt_prompt_root, alt_simple_root]:
        alt_path = alt_root / dataset / "filtered_nouns.json"
        if alt_path not in candidate_jsons:
            candidate_jsons.append(alt_path)

    filtered_json = next((p for p in candidate_jsons if p.exists()), None)
    if filtered_json is not None:
        if model is None:
            model = load_clip_model()
        with open(filtered_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        nouns = payload.get("filtered_nouns", [])
        emb = encode_texts(model, nouns)
        return kec.normalize_np(emb.astype(np.float32))

    raise FileNotFoundError(f"Could not locate filtered noun embeddings for {dataset}.")


def load_clip_model():
    model = kec.CLIPModel(model_name="ViT-B/32").to(DEVICE)
    model.eval()
    return model


def encode_texts(model, texts: List[str], batch_size: int = 256) -> np.ndarray:
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(DEVICE)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = F.normalize(features.float(), dim=1)
        outputs.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def prepare_text_sources(model, knowledge_bank: List[Dict]):
    concept_texts = [item["concept"] for item in knowledge_bank]
    caption_texts = [item["concept_caption"] for item in knowledge_bank]
    concept_emb = encode_texts(model, concept_texts)
    caption_emb = encode_texts(model, caption_texts)

    unary_proto = []
    binary_proto = []
    for item in knowledge_bank:
        unary = encode_texts(model, item.get("attributes_unary", []))
        binary = encode_texts(model, item.get("attributes_binary", []))
        if len(unary) == 0:
            unary_proto.append(np.zeros((512,), dtype=np.float32))
        else:
            unary_proto.append(kec.normalize_np(unary.mean(axis=0, keepdims=True))[0])
        if len(binary) == 0:
            binary_proto.append(np.zeros((512,), dtype=np.float32))
        else:
            binary_proto.append(kec.normalize_np(binary.mean(axis=0, keepdims=True))[0])
    unary_proto = kec.normalize_np(np.stack(unary_proto, axis=0))
    binary_proto = kec.normalize_np(np.stack(binary_proto, axis=0))
    return {
        "concept_emb": concept_emb.astype(np.float32),
        "caption_emb": caption_emb.astype(np.float32),
        "unary_proto": unary_proto.astype(np.float32),
        "binary_proto": binary_proto.astype(np.float32),
    }


def build_text_prototypes(
    model,
    knowledge_bank: List[Dict] = None,
    text_sources: Dict[str, np.ndarray] = None,
    concept_caption_weight: float = 0.7,
    unary_weight: float = 0.7,
    binary_weight: float = 0.3,
):
    if text_sources is None:
        if knowledge_bank is None:
            raise ValueError("Either knowledge_bank or text_sources must be provided.")
        text_sources = prepare_text_sources(model, knowledge_bank)
    concept_emb = text_sources["concept_emb"]
    caption_emb = text_sources["caption_emb"]
    unary_proto = text_sources["unary_proto"]
    binary_proto = text_sources["binary_proto"]

    concept_proto = kec.normalize_np(concept_emb + concept_caption_weight * caption_emb)
    attr_proto = kec.normalize_np(unary_weight * unary_proto + binary_weight * binary_proto)
    return {
        "concept_proto": concept_proto.astype(np.float32),
        "unary_proto": unary_proto.astype(np.float32),
        "binary_proto": binary_proto.astype(np.float32),
        "attr_proto": attr_proto.astype(np.float32),
    }


def _batched_attention(images: np.ndarray, prototypes: np.ndarray, temperature: float, batch_size: int = 2048):
    outputs = []
    confidence = []
    images_t = torch.from_numpy(images).to(DEVICE).float()
    proto_t = torch.from_numpy(prototypes).to(DEVICE).float()
    for start in range(0, images.shape[0], batch_size):
        batch = images_t[start : start + batch_size]
        sim = torch.matmul(batch, proto_t.T)
        weight = torch.softmax(sim / temperature, dim=1)
        view = torch.matmul(weight, proto_t)
        view = F.normalize(view, dim=1)
        outputs.append(view.cpu().numpy().astype(np.float32))
        confidence.append(weight.max(dim=1).values.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0), np.concatenate(confidence, axis=0)


def _batched_grounded_attr(images: np.ndarray, attr_proto: np.ndarray, concept_proto: np.ndarray, batch_size: int = 512):
    outputs = []
    attr_conf = []
    images_t = torch.from_numpy(images).to(DEVICE).float()
    attr_t = torch.from_numpy(attr_proto).to(DEVICE).float()
    concept_t = torch.from_numpy(concept_proto).to(DEVICE).float()
    for start in range(0, images.shape[0], batch_size):
        batch = images_t[start : start + batch_size]
        concept_weight = torch.softmax(torch.matmul(batch, concept_t.T) / 0.07, dim=1)
        grounded = F.normalize(batch.unsqueeze(1) * attr_t.unsqueeze(0), dim=2)
        fused = (concept_weight.unsqueeze(2) * grounded).sum(dim=1)
        fused = F.normalize(fused, dim=1)
        outputs.append(fused.cpu().numpy().astype(np.float32))
        attr_conf.append(concept_weight.max(dim=1).values.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0), np.concatenate(attr_conf, axis=0)


def compute_semantic_confidence_views(
    images_train: np.ndarray,
    images_test: np.ndarray,
    noun_embeddings: np.ndarray,
    text_proto: Dict[str, np.ndarray],
    concept_temp: float = 0.07,
    noun_temp: float = 0.005,
    grounded_temp: float = 0.07,
):
    concept_train, concept_conf_train = _batched_attention(images_train, text_proto["concept_proto"], concept_temp)
    concept_test, concept_conf_test = _batched_attention(images_test, text_proto["concept_proto"], concept_temp)
    attr_train, attr_conf_train = _batched_grounded_attr(
        images_train, text_proto["attr_proto"], text_proto["concept_proto"]
    )
    attr_test, attr_conf_test = _batched_grounded_attr(
        images_test, text_proto["attr_proto"], text_proto["concept_proto"]
    )
    if grounded_temp != 0.07:
        # Light rescaling to mimic sharper or softer attribute grounding without re-implementing the full routine.
        attr_conf_train = np.clip(attr_conf_train ** (0.07 / grounded_temp), 0.0, 1.0)
        attr_conf_test = np.clip(attr_conf_test ** (0.07 / grounded_temp), 0.0, 1.0)
    _, noun_conf_train = _batched_attention(images_train, noun_embeddings, noun_temp)
    _, noun_conf_test = _batched_attention(images_test, noun_embeddings, noun_temp)
    return {
        "concept_train": concept_train,
        "concept_test": concept_test,
        "attr_train": attr_train,
        "attr_test": attr_test,
        "concept_conf_train": concept_conf_train,
        "concept_conf_test": concept_conf_test,
        "attr_conf_train": attr_conf_train,
        "attr_conf_test": attr_conf_test,
        "noun_conf_train": noun_conf_train,
        "noun_conf_test": noun_conf_test,
    }


def adaptive_fusion(
    noun_view: np.ndarray,
    knowledge_raw: np.ndarray,
    noun_conf: np.ndarray,
    knowledge_conf: np.ndarray,
    gate_min: float = 0.15,
    gate_max: float = 0.85,
):
    gate = knowledge_conf / (knowledge_conf + noun_conf + 1e-8)
    gate = np.clip(gate, gate_min, gate_max).astype(np.float32)
    fused = (1.0 - gate[:, None]) * noun_view + gate[:, None] * knowledge_raw
    return kec.normalize_np(fused), gate


def compute_alignment_score(image: np.ndarray, semantic: np.ndarray):
    score = (image * semantic).sum(axis=1)
    score = np.clip((score + 1.0) / 2.0, 0.0, 1.0)
    return score.astype(np.float32)


def build_confidence_guidance(semantic_conf: np.ndarray, alignment: np.ndarray, beta_max: float = 0.6):
    beta = beta_max * semantic_conf * alignment
    return np.clip(beta, 0.0, beta_max).astype(np.float32)


def hybridize_semantics(images: np.ndarray, semantic_a: np.ndarray, semantic_b: np.ndarray, temperature: float = 0.07):
    sim_a = (images * semantic_a).sum(axis=1)
    sim_b = (images * semantic_b).sum(axis=1)
    logits = np.stack([sim_a, sim_b], axis=1) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    weight = np.exp(logits)
    weight = weight / (weight.sum(axis=1, keepdims=True) + 1e-12)
    alpha = weight[:, 1].astype(np.float32)
    hybrid = kec.normalize_np((1.0 - alpha[:, None]) * semantic_a + alpha[:, None] * semantic_b)
    return hybrid, alpha


def semantic_neighbor_score(images: np.ndarray, semantic: np.ndarray, topk: int = 20):
    neighbor_mean = knn_neighbor_mean(images, semantic, topk=topk)
    score = (neighbor_mean * semantic).sum(axis=1)
    score = np.clip((score + 1.0) / 2.0, 0.0, 1.0)
    return score.astype(np.float32)


def mixture_of_semantic_experts(
    images: np.ndarray,
    expert_bank: Dict[str, np.ndarray],
    confidence_bank: Dict[str, np.ndarray],
    topk: int = 20,
    temperature: float = 0.07,
    conf_weight: float = 0.4,
    align_weight: float = 0.35,
    neighbor_weight: float = 0.25,
    prior_bias: Dict[str, float] = None,
):
    expert_names = list(expert_bank.keys())
    reliability_terms = []
    meta = {"experts": expert_names}
    prior_bias = prior_bias or {}
    align_store = {}
    neigh_store = {}
    conf_store = {}
    for name in expert_names:
        semantic = expert_bank[name]
        conf = confidence_bank[name]
        align = compute_alignment_score(images, semantic)
        neigh = semantic_neighbor_score(images, semantic, topk=topk)
        conf_store[name] = conf
        align_store[name] = align
        neigh_store[name] = neigh
        rel = (
            conf_weight * conf
            + align_weight * align
            + neighbor_weight * neigh
            + float(prior_bias.get(name, 0.0))
        )
        reliability_terms.append(rel.astype(np.float32))
    logits = np.stack(reliability_terms, axis=1) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
    fused = np.zeros_like(next(iter(expert_bank.values())), dtype=np.float32)
    for idx, name in enumerate(expert_names):
        fused += weights[:, idx : idx + 1] * expert_bank[name]
    fused = kec.normalize_np(fused)
    meta.update(
        {
            "mean_weights": {name: float(weights[:, idx].mean()) for idx, name in enumerate(expert_names)},
            "mean_alignment": {name: float(align_store[name].mean()) for name in expert_names},
            "mean_neighbor_score": {name: float(neigh_store[name].mean()) for name in expert_names},
            "mean_confidence": {name: float(conf_store[name].mean()) for name in expert_names},
        }
    )
    return fused, weights.astype(np.float32), meta


def preserve_semantic_core(
    fused_semantic: np.ndarray,
    preserve_semantic: np.ndarray,
    preserve_strength,
):
    strength = np.asarray(preserve_strength, dtype=np.float32)
    if strength.ndim == 0:
        if float(strength) <= 0.0:
            return fused_semantic
        strength = np.full((fused_semantic.shape[0],), float(strength), dtype=np.float32)
    strength = np.clip(strength, 0.0, 1.0)
    preserved = (1.0 - strength[:, None]) * fused_semantic + strength[:, None] * preserve_semantic
    return kec.normalize_np(preserved)


def knn_neighbor_mean(features: np.ndarray, semantic: np.ndarray, topk: int = 20):
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(np.ascontiguousarray(features.astype(np.float32)))
    _, inds = index.search(np.ascontiguousarray(features.astype(np.float32)), min(topk + 1, features.shape[0]))
    neighbor_inds = inds[:, 1:]
    neighbor_mean = semantic[neighbor_inds].mean(axis=1)
    return kec.normalize_np(neighbor_mean)


def graph_diffuse(
    images: np.ndarray,
    semantic: np.ndarray,
    noun_conf: np.ndarray,
    knowledge_conf: np.ndarray,
    topk: int = 20,
    smooth_min: float = 0.05,
    smooth_max: float = 0.35,
    certainty_power: float = 1.0,
):
    neighbor_mean = knn_neighbor_mean(images, semantic, topk=topk)
    certainty = np.maximum(noun_conf, knowledge_conf)
    smooth_weight = smooth_min + (smooth_max - smooth_min) * np.power(1.0 - certainty, certainty_power)
    smooth_weight = np.clip(smooth_weight, smooth_min, smooth_max).astype(np.float32)
    fused = (1.0 - smooth_weight[:, None]) * semantic + smooth_weight[:, None] * neighbor_mean
    return kec.normalize_np(fused), smooth_weight


def _anchor_hard_assign(features: np.ndarray, anchor_num: int, seed: int):
    features = np.ascontiguousarray(features.astype(np.float32))
    anchor_num = int(min(anchor_num, features.shape[0]))
    km = faiss.Kmeans(
        features.shape[1],
        anchor_num,
        gpu=torch.cuda.is_available(),
        spherical=True,
        niter=80,
        nredo=3,
        verbose=False,
        seed=seed,
    )
    km.train(features)
    _, inds = km.index.search(features, 1)
    return inds.reshape(-1).astype(np.int64), km.centroids.reshape(anchor_num, features.shape[1]).astype(np.float32)


def prototype_calibrate(
    images: np.ndarray,
    semantic: np.ndarray,
    noun_conf: np.ndarray,
    knowledge_conf: np.ndarray,
    num_classes: int,
    seed: int,
    anchor_multiplier: float = 4.0,
    anchor_cap: int = 512,
    mix_min: float = 0.10,
    mix_max: float = 0.40,
):
    anchor_num = int(min(max(int(num_classes * anchor_multiplier), 64), anchor_cap, images.shape[0]))
    anchor_assign, _ = _anchor_hard_assign(images, anchor_num=anchor_num, seed=seed)
    proto = np.zeros((anchor_num, semantic.shape[1]), dtype=np.float32)
    for idx in range(anchor_num):
        mask = anchor_assign == idx
        if mask.any():
            proto[idx] = semantic[mask].mean(axis=0)
    proto = kec.normalize_np(proto)
    proto_view = proto[anchor_assign]
    certainty = np.maximum(noun_conf, knowledge_conf)
    mix = mix_min + (mix_max - mix_min) * (1.0 - certainty)
    mix = np.clip(mix, mix_min, mix_max).astype(np.float32)
    fused = (1.0 - mix[:, None]) * semantic + mix[:, None] * proto_view
    return kec.normalize_np(fused), mix, anchor_num


def blend_semantics(primary: np.ndarray, residual: np.ndarray, rho: float = 0.0):
    if rho <= 0.0:
        return primary
    return kec.normalize_np((1.0 - rho) * primary + rho * residual)


def prepare_semantic_context(
    dataset: str,
    images_train: np.ndarray,
    images_test: np.ndarray,
    model=None,
    knowledge_root: Path = None,
):
    created_model = False
    if model is None:
        model = load_clip_model()
        created_model = True
    noun_embeddings = load_filtered_nouns_embedding(dataset, model=model, results_root=knowledge_root)
    knowledge_bank = load_knowledge_bank(dataset, results_root=knowledge_root)
    text_sources = prepare_text_sources(model, knowledge_bank)
    caches = load_existing_kec_artifacts(dataset, results_root=knowledge_root)
    context = {
        "dataset": dataset,
        "images_train": images_train,
        "images_test": images_test,
        "model": model,
        "noun_embeddings": noun_embeddings,
        "knowledge_bank": knowledge_bank,
        "text_sources": text_sources,
        "caches": caches,
        "created_model": created_model,
        "knowledge_root": str(knowledge_root or KEC_RESULTS_DIR),
    }
    return context


def build_custom_semantic_variant(context: Dict, config: Dict, seed: int = 42):
    text_proto = build_text_prototypes(
        model=context["model"],
        text_sources=context["text_sources"],
        concept_caption_weight=config.get("concept_caption_weight", 0.7),
        unary_weight=config.get("unary_weight", 0.7),
        binary_weight=config.get("binary_weight", 0.3),
    )
    conf_views = compute_semantic_confidence_views(
        images_train=context["images_train"],
        images_test=context["images_test"],
        noun_embeddings=context["noun_embeddings"],
        text_proto=text_proto,
        concept_temp=config.get("concept_temp", 0.07),
        noun_temp=config.get("noun_temp", 0.005),
        grounded_temp=config.get("grounded_temp", 0.07),
    )
    concept_conf_alpha = config.get("concept_conf_alpha", 0.6)
    knowledge_conf_train = concept_conf_alpha * conf_views["concept_conf_train"] + (1.0 - concept_conf_alpha) * conf_views["attr_conf_train"]
    knowledge_conf_test = concept_conf_alpha * conf_views["concept_conf_test"] + (1.0 - concept_conf_alpha) * conf_views["attr_conf_test"]

    adaptive_train, gate_train = adaptive_fusion(
        context["caches"]["tac_retrieval_train"],
        context["caches"]["knowledge_train_raw"],
        conf_views["noun_conf_train"],
        knowledge_conf_train,
        gate_min=config.get("gate_min", 0.15),
        gate_max=config.get("gate_max", 0.85),
    )
    adaptive_test, gate_test = adaptive_fusion(
        context["caches"]["tac_retrieval_test"],
        context["caches"]["knowledge_test_raw"],
        conf_views["noun_conf_test"],
        knowledge_conf_test,
        gate_min=config.get("gate_min", 0.15),
        gate_max=config.get("gate_max", 0.85),
    )

    adaptive_train = blend_semantics(adaptive_train, context["caches"]["knowledge_train"], config.get("knowledge_residual_rho", 0.0))
    adaptive_test = blend_semantics(adaptive_test, context["caches"]["knowledge_test"], config.get("knowledge_residual_rho", 0.0))

    diffuse_train, smooth_train = graph_diffuse(
        context["images_train"],
        adaptive_train,
        conf_views["noun_conf_train"],
        knowledge_conf_train,
        topk=config.get("graph_topk", 20),
        smooth_min=config.get("smooth_min", 0.05),
        smooth_max=config.get("smooth_max", 0.35),
        certainty_power=config.get("certainty_power", 1.0),
    )
    diffuse_test, smooth_test = graph_diffuse(
        context["images_test"],
        adaptive_test,
        conf_views["noun_conf_test"],
        knowledge_conf_test,
        topk=config.get("graph_topk", 20),
        smooth_min=config.get("smooth_min", 0.05),
        smooth_max=config.get("smooth_max", 0.35),
        certainty_power=config.get("certainty_power", 1.0),
    )
    diffuse_train = blend_semantics(diffuse_train, adaptive_train, config.get("graph_residual_rho", 0.0))
    diffuse_test = blend_semantics(diffuse_test, adaptive_test, config.get("graph_residual_rho", 0.0))

    proto_train, proto_mix_train, anchor_num_train = prototype_calibrate(
        context["images_train"],
        diffuse_train,
        conf_views["noun_conf_train"],
        knowledge_conf_train,
        num_classes=len(context["knowledge_bank"]),
        seed=seed,
        anchor_multiplier=config.get("anchor_multiplier", 4.0),
        anchor_cap=config.get("anchor_cap", 512),
        mix_min=config.get("proto_mix_min", 0.10),
        mix_max=config.get("proto_mix_max", 0.40),
    )
    proto_test, proto_mix_test, anchor_num_test = prototype_calibrate(
        context["images_test"],
        diffuse_test,
        conf_views["noun_conf_test"],
        knowledge_conf_test,
        num_classes=len(context["knowledge_bank"]),
        seed=seed,
        anchor_multiplier=config.get("anchor_multiplier", 4.0),
        anchor_cap=config.get("anchor_cap", 512),
        mix_min=config.get("proto_mix_min", 0.10),
        mix_max=config.get("proto_mix_max", 0.40),
    )
    proto_train = blend_semantics(proto_train, adaptive_train, config.get("proto_residual_rho", 0.0))
    proto_test = blend_semantics(proto_test, adaptive_test, config.get("proto_residual_rho", 0.0))

    semantic_conf_train = np.maximum(conf_views["noun_conf_train"], knowledge_conf_train)
    semantic_conf_test = np.maximum(conf_views["noun_conf_test"], knowledge_conf_test)

    family = config.get("family", "AG")
    if family == "AG":
        semantic_train = diffuse_train
        semantic_test = diffuse_test
    elif family == "AGP":
        semantic_train = proto_train
        semantic_test = proto_test
    elif family == "HYB":
        semantic_train, _ = hybridize_semantics(context["images_train"], diffuse_train, proto_train, temperature=config.get("hybrid_temp", 0.07))
        semantic_test, _ = hybridize_semantics(context["images_test"], diffuse_test, proto_test, temperature=config.get("hybrid_temp", 0.07))
    elif family in {"MOE", "MOEP"}:
        expert_train = {
            "kec": context["caches"]["knowledge_train"],
            "ag": diffuse_train,
            "agp": proto_train,
        }
        expert_test = {
            "kec": context["caches"]["knowledge_test"],
            "ag": diffuse_test,
            "agp": proto_test,
        }
        confidence_train = {
            "kec": knowledge_conf_train,
            "ag": 0.5 * (semantic_conf_train + (1.0 - smooth_train)),
            "agp": 0.5 * (semantic_conf_train + (1.0 - proto_mix_train)),
        }
        confidence_test = {
            "kec": knowledge_conf_test,
            "ag": 0.5 * (semantic_conf_test + (1.0 - smooth_test)),
            "agp": 0.5 * (semantic_conf_test + (1.0 - proto_mix_test)),
        }
        if config.get("include_adaptive_expert", False):
            expert_train["adaptive"] = adaptive_train
            expert_test["adaptive"] = adaptive_test
            confidence_train["adaptive"] = semantic_conf_train
            confidence_test["adaptive"] = semantic_conf_test
        prior_bias = config.get("expert_prior_bias", {})
        semantic_train, moe_weights_train, moe_meta_train = mixture_of_semantic_experts(
            context["images_train"],
            expert_train,
            confidence_train,
            topk=config.get("moe_topk", config.get("graph_topk", 20)),
            temperature=config.get("moe_temp", 0.07),
            conf_weight=config.get("moe_conf_weight", 0.4),
            align_weight=config.get("moe_align_weight", 0.35),
            neighbor_weight=config.get("moe_neighbor_weight", 0.25),
            prior_bias=prior_bias,
        )
        semantic_test, moe_weights_test, moe_meta_test = mixture_of_semantic_experts(
            context["images_test"],
            expert_test,
            confidence_test,
            topk=config.get("moe_topk", config.get("graph_topk", 20)),
            temperature=config.get("moe_temp", 0.07),
            conf_weight=config.get("moe_conf_weight", 0.4),
            align_weight=config.get("moe_align_weight", 0.35),
            neighbor_weight=config.get("moe_neighbor_weight", 0.25),
            prior_bias=prior_bias,
        )
        if family == "MOEP":
            preserve_base_train = np.clip(
                config.get("preserve_min", 0.10)
                + config.get("preserve_scale", 0.25) * (1.0 - knowledge_conf_train),
                0.0,
                config.get("preserve_cap", 0.40),
            )
            preserve_base_test = np.clip(
                config.get("preserve_min", 0.10)
                + config.get("preserve_scale", 0.25) * (1.0 - knowledge_conf_test),
                0.0,
                config.get("preserve_cap", 0.40),
            )
            semantic_train = preserve_semantic_core(semantic_train, context["caches"]["knowledge_train"], preserve_base_train)
            semantic_test = preserve_semantic_core(semantic_test, context["caches"]["knowledge_test"], preserve_base_test)
        meta_moe = {
            "moe_train": moe_meta_train,
            "moe_test": moe_meta_test,
        }
    else:
        raise ValueError(f"Unknown family: {family}")

    meta = {
        "mean_gate_train": float(gate_train.mean()),
        "mean_gate_test": float(gate_test.mean()),
        "mean_smooth_train": float(smooth_train.mean()),
        "mean_smooth_test": float(smooth_test.mean()),
        "mean_proto_mix_train": float(proto_mix_train.mean()),
        "mean_proto_mix_test": float(proto_mix_test.mean()),
        "anchor_num_train": int(anchor_num_train),
        "anchor_num_test": int(anchor_num_test),
    }
    if family in {"MOE", "MOEP"}:
        meta.update(meta_moe)
    return semantic_train.astype(np.float32), semantic_test.astype(np.float32), meta


def build_semantic_methods(
    dataset: str,
    images_train: np.ndarray,
    images_test: np.ndarray,
    seed: int = 42,
):
    model = load_clip_model()
    noun_embeddings = load_filtered_nouns_embedding(dataset)
    knowledge_bank = load_knowledge_bank(dataset)
    text_proto = build_text_prototypes(model, knowledge_bank)
    caches = load_existing_kec_artifacts(dataset)
    conf_views = compute_semantic_confidence_views(
        images_train=images_train,
        images_test=images_test,
        noun_embeddings=noun_embeddings,
        text_proto=text_proto,
    )

    knowledge_conf_train = 0.6 * conf_views["concept_conf_train"] + 0.4 * conf_views["attr_conf_train"]
    knowledge_conf_test = 0.6 * conf_views["concept_conf_test"] + 0.4 * conf_views["attr_conf_test"]

    adaptive_train, gate_train = adaptive_fusion(
        caches["tac_retrieval_train"],
        caches["knowledge_train_raw"],
        conf_views["noun_conf_train"],
        knowledge_conf_train,
    )
    adaptive_test, gate_test = adaptive_fusion(
        caches["tac_retrieval_test"],
        caches["knowledge_test_raw"],
        conf_views["noun_conf_test"],
        knowledge_conf_test,
    )

    diffuse_train, smooth_train = graph_diffuse(
        images_train,
        adaptive_train,
        conf_views["noun_conf_train"],
        knowledge_conf_train,
    )
    diffuse_test, smooth_test = graph_diffuse(
        images_test,
        adaptive_test,
        conf_views["noun_conf_test"],
        knowledge_conf_test,
    )

    proto_train, proto_mix_train, anchor_num_train = prototype_calibrate(
        images_train,
        diffuse_train,
        conf_views["noun_conf_train"],
        knowledge_conf_train,
        num_classes=len(knowledge_bank),
        seed=seed,
    )
    proto_test, proto_mix_test, anchor_num_test = prototype_calibrate(
        images_test,
        diffuse_test,
        conf_views["noun_conf_test"],
        knowledge_conf_test,
        num_classes=len(knowledge_bank),
        seed=seed,
    )

    semantic_conf_train = np.maximum(conf_views["noun_conf_train"], knowledge_conf_train)
    semantic_conf_test = np.maximum(conf_views["noun_conf_test"], knowledge_conf_test)
    alignment_ag_train = compute_alignment_score(images_train, diffuse_train)
    alignment_ag_test = compute_alignment_score(images_test, diffuse_test)
    alignment_agp_train = compute_alignment_score(images_train, proto_train)
    alignment_agp_test = compute_alignment_score(images_test, proto_test)

    gated_ag_train = build_confidence_guidance(semantic_conf_train, alignment_ag_train, beta_max=0.6)
    gated_ag_test = build_confidence_guidance(semantic_conf_test, alignment_ag_test, beta_max=0.6)
    gated_agp_train = build_confidence_guidance(semantic_conf_train, alignment_agp_train, beta_max=0.6)
    gated_agp_test = build_confidence_guidance(semantic_conf_test, alignment_agp_test, beta_max=0.6)

    hybrid_train, hybrid_alpha_train = hybridize_semantics(images_train, diffuse_train, proto_train)
    hybrid_test, hybrid_alpha_test = hybridize_semantics(images_test, diffuse_test, proto_test)
    hybrid_align_train = compute_alignment_score(images_train, hybrid_train)
    hybrid_align_test = compute_alignment_score(images_test, hybrid_test)
    hybrid_gate_train = build_confidence_guidance(semantic_conf_train, hybrid_align_train, beta_max=0.6)
    hybrid_gate_test = build_confidence_guidance(semantic_conf_test, hybrid_align_test, beta_max=0.6)

    moe_train, moe_weights_train, moe_meta_train = mixture_of_semantic_experts(
        images_train,
        {
            "kec": caches["knowledge_train"],
            "ag": diffuse_train,
            "agp": proto_train,
        },
        {
            "kec": knowledge_conf_train,
            "ag": 0.5 * (semantic_conf_train + (1.0 - smooth_train)),
            "agp": 0.5 * (semantic_conf_train + (1.0 - proto_mix_train)),
        },
        topk=20,
        temperature=0.07,
    )
    moe_test, moe_weights_test, moe_meta_test = mixture_of_semantic_experts(
        images_test,
        {
            "kec": caches["knowledge_test"],
            "ag": diffuse_test,
            "agp": proto_test,
        },
        {
            "kec": knowledge_conf_test,
            "ag": 0.5 * (semantic_conf_test + (1.0 - smooth_test)),
            "agp": 0.5 * (semantic_conf_test + (1.0 - proto_mix_test)),
        },
        topk=20,
        temperature=0.07,
    )
    moep_train = preserve_semantic_core(
        moe_train,
        caches["knowledge_train"],
        np.clip(0.10 + 0.25 * (1.0 - knowledge_conf_train), 0.0, 0.40),
    )
    moep_test = preserve_semantic_core(
        moe_test,
        caches["knowledge_test"],
        np.clip(0.10 + 0.25 * (1.0 - knowledge_conf_test), 0.0, 0.40),
    )
    moe_align_train = compute_alignment_score(images_train, moe_train)
    moe_align_test = compute_alignment_score(images_test, moe_test)
    moep_align_train = compute_alignment_score(images_train, moep_train)
    moep_align_test = compute_alignment_score(images_test, moep_test)
    moe_gate_train = build_confidence_guidance(semantic_conf_train, moe_align_train, beta_max=0.6)
    moe_gate_test = build_confidence_guidance(semantic_conf_test, moe_align_test, beta_max=0.6)
    moep_gate_train = build_confidence_guidance(semantic_conf_train, moep_align_train, beta_max=0.6)
    moep_gate_test = build_confidence_guidance(semantic_conf_test, moep_align_test, beta_max=0.6)

    method_payload = {
        "TAC": {
            "train": caches["tac_retrieval_train"],
            "test": caches["tac_retrieval_test"],
            "meta": {"train_image_guidance_options": [0.0]},
        },
        "KEC": {
            "train": caches["knowledge_train"],
            "test": caches["knowledge_test"],
            "meta": {"train_image_guidance_options": [0.5]},
        },
        "SAGE-A": {
            "train": adaptive_train,
            "test": adaptive_test,
            "meta": {
                "train_image_guidance_options": [0.5],
                "mean_gate_train": float(gate_train.mean()),
                "mean_gate_test": float(gate_test.mean()),
            },
        },
        "SAGE-AG": {
            "train": diffuse_train,
            "test": diffuse_test,
            "meta": {
                "train_image_guidance_options": [0.0, 0.5],
                "mean_gate_train": float(gate_train.mean()),
                "mean_gate_test": float(gate_test.mean()),
                "mean_smooth_train": float(smooth_train.mean()),
                "mean_smooth_test": float(smooth_test.mean()),
            },
        },
        "SAGE-AGP": {
            "train": proto_train,
            "test": proto_test,
            "meta": {
                "train_image_guidance_options": [0.0, 0.25, 0.5],
                "mean_gate_train": float(gate_train.mean()),
                "mean_gate_test": float(gate_test.mean()),
                "mean_smooth_train": float(smooth_train.mean()),
                "mean_smooth_test": float(smooth_test.mean()),
                "mean_proto_mix_train": float(proto_mix_train.mean()),
                "mean_proto_mix_test": float(proto_mix_test.mean()),
                "anchor_num_train": int(anchor_num_train),
                "anchor_num_test": int(anchor_num_test),
            },
        },
        "SAGE-AGC": {
            "train": diffuse_train,
            "test": diffuse_test,
            "meta": {
                "train_variants": [
                    {
                        "variant_suffix": "adaptive",
                        "image_guidance_train": gated_ag_train,
                        "image_guidance_test": gated_ag_test,
                    }
                ],
                "mean_adaptive_gate_train": float(gated_ag_train.mean()),
                "mean_adaptive_gate_test": float(gated_ag_test.mean()),
            },
        },
        "SAGE-AGPC": {
            "train": proto_train,
            "test": proto_test,
            "meta": {
                "train_variants": [
                    {
                        "variant_suffix": "adaptive",
                        "image_guidance_train": gated_agp_train,
                        "image_guidance_test": gated_agp_test,
                    }
                ],
                "mean_adaptive_gate_train": float(gated_agp_train.mean()),
                "mean_adaptive_gate_test": float(gated_agp_test.mean()),
            },
        },
        "SAGE-HYB": {
            "train": hybrid_train,
            "test": hybrid_test,
            "meta": {
                "train_variants": [
                    {
                        "variant_suffix": "adaptive",
                        "image_guidance_train": hybrid_gate_train,
                        "image_guidance_test": hybrid_gate_test,
                    }
                ],
                "mean_adaptive_gate_train": float(hybrid_gate_train.mean()),
                "mean_adaptive_gate_test": float(hybrid_gate_test.mean()),
                "mean_hybrid_alpha_train": float(hybrid_alpha_train.mean()),
                "mean_hybrid_alpha_test": float(hybrid_alpha_test.mean()),
            },
        },
        "SAGE-MOE": {
            "train": moe_train,
            "test": moe_test,
            "meta": {
                "train_variants": [
                    {
                        "variant_suffix": "adaptive",
                        "image_guidance_train": moe_gate_train,
                        "image_guidance_test": moe_gate_test,
                    }
                ],
                "mean_adaptive_gate_train": float(moe_gate_train.mean()),
                "mean_adaptive_gate_test": float(moe_gate_test.mean()),
                "moe_train": moe_meta_train,
                "moe_test": moe_meta_test,
            },
        },
        "SAGE-MOEP": {
            "train": moep_train,
            "test": moep_test,
            "meta": {
                "train_variants": [
                    {
                        "variant_suffix": "adaptive",
                        "image_guidance_train": moep_gate_train,
                        "image_guidance_test": moep_gate_test,
                    }
                ],
                "mean_adaptive_gate_train": float(moep_gate_train.mean()),
                "mean_adaptive_gate_test": float(moep_gate_test.mean()),
                "moe_train": moe_meta_train,
                "moe_test": moe_meta_test,
            },
        },
    }

    extra = {
        "knowledge_bank_size": len(knowledge_bank),
        "text_proto": text_proto,
        "confidence": conf_views,
    }
    return method_payload, extra


def run_concat_eval(images_test: np.ndarray, semantic_test: np.ndarray, labels_test: np.ndarray, cluster_num: int):
    metrics, preds, fused = kec.run_concat_kmeans_eval(images_test, semantic_test, labels_test, cluster_num)
    return metrics, preds, fused


def run_train_head_eval(
    dataset: str,
    semantic_train: np.ndarray,
    semantic_test: np.ndarray,
    images_train: np.ndarray,
    images_test: np.ndarray,
    labels_test: np.ndarray,
    cluster_num: int,
    seed: int,
    image_guidance_weight: float = 0.5,
    image_guidance_weight_test=None,
):
    if image_guidance_weight_test is None:
        image_guidance_weight_test = image_guidance_weight

    if np.isscalar(image_guidance_weight) and float(image_guidance_weight) == 0.0:
        guided_train = images_train
    else:
        guided_train = kec.normalize_np(images_train + np.asarray(image_guidance_weight, dtype=np.float32).reshape(-1, 1) * semantic_train)

    if np.isscalar(image_guidance_weight_test) and float(image_guidance_weight_test) == 0.0:
        guided_test = images_test
    else:
        guided_test = kec.normalize_np(images_test + np.asarray(image_guidance_weight_test, dtype=np.float32).reshape(-1, 1) * semantic_test)
    return kec.run_train_head_eval(
        dataset=dataset,
        text_train=semantic_train,
        image_train=guided_train,
        image_test=guided_test,
        labels_test=labels_test,
        cluster_num=cluster_num,
        seed=seed,
    )
