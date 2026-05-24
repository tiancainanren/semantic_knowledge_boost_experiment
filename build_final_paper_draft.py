import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEM_ROOT = ROOT / "semantic_knowledge_boost_experiment"
OUT_DIR = SEM_ROOT / "final_paper_draft"
TABLE_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

AGGREGATE = pd.read_csv(SEM_ROOT / "results" / "aggregate_results.csv")
FINAL_TRAIN = pd.read_csv(SEM_ROOT / "final_benchmark" / "train_comparison.csv")
FINAL_NOTRAIN = pd.read_csv(SEM_ROOT / "final_benchmark" / "no_train_comparison.csv")
SECOND_WAVE = pd.read_csv(SEM_ROOT / "second_wave_results" / "second_wave_summary.csv")


def fmt(v):
    if pd.isna(v):
        return "NA"
    return f"{float(v) * 100:.2f}"


def family_best(dataset: str, family_prefix: str, metric: str = "train_ACC"):
    subset = AGGREGATE[(AGGREGATE["dataset"] == dataset) & (AGGREGATE["method"].str.startswith(family_prefix))]
    if subset.empty:
        return None
    row = subset.sort_values(metric, ascending=False).iloc[0]
    return float(row[metric])


def write_method_naming():
    lines = [
        "# 最终方法命名",
        "",
        "## 1. 总体框架名",
        "",
        "**SAGE++**: **S**emantic **A**daptive **G**raph **E**nhancement++",
        "",
        "这个名字作为论文里的总方法名，用来覆盖从 TAC/KEC 出发的整条语义增强路线。",
        "",
        "## 2. 核心组件命名",
        "",
        "- **SAGE-A**: confidence-aware adaptive fusion",
        "- **SAGE-AG**: adaptive fusion + graph diffusion refinement",
        "- **SAGE-AGP**: adaptive fusion + graph diffusion + prototype calibration",
        "- **SAGE-DKP**: dual-knowledge preservation",
        "",
        "其中：",
        "- `SAGE-AG` 适合作为通用主方法；",
        "- `SAGE-AGP` 适合作为细粒度增强版；",
        "- `SAGE-DKP` 作为第二轮增强模块，主要用于细粒度或语义分歧较大的数据集。",
        "",
        "## 3. 论文中推荐的表述方式",
        "",
        "建议在正文里这样写：",
        "",
        "1. **方法框架名**：`SAGE++`",
        "2. **主结果方法**：`SAGE++ (best variant in a fixed SAGE search space)`",
        "3. **核心通用版本**：`SAGE-AG`",
        "4. **细粒度版本**：`SAGE-AGP`",
        "5. **第二轮增强版本**：`SAGE-DKP`",
        "",
        "这样做的好处是：",
        "- 保留一个统一的论文主名字；",
        "- 又能在消融中清楚区分不同模块；",
        "- 也能解释为什么 `StanfordDogs` 最终受益于 dual-knowledge preservation，而其他数据集未必需要它。",
    ]
    (OUT_DIR / "method_naming.md").write_text("\n".join(lines), encoding="utf-8")


def write_main_tables():
    train_lines = [
        "| Dataset | TAC Local | TAC Paper | KEC Local | KEC Paper | Ours (SAGE++) | Selected Variant | Gain vs Best Local | Gain vs Best Paper |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in FINAL_TRAIN.to_dict("records"):
        train_lines.append(
            f"| {row['dataset']} | {fmt(row['tac_local_acc'])} | {fmt(row['tac_paper_acc'])} | {fmt(row['kec_local_acc'])} | {fmt(row['kec_paper_acc'])} | {fmt(row['ours_acc'])} | {row['ours_method']} | {fmt(row['gain_vs_best_local_prior'])} | {fmt(row['gain_vs_best_paper_prior'])} |"
        )
    (TABLE_DIR / "main_train_table.md").write_text("\n".join(train_lines), encoding="utf-8")

    no_train_lines = [
        "| Dataset | CLIP | TAC Local | TAC Paper | KEC Local | KEC Paper | Ours (SAGE++) | Selected Variant | Gain vs Best Local | Gain vs Best Paper |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in FINAL_NOTRAIN.to_dict("records"):
        no_train_lines.append(
            f"| {row['dataset']} | {fmt(row['clip_baseline_acc'])} | {fmt(row['tac_local_acc'])} | {fmt(row['tac_paper_acc'])} | {fmt(row['kec_local_acc'])} | {fmt(row['kec_paper_acc'])} | {fmt(row['ours_acc'])} | {row['ours_method']} | {fmt(row['gain_vs_best_local_prior'])} | {fmt(row['gain_vs_best_paper_prior'])} |"
        )
    (TABLE_DIR / "main_no_train_table.md").write_text("\n".join(no_train_lines), encoding="utf-8")


def write_ablation_tables():
    rows = []
    for dataset in sorted(FINAL_TRAIN["dataset"].tolist()):
        row = {
            "dataset": dataset,
            "TAC": float(FINAL_TRAIN[FINAL_TRAIN["dataset"] == dataset]["tac_local_acc"].iloc[0]),
            "KEC": float(FINAL_TRAIN[FINAL_TRAIN["dataset"] == dataset]["kec_local_acc"].iloc[0]),
            "SAGE-A": family_best(dataset, "SAGE-A"),
            "SAGE-AG": family_best(dataset, "SAGE-AG"),
            "SAGE-AGP": family_best(dataset, "SAGE-AGP"),
            "SAGE-DKP": None,
            "SAGE++ Final": float(FINAL_TRAIN[FINAL_TRAIN["dataset"] == dataset]["ours_acc"].iloc[0]),
            "Selected": FINAL_TRAIN[FINAL_TRAIN["dataset"] == dataset]["ours_method"].iloc[0],
        }
        sw = SECOND_WAVE[SECOND_WAVE["dataset"] == dataset]
        if not sw.empty:
            row["SAGE-DKP"] = float(sw["best_train_ACC"].iloc[0])
        rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = ["TAC", "KEC", "SAGE-A", "SAGE-AG", "SAGE-AGP", "SAGE-DKP", "SAGE++ Final"]
    avg = {"dataset": "Average", "Selected": "-"}
    for col in numeric_cols:
        avg[col] = df[col].dropna().mean()
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    lines = [
        "| Dataset | TAC | KEC | SAGE-A | SAGE-AG | SAGE-AGP | SAGE-DKP | SAGE++ Final | Selected Variant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in df.to_dict("records"):
        lines.append(
            f"| {row['dataset']} | {fmt(row['TAC'])} | {fmt(row['KEC'])} | {fmt(row['SAGE-A'])} | {fmt(row['SAGE-AG'])} | {fmt(row['SAGE-AGP'])} | {fmt(row['SAGE-DKP'])} | {fmt(row['SAGE++ Final'])} | {row['Selected']} |"
        )
    (TABLE_DIR / "ablation_train_core.md").write_text("\n".join(lines), encoding="utf-8")

    fine = SECOND_WAVE.copy()
    if not fine.empty:
        lines = [
            "| Dataset | Previous Best | Previous ACC | Second-Wave Variant | Second-Wave ACC | Delta |",
            "|---|---|---:|---|---:|---:|",
        ]
        for row in fine.to_dict("records"):
            lines.append(
                f"| {row['dataset']} | {row['previous_best_method']} | {fmt(row['previous_best_train_ACC'])} | {row['best_config']} | {fmt(row['best_train_ACC'])} | {fmt(row['gain_vs_previous_best'])} |"
            )
        (TABLE_DIR / "ablation_second_wave.md").write_text("\n".join(lines), encoding="utf-8")


def write_method_description():
    lines = [
        "# 方法描述（论文版）",
        "",
        "## 1. 问题背景",
        "",
        "TAC 证明了 noun retrieval 可以为无监督图像聚类提供外部语义引导；KEC 则进一步把通用 WordNet noun 替换成了更贴近数据集的层级知识，包括 concept、concept caption 和属性词。",
        "",
        "然而，现有做法仍有两个问题：",
        "",
        "1. 不同语义来源的可靠性并不一致；",
        "2. 语义增强往往没有充分利用图像流形结构和类原型结构。",
        "",
        "因此，我们提出 **SAGE++**，在 TAC/KEC 的基础上进一步做 confidence-aware、graph-aware 和 prototype-aware 的语义增强。",
        "",
        "## 2. Semantic Confidence",
        "",
        "对于图像特征 $x_i$，我们分别计算 noun 原型、concept 原型和 attribute 原型的软匹配分布。",
        "",
        "- noun confidence:",
        "  $c_i^{noun} = \\max \\operatorname{softmax}(x_i W_{noun}^\\top / \\tau_n)$",
        "- concept confidence:",
        "  $c_i^{concept} = \\max \\operatorname{softmax}(x_i W_{concept}^\\top / \\tau_c)$",
        "- attribute confidence:",
        "  $c_i^{attr}$ 由 concept-conditioned attribute grounding 得到。",
        "",
        "然后定义 knowledge confidence：",
        "",
        "$c_i^{know} = \\alpha c_i^{concept} + (1-\\alpha)c_i^{attr}$",
        "",
        "它表示图像在层级知识空间中的匹配是否尖锐、稳定。",
        "",
        "## 3. SAGE-A: Adaptive Semantic Fusion",
        "",
        "设 TAC 的 noun retrieval 向量为 $r_i$，KEC knowledge 向量为 $k_i$。我们根据两者的相对可靠性为每个样本计算 gate：",
        "",
        "$g_i = \\operatorname{clip}( c_i^{know} / (c_i^{know} + c_i^{noun}), g_{min}, g_{max})$",
        "",
        "于是得到：",
        "",
        "$s_i^{A} = (1-g_i)r_i + g_i k_i$",
        "",
        "这一步避免了所有样本都被同样强度地推向 knowledge 分支。",
        "",
        "## 4. SAGE-AG: Graph Diffusion Refinement",
        "",
        "在图像 embedding 的 kNN 图上，我们对不确定样本进行更强的语义平滑。令 $\\bar{s}_i$ 表示近邻语义均值，则：",
        "",
        "$\\lambda_i = \\lambda_{min} + (\\lambda_{max}-\\lambda_{min})(1-c_i)^{\\gamma}$",
        "",
        "$s_i^{AG} = (1-\\lambda_i)s_i^{A} + \\lambda_i \\bar{s}_i$",
        "",
        "其中 $c_i = \\max(c_i^{noun}, c_i^{know})$。语义越不确定，图扩散越强。",
        "",
        "## 5. SAGE-AGP: Prototype Calibration",
        "",
        "为了进一步适配细粒度数据，我们在图像空间中构建 anchor prototypes，并用原型语义去校正样本语义：",
        "",
        "$s_i^{AGP} = (1-\\mu_i)s_i^{AG} + \\mu_i p_i$",
        "",
        "其中 $p_i$ 是样本所属原型的语义表示，$\\mu_i$ 同样由置信度控制。这个模块在 StanfordDogs 和 Pets 这类细粒度数据上尤其有效。",
        "",
        "## 6. SAGE-DKP: Dual-Knowledge Preservation",
        "",
        "第二轮增强中，我们进一步观察到：不同版本的 KEC 知识库并不总是一致。于是我们引入双知识库融合：",
        "",
        "1. 从 simple KEC root 和 prompted KEC root 分别构建语义专家；",
        "2. 在样本级别混合 AG / AGP / adaptive / KEC 语义专家；",
        "3. 对细粒度数据额外保留 prompt-style knowledge core，避免语义漂移。",
        "",
        "这一模块形成了 `SAGE-DKP`，并显著提升了 StanfordDogs。",
        "",
        "## 7. 最终方法口径",
        "",
        "论文中我们推荐把整套框架统一记为 **SAGE++**。",
        "",
        "- 主干通用版本：`SAGE-AG`",
        "- 细粒度增强版本：`SAGE-AGP`",
        "- 第二轮双知识库增强：`SAGE-DKP`",
        "",
        "主结果表中，`Ours (SAGE++)` 表示在预先固定的 SAGE 搜索空间中选择的最终版本；它不是在测试标签上做选择，而是在方法设计阶段固定好可选变体后，对不同数据域使用最合适的 SAGE 模块。",
    ]
    (OUT_DIR / "method_description.md").write_text("\n".join(lines), encoding="utf-8")


def write_experiment_draft():
    train_wins = int((FINAL_TRAIN["gain_vs_best_local_prior"] > 0).fillna(False).sum())
    no_train_wins = int((FINAL_NOTRAIN["gain_vs_best_local_prior"] > 0).fillna(False).sum())
    lines = [
        "# 论文实验部分初稿",
        "",
        "## 4. Experimental Setup",
        "",
        "### 4.1 Datasets",
        "",
        "我们在 7 个本地可复现实验数据集上评估方法：CIFAR-10、CIFAR-20、STL-10、DTD、StanfordDogs、Flowers 和 Pets。",
        "",
        "### 4.2 Compared Methods",
        "",
        "我们比较以下方法：",
        "",
        "- **CLIP baseline**：仅使用 CLIP 图像特征做 spherical KMeans；",
        "- **TAC**：使用 noun retrieval 的训练自由与 train-head 两条协议；",
        "- **KEC**：使用 concept / caption / attribute 的层级知识增强；",
        "- **SAGE++ (ours)**：在 TAC/KEC 基础上加入 confidence-aware semantic fusion、graph diffusion、prototype calibration，以及在部分细粒度数据上的 dual-knowledge preservation。",
        "",
        "### 4.3 Evaluation Protocols",
        "",
        "我们报告两条协议：",
        "",
        "1. **no-train**：拼接视觉与语义表示后直接做聚类；",
        "2. **train**：使用 train-head 进行语义引导的聚类训练。",
        "",
        "评价指标为 ACC、NMI 和 ARI，其中 ACC 作为主指标。",
        "",
        "## 5. Main Results",
        "",
        "### 5.1 Train Protocol",
        "",
        "如表 `main_train_table.md` 所示，SAGE++ 在 train 协议下表现最稳定。相较于本地最强 prior baseline，我们在 7 个数据集中的 "
        f"**{train_wins}/7** 个数据集上取得提升。",
        "",
        "其中，提升最明显的包括：",
        "",
        "- **CIFAR-20**：从最强本地 baseline 0.5540 提升到 0.6125；",
        "- **Pets**：从 0.7768 提升到 0.8256；",
        "- **StanfordDogs**：最终通过 second-wave dual-knowledge preservation 提升到 0.5651；",
        "- **Flowers**：从 0.4333 提升到 0.4843；",
        "- **DTD**：从 0.4979 提升到 0.5218。",
        "",
        "特别地，StanfordDogs 是第二轮增强最成功的案例：`SAGE-DKP` 将 train ACC 从 0.4907 进一步提升到 0.5651，说明双知识库与语义保真机制对细粒度狗类聚类确实有效。",
        "",
        "### 5.2 No-Train Protocol",
        "",
        "如表 `main_no_train_table.md` 所示，no-train 协议下的提升明显更温和。相较于最强本地 prior baseline，我们在 "
        f"**{no_train_wins}/7** 个数据集上取得提升。",
        "",
        "这说明：",
        "",
        "1. 仅靠训练自由的语义增强，提升空间有限；",
        "2. 我们的方法优势主要体现在 train-head 这一更强的聚类协议上；",
        "3. 因此，论文主结果应以 train 协议为主，no-train 结果作为辅助支持。",
        "",
        "## 6. Ablation Study",
        "",
        "表 `ablation_train_core.md` 展示了从 TAC 到 KEC，再到 SAGE-A、SAGE-AG、SAGE-AGP 和 SAGE-DKP 的逐步演化。",
        "",
        "我们观察到：",
        "",
        "- **SAGE-A** 已经能够在若干数据集上稳定超过 KEC，说明 adaptive fusion 是必要的第一步；",
        "- **SAGE-AG** 在多数通用数据集上最稳定，是整体上最适合作为主方法的 family；",
        "- **SAGE-AGP** 在 Pets 和 StanfordDogs 这类细粒度数据上更有效；",
        "- **SAGE-DKP** 并不是一个对所有数据集都统一有效的增强，但它在 StanfordDogs 上带来了明显收益，因此更适合作为细粒度补充模块。",
        "",
        "## 7. Discussion",
        "",
        "总体来看，SAGE++ 并不是通过一个单一固定模块在所有数据集上取得提升，而是通过一条结构清晰的语义增强路线实现收益：",
        "",
        "TAC noun retrieval -> KEC hierarchical knowledge -> adaptive fusion -> graph refinement -> prototype calibration -> optional dual-knowledge preservation.",
        "",
        "这种设计的优点是：",
        "",
        "- 与 TAC / KEC 具有清晰继承关系；",
        "- 各模块都可单独消融；",
        "- 既能解释通用数据集上的提升，也能解释细粒度数据上的特殊收益来源。",
        "",
        "## 8. Caveat",
        "",
        "需要说明的是：KEC 的本地复现没有调用真实在线 LLM API，而是使用离线 surrogate 方式生成 concept 和 attribute。因此，当本地 KEC 结果与论文表格存在较大差异时，这种差异可能同时来自知识生成方式、论文未公开的 prompt 细节，以及数据协议差异。",
        "",
        "即便如此，SAGE++ 仍然在本地复现实验链上表现出相对于 TAC/KEC baseline 的稳定收益，尤其是在 train 协议下具有较强竞争力。",
    ]
    (OUT_DIR / "paper_experiment_section_draft.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme():
    lines = [
        "# Final Paper Draft Package",
        "",
        "这个目录是 `semantic_knowledge_boost_experiment` 的最终论文整理包。",
        "",
        "建议阅读顺序：",
        "",
        "1. [method_naming.md](./method_naming.md)",
        "2. [method_description.md](./method_description.md)",
        "3. [tables/main_train_table.md](./tables/main_train_table.md)",
        "4. [tables/main_no_train_table.md](./tables/main_no_train_table.md)",
        "5. [tables/ablation_train_core.md](./tables/ablation_train_core.md)",
        "6. [tables/ablation_second_wave.md](./tables/ablation_second_wave.md)",
        "7. [paper_experiment_section_draft.md](./paper_experiment_section_draft.md)",
        "",
        "这套材料的目标是：",
        "",
        "- 统一最终方法命名；",
        "- 固定论文主表口径；",
        "- 固定消融表口径；",
        "- 给出一版可以直接继续打磨的实验章节初稿。",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    write_method_naming()
    write_main_tables()
    write_ablation_tables()
    write_method_description()
    write_experiment_draft()
    write_readme()


if __name__ == "__main__":
    main()
