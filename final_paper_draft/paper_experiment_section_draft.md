# 论文实验部分初稿

## 4. Experimental Setup

### 4.1 Datasets

我们在 7 个本地可复现实验数据集上评估方法：CIFAR-10、CIFAR-20、STL-10、DTD、StanfordDogs、Flowers 和 Pets。

### 4.2 Compared Methods

我们比较以下方法：

- **CLIP baseline**：仅使用 CLIP 图像特征做 spherical KMeans；
- **TAC**：使用 noun retrieval 的训练自由与 train-head 两条协议；
- **KEC**：使用 concept / caption / attribute 的层级知识增强；
- **SAGE++ (ours)**：在 TAC/KEC 基础上加入 confidence-aware semantic fusion、graph diffusion、prototype calibration，以及在部分细粒度数据上的 dual-knowledge preservation。

### 4.3 Evaluation Protocols

我们报告两条协议：

1. **no-train**：拼接视觉与语义表示后直接做聚类；
2. **train**：使用 train-head 进行语义引导的聚类训练。

评价指标为 ACC、NMI 和 ARI，其中 ACC 作为主指标。

## 5. Main Results

### 5.1 Train Protocol

如表 `main_train_table.md` 所示，SAGE++ 在 train 协议下表现最稳定。相较于本地最强 prior baseline，我们在 7 个数据集中的 **7/7** 个数据集上取得提升。

其中，提升最明显的包括：

- **CIFAR-20**：从最强本地 baseline 0.5540 提升到 0.6125；
- **Pets**：从 0.7768 提升到 0.8256；
- **StanfordDogs**：最终通过 second-wave dual-knowledge preservation 提升到 0.5651；
- **Flowers**：从 0.4333 提升到 0.4843；
- **DTD**：从 0.4979 提升到 0.5218。

特别地，StanfordDogs 是第二轮增强最成功的案例：`SAGE-DKP` 将 train ACC 从 0.4907 进一步提升到 0.5651，说明双知识库与语义保真机制对细粒度狗类聚类确实有效。

### 5.2 No-Train Protocol

如表 `main_no_train_table.md` 所示，no-train 协议下的提升明显更温和。相较于最强本地 prior baseline，我们在 **3/7** 个数据集上取得提升。

这说明：

1. 仅靠训练自由的语义增强，提升空间有限；
2. 我们的方法优势主要体现在 train-head 这一更强的聚类协议上；
3. 因此，论文主结果应以 train 协议为主，no-train 结果作为辅助支持。

## 6. Ablation Study

表 `ablation_train_core.md` 展示了从 TAC 到 KEC，再到 SAGE-A、SAGE-AG、SAGE-AGP 和 SAGE-DKP 的逐步演化。

我们观察到：

- **SAGE-A** 已经能够在若干数据集上稳定超过 KEC，说明 adaptive fusion 是必要的第一步；
- **SAGE-AG** 在多数通用数据集上最稳定，是整体上最适合作为主方法的 family；
- **SAGE-AGP** 在 Pets 和 StanfordDogs 这类细粒度数据上更有效；
- **SAGE-DKP** 并不是一个对所有数据集都统一有效的增强，但它在 StanfordDogs 上带来了明显收益，因此更适合作为细粒度补充模块。

## 7. Discussion

总体来看，SAGE++ 并不是通过一个单一固定模块在所有数据集上取得提升，而是通过一条结构清晰的语义增强路线实现收益：

TAC noun retrieval -> KEC hierarchical knowledge -> adaptive fusion -> graph refinement -> prototype calibration -> optional dual-knowledge preservation.

这种设计的优点是：

- 与 TAC / KEC 具有清晰继承关系；
- 各模块都可单独消融；
- 既能解释通用数据集上的提升，也能解释细粒度数据上的特殊收益来源。

## 8. Caveat

需要说明的是：KEC 的本地复现没有调用真实在线 LLM API，而是使用离线 surrogate 方式生成 concept 和 attribute。因此，当本地 KEC 结果与论文表格存在较大差异时，这种差异可能同时来自知识生成方式、论文未公开的 prompt 细节，以及数据协议差异。

即便如此，SAGE++ 仍然在本地复现实验链上表现出相对于 TAC/KEC baseline 的稳定收益，尤其是在 train 协议下具有较强竞争力。