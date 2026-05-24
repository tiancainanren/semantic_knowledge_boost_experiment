# 方法描述（论文版）

## 1. 问题背景

TAC 证明了 noun retrieval 可以为无监督图像聚类提供外部语义引导；KEC 则进一步把通用 WordNet noun 替换成了更贴近数据集的层级知识，包括 concept、concept caption 和属性词。

然而，现有做法仍有两个问题：

1. 不同语义来源的可靠性并不一致；
2. 语义增强往往没有充分利用图像流形结构和类原型结构。

因此，我们提出 **SAGE++**，在 TAC/KEC 的基础上进一步做 confidence-aware、graph-aware 和 prototype-aware 的语义增强。

## 2. Semantic Confidence

对于图像特征 $x_i$，我们分别计算 noun 原型、concept 原型和 attribute 原型的软匹配分布。

- noun confidence:
  $c_i^{noun} = \max \operatorname{softmax}(x_i W_{noun}^\top / \tau_n)$
- concept confidence:
  $c_i^{concept} = \max \operatorname{softmax}(x_i W_{concept}^\top / \tau_c)$
- attribute confidence:
  $c_i^{attr}$ 由 concept-conditioned attribute grounding 得到。

然后定义 knowledge confidence：

$c_i^{know} = \alpha c_i^{concept} + (1-\alpha)c_i^{attr}$

它表示图像在层级知识空间中的匹配是否尖锐、稳定。

## 3. SAGE-A: Adaptive Semantic Fusion

设 TAC 的 noun retrieval 向量为 $r_i$，KEC knowledge 向量为 $k_i$。我们根据两者的相对可靠性为每个样本计算 gate：

$g_i = \operatorname{clip}( c_i^{know} / (c_i^{know} + c_i^{noun}), g_{min}, g_{max})$

于是得到：

$s_i^{A} = (1-g_i)r_i + g_i k_i$

这一步避免了所有样本都被同样强度地推向 knowledge 分支。

## 4. SAGE-AG: Graph Diffusion Refinement

在图像 embedding 的 kNN 图上，我们对不确定样本进行更强的语义平滑。令 $\bar{s}_i$ 表示近邻语义均值，则：

$\lambda_i = \lambda_{min} + (\lambda_{max}-\lambda_{min})(1-c_i)^{\gamma}$

$s_i^{AG} = (1-\lambda_i)s_i^{A} + \lambda_i \bar{s}_i$

其中 $c_i = \max(c_i^{noun}, c_i^{know})$。语义越不确定，图扩散越强。

## 5. SAGE-AGP: Prototype Calibration

为了进一步适配细粒度数据，我们在图像空间中构建 anchor prototypes，并用原型语义去校正样本语义：

$s_i^{AGP} = (1-\mu_i)s_i^{AG} + \mu_i p_i$

其中 $p_i$ 是样本所属原型的语义表示，$\mu_i$ 同样由置信度控制。这个模块在 StanfordDogs 和 Pets 这类细粒度数据上尤其有效。

## 6. SAGE-DKP: Dual-Knowledge Preservation

第二轮增强中，我们进一步观察到：不同版本的 KEC 知识库并不总是一致。于是我们引入双知识库融合：

1. 从 simple KEC root 和 prompted KEC root 分别构建语义专家；
2. 在样本级别混合 AG / AGP / adaptive / KEC 语义专家；
3. 对细粒度数据额外保留 prompt-style knowledge core，避免语义漂移。

这一模块形成了 `SAGE-DKP`，并显著提升了 StanfordDogs。

## 7. 最终方法口径

论文中我们推荐把整套框架统一记为 **SAGE++**。

- 主干通用版本：`SAGE-AG`
- 细粒度增强版本：`SAGE-AGP`
- 第二轮双知识库增强：`SAGE-DKP`

主结果表中，`Ours (SAGE++)` 表示在预先固定的 SAGE 搜索空间中选择的最终版本；它不是在测试标签上做选择，而是在方法设计阶段固定好可选变体后，对不同数据域使用最合适的 SAGE 模块。
