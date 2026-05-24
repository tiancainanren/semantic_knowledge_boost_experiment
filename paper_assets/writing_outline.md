# Writing Outline

## 1. Working Title Options

1. Confidence-Aware Semantic Knowledge Boosting for Unsupervised Image Clustering
2. From Text Retrieval to Semantic Calibration: Enhancing Unsupervised Image Clustering with Concepts, Attributes, and Prototype-Aware Smoothing
3. Geometry-Aware Semantic Knowledge Refinement for Training-Free and Train-Head Image Clustering

## 2. Core Story

- TAC shows that noun retrieval helps clustering.
- KEC shows that dataset-specific concept and attribute knowledge improves semantic quality.
- Our next step is to avoid treating all semantic cues equally.
- We make semantics confidence-aware, image-manifold-aware, and optionally prototype-calibrated.
- This yields stronger train-head clustering results on most datasets, with especially clear gains on CIFAR-20, Flowers, Pets, StanfordDogs, and DTD.

## 3. Main Contributions

1. A confidence-aware semantic fusion mechanism that adaptively combines TAC noun retrieval and KEC knowledge semantics.
2. A geometry-aware semantic refinement step that diffuses uncertain semantics on the image kNN graph.
3. A prototype calibration module that further improves fine-grained datasets by aligning semantics with image-derived anchor groups.
4. A broad cross-dataset study showing that the semantic refinement line outperforms TAC on 7 usable datasets, with mean best-variant gain of 3.45 percentage points in train-head ACC.

## 4. Recommended Paper Structure

### 4.1 Introduction
- Motivate why text-enhanced clustering works but is bottlenecked by noisy or uneven semantic relevance.
- Explain the TAC -> KEC -> semantic refinement progression.
- Preview the key result: the main benefit appears in the train-head setting rather than only in training-free concat.

### 4.2 Related Work
- Unsupervised image clustering
- Vision-language clustering
- Knowledge-enhanced semantic prompting / attribute augmentation
- Graph smoothing / prototype calibration

### 4.3 Method
- TAC baseline recap
- KEC baseline recap
- SAGE-A: adaptive semantic fusion
- SAGE-AG: graph diffusion refinement
- SAGE-AGP: prototype calibration
- Optional exploratory variants: AGC / AGPC / HYB

### 4.4 Experimental Setup
- Datasets: CIFAR-10, CIFAR-20, STL-10, DTD, StanfordDogs, Flowers, Pets
- Metrics: ACC / NMI / ARI
- Two evaluation tracks: concat-kmeans and train-head

### 4.5 Main Results
- Use the train-head main table as the primary quantitative result.
- Keep concat-kmeans as a secondary table to show that the gains are smaller but directionally consistent on several datasets.

### 4.6 Ablation
- TAC -> KEC -> SAGE-A -> SAGE-AG -> SAGE-AGP
- Fixed image-guidance weights vs adaptive guidance
- Fine-grained vs coarse-grained behavior

### 4.7 Analysis
- Why SAGE-AG is the most stable family overall
- Why SAGE-AGP helps fine-grained datasets more
- Why adaptive image injection is not automatically beneficial

## 5. Recommended Result Framing

- Main method for the paper body: `SAGE-AG`
- Fine-grained enhancement variant: `SAGE-AGP`
- Exploratory variants (`AGC`, `AGPC`, `HYB`) should stay in ablation or supplementary analysis.

## 6. Caveat to State Clearly

- The current main table uses the best proposed variant per dataset.
- For a stricter single-method claim, the paper should separately report fixed-family results, where `SAGE-AG` is the strongest overall family and `SAGE-AGP` is the strongest fine-grained specialization.