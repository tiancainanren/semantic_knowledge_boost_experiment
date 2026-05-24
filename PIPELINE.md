# Semantic Knowledge Boost Pipeline

## 1. Goal

This experiment line extends the TAC and KEC direction without replacing the original backbone:

- `TAC` introduces noun-based retrieval semantics.
- `KEC` replaces plain WordNet nouns with dataset-specific concept and attribute knowledge.
- This stage asks a more specific question:
  can we make the semantic branch *more reliable per sample* and *more consistent with image geometry* before fusing it into clustering?

The answer implemented here is a small method family we can describe as:

- `SAGE-A`: adaptive semantic fusion
- `SAGE-AG`: adaptive fusion + graph diffusion
- `SAGE-AGP`: adaptive fusion + graph diffusion + prototype calibration
- `SAGE-AGC`: adaptive graph semantic + confidence-guided image injection
- `SAGE-AGPC`: adaptive graph+prototype semantic + confidence-guided image injection
- `SAGE-HYB`: hybrid semantic branch that switches between graph and prototype views

## 2. Inputs

For each dataset we reuse:

- image embeddings from the original TAC pipeline
- TAC noun-retrieval embeddings
- KEC knowledge embeddings
- KEC `knowledge_bank.json`

Those are loaded from:

- `data/{dataset}_image_embedding_train.npy`
- `data/{dataset}_image_embedding_test.npy`
- `hierarchical_textual_knowledge_experiment/results/{dataset}/artifacts/`
- `hierarchical_textual_knowledge_experiment/results/{dataset}/knowledge_bank.json`

## 3. Detailed Pipeline

### Step A. Build semantic prototypes from the KEC knowledge bank

For each class-level concept entry we encode:

- `concept`
- `concept_caption`
- unary attributes
- binary attributes

Then we construct:

- concept prototype
- unary attribute prototype
- binary attribute prototype
- final attribute prototype

Code:
- [build_text_prototypes](./src/methods.py)

### Step B. Compute multi-view semantic responses for every image

For each image embedding we compute:

1. concept-view response
2. attribute-view response
3. noun confidence from the TAC noun bank
4. concept confidence
5. attribute confidence

The idea is to know not only *what semantic feature we get*, but also *how trustworthy it looks*.

#### What exactly is "semantic confidence"?

This is an important point.
In this project, "semantic confidence" is **not** a ground-truth probability and **not** a calibration target learned from labels.
It is a purely unsupervised score derived from how *peaked* the image-to-text matching distribution is.

In other words:

- if an image clearly matches a small subset of text prototypes, confidence should be high
- if an image matches many text prototypes similarly, confidence should be low

So confidence here means:

> how concentrated the semantic assignment looks in CLIP-style similarity space

not:

> how certain the model is with respect to the true class label

#### 1. Noun confidence

For image feature `x_i` and noun prototype bank `{w_j}`, we compute:

`p_noun(j | i) = softmax( <x_i, w_j> / tau_noun )`

Then the noun confidence is:

`noun_conf_i = max_j p_noun(j | i)`

Interpretation:

- if one noun dominates, `noun_conf_i` is high
- if many nouns are equally plausible, `noun_conf_i` is low

This matches the implementation in `_batched_attention`, where we keep the max softmax weight as confidence.

#### 2. Concept confidence

For concept prototypes `{c_q}`, we compute:

`p_concept(q | i) = softmax( <x_i, c_q> / tau_concept )`

and define:

`concept_conf_i = max_q p_concept(q | i)`

This says how confidently the image can be attached to one representative concept.

#### 3. Attribute confidence

Attributes are not matched independently in the same way as nouns.
Instead, the code first computes concept weights, then grounds attribute prototypes through those concept weights.

Concretely, attribute grounding uses:

`p_concept(q | i) = softmax( <x_i, c_q> / tau_ground )`

and then builds an attribute-aware semantic view by weighting grounded attribute responses with these concept weights.

So in the current implementation, the attribute confidence is:

`attr_conf_i = max_q p_concept(q | i)`

This means:

- attribute confidence measures how stably the image can first be attached to a concept,
- because the attribute branch is concept-conditioned.

So it is better to read `attr_conf` as:

> confidence of the concept-conditioned attribute grounding

rather than:

> confidence of one isolated attribute token

#### 4. Knowledge confidence

The concept branch and attribute branch are combined into one overall confidence for the KEC-style semantic branch:

`knowledge_conf_i = alpha * concept_conf_i + (1 - alpha) * attr_conf_i`

In the default implementation:

`alpha = 0.6`

so:

`knowledge_conf_i = 0.6 * concept_conf_i + 0.4 * attr_conf_i`

Why this form:

- concept confidence is slightly more stable and global
- attribute confidence is more discriminative but also noisier
- the weighted sum balances semantic stability and fine-grained detail

#### 5. Semantic confidence

Later, when we need a single confidence for downstream guidance, we use:

`semantic_conf_i = max(noun_conf_i, knowledge_conf_i)`

This is a conservative choice:

- if either TAC noun semantics or KEC knowledge semantics is highly reliable, we allow the sample to benefit from that strong signal
- if both are weak, the sample is treated as uncertain

#### 6. What confidence is used for

These confidence values are not just diagnostic.
They directly control how much semantic information we trust in each module:

- `SAGE-A`: confidence decides noun-vs-knowledge fusion gate
- `SAGE-AG`: confidence decides graph smoothing strength
- `SAGE-AGP`: confidence decides prototype calibration strength
- `SAGE-AGC / AGPC`: confidence helps decide semantic-to-image injection strength

So confidence is the central control variable of the whole pipeline.

Code:
- [compute_semantic_confidence_views](./src/methods.py)

### Step C. Adaptive fusion: TAC retrieval vs KEC raw knowledge

Instead of hard-switching to TAC or hard-switching to KEC, we estimate a gate:

`gate = knowledge_conf / (knowledge_conf + noun_conf)`

and then clamp it into a stable interval:

`gate = clip(gate, gate_min, gate_max)`

with defaults:

- `gate_min = 0.15`
- `gate_max = 0.85`

Then per sample:

`semantic = (1 - gate) * tac_retrieval + gate * kec_raw`

This is `SAGE-A`.

Why this matters:
- if the noun branch is already confident, we keep more TAC signal
- if the knowledge branch is more confident, we trust concept/attribute semantics more
- clipping prevents extremely hard switching caused by noisy confidence spikes

Code:
- [adaptive_fusion](./src/methods.py)

### Step D. Graph diffusion on the image manifold

We build a kNN graph in image-embedding space.
For each sample we average the semantic features of nearby image neighbors and use a confidence-dependent smoothing weight:

`semantic_graph = (1 - alpha) * semantic + alpha * neighbor_mean`

Here `alpha` is not fixed.
It is computed from semantic uncertainty:

`certainty_i = max(noun_conf_i, knowledge_conf_i)`

`alpha_i = alpha_min + (alpha_max - alpha_min) * (1 - certainty_i)^p`

So:

- high certainty -> small `alpha_i`
- low certainty -> large `alpha_i`

In plain language:

- if the semantic branch already looks reliable, keep it mostly unchanged
- if the semantic branch looks ambiguous, borrow more information from image-space neighbors

This gives `SAGE-AG`.

Intuition:
- semantics should respect local image geometry
- uncertain samples benefit more from neighborhood correction

Code:
- [graph_diffuse](./src/methods.py)

### Step E. Prototype calibration

We cluster image features into anchor-style prototypes, then use each sample's assigned prototype to calibrate the semantic representation:

`semantic_proto = (1 - beta) * semantic_graph + beta * prototype_semantic`

Again `beta` is confidence-dependent:

`beta_i = beta_min + (beta_max - beta_min) * (1 - certainty_i)`

So prototype calibration is stronger when the semantic branch is less certain.

This gives `SAGE-AGP`.

Why it helps:
- graph diffusion is local
- prototype calibration adds a coarser global prior
- this is especially useful on fine-grained datasets where local noise is common

Code:
- [prototype_calibrate](./src/methods.py)

### Step F. Adaptive image guidance for train-head

For train-head style clustering, we also explored injecting semantic information back into the image branch:

`guided_image = normalize(image + gamma * semantic)`

Here `gamma` can be:

- a fixed scalar for the whole dataset
- or a sample-wise value

For adaptive variants we define:

`gamma_i = gamma_max * semantic_conf_i * alignment_i`

where

- `semantic_conf_i = max(noun_conf_i, knowledge_conf_i)`
- `alignment_i = clip((<x_i, s_i> + 1) / 2, 0, 1)`

This means semantic injection becomes strong only when:

1. the semantic branch itself is confident, and
2. the semantic vector is already aligned with the image vector

This was designed to avoid over-injecting noisy semantic corrections into the image branch.

We did this in two ways:

1. fixed `gamma`:
   - `0.0`, `0.25`, `0.5`
2. sample-wise adaptive `gamma`:
   - based on semantic confidence
   - based on image-semantic alignment

This produced:

- `SAGE-AGC`
- `SAGE-AGPC`

Code:
- [build_confidence_guidance](./src/methods.py)
- [run_train_head_eval](./src/methods.py)

### Step G. Hybrid semantic branch

We also tried a hybrid semantic branch that mixes:

- graph-diffused semantics
- prototype-calibrated semantics

per sample, using image-semantic alignment as a soft selector.

Concretely, we compute:

- `sim_AG_i = <x_i, s_i^AG>`
- `sim_AGP_i = <x_i, s_i^AGP>`

then apply a two-way softmax:

`[w_AG_i, w_AGP_i] = softmax([sim_AG_i, sim_AGP_i] / tau_h)`

and form:

`s_i^HYB = w_AG_i * s_i^AG + w_AGP_i * s_i^AGP`

So the hybrid branch lets each image decide whether local graph smoothing or prototype calibration looks more compatible with its visual representation.

This is `SAGE-HYB`.

Code:
- [hybridize_semantics](./src/methods.py)

## 4. Evaluation Protocol

Each method family is evaluated in two ways:

1. `concat_kmeans`
   - direct clustering on fused image + semantic features
2. `train_head`
   - reusing the KEC/TAC train-head evaluation path

For every dataset we save:

- per-method predictions
- per-method metrics
- dataset-level summary
- aggregate summary across all datasets

Driver:
- [run_experiment.py](./run_experiment.py)

## 5. What We Actually Explored

This round was not just one method.
We actually ran the following progression:

1. `TAC`
2. `KEC`
3. `SAGE-A`
4. `SAGE-AG`
5. `SAGE-AGP`
6. `SAGE-AGC`
7. `SAGE-AGPC`
8. `SAGE-HYB`

And for train-head we also explored different semantic injection strengths.

## 6. Main Findings

### Training-free line

The `concat_kmeans` line improves, but usually moderately.

This is expected:
- we are improving the semantic branch
- but we are still not learning a new feature space end to end

### Train-head line

The strongest gains come from using the improved semantic branch inside the train-head pipeline.

Most consistent family:
- `SAGE-AG`

Best family for fine-grained datasets:
- `SAGE-AGP`

### Adaptive image injection is not automatically better

The adaptive `AGC / AGPC / HYB` variants are useful explorations, but they did not become the dominant best method across datasets.

That is still a useful result:
- the big gain seems to come more from **better semantic construction**
- not from making the image-guidance mechanism more elaborate

## 7. Final Result Files

The latest full results are summarized in:

- [aggregate_results.csv](./results/aggregate_results.csv)
- [best_by_dataset.csv](./results/best_by_dataset.csv)
- [final_comparison.md](./results/final_comparison.md)

Per-dataset details live under:

- `semantic_knowledge_boost_experiment/results/{dataset}/`

## 8. Practical Takeaway

If we want a single main story for writing:

1. TAC proves text retrieval helps clustering.
2. KEC proves concept/attribute knowledge is better than plain noun lists.
3. Our work shows semantic knowledge should be:
   - confidence-aware
   - geometry-aware
   - optionally prototype-calibrated
4. This yields larger and more stable gains, especially in the train-head regime.

## 9. One-Sentence Definition You Can Reuse

If you need a compact sentence for a report or paper, the cleanest definition is:

> Semantic confidence is the maximum soft assignment probability produced when an image feature is matched to a semantic prototype bank, and it is used as an unsupervised reliability signal to control semantic fusion, graph smoothing, prototype calibration, and image-branch guidance.
