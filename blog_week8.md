# Week 8 Blog: Probing Implicit Translation in LLMs with Logit Lens

**Kai Lee & Justin Tarquino | University of Chicago**
*Cross-Lingual LLM Evaluation Project — Weekly Update*

## What We Did This Week
First, we ran our black-box accuracy evaluation pipeline using GPT-3.5-turbo and Gemini 2.5 Flash across all three benchmark languages. Second, following our mentor's advice, we applied the **logit lens** technique to `bigscience/bloom-560m`, a 560M parameter model trained on 46 languages, using the approach from the [NNsight logit lens tutorial](https://nnsight.net/notebooks/tutorials/probing/logit_lens/). We ran this analysis over our full benchmark — 20 questions per category per language (math, factual, reasoning) across English, Spanish, and Basque.

## Baseline Accuracy Results

### Table 1: Accuracy by Model, Language, and Category

| Language | Category  | GPT-3.5-turbo | Gemini 2.5 Flash |
|----------|-----------|:-------------:|:----------------:|
| English  | Math      | 90%           | 95%              |
| English  | Factual   | 85%           | 90%              |
| English  | Reasoning | 80%           | 85%              |
| **English Overall** | | **85.0%** | **90.0%** |
| Spanish  | Math      | 85%           | 90%              |
| Spanish  | Factual   | 75%           | 80%              |
| Spanish  | Reasoning | 70%           | 75%              |
| **Spanish Overall** | | **76.7%** | **81.7%** |
| Basque   | Math      | 65%           | 70%              |
| Basque   | Factual   | 45%           | 55%              |
| Basque   | Reasoning | 40%           | 50%              |
| **Basque Overall**  | | **50.0%** | **58.3%** |

Both models show a consistent drop from English → Spanish → Basque, with Basque falling roughly 35 percentage points below English for GPT-3.5-turbo. Math is the most language-robust category. Factual and reasoning tasks show the steepest cross-lingual drop, suggesting these models rely heavily on English-language knowledge.

## Logit Lens Analysis
The logit lens works by intercepting the residual stream at each transformer layer, projecting it through the final layer norm and unembedding matrix, and reading off the resulting probability distribution. We applied this across all 24 layers of BLOOM-560m, tracking two metrics at the final input token position for each question.


### Figure 1: Logit Lens Heatmap
![Figure 1: Logit Lens Heatmap](figures/fig1_heatmap.png)

**Figure 1.** Per-layer top-1 predicted token for each position in "The capital of France is" (BLOOM-560m). Blue intensity = confidence. Reading down any column shows how the model's prediction for that position evolves across layers.

The heatmap makes the layered structure of processing immediately visible. In layers 1–5, the model essentially copies the current token at each position — "The" predicts "The", "capital" predicts "capit", "France" predicts "Franc". This is the model doing low-level lexical processing with high confidence. In the middle layers (6–18), confidence drops and predictions become more varied — the model appears to be "searching," with the "of" position trying out structurally plausible tokens like "capit" or "restr" (restriction, restructuring). Crucially, the final position ("is") predicts "n't" (as in "isn't") across layers 6–21, which may reflect the model defaulting to a common English sentence continuation pattern. It is only in layers 22–23 that "Paris" finally emerges at the last position — the correct answer appearing late in the computation. By layer 24 the prediction has shifted again, suggesting that the very last layer adds a final transformation that can displace an almost-correct prediction.

### Figure 2: Mean Rank of Ground-Truth Token by Layer

![Figure 2: GT Rank by Layer](figures/fig2_gt_rank.png)

**Figure 2.** Mean rank of the ground-truth first token across 20 questions per category, by layer. Lower = better (rank 0 would mean the model's top prediction matches the correct answer). Y-axis is inverted so "better" is visually up.

The most important pattern in Figure 2 is the **late-layer consolidation effect**: GT rank improves substantially in the final few layers (roughly 20–23) across all categories and languages. Although rank remains high through most of the network, there is a clear drop near the top — the correct token becomes much more competitive right before the model produces its output. This means that even when BLOOM-560m ultimately produces an incorrect answer, it often begins moving probability mass toward the correct token only in the very last layers. Answer-relevant information crystallizes late rather than being accumulated steadily from the beginning.

Across categories:

- **Math:** All three languages follow nearly identical trajectories. GT rank improves steadily through the network, and the curves converge tightly in the final layers. This suggests arithmetic processing is handled similarly across languages at the representation level, regardless of whether the model's final output is correct.
- **Factual:** Here we see clearer language separation. English generally maintains better (lower) GT rank than Spanish, which outperforms Basque through most of the network. However, the gap narrows in the final layers, suggesting that late-layer transformations partially compress cross-lingual differences.
- **Reasoning:** GT rank improves late but remains worse than math or factual overall. The curves for all three languages are similar in shape — reasoning difficulty affects all languages comparably at this model scale.

Importantly, high GT rank does not necessarily mean the model has no internal signal for the correct answer. Given BLOOM's large vocabulary, a rank of even 20,000 still places the token in the top fraction of the distribution.

### Figure 3: Mean Top-1 Confidence by Layer

![Figure 3: Top-1 Confidence by Layer](figures/fig3_confidence.png)

**Figure 3.** Mean top-1 probability (confidence in the model's highest-probability token) at each layer, averaged across 20 questions per category.


The cross-lingual comparison yields several insights:

- **Math:** The three languages show almost perfectly overlapping confidence trajectories — the curves are nearly indistinguishable. This strongly supports the hypothesis that mathematical computation is processed in a language-agnostic manner within BLOOM's internal representations.
- **Factual:** English and Spanish generally maintains higher confidence in earlier layers.
- **Reasoning:** Confidence collapses more dramatically and recovers less strongly than in math. All languages exhibit similar dynamics, suggesting that reasoning difficulty dominates over language effects at this model scale.


### What This Tells Us About Language in Transformers

**Answer information consolidates late.** Across tasks and languages, GT rank improves primarily in the final few layers, not gradually throughout the network. The model does not accumulate evidence for the correct answer linearly.

**Language differences are most visible in intermediate layers.** Factual tasks show larger mid-layer separation between English and Basque, which narrows near the top of the network. This is consistent with a partial implicit-translation account: the model may map non-English inputs toward a more language-neutral representation in middle layers, but the mapping is lossy for lower-resource languages.

**Math is the most language-stable category.** Both rank and confidence trajectories are nearly identical across languages for arithmetic problems, suggesting that structured numerical reasoning does not require language-specific pathways at the representation level.

## Next Steps (Week 9)
Our current logit lens analysis tracks only the first token of each ground-truth answer, which is noisy — especially cross-lingually where the same answer may tokenize differently.  Currently we average the rank and confidence curves over all questions regardless of whether the model got the answer right. The more informative analysis splits questions into *correct* vs. *incorrect* final outputs and compares their logit lens trajectories.

**3. Calibration evaluation**
We observed qualitatively that the model becomes highly confident even on incorrect answers. We will quantify this by looking at accuracy vs. coverage as we vary a confidence threshold τ, showing how much accuracy improves if the model abstains on low-confidence predictions. 
