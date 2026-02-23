# Week 8 Blog: Probing Implicit Translation in LLMs with Logit Lens

**Kai Lee & Justin Tarquino | University of Chicago**
*Cross-Lingual LLM Evaluation Project — Weekly Update*

## What We Did This Week

This week we 1. Performed black-box accuracy evaluation toward mechanistic interpretability. 2. Following our mentors advice we we applied the **logit lens** technique to `bigscience/bloom-560m`, a 560M parameter model trained on 46 languages, following the approach from the [NNsight logit lens tutorial](https://nnsight.net/notebooks/tutorials/probing/logit_lens/). We ran the analysis over our full benchmark — 20 questions per category per language (math, factual, reasoning) across English, Spanish, and Basque. We also ran our baseline black-box evaluation using GPT-3.5-turbo and Gemini 2.5 Flash.

---

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

Both models show a consistent drop from English → Spanish → Basque, with Basque falling roughly 35 percentage points below English for GPT-3.5-turbo. Math is the most language-robust category. Factual and reasoning tasks show the steepest cross-lingual drop, suggesting these models rely heavily on English-language knowledge. Gemini 2.5 Flash outperforms GPT-3.5-turbo in all conditions but shows the same pattern of degradation.

---

The logit lens works by intercepting the residual stream at each transformer layer, projecting it through the final layer norm and unembedding matrix, and reading off the resulting probability distribution. We applied this across all 24 layers of BLOOM-560m, tracking two metrics at the final input token position for each question:


### Figure 1: Logit Lens Heatmap

![Figure 1: Logit Lens Heatmap](figures/fig1_heatmap.png)

**Figure 1.** Per-layer top-1 predicted token for each position in "The capital of France is" (BLOOM-560m). Blue intensity = confidence. Reading down any column shows how the model's prediction for that position evolves across layers.

The heatmap makes the layered structure of processing immediately visible. In layers 1–5, the model essentially copies the current token at each position — "The" predicts "The", "capital" predicts "capit", "France" predicts "Franc". This is the model doing low-level lexical processing with high confidence. In the middle layers (6–18), confidence drops and predictions become more varied — the model appears to be "searching," with the "of" position trying out structurally plausible tokens like "capit" or "restr" (restriction, restructuring). Crucially, the final position ("is") predicts "n't" (as in "isn't") across layers 6–21, which may reflect the model defaulting to a common English sentence continuation pattern. It is only in layers 22–23 that "Paris" finally emerges at the last position — the correct answer appearing late in the computation. By layer 24 the prediction has shifted again, suggesting that the very last layer adds a final transformation that can displace an almost-correct prediction.

---

### Figure 2: Mean Rank of Ground-Truth Token by Layer

![Figure 2: GT Rank by Layer](figures/fig2_gt_rank.png)

**Figure 2.** Mean rank of the ground-truth first token across 20 questions per category, by layer. Lower = better (rank 0 would mean the model's top prediction matches the correct answer). Y-axis is inverted so "better" is visually up.

The most striking feature of Figure 2 is how high the GT rank remains throughout — on the order of tens of thousands for most categories. This tells us that BLOOM-560m, at 560M parameters, largely does not place the correct answer token near the top of its distribution for these questions, even in its final layers. This is not surprising: our benchmark questions require specific factual recall and multi-step reasoning, tasks that small models handle poorly. What the figure does reveal is the *shape* of each language's trajectory:

- **Math:** All three languages track together closely, with similarly high and noisy rank curves. This confirms that numerical reasoning is processed in a language-similar way internally, even if the model cannot reliably produce the correct answer.
- **Factual:** The curves diverge more. English and Spanish show somewhat lower ranks in later layers than Basque, meaning the model at least partially narrows toward the correct answer for higher-resource languages.
- **Reasoning:** Rank curves for all three languages stay high and flat throughout all 24 layers. The model shows essentially no internal signal pointing toward the correct answer for multi-step reasoning — the problem is too hard for a model this size, regardless of language.

This is an important finding: the cross-lingual accuracy gap in Table 1 is not simply explained by the model "knowing the answer but outputting it in the wrong language." For most questions, BLOOM-560m does not identify the correct answer token internally in any language.

---

### Figure 3: Mean Top-1 Confidence by Layer

![Figure 3: Top-1 Confidence by Layer](figures/fig3_confidence.png)

**Figure 3.** Mean top-1 probability (confidence in whatever the model's best guess is) at each layer, averaged across 20 questions per category.

Figure 3 reveals a pattern that runs counter to simple intuitions about how transformers build up answers. Rather than confidence rising steadily from layer 1 to layer 24, it **starts near 1.0, collapses in the middle layers, and then recovers unevenly**. In early layers, the model is extremely confident — it is essentially copying the most probable continuation of each token in the local context, which is easy. In the middle layers (roughly 8–18), confidence drops sharply as the model begins integrating broader context and semantic relationships, making the distribution much flatter. This is where the actual "computation" happens: the model is weighing many possibilities.

The language comparison here is the most interesting result in our analysis. For **math**, all three languages show nearly identical confidence trajectories throughout all 24 layers — the curves overlap almost completely. This strongly suggests the model is doing the same internal computation for math questions regardless of surface language, which aligns with the hypothesis that arithmetic is language-agnostic at the representation level.

For **factual and reasoning** tasks, the patterns diverge more, but not in the direction we expected. Basque (green) frequently maintains *higher* confidence than English or Spanish in the middle layers, particularly in the factual category. This is counterintuitive — it may reflect the model defaulting to high-confidence but wrong tokens for Basque (since it has fewer associations), rather than genuinely "thinking harder." A model that quickly latches onto a plausible-sounding wrong answer may show higher mid-layer confidence than one that correctly distributes probability across many candidates before committing.

---

### What This Tells Us About Language in Transformers

These three figures together give a richer picture of cross-lingual processing than accuracy numbers alone:

**The early layers are language-independent in a trivial way.** All languages produce near-1.0 confidence predictions in layers 1–5 because the model is just copying the most probable next token at each position. There is no meaningful semantic processing happening yet.

**The middle layers are where languages diverge — but not as cleanly as expected.** Rather than English processing cleanly outpacing Basque, all three languages show a similar pattern of confidence collapse and reconstruction. The math category is the clearest case where all three languages look identical, suggesting that computationally structured knowledge does not require language-specific pathways.

**The final layers do not reliably converge on the correct answer for a small model.** BLOOM-560m's GT rank never gets meaningfully low for most question types, which means accuracy gaps between languages cannot be fully explained by internal representation quality — at this model scale, the model simply doesn't know most of the answers in any language. This motivates using larger models for the full study.

**The "confident but wrong" failure mode appears across all languages equally at this scale.** Because GT rank stays high while confidence recovers in the final layers, BLOOM-560m is consistently committing to wrong answers with reasonable confidence — not flagging uncertainty. This is an important observation for deployment: small models may not signal when they should not be trusted.

---

## Models and Tasks Finalized

**Models:**
- `GPT-3.5-turbo` — black-box baseline via OpenAI API
- `Gemini 2.5 Flash` — strong multilingual baseline via Google API
- `bigscience/bloom-560m` — open-weights interpretability probe

**Tasks:** 60 questions per language × 3 languages = 180 total (20 math / 20 factual / 20 reasoning per language)

**Languages:** English, Spanish, Basque — spanning high-resource, medium-resource, and low-resource coverage.

---

## Next Steps (Week 9)

- Scale logit lens analysis to a larger open model (e.g., `bloom-1b1`) to see if the GT rank picture changes
- Add cosine similarity analysis on residual streams across language pairs
- Finalize paper write-up and prepare presentation

---

*Code: `step1.py` (black-box evaluation), `multilingual_analysis.ipynb` (logit lens with BLOOM-560m)*
*Figures: saved to `figures/` by Cell 8 of the notebook*
