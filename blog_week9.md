# Week 9 Blog: Residual Stream Extraction and Cross-Lingual RSA on Mistral-7B
## Addressing Last Week's Feedback
In week Week 7 we used GPT-3.5-turbo and Gemini 2.5 Flash for black-box accuracy evaluation, and then used logit lens on BLOOM-560M. Darin pointed out to us that a 560M parameter model may process language very differently from a billion-parameter instruction-tuned model. Thus, comparing black-box and white-box results was not a good strategy.

This week we used **Mistral-7B-Instruct-v0.2** as our single model for all evals. Because the same model is doing the answering *and* the internal probing, the white-box analysis now directly explains the black-box behavior. 

## Black-Box Accuracy Results

We re-ran the full 180-question benchmark from last-week with Mistral-7B-Instruct-v0.2 via Modal.

### Table 1: Accuracy by Language and Category (Mistral-7B-Instruct-v0.2)

| Language | Math | Factual | Reasoning | Overall |
|----------|:----:|:-------:|:---------:|:-------:|
| English  | 40%  | **95%** | 65%       | **67%** |
| Spanish  | 50%  | **95%** | 35%       | **60%** |
| Basque   | 20%  | **0%**  | 5%        | **8%**  |


The Basque collapse is particularly interesting: the model clearly lacks proficiency, likely because it was not trained extensively on Basque. This observation contributes to our residual stream analysis and raises an important question: does the model internally interpret Basque differently, or does it understand the underlying concepts but fail during generation?


## Logit Lens Analysis

Simmilar to last week's analysis, we track two metrics at the last token position of each question: the rank of the ground-truth first token (lower = better) and the top-1 confidence (how decisive the model is).

### Table 2: Logit Lens — Mean GT Rank and Top-1 Confidence

| Language | Category  | Rank @ L1 | Rank @ L32 | Conf @ L1 | Conf @ L32 |
|----------|-----------|:---------:|:----------:|:---------:|:----------:|
| English  | Math      | 1,160     | 261        | 0.001     | **0.942**  |
| English  | Factual   | 12,041    | **26**     | 0.001     | **0.978**  |
| English  | Reasoning | 5,652     | 186        | 0.001     | 0.870      |
| Spanish  | Math      | 1,143     | 452        | 0.001     | 0.810      |
| Spanish  | Factual   | 14,912    | 219        | 0.001     | 0.768      |
| Spanish  | Reasoning | 11,241    | 309        | 0.001     | 0.688      |
| Basque   | Math      | **339**   | **62**     | 0.001     | 0.703      |
| Basque   | Factual   | 10,434    | 2,440      | 0.001     | 0.848      |
| Basque   | Reasoning | 5,690     | 108        | 0.001     | 0.539      |

Several findings stand out:

**Math is language-agnostic in early layers.** Basque math questions start with a lower initial GT rank (339) than English (1,160) or Spanish (1,143). This makes sense: mathematical answers are numerals, and numerals tokenize the same way regardless of the surrounding language. The model's residual stream "finds" the numeric answer at relatively shallow depth.

**English factual processing converges sharply.** English factual questions reach a mean GT rank of just 26 at layer 32 — the model is essentially pointing directly at the correct answer token by the final layer. Spanish factual rank (219) is 8× worse, and Basque factual rank (2,440) is 94× worse. This mirrors the 95%/95%/0% accuracy pattern and confirms that the behavioral gap is reflected in the model's internal representations.

**Confidence exhibits complimentary behavior.** All languages start at near-zero confidence (0.001) — the early layers produce diffuse distributions. By layer 32, English math reaches 94.2% top-1 confidence, Spanish math 81.0%, and Basque math 70.3%. For reasoning, Basque top-1 confidence (53.9%) is notably lower than English (87.0%), suggesting the model is less "committed" to any particular answer when processing Basque — consistent with the 5% accuracy.

## Residual Stream Extraction and RSA

RSA (Representational Similarity Analysis) addresses the question: do the model's internal representations of semantically equivalent questions across languages look structurally similar?

For each question, we extracted the final-token residual stream vector at every layer (32 layers × 4096 dimensions). For each language-category group (e.g., all 20 English math questions), we computed a **Representational Dissimilarity Matrix (RDM)** — a 20×20 matrix of pairwise cosine distances between questions at a given layer. We then measured the Spearman correlation between the RDMs of two languages at each layer. A high correlation means the model represents the relational structure between questions similarly across languages.

### Table 3: RSA Spearman r Between Language Pairs by Layer

| Category  | Pair  | Layer 1 | Layer 8 | Layer 16 | Layer 24 | Layer 32 |
|-----------|-------|:-------:|:-------:|:--------:|:--------:|:--------:|
| **Math**  | EN–ES | 0.906   | 0.871   | 0.834    | 0.858    | 0.785    |
|           | EN–EU | 0.807   | 0.683   | 0.299    | 0.352    | 0.389    |
|           | SP–EU | 0.846   | 0.630   | 0.496    | 0.465    | 0.273    |
| **Factual** | EN–ES | 0.718 | 0.840   | 0.704    | 0.795    | 0.782    |
|           | EN–EU | 0.508   | 0.406   | **−0.032** | 0.142  | **−0.049** |
|           | SP–EU | 0.694   | 0.384   | 0.056    | 0.166    | 0.013    |
| **Reasoning** | EN–ES | 0.394 | 0.810 | 0.706   | 0.739    | 0.558    |
|           | EN–EU | 0.303   | **−0.097** | 0.088 | 0.023   | 0.018    |
|           | SP–EU | 0.584   | **−0.035** | 0.104 | 0.022  | **−0.080** |

### What the RSA Results Show

**Finding 1 — Math has the strongest and most durable cross-lingual alignment.** The EN–ES Spearman r starts at 0.906 in layer 1 and remains at 0.785 in layer 32. This is the highest final-layer correlation across all categories and language pairs. Even EN–EU math maintains a positive correlation (0.389) through all 32 layers, despite Basque having very different surface forms. The model internally organizes math questions by their mathematical structure, not by the language they are written in.

**Finding 2 — Basque-English alignment collapses in mid-to-late layers for factual and reasoning.** The EN–EU factual correlation drops from 0.508 at layer 1 to −0.049 at layer 32 — statistically indistinguishable from random, and occasionally *anti-correlated*. This means the model's representation of factual questions in Basque becomes *structurally dissimilar* to English as processing deepens. The model may be tokenizing Basque input but ultimately processing it as noise rather than as meaningful content, leading to representational structure that is orthogonal to the English computation.

**Finding 3 — Reasoning EN–ES peaks in the mid-layers.** The EN–ES reasoning correlation is only 0.394 at layer 1 but climbs to 0.810 at layer 8 before declining to 0.558 at layer 32. This U-shape suggests the model constructs a shared cross-lingual reasoning representation in middle layers, then diverges again in final layers as it resolves the answer in a language-specific way. This is consistent with the "implicit translation" hypothesis from our Week 8 blog — reasoning problems in English and Spanish are processed through a shared internal format, but the output stage is language-dependent.

**Finding 4 — Early layers show surprising Basque alignment.** Even for factual and reasoning, the Basque RSA correlations at layer 1 are moderate (0.508 for EN–EU factual, 0.303 for EN–EU reasoning). The early transformer layers appear to extract some language-agnostic structure from the input tokens before the more language-specific processing diverges in middle and final layers. This could reflect shared BPE subword structure across languages or positional encoding effects.



