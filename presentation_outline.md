# Cross-Lingual LLM Evaluation: Class Presentation Outline

**Kai Lee & Justin Tarquino**

*This presentation builds on our project proposal (COLM 2025): **“Evaluating Cross-Lingual Performance and Implicit Translation in Large Language Models.”***

---

## 1. Background & Motivation

### From the proposal (abstract)
> Large language models (LLMs) are trained predominantly on English data but are increasingly deployed in multilingual and non-English contexts. Standard evaluation benchmarks remain heavily English-centric, leaving open questions about how well LLMs generalize across languages and **whether they rely on implicit translation mechanisms** when processing non-English inputs. This project proposes a focused multilingual evaluation framework to systematically measure cross-lingual performance and **probe indications of implicit internal translation**. We aim to build a lightweight multilingual benchmark and leaderboard that evaluates LLMs across aligned tasks in several languages.

### Why we chose this (proposal motivation)
- **Gap between evaluation and use:** LLMs are deployed in multilingual settings, but most benchmarks are English-centric → hidden performance disparities, linguistic bias, unclear generalization.
- **Implicit translation hypothesis:** LLMs may rely on *implicit internal translation* when handling non-English inputs — with implications for **fairness, reliability, and deployment**.
- **Goal:** Measure cross-lingual performance and probe whether internal representations show convergence across languages (implicit translation) or language-specific processing.

### Research question (refined through the project)
- **Do LLMs process semantically equivalent content similarly across languages internally?**  
  We care about both *behavior* (accuracy) and *mechanisms* (how representations evolve across layers).

### Why this matters
- **Fairness & access:** Many users interact with LLMs in non-English languages. If models “understand” less in those languages, outputs can be worse or inconsistent even when the underlying task is the same.
- **Interpretability:** If we can show *where* and *how* cross-lingual processing diverges (e.g., early vs. late layers, math vs. factual), we get clues about what to improve (data, architecture, or training).
- **Hypothesis we test:**  
  - **Math** may be more language-agnostic (numerals, structure).  
  - **Factual** and **reasoning** may depend more on language-specific knowledge and thus show larger gaps and different internal structure across languages.

### High-level approach
1. **Black-box:** Accuracy on the same benchmark in English, Spanish, and Basque.  
2. **White-box:** Logit lens (what the model “predicts” at each layer) and **RSA** (Representational Similarity Analysis) on residual stream vectors to see if *relational structure* between questions is shared across languages — directly probing **implicit translation** in internal representations.

---

## 2. Experimental Design

### From proposal to execution
- **Proposal:** Parallel/semantically aligned task sets in multiple languages (e.g., English, Spanish, French, one low-resource); black-box evaluation; probe implicit translation by comparing outputs and convergence patterns.
- **What we did:** We implemented aligned tasks in **English, Spanish, and Basque** (Basque as our low-resource language instead of a separate fourth). We added **white-box probing** (logit lens, residual stream RSA) so we can test *internal* convergence, not only output-level consistency. Benchmark: 180 questions, 3 categories (math, factual, reasoning).

### Why these three languages?

| Language | Rationale |
|----------|-----------|
| **English** | Dominant training language; strong baseline. |
| **Spanish** | High-resource, typologically similar to English (Indo-European). Lets us test a “closer” language pair (EN–ES). |
| **Basque** | **Low-resource**, language-isolate (no close relatives). Likely much less data in training. Lets us test whether the model has a *representation* for Basque content or mostly treats it as noise. |

Together they give a **gradient**: high-resource (EN) → medium (ES) → low-resource (EU), so we can see how both accuracy and internal representations scale with language support.

### Why these task categories?

| Category   | Content (20 Q each) | Why include it |
|-----------|----------------------|-----------------|
| **Math**  | Arithmetic, algebra, geometry | Answers are numerals/symbols; minimal lexical variation. Good test of “language-agnostic” processing. |
| **Factual** | World knowledge, science, geography | Requires retrieval of facts; often English-centric in training. Expect larger cross-lingual gaps. |
| **Reasoning** | Logic puzzles, lateral thinking, trick questions | Tests structure and inference; we can ask whether reasoning is shared in mid-layers (“implicit translation”) then language-specific at output. |

**Total:** 60 questions per language × 3 languages = **180 questions**, aligned by `id` so each question has semantically equivalent EN/ES/EU versions (translated via GPT).

### Why Mistral-7B-Instruct-v0.2 (unified model)?

- **Week 7/8 issue:** Black-box evals used **GPT-3.5-turbo** and **Gemini 2.5 Flash**; white-box probing used **BLOOM-560M**. Different models → we couldn’t say that the probing *explained* the black-box results.
- **Fix (Week 9):** Use **one model** for both:
  - **Accuracy:** Mistral-7B-Instruct-v0.2 answers all 180 questions (via Modal).
  - **Logit lens & RSA:** Same model’s residual stream is probed.
- So: **White-box analysis now directly explains black-box behavior** for that model.

### Why RSA (Representational Similarity Analysis)?

- **Question:** Do internal representations of “same question in different languages” have similar *structure* (which questions are similar to which)?
- **Method:** For each (language, category) we have 20 questions. At each layer we take the last-token residual stream (4096-d), build a 20×20 **Representational Dissimilarity Matrix (RDM)** from pairwise cosine distances, then compute **Spearman r** between RDMs of two languages. High r ⇒ similar relational structure across languages.
- **Interpretation:** Even if absolute vectors differ, shared *structure* suggests the model is organizing the content in a comparable way (e.g., by problem type in math, or by topic in factual).

### Experimental pipeline (end-to-end)

1. **Input data**  
   Load aligned benchmark: `data/english.jsonl`, `data/spanish.jsonl`, `data/basque.jsonl` (60 Q per language, 20 per category). Each row has `id`, `category`, `question`, `ground_truth`.

2. **Black-box (accuracy)**  
   - **step1.py:** Submit all 180 questions to Modal and run **Mistral-7B-Instruct-v0.2** with greedy decoding (same prompt format as in white-box runs).  
   - Save `mistral_inference_results.json` (question, model_answer, ground_truth, language, category).  
   - **Locally:** A **GPT-4o-mini** judge compares each model answer to the ground truth; we then aggregate accuracy by language and category.

3. **White-box (Modal GPU)**  
   **modal_bloom.py** runs three jobs on the same 180 questions with the same model:
   - **Inference** (optional for step1): same as above → `mistral_inference_results.json`.  
   - **Hidden states:** For each question we run a forward pass with hidden states enabled, then save the **last-token residual stream** at every layer (32 layers × 4096 dims) to `mistral_hidden_states.json`.  
   - **Logit lens:** At each layer we project the last-token hidden state through the final layer norm and lm_head, apply softmax, then record the **rank of the ground-truth first token** and the **top-1 probability**; results go to `mistral_logit_lens_results.json`.

4. **Analysis (notebook, no GPU)**  
   **multilingual_analysis_large.ipynb** loads the JSON outputs:
   - Using the **logit lens** outputs we plot mean ground-truth rank and mean top-1 confidence by layer (broken down by language and category), and optionally print L1 vs L32 summary tables.  
   - Using the **hidden states** we build a 20×20 representational dissimilarity matrix (RDM) from pairwise cosine distances for each (language, category) at each layer, then compute **Spearman r** between RDMs for each language pair (EN–ES, EN–EU, ES–EU) by layer to produce RSA tables and plots.

**Single model throughout:** Mistral-7B-Instruct-v0.2 is used for inference, hidden-state extraction, and logit lens so black-box and white-box results are directly comparable.

---

## 3. How the Experiment Progressed

### Proposal timeline (original plan)
- **Week 6:** Curate aligned multilingual task sets; finalize models and evaluation metrics.
- **Week 7:** Run multilingual evaluations; analyze performance gaps and consistency.
- **Week 8:** Probe implicit translation effects; build leaderboard and visualizations.
- **Week 9:** Finalize analysis and prepare presentation and final report.

### What we did (actual progress)

| Phase | What we did |
|-------|-------------|
| **Data** | Built 60 EN questions (20 math, 20 factual, 20 reasoning), translated to Spanish and Basque (semantically aligned triplets). |
| **Week 7 / 8** | Black-box: GPT-3.5-turbo & Gemini 2.5 Flash on full 180-Q benchmark. White-box: Logit lens on **BLOOM-560M** (24 layers). Noted that comparing different models was a confound. |
| **Mentor feedback** | Use a single model for both accuracy and probing so white-box explains black-box. |
| **Week 9** | Switched to **Mistral-7B-Instruct-v0.2** for everything: re-ran 180-Q accuracy, logit lens at L1 and L32, and **residual stream extraction + RSA** across all 32 layers for EN–ES, EN–EU, ES–EU by category. |

*Leaderboard and visualizations from the proposal are a natural next step (see §5).*

---

## 4. Current Results

### 4.1 Black-box accuracy (Mistral-7B-Instruct-v0.2)

| Language | Math | Factual | Reasoning | **Overall** |
|----------|:----:|:-------:|:---------:|:-----------:|
| English  | 40%  | **95%** | 65%       | **67%**     |
| Spanish  | 50%  | **95%** | 35%       | **60%**     |
| Basque   | 20%  | **0%**  | 5%        | **8%**      |

- **Basque collapse:** Especially on factual (0%); overall 8%. Suggests limited Basque in training; raises the question: does the model still *represent* Basque content meaningfully internally, or mainly fail at generation?
- **EN vs ES:** Factual equal (95%); math/reasoning somewhat lower in Spanish; overall 67% vs 60%.

### 4.2 Logit lens (GT rank and top-1 confidence at last token, L1 vs L32)

- **Math:** Basque actually has *better* (lower) GT rank at L1 (339 vs ~1,150 for EN/ES) — numerals tokenize similarly; model “finds” numeric answer early. By L32, EN has highest confidence (0.942), then ES (0.810), then EU (0.703).
- **Factual:** EN converges sharply (GT rank 26 at L32); Spanish 219; Basque 2,440. Mirrors 95% / 95% / 0% accuracy — behavioral gap is reflected in internal predictions.
- **Reasoning:** Basque has lowest L32 confidence (0.539 vs 0.870 EN), consistent with 5% accuracy and “less committed” answers.

### 4.3 RSA (cross-lingual representational similarity by layer)

- **Math:** Strongest, most stable alignment. EN–ES r stays high (0.906 → 0.785); EN–EU stays positive (0.807 → 0.389). Model organizes math by *structure*, not surface language.
- **Factual & reasoning with Basque:** EN–EU and ES–EU correlations **collapse** in mid/late layers (e.g. EN–EU factual 0.508 → −0.049 at L32). Basque representations become structurally *dissimilar* to EN — consistent with “tokenizing but not meaningfully processing” Basque.
- **Reasoning EN–ES:** r peaks in *mid*-layers (e.g. 0.810 at L8) then drops by L32 (0.558). Supports **implicit translation**: shared reasoning representation in the middle, then language-specific resolution at output.

---

## 5. Next Steps

1. **Correct vs incorrect split:** For each language/category, compare RSA and logit-lens trajectories for *correct* vs *incorrect* answers. Does successful reasoning show earlier or stronger cross-lingual alignment?
2. **More layers / finer granularity:** Report RSA at every layer (or every 4) and plot curves; same for logit lens to see exactly where EN–EU factual/reasoning diverge.
3. **Other models:** Replicate pipeline on another 7B (e.g. Llama, Qwen) to see if “math robust, Basque factual collapse” is general or Mistral-specific.
4. **Intervention:** If Basque representations are “wrong” in late layers, test whether patching in English or Spanish residual stream at a given layer improves Basque accuracy (activation patching / representation substitution).
5. **Write-up:** Turn weekly blogs into a short report (motivation, design, results, RSA interpretation, next steps) for submission or portfolio.
6. **Leaderboard & visualizations:** As in the original proposal — public benchmark and cross-lingual consistency visualizations.

---

## 6. Related Work (from proposal)

We planned to build on the following:

- **Multilingual Evaluation of Large Language Models: Challenges and Gaps** — [ACL Anthology 2025.emnlp-main.69](https://aclanthology.org/2025.emnlp-main.69/)
- **Analyzing Cross-Lingual Generalization in Large Language Models** — [ACL Anthology 2025.emnlp-main.762](https://aclanthology.org/2025.emnlp-main.762/)
- **MMLU-ProX: A Multilingual Benchmark for Advanced LLM Evaluation** — [arXiv:2503.10497](https://arxiv.org/abs/2503.10497)
- **INCLUDE: Evaluating Multilingual Language Models Across 44 Languages** — [OpenReview](https://openreview.net/forum?id=k3gCieTXeY)
- **Mexa: Cross-Lingual Alignment and Evaluation of English-Centric LLMs** — [OpenReview](https://openreview.net/forum?id=hsMkpzr9Oy)

---

## Quick reference: Key numbers for slides

- **Proposal:** COLM 2025 — “Evaluating Cross-Lingual Performance and Implicit Translation in LLMs”; lightweight multilingual benchmark + implicit translation probing.
- **Benchmark:** 180 questions (60 per language × 3 categories); aligned EN/ES/EU.
- **Model:** Mistral-7B-Instruct-v0.2 (single model for accuracy + probing).
- **Basque overall accuracy:** 8% (0% factual).
- **RSA:** Math EN–ES ~0.79 at L32; EN–EU factual/reasoning ≈ 0 or negative at L32.
- **Logit lens:** Basque math GT rank at L1 = 339 (better than EN/ES); Basque factual GT rank at L32 = 2,440 (vs EN 26).
- **Implicit translation:** EN–ES reasoning RSA peaks in mid-layers (0.81 @ L8) → supports shared internal representation, then language-specific output.
