"""
modal_bloom.py — Modal GPU app for Mistral-7B-Instruct-v0.2 inference,
hidden states extraction, and logit lens.

Model specs: 32 transformer layers, d_model=4096, vocab=32000

Usage:
    pip install modal
    modal setup          # one-time browser auth
    modal run modal_bloom.py

Outputs (written locally):
    mistral_inference_results.json   — for step1.py accuracy evaluation
    mistral_hidden_states.json       — for multilingual_analysis_large.ipynb RSA
    mistral_logit_lens_results.json  — for multilingual_analysis_large.ipynb logit lens
"""

import json
import modal

app = modal.App("mistral-multilingual")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.41.0",
        "accelerate==0.30.0",
        "scipy",
        "numpy",
    )
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_CACHE_DIR = "/model_cache"

model_volume = modal.Volume.from_name("mistral-weights-cache", create_if_missing=True)


def _format_prompt(question: str) -> str:
    """Mistral instruct format."""
    return f"<s>[INST] {question} [/INST]"


def _load_model():
    """Load Mistral-7B-Instruct-v0.2 in float16 on GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE_DIR,
    )
    model.eval()
    return model, tokenizer


@app.function(
    image=image,
    gpu="A100",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=900,
    memory=20480,
)
def run_inference(questions: list[dict]) -> list[dict]:
    """
    Greedy-decode each question with Mistral-7B-Instruct-v0.2.

    Returns list of dicts with model_answer field.
    """
    import torch

    model, tokenizer = _load_model()
    model_volume.commit()

    results = []
    for item in questions:
        prompt = _format_prompt(item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append({
            "id": item["id"],
            "language": item["language"],
            "category": item["category"],
            "question": item["question"],
            "ground_truth": item.get("ground_truth", ""),
            "model_answer": answer,
        })

    return results


@app.function(
    image=image,
    gpu="A100",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=900,
    memory=20480,
)
def run_hidden_states(questions: list[dict]) -> list[dict]:
    """
    Extract residual stream hidden states at every layer for each question.

    Mistral-7B-Instruct-v0.2: 32 transformer layers, d_model=4096.
    Returns hidden_states: (n_layers, d_model) per question, JSON-serializable.
    """
    import torch

    model, tokenizer = _load_model()
    model_volume.commit()

    results = []
    for item in questions:
        prompt = _format_prompt(item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors — index 0 = embedding
        vecs = []
        for h in outputs.hidden_states[1:]:
            last_token_vec = h[0, -1, :].float().cpu().numpy().tolist()
            vecs.append(last_token_vec)

        results.append({
            "id": item["id"],
            "language": item["language"],
            "category": item["category"],
            "hidden_states": vecs,  # (n_layers, d_model)
        })

    return results


@app.function(
    image=image,
    gpu="A100",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=900,
    memory=20480,
)
def run_logit_lens(questions: list[dict]) -> list[dict]:
    """
    Run logit lens analysis on Mistral-7B-Instruct-v0.2 for each question.

    Mistral uses LlamaForCausalLM-style architecture:
      logit lens projection = model.lm_head(model.model.norm(h))

    Records rank of GT first token and top-1 probability at every layer.
    """
    import torch

    model, tokenizer = _load_model()
    model_volume.commit()

    def gt_first_token_id(ground_truth: str):
        # Mistral tokenizer: leading space helps get the right token
        ids = tokenizer.encode(" " + ground_truth.strip(), add_special_tokens=False)
        return ids[0] if ids else None

    results = []
    for item in questions:
        gt_token_id = gt_first_token_id(item.get("ground_truth", ""))
        prompt = _format_prompt(item["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        ranks = []
        top1_probs = []

        for h in outputs.hidden_states[1:]:
            h_last = h[0, -1, :]  # keep float16 — norm + lm_head weights are float16
            # Mistral/LlamaForCausalLM logit lens: model.model.norm + model.lm_head
            logits = model.lm_head(model.model.norm(h_last))
            probs = logits.float().softmax(dim=-1)  # cast to float32 for softmax stability

            top1_probs.append(probs.max().item())

            if gt_token_id is not None:
                rank = (probs.argsort(descending=True) == gt_token_id).nonzero(as_tuple=True)[0].item()
            else:
                rank = -1
            ranks.append(rank)

        results.append({
            "id": item["id"],
            "language": item["language"],
            "category": item["category"],
            "ground_truth": item.get("ground_truth", ""),
            "ranks": ranks,
            "top1_probs": top1_probs,
        })

    return results


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def run_all_jobs():
    """
    Load all 3 JSONL files, run all three GPU jobs, write output JSON files.
    Set SMOKE_TEST=true for a quick 9-question test run.
    """
    import os

    SMOKE_TEST = os.environ.get("SMOKE_TEST", "false").lower() == "true"

    print("Loading benchmark data...")
    all_questions = []
    for lang, file_path in [
        ("English", "data/english.jsonl"),
        ("Spanish", "data/spanish.jsonl"),
        ("Basque",  "data/basque.jsonl"),
    ]:
        with open(file_path, encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        if SMOKE_TEST:
            by_cat = {}
            for item in items:
                by_cat.setdefault(item["category"], []).append(item)
            items = [v[0] for v in by_cat.values()]
        for item in items:
            item["language"] = lang
        all_questions.extend(items)

    print(f"Loaded {len(all_questions)} questions. Sending to Modal...")

    print("\n[1/3] Running Mistral-7B-Instruct inference...")
    inference_results = run_inference.remote(all_questions)
    with open("mistral_inference_results.json", "w") as f:
        json.dump(inference_results, f, indent=2)
    print(f"  Wrote mistral_inference_results.json ({len(inference_results)} entries)")

    print("\n[2/3] Running hidden states extraction...")
    hidden_states_results = run_hidden_states.remote(all_questions)
    with open("mistral_hidden_states.json", "w") as f:
        json.dump(hidden_states_results, f, indent=2)
    print(f"  Wrote mistral_hidden_states.json ({len(hidden_states_results)} entries)")

    print("\n[3/3] Running logit lens...")
    logit_lens_results = run_logit_lens.remote(all_questions)
    with open("mistral_logit_lens_results.json", "w") as f:
        json.dump(logit_lens_results, f, indent=2)
    print(f"  Wrote mistral_logit_lens_results.json ({len(logit_lens_results)} entries)")

    print("\nDone. All output files written locally.")
    if SMOKE_TEST:
        print("(SMOKE_TEST mode — 1 question per category per language)")
