"""
modal_qwen.py — Modal GPU app for Qwen/Qwen2-7B-Instruct inference,
hidden states extraction, and logit lens.

Model specs: 28 transformer layers, d_model=3584, vocab=152064
Public model — no HF token required.

Usage:
    pip install modal
    modal setup          # one-time browser auth
    modal run modal_qwen.py

Outputs (written locally):
    qwen_inference_results.json   — for step1.py accuracy evaluation
    qwen_hidden_states.json       — for multilingual_analysis_large.ipynb RSA
    qwen_logit_lens_results.json  — for multilingual_analysis_large.ipynb logit lens
"""

import json
import modal

app = modal.App("qwen-multilingual")

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

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
MODEL_CACHE_DIR = "/model_cache"

model_volume = modal.Volume.from_name("qwen-weights-cache", create_if_missing=True)


def _format_prompt(question: str, tokenizer) -> str:
    """Qwen2-Instruct chat template format."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_model():
    """Load Qwen/Qwen2-7B-Instruct in float16 on GPU."""
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


def _run_inference(model, tokenizer, questions):
    import torch

    results = []
    for item in questions:
        prompt = _format_prompt(item["question"], tokenizer)
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


def _run_hidden_states(model, tokenizer, questions):
    import torch

    results = []
    for item in questions:
        prompt = _format_prompt(item["question"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        vecs = []
        for h in outputs.hidden_states[1:]:
            last_token_vec = h[0, -1, :].float().cpu().numpy().tolist()
            vecs.append(last_token_vec)

        results.append({
            "id": item["id"],
            "language": item["language"],
            "category": item["category"],
            "hidden_states": vecs,
        })
    return results


def _run_logit_lens(model, tokenizer, questions):
    import torch

    def gt_first_token_id(ground_truth: str):
        ids = tokenizer.encode(" " + ground_truth.strip(), add_special_tokens=False)
        return ids[0] if ids else None

    results = []
    for item in questions:
        gt_token_id = gt_first_token_id(item.get("ground_truth", ""))
        prompt = _format_prompt(item["question"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        ranks = []
        top1_probs = []

        for h in outputs.hidden_states[1:]:
            h_last = h[0, -1, :]
            logits = model.lm_head(model.model.norm(h_last))
            probs = logits.float().softmax(dim=-1)

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


@app.function(
    image=image,
    gpu="A100",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=2700,
    memory=20480,
)
def run_all(questions: list[dict]) -> tuple[list, list, list]:
    """
    Run inference, hidden states, and logit lens in a single container.
    Model loads once — no repeated downloads between jobs.
    Returns (inference_results, hidden_states_results, logit_lens_results).
    """
    print("Loading model...")
    model, tokenizer = _load_model()
    model_volume.commit()  # persist weights to volume for future runs
    print("Model loaded.")

    print("[1/3] Running inference...")
    inference = _run_inference(model, tokenizer, questions)
    print(f"  Done ({len(inference)} entries)")

    print("[2/3] Running hidden states...")
    hidden = _run_hidden_states(model, tokenizer, questions)
    print(f"  Done ({len(hidden)} entries)")

    print("[3/3] Running logit lens...")
    logit = _run_logit_lens(model, tokenizer, questions)
    print(f"  Done ({len(logit)} entries)")

    return inference, hidden, logit


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def run_all_jobs():
    """
    Load all 3 JSONL files, run all GPU jobs in one container, write output JSON files.
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

    os.makedirs("results", exist_ok=True)

    inference_results, hidden_states_results, logit_lens_results = run_all.remote(all_questions)

    with open("results/qwen_inference_results.json", "w") as f:
        json.dump(inference_results, f, indent=2)
    print(f"Wrote results/qwen_inference_results.json ({len(inference_results)} entries)")

    with open("results/qwen_hidden_states.json", "w") as f:
        json.dump(hidden_states_results, f, indent=2)
    print(f"Wrote results/qwen_hidden_states.json ({len(hidden_states_results)} entries)")

    with open("results/qwen_logit_lens_results.json", "w") as f:
        json.dump(logit_lens_results, f, indent=2)
    print(f"Wrote results/qwen_logit_lens_results.json ({len(logit_lens_results)} entries)")

    print("\nDone. All output files written locally.")
    if SMOKE_TEST:
        print("(SMOKE_TEST mode — 1 question per category per language)")
