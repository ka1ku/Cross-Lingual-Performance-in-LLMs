"""
step1.py — Black-box accuracy evaluation using Mistral-7B-Instruct-v0.2 via Modal.

Mistral-7B-Instruct-v0.2 is the same model used for logit lens and RSA in
multilingual_analysis_large.ipynb, making the black-box baseline directly
comparable to the white-box probing results.

Usage:
    # Smoke test (3 questions per language, 1 per category):
    SMOKE_TEST=true modal run step1.py

    # Full run (60 questions per language, 180 total):
    modal run step1.py
"""

import json
import os
import modal

# ── Modal setup ───────────────────────────────────────────────────────────────

app = modal.App("step1-accuracy-eval")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers==4.41.0",
        "accelerate==0.30.0",
        "numpy",
    )
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_CACHE_DIR = "/model_cache"

model_volume = modal.Volume.from_name("mistral-weights-cache", create_if_missing=True)


def _format_prompt(question: str) -> str:
    """Mistral instruct format: <s>[INST] question [/INST]"""
    return f"<s>[INST] {question} [/INST]"


@app.function(
    image=image,
    gpu="A100",
    volumes={MODEL_CACHE_DIR: model_volume},
    timeout=900,
    memory=20480,
)
def run_inference(questions: list[dict]) -> list[dict]:
    """Greedy-decode each question with Mistral-7B-Instruct-v0.2 and return answers."""
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


# ── Judge (runs locally) ──────────────────────────────────────────────────────

def evaluate_with_judge(question, model_answer, ground_truth):
    """Use GPT-4o-mini as a cheap judge to score the model's answer."""
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(override=True)
    if not model_answer or "API Error" in model_answer:
        return False
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f'''
    You are an objective evaluator.
    Question: "{question}"
    Expected Correct Answer Concept: "{ground_truth}"
    Model's Answer: "{model_answer}"

    Does the Model's Answer contain or convey the correct concept required by the Expected Correct Answer?
    Some Model answers might be verbose or slightly different, but if they get the core specific answer right, return YES.
    If they are completely wrong, return NO.

    Answer ONLY with exactly "YES" or "NO".
    '''
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return "YES" in response.choices[0].message.content.upper()
    except Exception as e:
        print("Judge error:", str(e))
        return False


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
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
            # Pick 1 per category to cover all 3 categories in the smoke test
            by_cat = {}
            for item in items:
                by_cat.setdefault(item["category"], []).append(item)
            items = [v[0] for v in by_cat.values()]
        for item in items:
            item["language"] = lang
        all_questions.extend(items)

    total = len(all_questions)
    print(f"Loaded {total} questions. Sending to Modal (Mistral-7B-Instruct-v0.2)...")

    inference_results = run_inference.remote(all_questions)

    print(f"Got {len(inference_results)} answers back. Judging with GPT-4o-mini...")

    metrics = {
        lang: {cat: {"correct": 0, "total": 0} for cat in ["math", "factual", "reasoning"]}
        for lang in ["English", "Spanish", "Basque"]
    }

    for i, r in enumerate(inference_results):
        correct = evaluate_with_judge(r["question"], r["model_answer"], r["ground_truth"])
        metrics[r["language"]][r["category"]]["correct"] += int(correct)
        metrics[r["language"]][r["category"]]["total"]   += 1
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Judged {i+1}/{total}")

    with open("mistral_inference_results.json", "w") as f:
        json.dump(inference_results, f, indent=2)
    print("\nSaved mistral_inference_results.json")

    print("\n=== RESULTS (Mistral-7B-Instruct-v0.2) ===")
    for lang in ["English", "Spanish", "Basque"]:
        print(f"\n-- {lang} --")
        for cat in ["math", "factual", "reasoning"]:
            m = metrics[lang][cat]
            n = m["total"]
            pct = (m["correct"] / n * 100) if n > 0 else 0.0
            print(f"  Category: {cat:10s} | Mistral-7B-Instruct: {pct:5.1f}%  ({m['correct']}/{n})")

    if SMOKE_TEST:
        print("\n(SMOKE_TEST mode — 1 question per category per language)")
