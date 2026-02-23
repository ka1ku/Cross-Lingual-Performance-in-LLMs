import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

load_dotenv(override=True)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use the cheapest models
OAI_MODEL = "gpt-3.5-turbo-0125"  # Cheapest chat model as requested
GEMINI_MODEL = "gemini-2.5-flash" # Modern low-end gemini

def get_openai_response(prompt):
    try:
        response = openai_client.chat.completions.create(
            model=OAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return f"API Error: {e}"

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return f"API Error: {e}"

def evaluate_with_judge(question, model_answer, ground_truth):
    if "API Error" in model_answer or "object has no attribute" in model_answer:
        return False
    # Using gpt-4o-mini as a cheap judge
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
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return "YES" in response.choices[0].message.content.upper()
    except Exception as e:
        print("Judge error:", str(e))
        return False

def evaluate_question(item, language):
    q_id = item["id"]
    category = item["category"]
    question = item["question"]
    gt = item.get("ground_truth", "") # Read directly from JSON item
    
    oai_ans = get_openai_response(question)
    gemini_ans = get_gemini_response(question)
    
    # Judge
    oai_correct = evaluate_with_judge(question, oai_ans, gt)
    gemini_correct = evaluate_with_judge(question, gemini_ans, gt)
    
    return {
        "id": q_id,
        "language": language,
        "category": category,
        "oai_correct": oai_correct,
        "gemini_correct": gemini_correct
    }

def main():
    print("Loading data...")
    all_data = []
    for lang, file_path in [("English", "data/english.jsonl"), ("Spanish", "data/spanish.jsonl"), ("Basque", "data/basque.jsonl")]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                all_data.append((item, lang))
                
    print(f"Loaded {len(all_data)} questions. Starting evaluation...")
    results = []
    
    # We will use max_workers=5 to avoid rate limits on small/free tiers
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_question, item, lang): (item, lang) for item, lang in all_data}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                if (i+1) % 10 == 0:
                    print(f"Evaluated {i+1} / {len(all_data)} questions...")
            except Exception as e:
                print(f"Error evaluating a question: {e}")
                
    print(f"Evaluation complete in {time.time() - start_time:.2f} seconds.")

    # Calculate metrics
    metrics = {
        "English": {"math": {"oai": 0, "gemini": 0}, "factual": {"oai": 0, "gemini": 0}, "reasoning": {"oai": 0, "gemini": 0}},
        "Spanish": {"math": {"oai": 0, "gemini": 0}, "factual": {"oai": 0, "gemini": 0}, "reasoning": {"oai": 0, "gemini": 0}},
        "Basque": {"math": {"oai": 0, "gemini": 0}, "factual": {"oai": 0, "gemini": 0}, "reasoning": {"oai": 0, "gemini": 0}}
    }
    
    for r in results:
        lang = r["language"]
        cat = r["category"]
        if r["oai_correct"]:
            metrics[lang][cat]["oai"] += 1
        if r["gemini_correct"]:
            metrics[lang][cat]["gemini"] += 1
            
    print("\n=== RESULTS ===")
    for lang in ["English", "Spanish", "Basque"]:
        print(f"\n-- {lang} --")
        for cat in ["math", "factual", "reasoning"]:
            oai_score = metrics[lang][cat]["oai"] / 20.0 * 100
            gem_score = metrics[lang][cat]["gemini"] / 20.0 * 100
            print(f"Category: {cat:10s} | OpenAI ({OAI_MODEL}): {oai_score:5.1f}% | Gemini ({GEMINI_MODEL}): {gem_score:5.1f}%")

if __name__ == "__main__":
    main()
