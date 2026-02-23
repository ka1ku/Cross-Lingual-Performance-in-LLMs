import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LANGUAGE = "Basque"

def translate_to_niche(text, is_gt=False):
    if is_gt and (text.isdigit() or text == "x=7" or text == "3/8"):
        return text
    prompt = f"Translate the following text to {LANGUAGE}. Provide ONLY the correct {LANGUAGE} translation, nothing else. Do not add quotes.\n\nText: {text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        t = response.choices[0].message.content.strip()
        return t
    except Exception as e:
        print(f"Error translating: {e}")
        return text

out_lines = []
with open("data/english.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        print(f"Translating {obj['id']} to {LANGUAGE}...")
        obj["question"] = translate_to_niche(obj["question"])
        obj["ground_truth"] = translate_to_niche(obj["ground_truth"], is_gt=True)
        out_lines.append(json.dumps(obj, ensure_ascii=False))

with open(f"data/{LANGUAGE.lower()}.jsonl", "w", encoding="utf-8") as f:
    for line in out_lines:
        f.write(line + "\n")
        
print(f"Successfully generated data/{LANGUAGE.lower()}.jsonl")
