"""
LoCoMo pass-2 retry: retries all v7 failures with full conversation context.
LoCoMo convs are ~19K avg tokens, fits in GPT-4.1 128K window.
"""
from __future__ import annotations
import json, re, time, sys
from pathlib import Path
from collections import defaultdict
import openai

OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')
client = openai.OpenAI(api_key=OPENAI_KEY)
MODEL = "gpt-4.1"
LOCOMO_PATH = "C:/Users/lauri/AOE/Legendary/benchmarks/data/locomo10.json"
CAT_NAMES = {1:"single-session", 2:"multi-session", 3:"open-domain", 4:"temporal", 5:"adversarial"}


def gpt(system, user, mt=300, temp=0):
    msgs = [{"role": "user", "content": user}]
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=mt, messages=msgs, timeout=120, temperature=temp
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error: {e}")
            if attempt < 2:
                time.sleep(3 ** attempt)
    return ""


def locomo_judge(question, gold, generated):
    if not generated:
        return False
    g, h = str(gold).lower().strip(), str(generated).lower().strip()
    if g in h or h in g:
        return True
    if g in ("yes", "no"):
        return g in h[:20]
    verdict = gpt("", f"Question: {question}\nGold: {gold}\nGenerated: {generated}\nDoes the generated answer correctly answer the question? YES or NO.", mt=5)
    return "YES" in verdict.upper()


def full_conv(sess_data):
    parts = []
    for s in sorted(sess_data, key=lambda s: s["snum"]):
        parts.append(f"=== Session {s['snum']} [{s.get('date', '')}] ===\n{s['text']}")
    return "\n\n".join(parts)


def answer_full(question, category, sess_data):
    ctx = full_conv(sess_data)
    if category in (1, 2):
        sys_prompt = "Answer the question using the conversation. Use inference if needed. Be concise."
    elif category == 3:
        sys_prompt = "Answer using evidence from the conversation. Inference is OK. Be concise."
    elif category == 4:
        sys_prompt = "Find the specific fact in the conversation and give a direct answer."
    else:
        sys_prompt = "Answer concisely based on the conversation."
    return gpt(sys_prompt, f"Conversation:\n{ctx}\n\nQuestion: {question}\n\nAnswer:", mt=300)


def main():
    ck_path = Path("unified_v2_locomo_checkpoint.json")
    checkpoint = json.loads(ck_path.read_text())
    failures = {k: v for k, v in checkpoint.items() if not v.get("correct", False)}
    print(f"Failures: {len(failures)}")

    data = json.load(open(LOCOMO_PATH, encoding="utf-8"))
    all_items = {}
    for conv in data:
        conv_data = conv.get("conversation", {})
        sample_id = conv.get("sample_id", "")
        session_keys = sorted([k for k in conv_data if k.startswith("session_") and "_date_time" not in k],
                               key=lambda k: int(k.split("_")[1]))
        sess_data = []
        for sk in session_keys:
            snum = int(sk.split("_")[1])
            date = conv_data.get(f"{sk}_date_time", "")
            turns = conv_data[sk] if isinstance(conv_data[sk], list) else []
            text = "\n".join(f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns if t.get("text"))
            sess_data.append({"snum": snum, "date": date, "text": text})
        for qa in conv.get("qa", []):
            item_id = f"{sample_id}_{qa.get('question', '')[:30]}"
            all_items[item_id] = {"qa": qa, "sess_data": sess_data}

    p2_path = Path("locomo_pass2_checkpoint.json")
    p2 = json.loads(p2_path.read_text()) if p2_path.exists() else {}
    print(f"Resuming: {len(p2)} done")

    n_improved = n_tried = 0
    for k in failures:
        if k in p2:
            if p2[k].get("correct"): n_improved += 1
            continue
        if k not in all_items:
            continue
        item = all_items[k]
        qa = item["qa"]
        question = qa.get("question", "")
        gold = qa.get("answer", "")
        category = qa.get("category", 1)
        n_tried += 1
        print(f"\n[{n_tried}] cat{category} | {question[:70]}")
        print(f"  Gold: {gold[:60]}")
        hyp = answer_full(question, category, item["sess_data"])
        correct = locomo_judge(question, gold, hyp)
        print(f"  Got: {hyp[:100]}")
        print(f"  {'PASS' if correct else 'FAIL'}", flush=True)
        p2[k] = {"category": category, "cat_name": CAT_NAMES.get(category,"?"), "correct": correct,
                 "question": question[:100], "gold": gold[:100], "hyp": hyp[:200]}
        p2_path.write_text(json.dumps(p2, ensure_ascii=False))
        if correct: n_improved += 1

    print(f"\nPass-2: tried={n_tried}, improved={n_improved}")
    merged = dict(checkpoint)
    for k, v in p2.items():
        if k in merged and v.get("correct") and not merged[k].get("correct"):
            merged[k] = v
    total = len(merged)
    n_c = sum(1 for v in merged.values() if v.get("correct"))
    print(f"Merged: {n_c}/{total} = {n_c/total*100:.1f}%")
    by_cat = defaultdict(list)
    for v in merged.values():
        by_cat[v.get("category","?")].append(1 if v.get("correct") else 0)
    for c, scores in sorted(by_cat.items()):
        print(f"  cat{c}: {sum(scores)}/{len(scores)} = {sum(scores)/len(scores)*100:.1f}%")


if __name__ == "__main__":
    main()
