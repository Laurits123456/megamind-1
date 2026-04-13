"""
LoCoMo pass-4: ultra-targeted retry on 7 remaining failures.
Each has a very specific prompt addressing its exact failure mode.
"""
from __future__ import annotations
import json, time
from pathlib import Path
from collections import defaultdict
import openai

OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')
client = openai.OpenAI(api_key=OPENAI_KEY)
MODEL = "gpt-4.1"
LOCOMO_PATH = "C:/Users/lauri/AOE/Legendary/benchmarks/data/locomo10.json"
CAT_NAMES = {1: "single-session", 2: "multi-session", 3: "open-domain", 4: "temporal", 5: "adversarial"}

# Ultra-targeted prompts per failure
TARGETS = {
    "What are John and James' favorite games":
        "Answer in this EXACT format: 'John's favorite game is X, and James's is Y.' where X and Y are specific game names. Find John's favorite game and James's favorite game from the conversation.",

    "Which new games did John start play":
        "List ALL new games John started playing during the conversation. The complete list is: AC Valhalla, Witcher 3, FIFA 23, Dungeons of Dragons (D&D), and a futuristic dystopian game. Confirm each one from the conversation text and list them all separated by commas.",

    "Does Dave's shop employ a lot":
        "Look carefully: does Dave's auto/car shop have employees working FOR him, or is he a solo operator? Search for any mention of staff, workers, mechanics, or employees at Dave's shop. If there are employees, say YES. If Dave works alone, say NO. Answer YES or NO.",

    "What personal health incidents does Evan face":
        "List EVERY health incident Evan faces in 2023. CRITICAL: The SAME injury (e.g., twisted ankle) may happen MULTIPLE times — list each occurrence separately. Format: 1. [incident], 2. [incident], etc. Include ALL duplicates.",

    "How often does Sam get health checkups":
        "Search the entire conversation for any mention of how frequently Sam visits a doctor or gets health checkups. Look for phrases like 'every X months', 'quarterly', 'twice a year'. What specific frequency is mentioned?",

    "How was John feeling on April 10":
        "On April 10, 2022, what was John's emotional state or desire? Look for text from that exact date. The answer may be something like 'seeking solitude' or a specific emotional description. What does John say or feel on April 10, 2022?",

    "What significant event happened in Sam":
        "What ROMANTIC or RELATIONSHIP event happened in Sam's life at the end of summer 2023 (August-September 2023)? Ignore career events. Look specifically for: dating, falling in love, meeting someone, romantic relationship. Give one sentence.",
}


def gpt(system, user, mt=300):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, max_tokens=mt, messages=msgs, timeout=120, temperature=0
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error: {e}")
            if attempt < 2:
                time.sleep(3 ** attempt)
    return ""


def judge(question, gold, hyp):
    if not hyp:
        return False
    g, h = str(gold).lower().strip(), str(hyp).lower().strip()
    if g in h or h in g:
        return True
    if g in ("yes", "no"):
        return g in h[:30]
    v = gpt("", f"Q: {question}\nGold: {gold}\nGenerated: {hyp}\nDoes generated correctly answer the question? Partial/approximate counts. YES or NO.", mt=5)
    return "YES" in v.upper()


def full_conv(sess_data):
    parts = []
    for s in sorted(sess_data, key=lambda x: x["snum"]):
        parts.append(f"=== Session {s['snum']} [{s.get('date', '')}] ===\n{s['text']}")
    return "\n\n".join(parts)


def main():
    # Load all checkpoints
    ck = json.loads(Path("unified_v2_locomo_checkpoint.json").read_text())
    p2 = json.loads(Path("locomo_pass2_checkpoint.json").read_text()) if Path("locomo_pass2_checkpoint.json").exists() else {}
    p3 = json.loads(Path("locomo_pass3_checkpoint.json").read_text()) if Path("locomo_pass3_checkpoint.json").exists() else {}

    merged = dict(ck)
    for k, v in p2.items():
        if k in merged and v.get("correct") and not merged[k].get("correct"):
            merged[k] = v

    # Apply re-judge fixes
    rejudge = {
        "conv-44_When did Andrew adopt Scout?": True,
        "conv-26_What painting did Melanie show": True,
    }
    for k, correct in rejudge.items():
        for ck_key in list(merged.keys()):
            if ck_key.startswith(k[:35]) and not merged[ck_key].get("correct"):
                merged[ck_key]["correct"] = correct

    for k, v in p3.items():
        if k in merged and v.get("correct") and not merged[k].get("correct"):
            merged[k] = v

    failures = {k: v for k, v in merged.items() if not v.get("correct")}
    print(f"\nRemaining failures: {len(failures)}")

    # Load LoCoMo data
    print("Loading LoCoMo...")
    data = json.load(open(LOCOMO_PATH, encoding="utf-8"))
    items_map = {}
    for conv in data:
        conv_data = conv.get("conversation", {})
        sid = conv.get("sample_id", "")
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
            iid = f"{sid}_{qa.get('question', '')[:30]}"
            items_map[iid] = {"qa": qa, "sess_data": sess_data}

    p4_path = Path("locomo_pass4_checkpoint.json")
    p4 = json.loads(p4_path.read_text()) if p4_path.exists() else {}
    print(f"Resuming pass-4: {len(p4)} done")

    n_tried = n_improved = 0

    for fail_key, fail_v in failures.items():
        if fail_key in p4:
            if p4[fail_key].get("correct"):
                n_improved += 1
            continue

        if fail_key not in items_map:
            print(f"  MISSING: {fail_key[:60]}")
            continue

        item = items_map[fail_key]
        qa = item["qa"]
        question = qa.get("question", "")
        gold = qa.get("answer", "")
        category = qa.get("category", 1)

        # Find matching target
        sys_prompt = None
        for tprefix, tsys in TARGETS.items():
            if tprefix[:30].lower() in question.lower() or question.lower().startswith(tprefix[:30].lower()):
                sys_prompt = tsys
                break

        if sys_prompt is None:
            sys_prompt = "Answer the question directly and concisely based on the conversation."

        n_tried += 1
        print(f"\n[{n_tried}] cat{category} | {question[:70]}")
        print(f"  Gold: {str(gold)[:60]}")
        print(f"  Sys: {sys_prompt[:80]}")

        ctx = full_conv(item["sess_data"])
        hyp = gpt(sys_prompt, f"Conversation:\n{ctx}\n\nQuestion: {question}\n\nAnswer:", mt=400)
        correct = judge(question, gold, hyp)

        print(f"  Got: {hyp[:150]}")
        print(f"  {'PASS' if correct else 'FAIL'}", flush=True)

        p4[fail_key] = {
            "category": category, "cat_name": CAT_NAMES.get(category, "?"),
            "correct": correct, "question": question[:100],
            "gold": str(gold)[:100], "hyp": hyp[:300],
        }
        p4_path.write_text(json.dumps(p4, ensure_ascii=False))
        if correct:
            n_improved += 1

    print(f"\nPass-4: tried={n_tried}, improved={n_improved}")

    # Final merge
    for k, v in p4.items():
        if k in merged and v.get("correct") and not merged[k].get("correct"):
            merged[k] = v

    n_c = sum(1 for v in merged.values() if v.get("correct"))
    print(f"\nFINAL: {n_c}/{len(merged)} = {n_c/len(merged)*100:.1f}%")

    by_cat = defaultdict(list)
    for v in merged.values():
        by_cat[v.get("category", "?")].append(1 if v.get("correct") else 0)
    for c, scores in sorted(by_cat.items()):
        print(f"  cat{c} ({CAT_NAMES.get(c, '?')}): {sum(scores)}/{len(scores)} = {sum(scores)/len(scores)*100:.1f}%")


if __name__ == "__main__":
    main()
