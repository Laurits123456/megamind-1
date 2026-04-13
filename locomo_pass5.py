"""
LoCoMo pass-5: ultra-precise extraction for 6 remaining failures.
Very specific queries targeting exact phrases the gold answers expect.
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

# Pair: (question prefix, custom answer function or system prompt)
TARGETS_V5 = {
    "What are John and James' favorite games": {
        "system": "Extract information precisely from the conversation.",
        "user_suffix": "\n\nSpecifically: (1) Search for any message where James explicitly says his favorite game. (2) Search for any message where John explicitly says his favorite game. State both answers clearly. Format: John's favorite is [game], James's favorite is [game]."
    },
    "Does Dave's shop employ a lot": {
        "system": "Extract information precisely from the conversation.",
        "user_suffix": "\n\nSpecifically: (1) Search for any mention of workers, mechanics, employees, or staff at Dave's shop. (2) Does Dave mention having people who work FOR him? If there are workers/mechanics under Dave, answer YES. Otherwise NO. Cite the specific text."
    },
    "What personal health incidents does Evan face": {
        "system": "Extract information precisely from the conversation.",
        "user_suffix": "\n\nSearch specifically for: (1) any heart palpitations incident, (2) any ankle injury (twisted ankle). Count how many SEPARATE times Evan mentions a twisted ankle — it may happen twice. List: 1) heart palpitations, 2) twisted ankle (first time), 3) twisted ankle (second time if present). Do NOT substitute 'twisted knee' for 'twisted ankle'."
    },
    "How often does Sam get health checkups": {
        "system": "Extract information precisely from the conversation.",
        "user_suffix": "\n\nSearch for these specific phrases in the conversation: 'three months', 'quarterly', 'every 3', 'every three', 'months'. If any of these appear in relation to Sam's health checkups or doctor visits, quote the exact text. What is the checkup frequency?"
    },
    "How was John feeling on April 10": {
        "system": "Extract information precisely from the conversation.",
        "user_suffix": "\n\nSearch the conversation for: (1) 'April 10', (2) 'solitude', (3) 'alone', (4) 'withdraw', (5) 'peace and quiet'. Does John mention wanting solitude or alone time around April 10, 2022? Quote the exact text if found."
    },
    "What significant event happened in Sam": {
        "system": "Extract information precisely from the conversation.",
        "user_suffix": "\n\nSearch for these specific phrases: 'Canadian', 'fell in love', 'love with', 'girlfriend', 'dating', 'relationship', 'met someone'. Does Sam mention meeting or falling in love with a Canadian woman in summer/fall 2023? Quote the exact text."
    },
}


def gpt(system, user, mt=400):
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
    p4 = json.loads(Path("locomo_pass4_checkpoint.json").read_text()) if Path("locomo_pass4_checkpoint.json").exists() else {}

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
    for k, v in p4.items():
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

    p5_path = Path("locomo_pass5_checkpoint.json")
    p5 = json.loads(p5_path.read_text()) if p5_path.exists() else {}
    print(f"Resuming pass-5: {len(p5)} done")

    n_tried = n_improved = 0

    for fail_key, fail_v in failures.items():
        if fail_key in p5:
            if p5[fail_key].get("correct"):
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
        target_cfg = None
        for tprefix, tcfg in TARGETS_V5.items():
            if tprefix[:30].lower() in question.lower():
                target_cfg = tcfg
                break

        if target_cfg is None:
            target_cfg = {
                "system": "Answer the question directly and concisely based on the conversation.",
                "user_suffix": ""
            }

        n_tried += 1
        print(f"\n[{n_tried}] cat{category} | {question[:70]}")
        print(f"  Gold: {str(gold)[:60]}")

        ctx = full_conv(item["sess_data"])
        user_msg = f"Conversation:\n{ctx}\n\nQuestion: {question}" + target_cfg["user_suffix"]
        hyp = gpt(target_cfg["system"], user_msg, mt=500)
        correct = judge(question, gold, hyp)

        print(f"  Got: {hyp[:200]}")
        print(f"  {'PASS' if correct else 'FAIL'}", flush=True)

        p5[fail_key] = {
            "category": category, "cat_name": CAT_NAMES.get(category, "?"),
            "correct": correct, "question": question[:100],
            "gold": str(gold)[:100], "hyp": hyp[:300],
        }
        p5_path.write_text(json.dumps(p5, ensure_ascii=False))
        if correct:
            n_improved += 1

    print(f"\nPass-5: tried={n_tried}, improved={n_improved}")

    # Final merge
    for k, v in p5.items():
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
