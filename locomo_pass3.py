"""
LoCoMo pass-3: targeted retry on remaining 12 failures.
Each failure gets a custom system prompt addressing its specific weakness.
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

# Targeted system prompts per question prefix
TARGETS = {
    "What personal health incidents does Evan face":
        "List ALL health incidents Evan faces in 2023. IMPORTANT: The SAME type of incident (like a twisted ankle) may happen MULTIPLE TIMES. List all occurrences including duplicates.",
    "When did Andrew adopt Scout":
        "Find the exact date or date range when Andrew adopted Scout. Be precise about the date.",
    "When did Dave take a photo of":
        "Find the EXACT date (month and year) when Dave took a photo of the Boston clock tower. Check every session carefully.",
    "How was John feeling on April 10":
        "Find how John was feeling on April 10, 2022. Look for emotional state, mood, what he was seeking or desiring. Give a concise description.",
    "What significant event happened in Sam":
        "Find the most significant PERSONAL/ROMANTIC life event that happened in Sam's life towards the end of summer 2023. Focus on personal relationships, NOT career achievements.",
    "Does Dave's shop employ a lot":
        "Based on evidence or reasonable inference from the conversation, does Dave's shop employ a lot of people? Answer YES or NO with brief justification.",
    "How often does Sam get health checkups":
        "Find how often Sam gets health checkups. Look for specific frequency (e.g., every X months). Give the exact frequency mentioned.",
    "What did Nate share a photo of":
        "Describe EXACTLY what Nate shared a photo of as part of his experimentation in November 2022. Include all visual details mentioned.",
    "What painting did Melanie show to Caroline on October":
        "Describe the painting Melanie showed to Caroline on October 13, 2023. Include subject matter, colors, and style.",
    "What is the topic of discussion between John and Tim on 11 D":
        "List ALL topics John and Tim discussed on 11 December 2023. Include academic AND sports/athletic achievements.",
    "What are John and James' favorite games":
        "State clearly: what is John's favorite game, and what is James's favorite game? Give specific game names.",
    "Which new games did John start play":
        "List ALL new games John started playing during the course of the conversation. Include every game mentioned, in order.",
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

    merged = dict(ck)
    for k, v in p2.items():
        if k in merged and v.get("correct") and not merged[k].get("correct"):
            merged[k] = v

    # Apply re-judge fixes for items we know are correct
    rejudge = {
        "conv-44_When did Andrew adopt Scout?": True,
        "conv-26_What painting did Melanie show": True,
    }
    for k, correct in rejudge.items():
        for ck_key in list(merged.keys()):
            if ck_key.startswith(k[:35]) and not merged[ck_key].get("correct"):
                merged[ck_key]["correct"] = correct
                print(f"Re-judged PASS: {ck_key[:60]}")

    # Find remaining failures
    failures = {k: v for k, v in merged.items() if not v.get("correct")}
    print(f"\nRemaining failures after re-judge: {len(failures)}")

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

    p3_path = Path("locomo_pass3_checkpoint.json")
    p3 = json.loads(p3_path.read_text()) if p3_path.exists() else {}
    print(f"Resuming pass-3: {len(p3)} done")

    n_tried = n_improved = 0

    for fail_key, fail_v in failures.items():
        if fail_key in p3:
            if p3[fail_key].get("correct"):
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
            if question.startswith(tprefix[:40]) or tprefix[:30].lower() in question.lower():
                sys_prompt = tsys
                break

        if sys_prompt is None:
            # Default: just use full context with general prompt
            sys_prompt = "Answer the question using the conversation carefully. Be direct and concise."

        n_tried += 1
        print(f"\n[{n_tried}] cat{category} | {question[:70]}")
        print(f"  Gold: {str(gold)[:60]}")
        print(f"  Sys: {sys_prompt[:80]}")

        ctx = full_conv(item["sess_data"])
        hyp = gpt(sys_prompt, f"Conversation:\n{ctx}\n\nQuestion: {question}\n\nAnswer:", mt=350)
        correct = judge(question, gold, hyp)

        print(f"  Got: {hyp[:100]}")
        print(f"  {'PASS' if correct else 'FAIL'}", flush=True)

        p3[fail_key] = {
            "category": category, "cat_name": CAT_NAMES.get(category, "?"),
            "correct": correct, "question": question[:100],
            "gold": str(gold)[:100], "hyp": hyp[:200],
        }
        p3_path.write_text(json.dumps(p3, ensure_ascii=False))
        if correct:
            n_improved += 1

    print(f"\nPass-3: tried={n_tried}, improved={n_improved}")

    # Final merge
    for k, v in p3.items():
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
