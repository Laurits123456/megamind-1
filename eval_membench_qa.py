"""
Membench QA eval — multiple choice QA on RecMultiSession.json
Format: 500 conversations, each with a multi-choice QA (A/B/C/D)
Strategy: full conversation context + GPT-4.1 picks best choice
"""
from __future__ import annotations
import json, re, time, sys
from pathlib import Path
import numpy as np
import openai
from sentence_transformers import SentenceTransformer

OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')
client = openai.OpenAI(api_key=OPENAI_KEY)

STOP = {'the','a','an','i','my','did','do','how','many','what','when','which','is','are','was','were',
        'have','has','been','in','on','at','to','of','for','and','or','does','had','this','that',
        'with','by','about','its','you','me','we','our','can','could','would','should','also',
        'just','get','make','set','use','well','now','then','than','from','they','them','their',
        'your','who','where','there','here','some','any','all','recommend','recommended','tell',
        'said','mentioned','ever','give','given','ask'}


def gpt(model, system, user, mt=10, temp=0):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=mt, messages=msgs, timeout=90, temperature=temp
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error: {e}")
            if attempt < 2:
                time.sleep(3 ** attempt)
    return ""


def get_turns(message_list):
    turns = []
    for si, session in enumerate(message_list):
        for msg in session:
            if isinstance(msg, dict):
                u = msg.get('user', '')
                a = msg.get('assistant', '')
                t = msg.get('time', '')
                if u or a:
                    turns.append({'si': si, 'time': t, 'user': u, 'asst': a})
    return turns


def full_conv(turns):
    parts = []
    for t in turns:
        sess = f"[Session {t['si']+1}, {t['time']}]" if t.get('time') else f"[Session {t['si']+1}]"
        parts.append(f"{sess}\nUser: {t['user']}\nAssistant: {t['asst']}")
    return "\n\n".join(parts)


def format_choices(choices):
    lines = []
    for letter in ['A', 'B', 'C', 'D']:
        if letter in choices:
            items = choices[letter]
            items_str = ", ".join(items) if isinstance(items, list) else str(items)
            lines.append(f"{letter}: {items_str}")
    return "\n".join(lines)


def answer(question, choices, turns):
    ctx = full_conv(turns)
    choices_str = format_choices(choices)
    system = (
        "You are answering multiple-choice questions about a conversation. "
        "Read the conversation carefully and pick the BEST answer choice. "
        "Respond with ONLY the single letter A, B, C, or D."
    )
    user = (
        f"Conversation:\n{ctx[:50000]}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choices_str}\n\n"
        f"Answer (A, B, C, or D):"
    )
    resp = gpt("gpt-4.1", system, user, mt=5)
    m = re.search(r'\b([ABCD])\b', resp.upper())
    if m:
        return m.group(1)
    for ch in resp.upper():
        if ch in 'ABCD':
            return ch
    return 'A'


def main():
    print("Loading Membench...")
    with open('data/membench/RecMultiSession.json', encoding='utf-8') as f:
        data = json.load(f)
    items = data['multi_agent']
    print(f"Total: {len(items)}")

    ck_path = Path('eval_membench_qa_checkpoint.json')
    checkpoint = {}
    if ck_path.exists():
        checkpoint = json.loads(ck_path.read_text(encoding='utf-8'))
        print(f"Resuming: {len(checkpoint)} done")

    n_correct = n_total = 0

    for item in items:
        tid = str(item.get('tid', ''))
        qa = item.get('QA', {})
        if isinstance(qa, list):
            qa = qa[0] if qa else {}
        if not qa:
            continue

        question = qa.get('question', '')
        choices = qa.get('choices', {})
        gt = qa.get('ground_truth', '')
        qid = str(qa.get('qid', tid))
        key = f"{tid}_{qid}"

        if not question or not choices or not gt:
            continue
        n_total += 1

        if key in checkpoint:
            if checkpoint[key].get('predicted') == gt:
                n_correct += 1
            continue

        turns = get_turns(item.get('message_list', []))
        predicted = answer(question, choices, turns)
        correct = predicted == gt

        checkpoint[key] = {
            'tid': tid, 'question': question[:100],
            'predicted': predicted, 'ground_truth': gt, 'correct': correct,
        }
        ck_path.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding='utf-8')

        if correct:
            n_correct += 1
        if n_total % 50 == 0:
            print(f"[{n_total}/{len(items)}] acc={n_correct/n_total*100:.1f}%", flush=True)
        elif not correct:
            print(f"  FAIL [{n_total}] Q={question[:60]} | GT={gt} | Got={predicted}", flush=True)

    print(f"\nMEMBENCH: {n_correct}/{n_total} = {n_correct/n_total*100:.2f}%")


if __name__ == "__main__":
    main()
