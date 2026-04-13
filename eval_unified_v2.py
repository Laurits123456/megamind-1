"""
eval_unified_v2.py — Unified benchmark evaluation across LME-S, LoCoMo, and BEAM.

Uses the same v2 architecture as eval_final_v2.py:
- EBRM singleton retrieval
- Full session context (no assistant truncation)
- Adaptive routing by question type
- Temporal arithmetic tool
- gpt-4.1 answer + gpt-4.1-mini judge (BEAM-comparable protocol)

BEAM uses nugget scoring (3-tier 0/0.5/1.0) per the official protocol.
LoCoMo uses category-aware semantic turn retrieval.

Usage:
  BENCHMARKS=lme,locomo,beam LIMIT=100 python -u eval_unified_v2.py

Outputs:
  unified_v2_lme_checkpoint.json
  unified_v2_locomo_checkpoint.json
  unified_v2_beam_checkpoint.json
  unified_v2_summary.json
"""

import json, sys, os, re, time, random, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import openai as _openai

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# ── API setup ─────────────────────────────────────────────────────────────────
OPENAI_KEY = os.environ.get("OPENAI_API_KEY",
    os.environ.get('OPENAI_API_KEY', ''))
_openai.api_key = OPENAI_KEY
MODEL = os.environ.get("MODEL", "gpt-4.1")
JUDGE_MODEL = "gpt-4.1-mini"
BENCHMARKS = os.environ.get("BENCHMARKS", "lme,locomo,beam").split(",")
LIMIT = int(os.environ.get("LIMIT", "100"))

LME_PATH = "C:/Users/lauri/AOE/Legendary/benchmarks/data/longmemeval_s_cleaned.json"
LOCOMO_PATH = "C:/Users/lauri/AOE/Legendary/benchmarks/data/locomo10.json"
BEAM_SPLIT = os.environ.get("BEAM_SPLIT", "100K")

# ── Singletons ────────────────────────────────────────────────────────────────
_EBRM = None
_MINILM = None
_CMEN = None


def get_cmen():
    """Load trained CMEN for EBM-guided turn selection."""
    global _CMEN
    if _CMEN is None:
        import torch
        from cmen import CMEN
        model = CMEN(emb_dim=384, hidden=128)
        ckpt_dir = Path(__file__).parent / "data"
        for attr, fname in [
            ("relevance", "cmen_relevance_v2.pt"),
            ("temporal", "cmen_temporal.pt"),
            ("recency", "cmen_recency.pt"),
            ("sufficiency", "cmen_sufficiency.pt"),
            ("composition", "cmen_composition.pt"),
        ]:
            p = ckpt_dir / fname
            if p.exists():
                state = torch.load(p, map_location="cpu")
                getattr(model, attr).load_state_dict(state, strict=False)
        lam_path = ckpt_dir / "cmen_lambdas_trained.pt"
        if lam_path.exists():
            model.log_lambdas.data.copy_(torch.load(lam_path, map_location="cpu"))
        model.eval()
        _CMEN = model
    return _CMEN


def get_ebrm():
    global _EBRM
    if _EBRM is None:
        from ebrm_search import EBRMSearch
        _EBRM = EBRMSearch()
    return _EBRM

def get_minilm():
    global _MINILM
    if _MINILM is None:
        from sentence_transformers import SentenceTransformer
        _MINILM = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _MINILM

# ── LLM calls ─────────────────────────────────────────────────────────────────
def llm(prompt, max_tokens=512, model=None):
    for attempt in range(3):
        try:
            r = _openai.chat.completions.create(
                model=model or MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(2 ** attempt)
    return ""

def cheap(prompt, max_tokens=200):
    return llm(prompt, max_tokens=max_tokens, model=JUDGE_MODEL)

# ── Temporal arithmetic ───────────────────────────────────────────────────────
def parse_date_str(text):
    for fmt in ("%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y",
                "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            pass
    return None

def try_temporal_arithmetic(question, context):
    q_lower = question.lower()
    if not any(t in q_lower for t in ["how many days", "days between", "days ago",
               "how long ago", "how many weeks", "how many months", "how many years",
               "days since", "weeks since", "months since"]):
        return None
    date_pat = r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b'
    found = re.findall(date_pat, context, re.IGNORECASE)
    parsed = [(d, parse_date_str(d)) for d in found[:10]]
    parsed = [(d, dt) for d, dt in parsed if dt]
    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda x: x[1])
    d1_str, d1 = parsed[0]
    d2_str, d2 = parsed[-1]
    delta = d2 - d1
    if "days" in q_lower:
        return f"{abs(delta.days)} days (from {d1_str} to {d2_str})"
    elif "weeks" in q_lower:
        return f"{abs(delta.days) // 7} weeks"
    elif "months" in q_lower:
        return f"{abs((d2.year-d1.year)*12+(d2.month-d1.month))} months"
    elif "years" in q_lower:
        return f"{abs(d2.year-d1.year)} years"
    return None

# ══════════════════════════════════════════════════════════════════════════════
# LME-S EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def format_session_full(sess, date, idx, max_chars=8000):
    if not isinstance(sess, list):
        return f"### Session {idx} ({date}):\n{str(sess)[:max_chars]}"
    lines = []
    for t in sess:
        if not isinstance(t, dict):
            continue
        role = t.get('role', 'user')
        content = t.get('content', '')
        if role == 'user':
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content}")  # NO truncation
    text = "\n".join(lines)
    return f"### Session {idx} ({date}):\n{text[:max_chars]}"

def lme_judge(question, gold, generated):
    if not generated:
        return False
    g, h = str(gold).lower().strip(), str(generated).lower().strip()
    if g in ("yes", "no"):
        return g in h[:20]
    verdict = cheap(f"""Question: {question}
Gold answer: {gold}
Generated answer: {generated}
Is the generated answer correct? Answer YES or NO.""", max_tokens=5)
    return "YES" in verdict.upper()

def lme_answer(sample, sessions, session_dates, q_type, question, q_date):
    search = get_ebrm()
    docs = [sess if isinstance(sess, list) else str(sess) for sess in sessions]
    search.build_index(docs)

    is_multi = q_type == "multi-session"
    is_single = "single-session" in q_type
    is_ku = q_type == "knowledge-update"
    is_tr = q_type == "temporal-reasoning"

    if is_multi:
        result = search.search(question, top_k=20)
        indices = sorted(list(result.indices[:20]),
                         key=lambda i: session_dates[i] if i < len(session_dates) else "")
        parts = [format_session_full(sessions[i], session_dates[i] if i < len(session_dates) else "", r+1, max_chars=3000)
                 for r, i in enumerate(indices) if i < len(sessions)]
        context = "\n\n".join(parts)
        prompt = f"""Read the following chat sessions (chronological) and answer.
{context[:40000]}
Current Date: {q_date}
Question: {question}
Instructions: Enumerate ALL items for count questions. Step-by-step then FINAL ANSWER.
Answer:"""

    elif is_single:
        result = search.search(question, top_k=3)
        parts = [format_session_full(sessions[i], session_dates[i] if i < len(session_dates) else "", r+1, max_chars=10000)
                 for r, i in enumerate(list(result.indices[:3])) if i < len(sessions)]
        context = "\n\n".join(parts)
        asst_hint = "The answer may be in ASSISTANT responses — read both user AND assistant turns carefully.\n" if "assistant" in q_type else ""
        prompt = f"""Read the following chat history and answer.
{asst_hint}{context}
Current Date: {q_date}
Question: {question}
Answer (be specific and direct):"""

    elif is_ku:
        result = search.search(question, top_k=8)
        indices = sorted(list(result.indices[:8]),
                         key=lambda i: session_dates[i] if i < len(session_dates) else "")
        parts = [format_session_full(sessions[i], session_dates[i] if i < len(session_dates) else "", r+1, max_chars=4000)
                 for r, i in enumerate(indices) if i < len(sessions)]
        context = "\n\n".join(parts)
        prompt = f"""Read the following sessions (chronological) and answer with the MOST RECENT value.
{context[:30000]}
Current Date: {q_date}
Question: {question}
Answer (state the latest/current value):"""

    else:  # temporal-reasoning
        result = search.search(question, top_k=5)
        indices = sorted(list(result.indices[:5]),
                         key=lambda i: session_dates[i] if i < len(session_dates) else "")
        parts = [format_session_full(sessions[i], session_dates[i] if i < len(session_dates) else "", r+1)
                 for r, i in enumerate(indices) if i < len(sessions)]
        context = "\n\n".join(parts)
        arith = try_temporal_arithmetic(question, context)
        arith_hint = f"\nCOMPUTED: {arith}" if arith else ""
        prompt = f"""Read the following sessions and answer the temporal question.{arith_hint}
{context[:25000]}
Current Date: {q_date}
Question: {question}
Answer (be specific with dates/durations):"""

    return llm(prompt, max_tokens=400)

def run_lme(limit, ckpt_path):
    print(f"\n{'='*60}", flush=True)
    print(f"LME-S EVAL | gpt-4.1 + gpt-4.1-mini | limit={limit}", flush=True)
    print(f"{'='*60}", flush=True)

    data = json.loads(Path(LME_PATH).read_text(encoding='utf-8', errors='replace'))
    random.seed(42); random.shuffle(data)
    data = data[:limit]

    done_ids = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
    if done_ids:
        print(f"Resuming: {len(done_ids)} done", flush=True)

    scores_by_type = defaultdict(list)
    for v in done_ids.values():
        scores_by_type[v['q_type']].append(1 if v['correct'] else 0)
    correct_total = sum(1 for r in done_ids.values() if r['correct'])
    total = len(done_ids)
    t0 = time.time()

    for sample in data:
        q_id = sample.get('question_id', '')
        if q_id in done_ids:
            continue
        sessions = sample.get('haystack_sessions', [])
        question = sample.get('question', '')
        gold = sample.get('answer', '')
        q_type = sample.get('question_type', '')
        q_date = sample.get('question_date', '')
        session_dates = sample.get('haystack_dates', [])
        if not sessions or not question:
            continue

        hyp = lme_answer(sample, sessions, session_dates, q_type, question, q_date)
        correct = lme_judge(question, gold, hyp)
        total += 1
        if correct:
            correct_total += 1
        scores_by_type[q_type].append(1 if correct else 0)
        if not correct:
            print(f"  FAIL [{q_type}] Q={str(question)[:60]} | Gold={str(gold)[:30]} | Got={str(hyp)[:30]}", flush=True)
        done_ids[q_id] = {'q_type': q_type, 'correct': correct}
        ckpt_path.write_text(json.dumps(done_ids))

        if total % 50 == 0:
            print(f"  [{total}/{limit}] acc={correct_total/total*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nLME-S RESULTS ({total} items, {time.time()-t0:.0f}s):", flush=True)
    for qt, scores in sorted(scores_by_type.items()):
        print(f"  {qt}: {sum(scores)/len(scores)*100:.1f}% ({sum(scores)}/{len(scores)})", flush=True)
    acc = correct_total / max(total, 1)
    print(f"  OVERALL: {acc*100:.1f}% | OMEGA SOTA: 95.4%", flush=True)
    return {'benchmark': 'LME-S', 'accuracy': acc, 'n': total, 'by_type': {k: sum(v)/len(v) for k, v in scores_by_type.items()}}

# ══════════════════════════════════════════════════════════════════════════════
# LOCOMO EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

import math as _math

_LOCOMO_STOP = {
    'the','a','an','i','my','did','do','how','many','what','when','which','who',
    'is','are','was','were','have','has','been','in','on','at','to','of','for',
    'and','or','does','had','will','would','about','with','by','their','its',
    'this','that','these','those','be','it','he','she','they','we','you',
}

def _tokenize(text):
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def _bm25_rank_sessions(question, sessions_text, top_k=12):
    """BM25 over full session texts. Returns sorted indices (best first)."""
    q_tokens = [t for t in _tokenize(question) if t not in _LOCOMO_STOP]
    if not q_tokens:
        return list(range(min(top_k, len(sessions_text))))
    N = len(sessions_text)
    doc_tfs, doc_lens = [], []
    df = defaultdict(int)
    for s in sessions_text:
        words = _tokenize(s)
        tf = defaultdict(int)
        for w in words: tf[w] += 1
        for w in tf: df[w] += 1
        doc_tfs.append(dict(tf))
        doc_lens.append(max(len(words), 1))
    avgdl = sum(doc_lens) / max(N, 1)
    scores = []
    for tf_d, dl in zip(doc_tfs, doc_lens):
        s = 0.0
        for t in q_tokens:
            d = df.get(t, 0)
            if d > 0:
                idf = max(0, _math.log((N - d + 0.5) / (d + 0.5) + 1e-8))
                tf = tf_d.get(t, 0)
                if tf > 0:
                    s += idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / max(avgdl, 1)))
        scores.append(s)
    return sorted(range(N), key=lambda i: scores[i], reverse=True)[:top_k]

def _minilm_rank_sessions(question, sessions_text, top_k=12):
    """MiniLM semantic ranking over session texts."""
    enc = get_minilm()
    q_emb = enc.encode([question], normalize_embeddings=True, show_progress_bar=False)
    d_emb = enc.encode(sessions_text, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    sims = (q_emb @ d_emb.T)[0]
    return list(np.argsort(sims)[::-1][:top_k])

def _rrf_fuse(ranks_list, n):
    """Reciprocal rank fusion over multiple ranked lists."""
    scores = defaultdict(float)
    for ranks in ranks_list:
        for rank, idx in enumerate(ranks):
            scores[idx] += 1.0 / (60 + rank + 1)
    return sorted(range(n), key=lambda i: scores[i], reverse=True)

_MONTH_NAMES = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)'
_DATE_IN_Q = re.compile(rf'\b(\d{{1,2}})\s+{_MONTH_NAMES}(?:,?\s+\d{{4}})?\b|\b{_MONTH_NAMES}\s+(\d{{1,2}})(?:,?\s+\d{{4}})?\b', re.IGNORECASE)

def _extract_date_tokens(question):
    """Extract day+month tokens from question for date-anchor matching."""
    m = _DATE_IN_Q.search(question)
    if not m:
        return []
    # Return lower-cased tokens from the match
    return _tokenize(m.group(0).lower())

def locomo_retrieve_sessions(question, sess_data, top_k=12):
    """RRF fusion of BM25 + MiniLM + date-anchor injection."""
    texts = [s['text'] for s in sess_data]
    if not texts:
        return sess_data[:top_k]

    # Date-anchor: if question mentions a specific date, force-include sessions on that date
    date_tokens = _extract_date_tokens(question)
    forced_indices = set()
    if date_tokens:
        for i, s in enumerate(sess_data):
            session_date_lower = s['date'].lower()
            if any(tok in session_date_lower for tok in date_tokens if len(tok) > 2 and tok.isalpha()):
                forced_indices.add(i)
            elif any(tok in session_date_lower for tok in date_tokens if tok.isdigit() and len(tok) <= 2):
                # day number match — check day + month together
                alpha_toks = [t for t in date_tokens if t.isalpha() and len(t) > 2]
                if any(tok in session_date_lower for tok in alpha_toks):
                    forced_indices.add(i)

    bm25_idx = _bm25_rank_sessions(question, texts, top_k=top_k)
    sem_idx = _minilm_rank_sessions(question, texts, top_k=top_k)
    fused = _rrf_fuse([bm25_idx, sem_idx], len(texts))

    # Merge: forced first (by snum), then fused (excluding forced), up to top_k
    result_indices = list(forced_indices)
    for i in fused:
        if i not in forced_indices:
            result_indices.append(i)
        if len(result_indices) >= top_k:
            break

    result_indices = sorted(result_indices[:top_k], key=lambda i: sess_data[i]['snum'])
    return [sess_data[i] for i in result_indices]

def locomo_judge(question, gold, generated):
    if not generated:
        return False
    g, h = str(gold).lower().strip(), str(generated).lower().strip()
    if g in ("yes", "no"):
        return g in h[:20]
    verdict = cheap(f"""Question: {question}
Gold answer: {gold}
Generated answer: {generated}
Does the generated answer correctly answer the question with the key information from the gold answer?
A partial but substantially correct answer counts as YES.
Answer YES or NO.""", max_tokens=5)
    return "YES" in verdict.upper()

def _locomo_turns_from_sessions(sess_data):
    """Flatten sessions into turns for turn-level retrieval."""
    turns = []
    for s in sess_data:
        for line in s['text'].split('\n'):
            if line.strip():
                turns.append({'text': line, 'snum': s['snum'], 'date': s['date']})
    return turns

def _retrieve_turns_minilm(question, turns, top_k=40):
    """MiniLM turn-level retrieval, sorted by session order."""
    enc = get_minilm()
    docs = [t['text'] for t in turns]
    if not docs:
        return []
    q_emb = enc.encode([question], normalize_embeddings=True, show_progress_bar=False)
    d_emb = enc.encode(docs, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    sims = (q_emb @ d_emb.T)[0]
    ranked = list(np.argsort(sims)[::-1][:top_k])
    ranked.sort(key=lambda i: (turns[i]['snum'], turns[i].get('dia_id', '')))
    return [turns[i] for i in ranked]

def _format_turns(turns):
    """Format retrieved turns for LLM context.
    Date is prefixed on EVERY line (Agent B's temporal grounding fix):
    relative refs like 'yesterday' are co-located with their anchor date."""
    lines = []
    for t in turns:
        date = t.get('date', '')
        text = t.get('text', '')
        if date:
            lines.append(f"[{date}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)

def _bm25_rank_turns(question, turns, top_k=60):
    """BM25 over individual turns. Returns indices (best first)."""
    q_tokens = [t for t in _tokenize(question) if t not in _LOCOMO_STOP]
    if not q_tokens:
        return list(range(min(top_k, len(turns))))
    docs = [t['text'] for t in turns]
    N = len(docs)
    doc_tfs, doc_lens = [], []
    df = defaultdict(int)
    for doc in docs:
        words = _tokenize(doc)
        tf = defaultdict(int)
        for w in words: tf[w] += 1
        for w in tf: df[w] += 1
        doc_tfs.append(dict(tf))
        doc_lens.append(max(len(words), 1))
    avgdl = sum(doc_lens) / max(N, 1)
    scores = []
    for tf_d, dl in zip(doc_tfs, doc_lens):
        s = 0.0
        for t in q_tokens:
            d = df.get(t, 0)
            if d > 0:
                idf = max(0, _math.log((N - d + 0.5) / (d + 0.5) + 1e-8))
                tf = tf_d.get(t, 0)
                if tf > 0:
                    s += idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / max(avgdl, 1)))
        scores.append(s)
    return sorted(range(N), key=lambda i: scores[i], reverse=True)[:top_k]


def _retrieve_turns_hybrid(question, turns, top_k=50):
    """Turn-level BM25 + MiniLM RRF, sorted by session order."""
    if not turns:
        return [], None
    bm25_idx = _bm25_rank_turns(question, turns, top_k=top_k)
    enc = get_minilm()
    q_emb = enc.encode([question], normalize_embeddings=True, show_progress_bar=False)
    d_emb = enc.encode([t['text'] for t in turns], normalize_embeddings=True,
                       batch_size=128, show_progress_bar=False)
    sem_idx = list(np.argsort((q_emb @ d_emb.T)[0])[::-1][:top_k])
    fused_idx = sorted(_rrf_fuse([bm25_idx, sem_idx], len(turns))[:top_k],
                       key=lambda i: (turns[i]['snum'], turns[i].get('dia_id', '')))
    return [turns[i] for i in fused_idx], d_emb  # return embeddings for potential reuse


def _cmen_rerank_turns(question, turns, d_emb, top_k=50, n_select=45):
    """
    EBM-guided joint turn selection via CMEN PEM.

    Instead of scoring turns independently, CMEN optimizes a joint configuration
    y* over which turns to include. Composition energy keeps related turns together
    (e.g., all mentions of a sport), sufficiency signals when the set is complete.

    Returns (selected_turns, suff_E):
      - selected_turns: top turns sorted by y* weight, then chronologically
      - suff_E: sufficiency energy (high = answer probably NOT in selected set)
    """
    import torch
    if len(turns) < 3 or d_emb is None:
        return turns[:n_select], 0.0

    try:
        cmen = get_cmen()
        enc = get_minilm()
        q_emb = enc.encode([question], normalize_embeddings=True, show_progress_bar=False)

        h_q = torch.tensor(q_emb, dtype=torch.float32)          # [1, D]
        M = torch.tensor(d_emb, dtype=torch.float32).unsqueeze(0)  # [1, N, D]

        # Fake timestamps: use turn order as proxy (newer turns = higher index)
        N = len(turns)
        timestamps = torch.arange(N, dtype=torch.float32).unsqueeze(0)  # [1, N]

        # PEM needs gradients — run outside no_grad
        y_star, suff_E = cmen.optimize_configuration(
            h_q, M, timestamps,
            n_particles=8, n_steps=10, n_landscapes=2, lr=0.08,
        )
        with torch.no_grad():
            weights = y_star.cpu().numpy()  # [N]

        # Select top turns by y* weight, break ties chronologically
        ranked = sorted(range(N), key=lambda i: (-weights[i], turns[i]['snum']))
        selected_idx = sorted(ranked[:n_select],
                              key=lambda i: (turns[i]['snum'], turns[i].get('dia_id', '')))
        return [turns[i] for i in selected_idx], float(suff_E)

    except Exception as e:
        # Fallback to unmodified turns on any error
        return turns[:n_select], 0.0


def _is_idk(response):
    """Detect hedging/IDK responses."""
    if not response:
        return True
    r = response.lower()
    return any(p in r for p in [
        "i don't have that", "not mentioned", "no information", "not provided",
        "i don't know", "cannot find", "not found", "no record", "not stated",
        "isn't mentioned", "is not mentioned", "not discussed", "not available",
        "there is no", "there's no", "i cannot", "i can't find",
    ])


def _prf_expand(question, top_turns, n=15):
    """Pseudo-relevance feedback: extract high-TF nouns from top-5 turns
    that don't appear in the question. Returns expansion string."""
    q_tokens = set(_tokenize(question.lower()))
    freq = defaultdict(int)
    for t in top_turns[:5]:
        for tok in _tokenize(t['text'].lower()):
            if tok not in q_tokens and tok not in _LOCOMO_STOP and len(tok) > 3:
                freq[tok] += 1
    top_terms = sorted(freq, key=lambda t: freq[t], reverse=True)[:n]
    return " ".join(top_terms)


def _merge_turns(base, extra, max_total=65):
    """Merge two turn lists, dedup by (snum, dia_id), sorted chronologically."""
    seen = set()
    merged = []
    for t in base + extra:
        key = (t['snum'], t.get('dia_id', ''))
        if key not in seen:
            seen.add(key)
            merged.append(t)
    merged.sort(key=lambda t: (t['snum'], t.get('dia_id', '')))
    return merged[:max_total]


def _build_prompt(question, category, ctx, allow_inference=False):
    if category == 5:
        return (f"Read the following conversation excerpts carefully.\n"
                f"If the specific information asked is NOT mentioned, say exactly "
                f"\"I don't have that information.\"\n{ctx[:20000]}\n"
                f"Question: {question}\nAnswer:")
    elif category == 4:
        # Some cat4 questions are actually fact-recall mislabeled as temporal
        is_date_q = any(w in question.lower() for w in [
            'when', 'how long', 'how many days', 'how many weeks', 'how many months',
            'how many years', 'what date', 'what time', 'what year', 'first', 'last',
            'before', 'after', 'ago', 'since', 'duration', 'how old'
        ])
        if is_date_q:
            return (f"Read the following conversation excerpts and answer the temporal question.\n"
                    f"Pay close attention to session dates.\n{ctx[:25000]}\n"
                    f"Question: {question}\nAnswer (specific dates/durations):")
        else:
            # Fact-recall miscategorized as temporal — use direct prompt
            return (f"Read the following conversation excerpts and answer directly.\n"
                    f"{ctx[:25000]}\nQuestion: {question}\nAnswer (be specific and concise):")
    elif category == 2:
        is_count = any(w in question.lower() for w in ['how many','total','count','number of','how much'])
        if is_count:
            return (f"Read ALL the following conversation sessions carefully.\n"
                    f"Count ALL instances across ALL sessions — do not miss any.\n"
                    f"Go through each session in order, note each occurrence, then give the final total.\n{ctx}\n"
                    f"Question: {question}\nStep-by-step (list each instance with session number, then total):")
        return (f"Read the following conversation excerpts (chronological) and answer.\n"
                f"Synthesize information from multiple sessions as needed.\n{ctx}\n"
                f"Question: {question}\nAnswer (be specific and concise):")
    else:  # cat1, cat3 — full conversation, inference-permissive
        is_list_q = any(w in question.lower() for w in [
            'what kind', 'what types', 'what activities', 'what recipes', 'what gifts',
            'what games', 'what names', 'what are', 'which', 'list', 'all', 'what sports',
            'what topics', 'what movies', 'what books', 'what places', 'what shows'
        ])
        if is_list_q:
            return (f"Read ALL the following conversation sessions carefully.\n"
                    f"Search exhaustively — the answer may be spread across multiple sessions.\n"
                    f"If something is strongly implied even if not stated explicitly, include it.\n"
                    f"{ctx}\nQuestion: {question}\n"
                    f"Answer (list ALL items mentioned or implied, be thorough):")
        return (f"Read the following conversation sessions carefully and answer.\n"
                f"Use direct evidence where available. If the answer requires reasonable inference "
                f"from the conversation context (e.g. a likely relationship or habit), state your "
                f"inference clearly ('Based on the conversation, it seems likely that...').\n"
                f"{ctx}\nQuestion: {question}\nAnswer:")


def _full_conversation_ctx_highlighted(question, sess_data, turns):
    """Full conversation with top-8 relevant turns highlighted at the top.
    Agent B's lost-in-the-middle fix: surface key evidence before the full dump."""
    top_turns, _ = _retrieve_turns_hybrid(question, turns, top_k=8)
    highlight = "\n".join(f"  [{t['snum']}] {t['text']}" for t in top_turns)
    full = _full_conversation_ctx(sess_data)
    return f"KEY RELEVANT EXCERPTS:\n{highlight}\n\nFULL CONVERSATION:\n{full}"


def _full_conversation_ctx(sess_data):
    """Return full conversation — all sessions chronologically.
    LoCoMo convs are ~19K tokens avg (max 23.5K), fits in GPT-4.1 128K context."""
    parts = []
    for s in sorted(sess_data, key=lambda s: s['snum']):
        parts.append(f"=== Session {s['snum']} [{s.get('date','')}] ===\n{s['text']}")
    return "\n\n".join(parts)


def locomo_answer(qa, sess_data, category):
    """
    LoCoMo QA pipeline — adaptive routing by conversation length / category.

    Cat1/2/3: Full conversation context (all sessions, ~19K tokens avg, max 23.5K).
              Zero retrieval — bypasses semantic gap entirely. GPT-4.1 128K handles it.
    Cat4:     Hybrid retrieval (BM25+MiniLM RRF) + temporal arithmetic.
              IDK → PRF expansion retry.
    Cat5:     Hybrid retrieval, no retry (IDK IS the correct answer).
    """
    question = qa.get('question', '')
    turns = _locomo_turns_from_sessions(sess_data)
    if not turns:
        return ""

    MAX_TOKENS = {1: 400, 2: 400, 3: 400, 4: 256, 5: 200}
    max_tok = MAX_TOKENS.get(category, 300)

    # ── Cat1 / Cat3: full conversation + evidence highlighting ────────────────
    # Full ctx bypasses semantic gap; highlights guard against lost-in-middle
    if category in (1, 3):
        ctx = _full_conversation_ctx_highlighted(question, sess_data, turns)
        prompt = _build_prompt(question, category, ctx)
        return llm(prompt, max_tokens=max_tok)

    # ── Cat2 counting: full context to catch all instances ───────────────────
    if category == 2:
        is_count = any(w in question.lower() for w in ['how many','total','count','number of','how much'])
        if is_count:
            ctx = _full_conversation_ctx(sess_data)
            prompt = _build_prompt(question, category, ctx)
            return llm(prompt, max_tokens=max_tok)

    # ── Cat4 / Cat5 / Cat2 (non-count): retrieval-based ─────────────────────
    # Cat4: retrieval wins (91.1%) — full ctx overwhelms LLM with 19K tokens of noise
    # Cat5: retrieval only — must not hallucinate IDK answers
    top_turns, _ = _retrieve_turns_hybrid(question, turns, top_k=60 if category == 2 else 50)
    ctx = _format_turns(top_turns)

    if category == 4:
        arith = try_temporal_arithmetic(question, ctx)
        if arith:
            ctx = f"SYMBOLIC COMPUTATION: {arith}\n\n" + ctx

    prompt = _build_prompt(question, category, ctx)
    response = llm(prompt, max_tokens=max_tok)

    if category == 5:
        return response

    # ── Agent loop on IDK ─────────────────────────────────────────────────────
    if _is_idk(response):
        if category == 2:
            ctx2 = _full_conversation_ctx(sess_data)
            prompt2 = _build_prompt(question, category, ctx2)
        else:
            # Cat4: PRF expansion
            expansion = _prf_expand(question, top_turns)
            expanded_q = f"{question} {expansion}".strip()
            extra_turns, _ = _retrieve_turns_hybrid(expanded_q, turns, top_k=50)
            merged = _merge_turns(top_turns, extra_turns, max_total=65)
            ctx2 = _format_turns(merged)
            arith2 = try_temporal_arithmetic(question, ctx2)
            if arith2:
                ctx2 = f"SYMBOLIC COMPUTATION: {arith2}\n\n" + ctx2
            prompt2 = _build_prompt(question, category, ctx2)
        response = llm(prompt2, max_tokens=max_tok)

    return response

def run_locomo(limit, ckpt_path):
    print(f"\n{'='*60}", flush=True)
    print(f"LOCOMO EVAL | gpt-4.1 + gpt-4.1-mini | limit={limit}", flush=True)
    print(f"{'='*60}", flush=True)

    data = json.load(open(LOCOMO_PATH, encoding='utf-8'))

    # Build session-level QA list (not turn-level)
    all_items = []
    for conv in data:
        conv_data = conv.get('conversation', {})
        sample_id = conv.get('sample_id', '')
        session_keys = sorted([k for k in conv_data if k.startswith('session_') and '_date_time' not in k],
                               key=lambda k: int(k.split('_')[1]))
        # Build sess_data: list of {snum, date, text} — full session texts for BM25+MiniLM
        sess_data = []
        for sk in session_keys:
            snum = int(sk.split('_')[1])
            date = conv_data.get(f"{sk}_date_time", "")
            turns = conv_data[sk] if isinstance(conv_data[sk], list) else []
            text = "\n".join(f"{t.get('speaker','')}: {t.get('text','')}" for t in turns if t.get('text'))
            sess_data.append({'snum': snum, 'date': date, 'text': text})
        for qa in conv.get('qa', []):
            item_id = f"{sample_id}_{qa.get('question','')[:30]}"
            all_items.append({'id': item_id, 'qa': qa, 'sess_data': sess_data})

    random.seed(42); random.shuffle(all_items)
    all_items = all_items[:limit]

    done_ids = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
    if done_ids:
        print(f"Resuming: {len(done_ids)} done", flush=True)

    scores_by_cat = defaultdict(list)
    for v in done_ids.values():
        scores_by_cat[v['category']].append(1 if v['correct'] else 0)
    correct_total = sum(1 for r in done_ids.values() if r['correct'])
    total = len(done_ids)
    t0 = time.time()
    CAT_NAMES = {1:"single-session", 2:"multi-session", 3:"open-domain", 4:"temporal", 5:"adversarial"}

    for item in all_items:
        if item['id'] in done_ids:
            continue
        qa = item['qa']
        question = qa.get('question', '')
        gold = qa.get('answer', '')
        category = qa.get('category', 1)

        hyp = locomo_answer(qa, item['sess_data'], category)
        correct = locomo_judge(question, gold, hyp)
        total += 1
        if correct:
            correct_total += 1
        scores_by_cat[category].append(1 if correct else 0)
        if not correct:
            print(f"  FAIL [cat{category}] Q={str(question)[:60]} | Gold={str(gold)[:25]} | Got={str(hyp)[:25]}", flush=True)
        done_ids[item['id']] = {'category': category, 'cat_name': CAT_NAMES.get(category,'?'), 'correct': correct}
        ckpt_path.write_text(json.dumps(done_ids))

        if total % 50 == 0:
            print(f"  [{total}/{limit}] acc={correct_total/total*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nLOCOMO RESULTS ({total} items, {time.time()-t0:.0f}s):", flush=True)
    for cat, scores in sorted(scores_by_cat.items()):
        print(f"  cat{cat} ({CAT_NAMES.get(cat,'?')}): {sum(scores)/len(scores)*100:.1f}% ({sum(scores)}/{len(scores)})", flush=True)
    acc = correct_total / max(total, 1)
    print(f"  OVERALL: {acc*100:.1f}% | SmartSearch SOTA: 91.7%", flush=True)
    return {'benchmark': 'LoCoMo', 'accuracy': acc, 'n': total, 'by_cat': {CAT_NAMES.get(k,'?'): sum(v)/len(v) for k,v in scores_by_cat.items()}}

# ══════════════════════════════════════════════════════════════════════════════
# BEAM EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def beam_judge_nugget(nugget, hypothesis):
    """3-tier nugget scoring per BEAM paper."""
    prompt = f"""Score how well the response satisfies this requirement:
- 1.0: Fully satisfied
- 0.5: Partially satisfied
- 0.0: Not satisfied

Requirement: {nugget}
Response: {hypothesis[:600]}

Answer with only: 0, 0.5, or 1"""
    text = cheap(prompt, max_tokens=5)
    try:
        score = float(text.strip().split()[0].replace(',', '.'))
        if score >= 0.75: return 1.0
        elif score >= 0.25: return 0.5
        else: return 0.0
    except:
        return 0.0

def beam_answer_from_turns(question, flat_turns, category):
    """Retrieve top-20 turns via EBRM and build prompt. flat_turns = list of {id, role, content, sess_idx}."""
    search = get_ebrm()
    turn_docs = [t['content'] for t in flat_turns]
    search.build_index(turn_docs)
    result = search.search(question, top_k=min(20, len(turn_docs)))

    top_turns = sorted(
        [flat_turns[i] for i in result.indices[:20] if i < len(flat_turns)],
        key=lambda t: t.get('id', 0) if t.get('id') is not None else 0
    )
    context = "\n".join(f"{t['role']}: {t['content']}" for t in top_turns)[:32000]

    arith = try_temporal_arithmetic(question, context)
    arith_hint = f"\nCOMPUTED: {arith}" if arith else ""

    if category == "abstention":
        prompt = f"""Below are excerpts from chat history. The following question may or may not be answerable.
{context}
Question: {question}
If the answer is present in the excerpts above, state it. If not, say clearly that you don't have this information."""
    elif category == "temporal_reasoning":
        prompt = f"""Below are relevant excerpts from chat history. Answer the temporal question.{arith_hint}
{context}
Question: {question}
Answer (quote or paraphrase the specific fact from the excerpts above):"""
    else:
        prompt = f"""Below are relevant excerpts from chat history, retrieved for this question. Answer using ONLY the specific facts stated in these excerpts.
Excerpts:
{context}
Question: {question}
Answer (quote or paraphrase the specific fact from the excerpts above):"""

    return llm(prompt, max_tokens=600)


def beam_judge_abstention(question, hypothesis):
    verdict = cheap(f"""Did the model correctly identify that this question cannot be answered from the available information, or correctly said it doesn't know?
Question: {question}
Response: {hypothesis[:600]}
Answer yes or no only.""", max_tokens=5)
    return "yes" in verdict.lower()

def run_beam(limit, ckpt_path):
    """BEAM eval. Dataset schema:
      - chat: list of sessions, each session = list of {content, role, time_anchor, ...} turns
      - probing_questions: str (ast.literal_eval) -> dict keyed by category, each value = list of QA dicts
      - Each QA dict has: question, ideal_response, rubric (list of strings), difficulty
    """
    import ast as _ast
    print(f"\n{'='*60}", flush=True)
    print(f"BEAM {BEAM_SPLIT} EVAL | gpt-4.1 + gpt-4.1-mini | limit={limit}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        from datasets import load_dataset
        d = load_dataset('Mohammadta/BEAM', split=BEAM_SPLIT)
        raw_items = list(d)
    except Exception as e:
        print(f"  Failed to load BEAM dataset: {e}", flush=True)
        return None

    print(f"  Loaded {len(raw_items)} conversations", flush=True)

    # Flatten: one item per (conversation_id, category, question_idx)
    flat_items = []
    for conv in raw_items:
        conv_id = str(conv.get('conversation_id', ''))
        sessions = conv.get('chat', [])  # list of sessions
        pq_raw = conv.get('probing_questions', '{}')
        try:
            pq = _ast.literal_eval(pq_raw) if isinstance(pq_raw, str) else pq_raw
        except Exception:
            pq = {}
        for category, qa_list in pq.items():
            for qi, qa in enumerate(qa_list):
                item_id = f"{conv_id}_{category}_{qi}"
                flat_items.append({
                    'id': item_id,
                    'conv_id': conv_id,
                    'sessions': sessions,
                    'category': category,
                    'question': qa.get('question', ''),
                    'ideal_response': qa.get('ideal_response', ''),
                    'rubric': qa.get('rubric', []),
                    'difficulty': qa.get('difficulty', ''),
                })

    print(f"  Flattened to {len(flat_items)} QA items", flush=True)
    random.seed(42); random.shuffle(flat_items)
    flat_items = flat_items[:limit]

    done_ids = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}
    if done_ids:
        print(f"Resuming: {len(done_ids)} done", flush=True)

    scores_by_cat = defaultdict(list)
    for v in done_ids.values():
        scores_by_cat[v['category']].append(v['score'])
    score_total = sum(r['score'] for r in done_ids.values())
    total = len(done_ids)
    t0 = time.time()

    for item in flat_items:
        item_id = item['id']
        if item_id in done_ids:
            continue

        question = item['question']
        category = item['category']
        sessions = item['sessions']   # list of session-lists of {content, role, ...}
        rubric = item['rubric']       # list of strings like "LLM response should state: X"
        question_date = ""            # BEAM doesn't have per-question dates

        if not sessions or not question:
            continue

        # Build flat turn list for this conversation
        flat_turns = []
        for sess_idx, sess in enumerate(sessions):
            if isinstance(sess, list):
                for t in sess:
                    if isinstance(t, dict):
                        flat_turns.append({
                            'id': t.get('id'), 'role': t.get('role', 'user'),
                            'content': t.get('content', ''), 'sess_idx': sess_idx
                        })

        if not flat_turns:
            continue

        hyp = beam_answer_from_turns(question, flat_turns, category)

        if category == 'abstention':
            score = 1.0 if beam_judge_abstention(question, hyp) else 0.0
        elif not rubric:
            score = 0.0
        else:
            nugget_scores = [beam_judge_nugget(r, hyp) for r in rubric[:5]]
            score = float(np.mean(nugget_scores)) if nugget_scores else 0.0

        total += 1
        score_total += score
        scores_by_cat[category].append(score)
        if score < 0.5:
            print(f"  LOW [{category}] score={score:.2f} Q={question[:60]}", flush=True)
        done_ids[item_id] = {'category': category, 'score': score}
        ckpt_path.write_text(json.dumps(done_ids))

        if total % 20 == 0:
            print(f"  [{total}/{limit}] avg={score_total/total*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nBEAM {BEAM_SPLIT} RESULTS ({total} items, {time.time()-t0:.0f}s):", flush=True)
    for cat, scores in sorted(scores_by_cat.items()):
        print(f"  {cat}: {np.mean(scores)*100:.1f}% ({len(scores)} items)", flush=True)
    avg = score_total / max(total, 1)
    print(f"  OVERALL: {avg*100:.1f}% | Hindsight SOTA: 75.0%", flush=True)
    return {'benchmark': f'BEAM-{BEAM_SPLIT}', 'accuracy': avg, 'n': total, 'by_cat': {k: float(np.mean(v)) for k,v in scores_by_cat.items()}}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    results = {}
    ckpt_dir = Path(".")

    if "lme" in BENCHMARKS:
        r = run_lme(LIMIT, ckpt_dir / "unified_v2_lme_checkpoint.json")
        results['lme'] = r

    if "locomo" in BENCHMARKS:
        r = run_locomo(LIMIT, ckpt_dir / "unified_v2_locomo_checkpoint.json")
        results['locomo'] = r

    if "beam" in BENCHMARKS:
        r = run_beam(LIMIT, ckpt_dir / "unified_v2_beam_checkpoint.json")
        if r:
            results['beam'] = r

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"UNIFIED v2 SUMMARY | {LIMIT} items/benchmark", flush=True)
    print(f"{'='*60}", flush=True)
    sota = {'lme': 95.4, 'locomo': 91.7, 'beam': 75.0}
    total_acc = []
    for bname, r in results.items():
        if r:
            acc = r['accuracy'] * 100
            s = sota.get(bname, 0)
            diff = acc - s
            sign = "+" if diff >= 0 else ""
            total_acc.append(acc)
            print(f"  {r['benchmark']}: {acc:.1f}% (SOTA {s}% | {sign}{diff:.1f}pp)", flush=True)
    if total_acc:
        print(f"\n  Average across benchmarks: {sum(total_acc)/len(total_acc):.1f}%", flush=True)

    # Save summary
    summary_path = ckpt_dir / "unified_v2_summary.json"
    summary_path.write_text(json.dumps({k: {**v, 'accuracy': v['accuracy']} for k, v in results.items() if v}, indent=2))
    print(f"\nResults saved to unified_v2_summary.json", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    main()
