"""
ASMR-E Answerer — query-type-aware answer generation.

Routes to different answering strategies based on query type:
  - Single-session: full session context -> LLM
  - Multi-session: query-guided extraction from each session -> synthesis
  - Temporal: full session with date emphasis -> LLM

This module handles ONLY the answer generation (Stage 3).
Retrieval (Stages 1-2) is handled by EBRM.
"""
from __future__ import annotations
from multi_session_reasoner import is_multi_session_query, multi_session_answer


def classify_query(question: str) -> str:
    """Classify query type for routing."""
    if is_multi_session_query(question):
        return "multi_session"

    q = question.lower()
    temporal_keywords = [
        "when", "what date", "what time", "how long ago",
        "before", "after", "first time", "most recent", "last time",
    ]
    if any(kw in q for kw in temporal_keywords):
        return "temporal"

    return "simple"


def answer_simple(question: str, session_text: str, llm_fn,
                   highlight_turns: list[int] = None, all_turns: list[str] = None) -> str:
    """Answer from a single session. If highlight_turns provided, mark key turns."""
    if highlight_turns and all_turns:
        # Present full session with key turns marked
        lines = []
        for i, turn in enumerate(all_turns):
            if i in highlight_turns:
                lines.append(f">>> {turn}")  # highlighted
            else:
                lines.append(turn)
        context = "\n".join(lines)[:15000]
        prompt = (
            f"Answer the question using the conversation below. "
            f"Lines marked with >>> are most likely to contain the answer.\n\n"
            f"CONVERSATION:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer directly and concisely."
        )
    else:
        prompt = (
            f"Answer the question using ONLY the conversation below.\n\n"
            f"CONVERSATION:\n{session_text[:15000]}\n\n"
            f"QUESTION: {question}\n\n"
            f"Answer directly and concisely. If the answer is not found, say \"I don't know\"."
        )
    try:
        return llm_fn(prompt).strip()
    except Exception:
        return "I don't know."


def answer_temporal(question: str, session_text: str, llm_fn) -> str:
    """Answer temporal question from session text."""
    prompt = (
        f"Answer the question about timing/dates using ONLY the conversation below.\n\n"
        f"CONVERSATION:\n{session_text[:15000]}\n\n"
        f"QUESTION: {question}\n\n"
        f"Look for specific dates, times, durations, or temporal references. "
        f"Answer directly and concisely."
    )
    try:
        return llm_fn(prompt).strip()
    except Exception:
        return "I don't know."


def answer_query(question: str, session_texts: list[str], llm_fn) -> tuple[str, str]:
    """
    Route to the appropriate answering strategy.

    Args:
        question: the user's question
        session_texts: list of session texts (from EBRM retrieval, ranked by relevance)
        llm_fn: callable(prompt) -> str

    Returns:
        (answer, query_type)
    """
    qtype = classify_query(question)

    if qtype == "multi_session":
        answer = multi_session_answer(question, session_texts[:10], llm_fn)
    elif qtype == "temporal":
        answer = answer_temporal(question, session_texts[0] if session_texts else "", llm_fn)
    else:
        answer = answer_simple(question, session_texts[0] if session_texts else "", llm_fn)

    return answer, qtype
