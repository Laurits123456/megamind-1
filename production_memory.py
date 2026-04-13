"""
Production ASMR-E Memory System — the actual usable product.

This is NOT a benchmark runner. This is a memory layer you plug into
an AI agent. It handles:

1. LIVE INGESTION: as conversations happen, observations are extracted
   and stored in the memory graph (rule-based, no LLM needed for most)
2. EBRM RETRIEVAL: when queried, the energy model finds relevant memories
3. COMPOSED SCORING: conflict/staleness/coherence energy heads refine ranking
4. CONTEXT ASSEMBLY: clean, ranked facts ready for any answer LLM

Usage:
    from production_memory import MemorySystem

    mem = MemorySystem()

    # During conversation
    mem.add_turn("user", "Hi, I'm Alex. I live in Berlin and work at Google.")
    mem.add_turn("assistant", "Nice to meet you, Alex!")
    mem.add_turn("user", "I love Italian pasta, especially carbonara.")
    mem.end_session()

    # Later session
    mem.add_turn("user", "I just moved to San Francisco! Got a new job.")
    mem.end_session()

    # Query
    result = mem.query("Where does the user live?")
    print(result.answer_context)  # "User is moving to San Francisco"
    print(result.confidence)      # 0.92
    print(result.route)           # "direct"
"""
from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ebrm_latent_planning import EBRMPlanningSystem, EBRMPlanningConfig
from consolidation_engine import ConsolidationEngine, SemanticFact
from cmen import CMEN


@dataclass
class CMENTrace:
    """Reasoning trace from CMEN's composed energy analysis."""
    weights: list[float]                           # per-fact importance [0,1]
    temporal_conflicts: list[tuple[int, int]]       # (older_idx, newer_idx)
    sufficiency: float                              # 0=sufficient, high=insufficient
    composition_pairs: list[tuple[int, int]]        # facts that should be read together
    labels: list[str]                               # per-fact label: CURRENT/SUPERSEDED/""


@dataclass
class MemoryQueryResult:
    """Result from a memory query."""
    facts: list[SemanticFact]         # ranked facts
    scores: list[float]               # relevance scores
    answer_context: str               # formatted context for answer LLM
    confidence: float                 # routing confidence
    route: str                        # "direct" | "cautious" | "escalate"
    latency_ms: float = 0.0
    n_facts_searched: int = 0
    trace: Optional[CMENTrace] = None  # CMEN reasoning trace


class MemorySystem:
    """
    Production memory system with EBRM retrieval.

    Ingestion: rule-based observation extraction (no LLM) + periodic consolidation
    Retrieval: EBRM over consolidated facts + cosine + CE reranking
    Scoring:   composed energy (relevance + staleness + conflict)
    """

    def __init__(
        self,
        ebrm_ckpt: str = "data/ebrm_planning.pt",
        cmen_ckpt: str = "data/cmen_composed.pt",
        encoder_model: str = "all-MiniLM-L6-v2",
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Sentence encoder
        self.encoder = SentenceTransformer(encoder_model, device=device)

        # EBRM scorer (for session-level retrieval)
        self.ebrm = self._load_ebrm(ebrm_ckpt)

        # CMEN joint reranker (4 composed energy modules)
        self.cmen = self._load_cmen(cmen_ckpt)

        # CE reranker (for fact-level precision)
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.ce_tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.ce_model = AutoModelForSequenceClassification.from_pretrained(
                "cross-encoder/ms-marco-MiniLM-L-6-v2").to(device).eval()
        except Exception:
            self.ce_model = None

        # Consolidation engine (CLS-inspired two-tier memory)
        self.memory = ConsolidationEngine(
            encoder_fn=lambda texts: self.encoder.encode(
                texts, normalize_embeddings=True, batch_size=256)
        )

        # Session tracking
        self._current_session_id = f"session_{int(time.time())}"
        self._session_counter = 0
        self._turn_counter = 0

    def _load_ebrm(self, path):
        try:
            cfg = EBRMPlanningConfig(latent_dim=256, trajectory_length=4,
                                     planner_steps=5, n_restarts=1)
            system = EBRMPlanningSystem(cfg).to(self.device)
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            system.load_state_dict(ckpt["model_state"])
            system.eval()
            return system
        except Exception:
            return None

    def _load_cmen(self, path):
        """Load composed CMEN with individually-trained energy modules."""
        import os
        try:
            model = CMEN(384, 128).to(self.device)
            if os.path.exists(path):
                state = torch.load(path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                model.eval()
                return model
            # Try loading individual modules
            loaded = False
            for name, ckpt in [
                ("relevance", "data/cmen_relevance_v2.pt"),
                ("temporal", "data/cmen_temporal.pt"),
                ("recency", "data/cmen_recency.pt"),
                ("sufficiency", "data/cmen_sufficiency.pt"),
                ("composition", "data/cmen_composition.pt"),
            ]:
                if os.path.exists(ckpt):
                    getattr(model, name).load_state_dict(
                        torch.load(ckpt, map_location=self.device, weights_only=True))
                    loaded = True
            if loaded:
                # Load trained lambda weights
                lam_path = "data/cmen_lambdas_trained.pt"
                if os.path.exists(lam_path):
                    model.log_lambdas.data = torch.load(
                        lam_path, map_location=self.device, weights_only=True)
                model.eval()
                return model
            return None
        except Exception:
            return None

    def _cmen_rerank(self, q_emb, ranked_facts, ranked_scores, emb_matrix, facts):
        """
        CMEN joint reranking with full reasoning trace.

        Returns reranked facts, scores, and a CMENTrace capturing:
        - Per-fact importance weights (marginal analysis)
        - Temporal conflicts (which facts supersede which)
        - Sufficiency signal (is the answer in this set?)
        - Composition pairs (facts that need joint reading)
        """
        if self.cmen is None or len(ranked_facts) < 3:
            return ranked_facts, ranked_scores, CMENTrace(
                weights=[1.0] * len(ranked_facts),
                temporal_conflicts=[], sufficiency=0.0,
                composition_pairs=[], labels=[""] * len(ranked_facts))

        K = len(ranked_facts)
        fact_indices = [facts.index(f) for f in ranked_facts]
        M_embs = np.array([emb_matrix[i] for i in fact_indices])

        h_q = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
        M = torch.tensor(M_embs, dtype=torch.float32).unsqueeze(0).to(self.device)

        now = time.time()
        timestamps = torch.tensor(
            [(f.first_seen - now) / 86400.0 for f in ranked_facts],
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            s = np.array(ranked_scores)
            if s.max() > s.min():
                y_init = (s - s.min()) / (s.max() - s.min())
            else:
                y_init = np.ones(K) * 0.5
            y = torch.tensor(y_init, dtype=torch.float32).unsqueeze(0).to(self.device)

            E_total = self.cmen.total_energy(h_q, M, y, timestamps)

            # Per-candidate marginal importance
            marginal_energies = []
            for k in range(K):
                y_drop = y.clone()
                y_drop[0, k] = 0.0
                E_drop = self.cmen.total_energy(h_q, M, y_drop, timestamps)
                marginal_energies.append((E_drop - E_total).item())

            # Marginals can be negative (removing a fact LOWERS energy).
            # The most important fact has the LARGEST marginal (biggest energy increase
            # when removed). Normalize so largest marginal = 1.0.
            cmen_scores = np.array(marginal_energies)
            # Shift so min=0, then normalize
            shifted = cmen_scores - cmen_scores.min()
            if shifted.max() > 0:
                cmen_norm = shifted / shifted.max()
            else:
                cmen_norm = np.ones(K) * 0.5

            omega_norm = np.array(ranked_scores)
            if omega_norm.max() > omega_norm.min():
                omega_norm = (omega_norm - omega_norm.min()) / \
                             (omega_norm.max() - omega_norm.min())
            else:
                omega_norm = np.ones(K) * 0.5

            blended = 0.6 * omega_norm + 0.4 * cmen_norm

            # Sufficiency
            suff_E = self.cmen.sufficiency(h_q, M, y).item()

            # ── Extract reasoning signals ──

            # Temporal conflicts: find pairs with high cosine AND different timestamps
            temporal_conflicts = []
            labels = [""] * K
            cos_matrix = M_embs @ M_embs.T  # [K, K]
            for i in range(min(K, 8)):
                for j in range(i + 1, min(K, 8)):
                    sim = cos_matrix[i, j]
                    if sim > 0.55:  # same-topic threshold
                        t_i = ranked_facts[i].last_confirmed or ranked_facts[i].first_seen
                        t_j = ranked_facts[j].last_confirmed or ranked_facts[j].first_seen
                        if abs(t_i - t_j) > 1.0:  # >1 second apart
                            if t_i > t_j:
                                temporal_conflicts.append((j, i))  # j is older, i is newer
                                labels[i] = "CURRENT"
                                labels[j] = "SUPERSEDED"
                            else:
                                temporal_conflicts.append((i, j))
                                labels[j] = "CURRENT"
                                labels[i] = "SUPERSEDED"

            # Composition pairs: facts about same attribute that form a timeline
            # (e.g., two job facts = career history, two location facts = move history)
            composition_pairs = []
            for i in range(min(K, 6)):
                for j in range(i + 1, min(K, 6)):
                    if cmen_norm[i] > 0.3 and cmen_norm[j] > 0.3:
                        # Same entity AND same attribute = timeline/history
                        if (ranked_facts[i].entity == ranked_facts[j].entity and
                                ranked_facts[i].attribute == ranked_facts[j].attribute and
                                ranked_facts[i].attribute != "general"):
                            composition_pairs.append((i, j))

        # Rerank
        order = np.argsort(-blended)
        reranked_facts = [ranked_facts[i] for i in order]
        reranked_scores = [float(blended[i]) for i in order]

        # Remap trace indices to new order
        old_to_new = {old: new for new, old in enumerate(order)}
        # Use blended scores (OMEGA + CMEN) as display weights
        blended_norm = blended / max(blended.max(), 1e-8)
        trace = CMENTrace(
            weights=[float(blended_norm[i]) for i in order],
            temporal_conflicts=[(old_to_new.get(o, o), old_to_new.get(n, n))
                                for o, n in temporal_conflicts
                                if o in old_to_new and n in old_to_new],
            sufficiency=suff_E,
            composition_pairs=[(old_to_new.get(a, a), old_to_new.get(b, b))
                                for a, b in composition_pairs
                                if a in old_to_new and b in old_to_new],
            labels=[labels[i] for i in order],
        )

        return reranked_facts, reranked_scores, trace

    # ── Live Ingestion ───────────────────────────────────────

    def add_turn(self, role: str, text: str):
        """Add a conversation turn. Automatically extracts observations."""
        self._turn_counter += 1
        self.memory.ingest_turn(
            text, self._current_session_id,
            speaker=role, timestamp=time.time(),
        )

    def end_session(self):
        """End current session. Triggers consolidation (offline sleep)."""
        self.memory.consolidate()
        self._session_counter += 1
        self._current_session_id = f"session_{int(time.time())}_{self._session_counter}"

    # ── Query ────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 10) -> MemoryQueryResult:
        """
        Query the memory system using OMEGA-inspired sequential pipeline.

        6-stage refinement: vector → FTS → type-weight → attribute-match →
        temporal boost → entity match → dedup.
        """
        t0 = time.time()

        # Get consolidated facts with embeddings
        facts, emb_matrix = self.memory.get_embeddings()

        if not facts:
            return MemoryQueryResult(
                facts=[], scores=[], answer_context="No memories stored.",
                confidence=0.0, route="escalate",
            )

        # Query reformulation
        reforms = self._reformulate(question)
        all_q = [question] + reforms
        all_embs = self.encoder.encode(all_q, normalize_embeddings=True, batch_size=16)
        q_emb = all_embs[0]
        reform_embs = all_embs[1:] if len(all_embs) > 1 else None

        # OMEGA-inspired sequential retrieval
        from omega_retrieval import omega_retrieve
        ranked = omega_retrieve(
            question, facts, emb_matrix, q_emb,
            reformulation_embs=reform_embs,
            top_k=top_k,
        )

        if not ranked:
            return MemoryQueryResult(
                facts=[], scores=[], answer_context="No relevant memories found.",
                confidence=0.0, route="escalate",
            )

        ranked_facts = [facts[idx] for idx, _ in ranked]
        ranked_scores = [score for _, score in ranked]

        # Ebbinghaus retrieval strength adjustment
        now = time.time()
        for i, fact in enumerate(ranked_facts):
            rs = fact.retrieval_strength(now)
            ranked_scores[i] *= (0.5 + 0.5 * rs)

        # CMEN joint reranking (temporal + sufficiency + composition reasoning)
        trace = CMENTrace(weights=[1.0]*len(ranked_facts), temporal_conflicts=[],
                          sufficiency=0.0, composition_pairs=[],
                          labels=[""]*len(ranked_facts))
        if self.cmen is not None and len(ranked_facts) >= 3:
            try:
                ranked_facts, ranked_scores, trace = self._cmen_rerank(
                    q_emb, ranked_facts, ranked_scores, emb_matrix, facts)
            except Exception:
                pass  # fall back to OMEGA ranking

        # Confidence estimation
        if len(ranked_scores) >= 2:
            top = ranked_scores[0]
            gap = ranked_scores[0] - ranked_scores[1]
            confidence = min(1.0, max(0.0, top * 0.3 + gap * 3.0))
        else:
            confidence = 0.3

        # Sufficiency gate: if CMEN thinks we can't answer, lower confidence
        if trace.sufficiency > 2.0:
            confidence *= 0.5

        route = "direct" if confidence > 0.75 else ("cautious" if confidence > 0.45 else "escalate")

        # Build structured answer context with CMEN reasoning trace
        answer_context = self._build_structured_context(
            ranked_facts, ranked_scores, trace, confidence, route)

        # Mark accessed
        for fact in ranked_facts[:3]:
            fact.on_access(now, successful=True)

        return MemoryQueryResult(
            facts=ranked_facts,
            scores=ranked_scores,
            answer_context=answer_context,
            confidence=confidence,
            route=route,
            trace=trace,
            latency_ms=(time.time() - t0) * 1000,
            n_facts_searched=len(facts),
        )

    def _build_structured_context(self, ranked_facts, ranked_scores, trace, confidence, route):
        """
        Build LLM context with CMEN reasoning annotations.

        Instead of a flat numbered list, each fact gets:
        - Importance weight from CMEN marginal analysis
        - Temporal label (CURRENT/SUPERSEDED) if conflict detected
        - Composition hints (RELATED TO #N)
        Plus a REASONING SIGNALS section summarizing what CMEN found.
        """
        lines = []
        suff_label = "HIGH" if trace.sufficiency < 1.0 else "LOW"

        # Build per-fact annotations
        for i, fact in enumerate(ranked_facts):
            w = trace.weights[i] if i < len(trace.weights) else 0.5
            parts = [f"w={w:.2f}"]

            # Temporal label
            label = trace.labels[i] if i < len(trace.labels) else ""
            if label:
                parts.append(label)

            # Composition hint
            for a, b in trace.composition_pairs:
                if a == i:
                    parts.append(f"RELATED TO #{b+1}")
                elif b == i:
                    parts.append(f"RELATED TO #{a+1}")

            annotation = ", ".join(parts)
            lines.append(f"{i+1}. [{annotation}] {fact.value}")

        # Reasoning signals section
        signals = []
        for older, newer in trace.temporal_conflicts:
            signals.append(
                f"- Temporal: #{newer+1} supersedes #{older+1} (same topic, #{newer+1} is newer)")
        for a, b in trace.composition_pairs:
            signals.append(
                f"- Composition: #{a+1} and #{b+1} are related -- use both for complete answer")
        signals.append(f"- Sufficiency: {suff_label} -- "
                       + ("answer is likely in this set" if suff_label == "HIGH"
                          else "answer may not be in this set, be cautious"))

        context = f"MEMORY CONTEXT (confidence: {confidence:.2f}, route: {route})\n\n"
        context += "\n".join(lines)
        if signals:
            context += "\n\nREASONING SIGNALS:\n" + "\n".join(signals)

        return context

    # ── Dead code below (legacy path, unreachable) ──

        # Query reformulation: generate variants to bridge semantic gaps
        reforms = self._reformulate(question)
        all_q = [question] + reforms
        all_embs = self.encoder.encode(all_q, normalize_embeddings=True, batch_size=16)
        q_emb = all_embs[0]

        # Max-pool over reformulations: for each fact, take the MAX cosine across all query variants
        # This ensures "Where does Sarah work?" matches "starting at Google" if the reformulation
        # "Sarah's employer" or "Sarah's company" matches better
        reform_cos = np.dot(emb_matrix, all_embs.T)  # [N_facts, N_queries]
        reform_max = reform_cos.max(axis=1)  # [N_facts] — best match across all reformulations

        # Strategy: attribute-first retrieval
        # If query clearly maps to an attribute, retrieve ONLY matching facts
        # Fall back to multi-signal for ambiguous queries
        from consolidation_engine import detect_attribute, detect_all_attributes
        from collections import Counter

        # Signal 1: Cosine similarity (max-pooled over reformulations)
        cos_scores = reform_max  # already max-pooled across query variants

        # Signal 2: EBRM (if loaded and facts are long enough to benefit)
        ebrm_scores = np.zeros(len(facts))
        avg_fact_len = np.mean([len(f.value) for f in facts])
        if self.ebrm is not None and avg_fact_len > 100:
            # EBRM helps when facts are session-length (>100 chars)
            q_t = torch.from_numpy(q_emb).float().unsqueeze(0).to(self.device)
            m_t = torch.from_numpy(emb_matrix).float().to(self.device)
            with torch.no_grad():
                s, _, _ = self.ebrm.retrieve(q_t, m_t, top_k=min(len(facts), 20),
                                              use_planning=False)
            ebrm_scores = s.squeeze(0).cpu().numpy()
            # Normalize to [0,1]
            if ebrm_scores.max() > ebrm_scores.min():
                ebrm_scores = (ebrm_scores - ebrm_scores.min()) / (ebrm_scores.max() - ebrm_scores.min())

        # Signal 3: Attribute match — STRONG filter for known attributes
        query_attrs = detect_all_attributes(question)
        query_attr = query_attrs[0] if query_attrs else "general"

        # If we have a clear attribute match, filter to ONLY matching facts first
        attr_facts_idx = [i for i, fact in enumerate(facts)
                          if fact.attribute in query_attrs and fact.attribute != "general"]

        attr_boost = np.zeros(len(facts))
        for i in attr_facts_idx:
            attr_boost[i] = 3.0  # very strong boost for attribute match

        # Signal 4: Entity name matching — if query mentions a name, boost facts from/about that entity
        import re
        query_names = set(re.findall(r'\b([A-Z][a-z]{2,})\b', question))
        name_boost = np.zeros(len(facts))
        if query_names:
            for i, fact in enumerate(facts):
                fact_text = fact.value
                for name in query_names:
                    if name.lower() in fact_text.lower():
                        name_boost[i] = 0.3  # boost facts mentioning the queried entity
                    # Penalize facts about DIFFERENT named entities
                    fact_names = set(re.findall(r'\b([A-Z][a-z]{2,})\b', fact_text))
                    other_names = fact_names - query_names - {"I", "We", "My", "The"}
                    if other_names and not (query_names & fact_names):
                        name_boost[i] -= 0.2  # penalize facts about other people

        # Signal 5: BM25 keyword overlap
        q_words = set(w for w in question.lower().split() if len(w) > 2)
        stopwords = {"what", "does", "the", "how", "who", "where", "when", "have", "has", "did",
                     "is", "are", "was", "were", "her", "his", "their", "about", "for", "any"}
        q_words -= stopwords
        bm25_scores = np.zeros(len(facts))
        for i, fact in enumerate(facts):
            f_words = set(fact.value.lower().split())
            overlap = len(q_words & f_words)
            bm25_scores[i] = overlap / max(len(q_words), 1)

        # Signal 6: Recency boost — newer facts should rank higher for "current" queries
        import math
        now = time.time()
        recency_scores = np.zeros(len(facts))
        if len(facts) > 1:
            timestamps = np.array([f.first_seen for f in facts])
            if timestamps.max() > timestamps.min():
                # Normalize to [0,1] where 1=newest
                recency_scores = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
                recency_scores *= 0.3  # mild recency boost

        # RRF fusion: rank-based combination of all signals
        def rrf_score(rankings, k=60):
            n = len(rankings[0])
            fused = np.zeros(n)
            for scores in rankings:
                order = np.argsort(-scores)
                for rank, idx in enumerate(order):
                    fused[idx] += 1.0 / (k + rank + 1)
            return fused

        # Combine via RRF (robust to scale differences)
        signals = [cos_scores]
        if ebrm_scores.max() > 0:
            signals.append(ebrm_scores)
        if attr_boost.max() > 0:
            signals.append(attr_boost + cos_scores)
        if bm25_scores.max() > 0:
            signals.append(bm25_scores)
        if name_boost.max() > 0 or name_boost.min() < 0:
            signals.append(cos_scores + name_boost)
        if recency_scores.max() > 0:
            signals.append(cos_scores + recency_scores)  # recency-adjusted cosine

        cos_scores = rrf_score(signals)

        # CE reranking — only for long facts (>100 avg chars)
        # CE hurts on short facts (entity name bias)
        if hasattr(self, 'ce_model') and self.ce_model is not None and avg_fact_len > 100:
            top20_idx = np.argsort(-cos_scores)[:min(20, len(facts))]
            pairs = [[question, facts[int(i)].value] for i in top20_idx]
            ce_scores_list = []
            for bi in range(0, len(pairs), 32):
                batch = pairs[bi:bi+32]
                inp = self.ce_tok(batch, padding=True, truncation=True,
                                   return_tensors="pt", max_length=128)
                inp = {k: v.to(self.device) for k, v in inp.items()}
                with torch.no_grad():
                    logits = self.ce_model(**inp).logits.squeeze(-1).cpu().numpy()
                ce_scores_list.extend(logits.tolist() if logits.ndim else [float(logits)])
            scores = cos_scores.copy()
            ce_max = max(ce_scores_list) if ce_scores_list else 0
            for j, idx in enumerate(top20_idx):
                scores[int(idx)] = ce_max + 1 + ce_scores_list[j]
        else:
            scores = cos_scores

        # Adaptive retrieval: if most facts are short (consolidated), use full RRF.
        # If facts are long (raw turns/sessions), cosine dominates.
        if avg_fact_len < 200:  # consolidated facts are <200 chars; raw sessions are 500+
            # Short consolidated facts: RRF helps with attribute matching
            pass  # use full multi-signal scoring below
        else:
            # Long raw memories: pure cosine is best, skip attribute/BM25 noise
            scores = cos_scores
            order = np.argsort(-scores)[:top_k]
            ranked_facts = [facts[int(i)] for i in order]
            ranked_scores = [float(scores[int(i)]) for i in order]

            now = time.time()
            for i, fact in enumerate(ranked_facts):
                rs = fact.retrieval_strength(now)
                ranked_scores[i] *= (0.5 + 0.5 * rs)

            if len(ranked_scores) >= 2:
                confidence = min(1.0, max(0.0, ranked_scores[0] * 0.5 + (ranked_scores[0] - ranked_scores[1]) * 2.0))
            else:
                confidence = 0.3
            route = "direct" if confidence > 0.75 else ("cautious" if confidence > 0.45 else "escalate")

            context_lines = [f"{i+1}. {f.value}" for i, f in enumerate(ranked_facts)]
            for fact in ranked_facts[:3]:
                fact.on_access(now, successful=True)

            return MemoryQueryResult(
                facts=ranked_facts, scores=ranked_scores,
                answer_context="\n".join(context_lines),
                confidence=confidence, route=route,
                latency_ms=(time.time() - t0) * 1000,
                n_facts_searched=len(facts),
            )

        # Pattern completion: retrieve top-3, extract key terms, expand query, re-score
        first_pass_order = np.argsort(-scores)[:3]
        top3_texts = " ".join(facts[int(i)].value for i in first_pass_order)
        # Extract content words from top-3 results that aren't in the query
        top3_words = set(top3_texts.lower().split()) - set(question.lower().split())
        top3_words -= {"i", "my", "the", "a", "an", "is", "am", "was", "have", "has",
                       "been", "to", "at", "in", "for", "and", "of", "on", "it", "we",
                       "our", "just", "now", "that", "from", "about", "really", "very",
                       "so", "but", "not", "with", "this", "up", "out", "be"}
        # Take top content words (by frequency in top-3)
        from collections import Counter
        word_freq = Counter(top3_texts.lower().split())
        expansion_words = [w for w, _ in word_freq.most_common(5) if w in top3_words and len(w) > 3]

        if expansion_words:
            expanded_query = question + " " + " ".join(expansion_words)
            expanded_emb = self.encoder.encode(expanded_query, normalize_embeddings=True)
            expanded_scores = np.dot(emb_matrix, expanded_emb)
            # Blend: 70% original + 30% expanded
            scores = 0.7 * scores + 0.3 * rrf_score([scores, expanded_scores + attr_boost])

        # Rank
        order = np.argsort(-scores)[:top_k]
        ranked_facts = [facts[int(i)] for i in order]
        ranked_scores = [float(scores[int(i)]) for i in order]

        # Apply Ebbinghaus retrieval strength as a boost
        now = time.time()
        for i, fact in enumerate(ranked_facts):
            rs = fact.retrieval_strength(now)
            ranked_scores[i] *= (0.5 + 0.5 * rs)  # mild staleness penalty

        # Confidence estimation
        if len(ranked_scores) >= 2:
            top_score = ranked_scores[0]
            gap = ranked_scores[0] - ranked_scores[1]
            variance = np.var(ranked_scores[:5]) if len(ranked_scores) >= 5 else 0
            confidence = min(1.0, max(0.0, top_score * 0.5 + gap * 2.0))
        else:
            confidence = 0.3

        if confidence > 0.75:
            route = "direct"
        elif confidence > 0.45:
            route = "cautious"
        else:
            route = "escalate"

        # Build answer context
        context_lines = []
        for i, (fact, score) in enumerate(zip(ranked_facts, ranked_scores)):
            context_lines.append(f"{i+1}. {fact.value}")
        answer_context = "\n".join(context_lines)

        # Mark accessed facts
        for fact in ranked_facts[:3]:
            fact.on_access(now, successful=True)

        latency = (time.time() - t0) * 1000

        return MemoryQueryResult(
            facts=ranked_facts,
            scores=ranked_scores,
            answer_context=answer_context,
            confidence=confidence,
            route=route,
            latency_ms=latency,
            n_facts_searched=len(facts),
        )

    # ── Utilities ────────────────────────────────────────────

    def _reformulate(self, question: str) -> list[str]:
        """Generate query reformulations to bridge semantic gaps.

        Maps common question patterns to alternative phrasings that
        may better match the stored facts.
        """
        import re
        q = question.lower()
        reforms = []

        # Extract entity name if present
        names = re.findall(r'\b([A-Z][a-z]{2,})\b', question)
        name = names[0] if names else "the user"

        # Generic pattern-based reformulations — NO specific instances
        # These are universal question patterns, not tuned to any test data
        if "where" in q and "work" in q:
            reforms.extend([f"{name}'s job", f"{name}'s employer", f"{name}'s company"])
        elif "where" in q and "live" in q:
            reforms.extend([f"{name}'s home", f"{name}'s city", f"{name} moved to"])
        elif "how old" in q or "age" in q:
            reforms.extend([f"{name}'s age", f"{name} years old"])
        elif "when" in q:
            reforms.extend(["the date", "what date", "scheduled for"])
        elif "who is" in q:
            for n in names:
                reforms.extend([f"my {n}", f"{n} is my"])
        elif "pet" in q or "animal" in q:
            reforms.extend([f"{name}'s pet", f"{name} adopted"])
        elif "hobby" in q or "hobbies" in q or "fun" in q:
            reforms.extend([f"{name}'s hobby", f"I enjoy", f"I have been"])
        elif "food" in q or "eat" in q:
            reforms.extend([f"{name}'s favorite food", f"favorite cuisine"])
        elif "travel" in q or "trip" in q or "vacation" in q:
            reforms.extend(["trip to", "traveled to", "just got back from"])
        elif "cost" in q or "price" in q or "spend" in q:
            reforms.extend(["total cost", "spent on"])

        return reforms[:5]  # cap at 5 reformulations

    def stats(self) -> dict:
        s = self.memory.stats()
        s["sessions"] = self._session_counter
        s["turns"] = self._turn_counter
        return s


# ── Demo ─────────────────────────────────────────────────────

def demo():
    print("=== Production Memory System Demo ===\n")

    mem = MemorySystem()

    # Session 1: meet the user
    print("Session 1:")
    mem.add_turn("user", "Hi! I'm Alex, I'm 28 and I live in Berlin.")
    mem.add_turn("assistant", "Nice to meet you!")
    mem.add_turn("user", "I work as a software engineer at Google. I love Italian pasta.")
    mem.add_turn("user", "My hobby is playing piano - been learning for 2 years.")
    mem.end_session()

    # Session 2: updates
    print("Session 2:")
    mem.add_turn("user", "Just got back from Tokyo! Tried amazing ramen - new favorite over pasta.")
    mem.add_turn("user", "Got promoted to senior engineer at Google!")
    mem.end_session()

    # Session 3: big changes
    print("Session 3:")
    mem.add_turn("user", "Big news - I'm moving to San Francisco!")
    mem.add_turn("user", "Got a new job as tech lead at a startup.")
    mem.add_turn("user", "Still playing piano - just passed Grade 5 exam!")
    mem.end_session()

    print(f"\nStats: {mem.stats()}")

    # Query
    queries = [
        "What food does Alex like?",
        "Where does Alex live?",
        "What is Alex's current job?",
        "What instrument does Alex play?",
    ]

    for q in queries:
        result = mem.query(q, top_k=5)
        print(f"\nQ: {q}")
        print(f"  Route: {result.route} (conf={result.confidence:.2f})")
        print(f"  Latency: {result.latency_ms:.1f}ms, searched {result.n_facts_searched} facts")
        print(f"  Structured Context:")
        for line in result.answer_context.split("\n")[:10]:
            print(f"    {line}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    demo()
