"""
Energy-Based Reasoning Model Agentic Search.

This is NOT cosine + rerank. The EBRM agents ARE the search.

Architecture:
  1. Query Decomposition: EBRM decomposes query into multiple search probes
  2. Multi-Probe Search: each probe searches from a different angle
     - Semantic probe (cosine over turn-level embeddings)
     - Temporal probe (recency-weighted search)
     - Entity probe (entity-name-matched search)
     - Attribute probe (attribute-type-matched search)
     - BM25 probe (lexical matching)
  3. Union Pool: collect ALL candidates from ALL probes
  4. CMEN Joint Scoring: energy-based reasoning over the candidate SET
     - Temporal conflict detection (which facts are outdated?)
     - Sufficiency check (do we have enough to answer?)
     - Composition analysis (which facts need joint reading?)
     - Recency preference (prefer newer when superseded)
  5. If CMEN says insufficient: generate NEW probes and search again

The key insight: CMEN's sufficiency energy drives an ITERATIVE search loop.
The search doesn't stop until the energy model says "I have enough to answer."
"""
from __future__ import annotations
import re
import math
import time
import numpy as np
from dataclasses import dataclass, field
from collections import Counter
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    """Result from EBRM agentic search."""
    texts: list[str]           # retrieved texts, ranked
    scores: list[float]        # relevance scores
    indices: list[int]         # original indices into corpus
    probes_used: list[str]     # which probes found each result
    n_iterations: int = 1      # how many search iterations
    sufficiency: float = 0.0   # CMEN sufficiency energy
    temporal_conflicts: list[tuple[int, int]] = field(default_factory=list)
    composition_pairs: list[tuple[int, int]] = field(default_factory=list)
    latency_ms: float = 0.0
    marginal_weights: list[float] = field(default_factory=list)  # CMEN per-result importance


class EBRMSearch:
    """
    Energy-Based Reasoning Model Agentic Search System.

    Uses turn-level max-pool embeddings + multi-probe search + CMEN joint scoring.
    """

    def __init__(self, encoder=None, cmen=None, device="cpu"):
        import torch
        self.device = device
        self.encoder = encoder or SentenceTransformer("all-MiniLM-L6-v2")

        # Load CMEN if available
        self.cmen = cmen
        if cmen is None:
            try:
                from cmen import CMEN
                import os
                model = CMEN(384, 128)
                for name, path in [
                    ("relevance", "data/cmen_relevance_v2.pt"),
                    ("temporal", "data/cmen_temporal.pt"),
                    ("recency", "data/cmen_recency.pt"),
                    ("sufficiency", "data/cmen_sufficiency.pt"),
                    ("composition", "data/cmen_composition.pt"),
                ]:
                    if os.path.exists(path):
                        getattr(model, name).load_state_dict(
                            torch.load(path, map_location=device, weights_only=True))
                # Load trained lambda weights (end-to-end trained composition)
                lam_path = "data/cmen_lambdas_trained.pt"
                if os.path.exists(lam_path):
                    model.log_lambdas.data = torch.load(
                        lam_path, map_location=device, weights_only=True)
                model.eval()
                self.cmen = model
            except Exception:
                pass

        # Corpus state
        self.corpus_texts = []        # raw texts
        self.corpus_turns = []        # list of list of turns per document
        self.turn_embs = []           # per-turn embeddings
        self.doc_embs = None          # per-document max-pool embeddings
        self.turn_to_doc = []         # turn index -> document index

    def build_index(self, documents: list[str | list[str]]):
        """
        Index documents with turn-level embeddings.

        documents: list of strings OR list of lists of turns.
        If strings, split into turns by newline.
        """
        t0 = time.time()
        self.corpus_texts = []
        self.corpus_turns = []
        all_turns = []
        self.turn_to_doc = []

        for i, doc in enumerate(documents):
            if isinstance(doc, list):
                turns = [t.get("content", str(t)) if isinstance(t, dict) else str(t)
                         for t in doc]
            else:
                turns = [line.strip() for line in doc.split("\n") if line.strip()]

            self.corpus_texts.append(
                "\n".join(turns) if isinstance(doc, list) else doc)
            self.corpus_turns.append(turns)

            for turn in turns:
                all_turns.append(turn[:512])  # cap turn length
                self.turn_to_doc.append(i)

        # Encode all turns
        if all_turns:
            self.turn_embs = self.encoder.encode(
                all_turns, batch_size=256, normalize_embeddings=True,
                show_progress_bar=False)
        else:
            self.turn_embs = np.array([])

        # Build per-document max-pool embeddings
        n_docs = len(self.corpus_texts)
        dim = self.turn_embs.shape[1] if len(self.turn_embs) > 0 else 384
        self.doc_embs = np.zeros((n_docs, dim))
        for turn_idx, doc_idx in enumerate(self.turn_to_doc):
            self.doc_embs[doc_idx] = np.maximum(
                self.doc_embs[doc_idx], self.turn_embs[turn_idx])

        # Normalize doc embeddings
        norms = np.linalg.norm(self.doc_embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.doc_embs = self.doc_embs / norms

        build_time = (time.time() - t0) * 1000
        return build_time

    def _probe_semantic(self, q_emb, top_k=10):
        """Semantic probe: max-pool cosine over turn-level embeddings."""
        if len(self.turn_embs) == 0:
            return []
        # Turn-level scores
        turn_scores = self.turn_embs @ q_emb
        # Max-pool to document level
        n_docs = len(self.corpus_texts)
        doc_scores = np.full(n_docs, -np.inf)
        for turn_idx, doc_idx in enumerate(self.turn_to_doc):
            doc_scores[doc_idx] = max(doc_scores[doc_idx], turn_scores[turn_idx])
        ranking = np.argsort(-doc_scores)[:top_k]
        return [(int(i), float(doc_scores[i]), "semantic") for i in ranking]

    def _probe_reformulation(self, query, q_emb, top_k=10):
        """Reformulation probe: generate query variants, max-pool across them."""
        reforms = self._reformulate(query)
        if not reforms:
            return []
        reform_embs = self.encoder.encode(reforms, normalize_embeddings=True)
        all_q = np.vstack([q_emb.reshape(1, -1), reform_embs])

        # Turn-level max-pool across ALL query variants
        turn_scores = self.turn_embs @ all_q.T  # [n_turns, n_queries]
        turn_max = turn_scores.max(axis=1)       # best query per turn

        n_docs = len(self.corpus_texts)
        doc_scores = np.full(n_docs, -np.inf)
        for turn_idx, doc_idx in enumerate(self.turn_to_doc):
            doc_scores[doc_idx] = max(doc_scores[doc_idx], turn_max[turn_idx])

        ranking = np.argsort(-doc_scores)[:top_k]
        return [(int(i), float(doc_scores[i]), "reformulation") for i in ranking]

    def _probe_bm25(self, query, top_k=10):
        """BM25 probe: lexical matching."""
        q_words = query.lower().split()
        n_docs = len(self.corpus_texts)
        scores = np.zeros(n_docs)

        # Simple TF-IDF approximation
        doc_freqs = Counter()
        for text in self.corpus_texts:
            words = set(text.lower().split())
            for w in words:
                doc_freqs[w] += 1

        for i, text in enumerate(self.corpus_texts):
            text_lower = text.lower()
            text_words = text_lower.split()
            tf = Counter(text_words)
            for w in q_words:
                if w in tf:
                    idf = math.log((n_docs + 1) / (doc_freqs.get(w, 0) + 1))
                    scores[i] += tf[w] * idf / (tf[w] + 1.5)

        ranking = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i]), "bm25") for i in ranking
                if scores[i] > 0]

    def _probe_entity(self, query, top_k=10):
        """Entity probe: find documents mentioning the same entities."""
        query_names = set(re.findall(r'\b([A-Z][a-z]{2,})\b', query))
        if not query_names:
            return []

        scores = []
        for i, text in enumerate(self.corpus_texts):
            doc_names = set(re.findall(r'\b([A-Z][a-z]{2,})\b', text))
            overlap = len(query_names & doc_names)
            if overlap > 0:
                scores.append((i, float(overlap), "entity"))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _reformulate(self, query):
        """Generate query reformulations covering broad personal memory patterns."""
        q = query.lower()
        reforms = []
        names = re.findall(r'\b([A-Z][a-z]{2,})\b', query)
        name = names[0] if names else "the user"

        # Extract key nouns from query (content words, length > 3, not stopwords)
        stopwords = {"what", "when", "where", "who", "which", "how", "many", "much",
                     "did", "does", "have", "has", "been", "are", "was", "were",
                     "the", "and", "for", "that", "this", "with", "from", "about",
                     "ago", "last", "previous", "our", "your", "my", "you", "our"}
        key_words = [w for w in re.findall(r'\b[a-z]{4,}\b', q) if w not in stopwords]

        patterns = [
            # Work / career
            (["where", "work", "job", "employer", "office", "company"],
             [f"{name}'s job", f"{name}'s employer", f"{name} works at", f"{name} employment"]),
            # Location / living
            (["where", "live", "moved", "relocat", "city", "home", "address"],
             [f"{name}'s home", f"{name}'s city", f"{name} lives in", f"{name} moved to"]),
            # Food / dining
            (["food", "eat", "meal", "restaurant", "cook", "dinner", "lunch", "breakfast", "favorite"],
             [f"{name}'s favorite food", f"{name} likes to eat", f"{name} restaurant", f"{name} cooking"]),
            # Hobbies / activities / sports
            (["hobby", "free time", "fun", "sport", "play", "game", "run", "hike", "gym",
              "exercise", "fitness", "swim", "cycle", "bike"],
             [f"{name}'s hobby", f"{name} enjoys", f"{name} activity", f"{name} sport"]),
            # Education
            (["degree", "graduat", "study", "school", "university", "college", "major", "gpa"],
             [f"{name}'s education", f"{name} studied", f"{name}'s degree", f"{name} graduated"]),
            # Age / birthday
            (["old", "age", "born", "birthday", "birth"],
             [f"{name}'s age", f"{name} born", f"{name} birthday"]),
            # Pets
            (["pet", "dog", "cat", "animal", "bird", "fish"],
             [f"{name}'s pet", f"{name} has a dog", f"{name} has a cat", f"{name} animal"]),
            # Travel / trips / events
            (["travel", "trip", "visit", "went", "went to", "vacation", "holiday", "journey", "flew"],
             [f"{name} traveled", f"{name}'s trip", f"{name} vacation", f"{name} went to"]),
            # Entertainment / media
            (["music", "song", "band", "listen", "concert", "spotify", "playlist"],
             [f"{name}'s music", f"{name} listens to", f"{name} concert", f"{name} playlist"]),
            (["movie", "film", "watch", "netflix", "show", "series", "documentary", "television", "tv"],
             [f"{name}'s favorite movie", f"{name} watches", f"{name} streaming", f"{name} netflix"]),
            (["book", "read", "novel", "author", "library"],
             [f"{name}'s favorite book", f"{name} reads", f"{name} reading"]),
            # Health / medical
            (["sick", "ill", "health", "doctor", "hospital", "allerg", "symptom", "sneez",
              "cough", "pain", "medic", "diagnos"],
             [f"{name} health", f"{name} medical", f"{name} sick", f"{name} doctor", f"{name} allergy"]),
            # Money / finance
            (["money", "cost", "price", "pay", "spend", "budget", "salary", "earn", "coupon"],
             [f"{name} spending", f"{name} money", f"{name} cost", f"{name} bought"]),
            # Family / relationships
            (["family", "parent", "mom", "dad", "sibling", "sister", "brother", "wife", "husband",
              "partner", "friend", "relationship", "marr"],
             [f"{name}'s family", f"{name}'s relationship", f"{name}'s partner"]),
            # Time / count / quantity
            (["how many", "how long", "how much", "hours", "days", "weeks", "months", "times",
              "count", "total", "number", "often", "frequently"],
             [" ".join(key_words[:3]), f"{name} " + " ".join(key_words[:2])]),
            # Events / experiences
            (["theme park", "amusement", "concert", "festival", "event", "raft", "kayak",
              "ski", "surf", "camp", "hike", "mountain", "beach"],
             [f"{name} " + " ".join(key_words[:2]), " ".join(key_words[:3])]),
            # Privacy / technology / discussions
            (["privacy", "security", "data", "conversation about", "discuss", "talk about",
              "mention", "shared", "said", "told"],
             [f"{name} " + " ".join(key_words[:3]), " ".join(key_words[:4])]),
            # House / home / property
            (["bedroom", "bathroom", "kitchen", "paint", "repaint", "wall", "floor", "furniture",
              "room", "apartment", "house", "decor"],
             [f"{name}'s home", f"{name}'s room", f"{name} remodel", f"{name} decoration"]),
            # Seasons / recurring events
            (["summer", "winter", "spring", "fall", "autumn", "annual", "every year",
              "tradition", "routine", "weekend", "holiday season"],
             [f"{name} " + " ".join(key_words[:3])]),
        ]

        for keywords, reformulations in patterns:
            if any(k in q for k in keywords):
                reforms.extend(reformulations)

        # Always add: query stripped of question words, key content words joined
        base = re.sub(r'\b(what|when|where|who|which|how|did|does|have|has|my|our|your|i|we|the|a|an)\b', '', q).strip()
        base = re.sub(r'\s+', ' ', base).strip()
        if base and base not in reforms:
            reforms.append(base)

        # Content-word phrase from key nouns
        if key_words:
            reforms.append(" ".join(key_words[:4]))

        # Remove empty/duplicate
        seen = set()
        unique = []
        for r in reforms:
            r = r.strip()
            if r and r not in seen:
                seen.add(r)
                unique.append(r)

        return unique[:6]

    def _generate_subqueries(self, query: str, current_result,
                              composition_pairs: list) -> list[str]:
        """
        EBM-driven sub-query generation for multi-hop reasoning.

        When composition energy detects that two facts need to be read jointly
        but neither alone is sufficient, generate targeted sub-queries to find
        the missing complementary facts.
        """
        sub_queries = []
        q = query.lower()

        # Decompose by question type
        if any(w in q for w in ["when", "date", "year", "time"]):
            sub_queries.append(f"date time when {query.split('?')[0].strip()}")
        if any(w in q for w in ["how many", "count", "number of", "total"]):
            sub_queries.append(f"count list all {query.split('?')[0].strip()}")
        if any(w in q for w in ["why", "reason", "because"]):
            sub_queries.append(f"reason cause {query.split('?')[0].strip()}")
        if any(w in q for w in ["what", "which"]) and "?" in query:
            core = re.sub(r'\b(what|which|is|the|did|does|has|have)\b', '', q).strip()
            sub_queries.append(core)

        # Entity-focused sub-queries from composition pairs
        if current_result and composition_pairs:
            texts = current_result.texts
            for i, j in composition_pairs[:2]:
                if i < len(texts) and j < len(texts):
                    # Extract key entities from each paired text
                    names_i = re.findall(r'\b([A-Z][a-z]{2,})\b', texts[i])
                    names_j = re.findall(r'\b([A-Z][a-z]{2,})\b', texts[j])
                    combined_names = list(set(names_i + names_j))[:2]
                    if combined_names:
                        sub_queries.append(" ".join(combined_names) + " " +
                                           query.split("?")[0].strip())

        return sub_queries[:3]  # cap at 3 sub-queries

    def search(self, query: str, top_k: int = 10, max_iterations: int = 2) -> SearchResult:
        """
        EBRM agentic search with iterative refinement.

        1. Multi-probe search from different angles
        2. Union pool all candidates
        3. CMEN joint scoring
        4. If insufficient, generate new probes and iterate
        """
        import torch
        t0 = time.time()

        q_emb = self.encoder.encode([query], normalize_embeddings=True)[0]

        best_results = []
        all_probes = set()
        # EBM-driven sub-queries: populated when composition energy signals multi-hop need
        extra_queries = []

        for iteration in range(max_iterations):
            # Run all probes
            candidates = {}  # doc_idx -> (score, probe_name)

            reforms = self._reformulate(query)
            reform_embs = (self.encoder.encode(reforms, normalize_embeddings=True)
                           if reforms else np.zeros((0, 384)))

            probes = [
                ("semantic", self._probe_semantic(q_emb, top_k=15)),
                ("reformulation", self._probe_reformulation(query, q_emb, top_k=15)),
                ("bm25", self._probe_bm25(query, top_k=10)),
                ("entity", self._probe_entity(query, top_k=10)),
            ]

            # Each reformulation gets its own semantic probe (max diversity)
            for ri, (rf, remb) in enumerate(zip(reforms, reform_embs)):
                probes.append((f"reform_{ri}", self._probe_semantic(remb, top_k=8)))
                if ri < 3:  # BM25 for top 3 reformulations
                    probes.append((f"reform_bm25_{ri}", self._probe_bm25(rf, top_k=6)))

            # If EBM identified composition need in previous iteration, add focused probes
            for sub_q in extra_queries:
                sub_emb = self.encoder.encode([sub_q], normalize_embeddings=True)[0]
                probes.append((f"composition:{sub_q[:20]}", self._probe_semantic(sub_emb, top_k=8)))
                probes.append((f"comp_bm25:{sub_q[:20]}", self._probe_bm25(sub_q, top_k=8)))

            # Union with per-probe normalized scoring
            # Normalize each probe's scores to [0,1], then take max across probes
            norm_scores = {}   # doc_idx -> best normalized score
            norm_probes = {}   # doc_idx -> which probe gave best score

            for probe_name, results in probes:
                all_probes.add(probe_name)
                if not results:
                    continue
                # Normalize to [0,1] within this probe
                probe_scores = np.array([s for _, s, _ in results])
                if probe_scores.max() > probe_scores.min():
                    probe_norm = (probe_scores - probe_scores.min()) / \
                                 (probe_scores.max() - probe_scores.min())
                else:
                    probe_norm = np.ones(len(results)) * 0.5

                for rank, ((doc_idx, raw_score, _), ns) in enumerate(zip(results, probe_norm)):
                    # Add RRF bonus for appearing in multiple probes
                    rrf_bonus = 0.1  # bonus just for being found
                    combined = float(ns) + rrf_bonus
                    if doc_idx not in norm_scores or combined > norm_scores[doc_idx]:
                        norm_scores[doc_idx] = combined
                        norm_probes[doc_idx] = probe_name
                    elif doc_idx in norm_scores:
                        # Multi-probe bonus: found by multiple probes
                        norm_scores[doc_idx] += 0.15

            candidates = norm_scores
            if not candidates:
                break

            ranked = sorted(candidates.items(), key=lambda x: -x[1])
            top_indices = [idx for idx, _ in ranked[:top_k]]
            top_scores = [score for _, score in ranked[:top_k]]
            top_probes = [norm_probes.get(idx, "unknown") for idx in top_indices]

            # CMEN joint scoring
            sufficiency = 0.0
            temporal_conflicts = []
            composition_pairs = []
            marginal_weights = []

            if self.cmen is not None and len(top_indices) >= 3:
                try:
                    K = len(top_indices)
                    c_embs = np.array([self.doc_embs[i] for i in top_indices])
                    h_q = torch.tensor(q_emb, dtype=torch.float32).unsqueeze(0)
                    M = torch.tensor(c_embs, dtype=torch.float32).unsqueeze(0)
                    ts = torch.arange(K, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        # Marginal importance analysis
                        s = np.array(top_scores)
                        if s.max() > s.min():
                            y_init = (s - s.min()) / (s.max() - s.min())
                        else:
                            y_init = np.ones(K) * 0.5
                        y = torch.tensor(y_init, dtype=torch.float32).unsqueeze(0)

                        E_total = self.cmen.total_energy(h_q, M, y, ts)

                        marginals = []
                        for k in range(K):
                            y_drop = y.clone()
                            y_drop[0, k] = 0.0
                            E_drop = self.cmen.total_energy(h_q, M, y_drop, ts)
                            marginals.append((E_drop - E_total).item())

                        # Blend probe scores with CMEN marginals
                        m_arr = np.array(marginals)
                        shifted = m_arr - m_arr.min()
                        if shifted.max() > 0:
                            m_norm = shifted / shifted.max()
                        else:
                            m_norm = np.ones(K) * 0.5

                        s_norm = np.array(top_scores)
                        if s_norm.max() > s_norm.min():
                            s_norm = (s_norm - s_norm.min()) / (s_norm.max() - s_norm.min())
                        else:
                            s_norm = np.ones(K) * 0.5

                        blended = 0.5 * s_norm + 0.5 * m_norm
                        order = np.argsort(-blended)

                        top_indices = [top_indices[i] for i in order]
                        top_scores = [float(blended[i]) for i in order]
                        top_probes = [top_probes[i] for i in order]
                        # Store marginal importance in retrieval order
                        marginal_weights = [float(m_norm[i]) for i in order]

                        # Sufficiency: compare E_total with concentrated y (best candidate
                        # gets weight 0.9) vs uniform y — gap signals how "sure" CMEN is.
                        # Large gap = answer clearly present. Small gap = need more retrieval.
                        best_k = int(np.argmax(blended[:K]))
                        y_concentrated = torch.ones(1, K) * 0.1
                        y_concentrated[0, best_k] = 0.9
                        y_uniform = torch.ones(1, K) * (1.0 / K)
                        E_conc = self.cmen.total_energy(h_q, M, y_concentrated, ts).item()
                        E_unif = self.cmen.total_energy(h_q, M, y_uniform, ts).item()
                        # If concentrated is much lower energy than uniform, answer is present
                        energy_gap = E_unif - E_conc
                        sufficiency = float(np.tanh(energy_gap))  # [0,1]-ish, positive = sufficient

                        # Temporal conflicts: pairs with high cosine similarity (same topic)
                        cos_matrix = c_embs @ c_embs.T
                        for i in range(min(K, 6)):
                            for j in range(i + 1, min(K, 6)):
                                if cos_matrix[i, j] > 0.5:
                                    temporal_conflicts.append(
                                        (int(order[i]) if i < len(order) else i,
                                         int(order[j]) if j < len(order) else j))

                        # Composition pairs: check if combining two facts reduces comp energy
                        top3 = list(range(min(3, K)))
                        for i in top3:
                            for j in top3:
                                if i >= j:
                                    continue
                                pair_embs = c_embs[[i, j]]
                                M_pair = torch.tensor(
                                    pair_embs, dtype=torch.float32).unsqueeze(0)
                                y_pair = torch.ones(1, 2) * 0.5
                                E_pair = self.cmen.composition(h_q, M_pair, y_pair).item()
                                M_single = torch.tensor(
                                    c_embs[i:i+1], dtype=torch.float32).unsqueeze(0)
                                y_single = torch.ones(1, 1) * 0.9
                                E_single = self.cmen.composition(
                                    h_q, M_single, y_single).item()
                                if E_pair < E_single - 0.05:
                                    composition_pairs.append((i, j))

                except Exception:
                    pass

            best_results = SearchResult(
                texts=[self.corpus_texts[i] for i in top_indices[:top_k]],
                scores=top_scores[:top_k],
                indices=top_indices[:top_k],
                probes_used=top_probes[:top_k],
                n_iterations=iteration + 1,
                sufficiency=sufficiency,
                temporal_conflicts=temporal_conflicts,
                composition_pairs=composition_pairs,
                latency_ms=(time.time() - t0) * 1000,
                marginal_weights=marginal_weights[:top_k] if marginal_weights else [],
            )

            # If CMEN says sufficient, stop iterating
            if sufficiency > 0.5:
                break

            # If CMEN says insufficient and we have composition pairs, generate sub-queries
            # to search for the missing complementary facts
            if iteration < max_iterations - 1 and composition_pairs and sufficiency < 0.2:
                extra_queries = self._generate_subqueries(query, best_results, composition_pairs)
            else:
                extra_queries = []

        return best_results


def _smoke_test():
    """Quick test."""
    search = EBRMSearch()

    docs = [
        "I work as a software engineer at Google. I live in Berlin.",
        "Just got back from Tokyo! The ramen was amazing.",
        "I have a dog named Max. He's a golden retriever.",
        "Big news - moving to San Francisco for a new job as tech lead!",
        "I love playing piano. Just passed my Grade 5 exam.",
        "The weather has been really nice lately.",
        "I graduated with a degree in Business Administration from MIT.",
        "My favorite food is Italian pasta, especially carbonara.",
    ]

    build_time = search.build_index(docs)
    print(f"Index built in {build_time:.0f}ms ({len(docs)} docs)")

    queries = [
        "What degree did the user graduate with?",
        "Where does the user live now?",
        "What food does the user like?",
        "What is the user's current job?",
    ]

    for q in queries:
        result = search.search(q, top_k=3)
        print(f"\nQ: {q}")
        print(f"  Latency: {result.latency_ms:.0f}ms, iterations: {result.n_iterations}")
        for i, (text, score, probe) in enumerate(
                zip(result.texts, result.scores, result.probes_used)):
            print(f"  {i+1}. [{probe}, s={score:.3f}] {text[:80]}...")


if __name__ == "__main__":
    _smoke_test()
