from __future__ import annotations

from difflib import SequenceMatcher

from .models import ColumnProfile, DatasetMeta, StandardTerm


def _score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _token_overlap(a: str, b: str) -> float:
    a_tokens = {token for token in a.replace("_", " ").split() if token}
    b_tokens = {token for token in b.replace("_", " ").split() if token}
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _is_candidate(query: str, candidate: str, score: float) -> bool:
    if query == candidate:
        return True
    if query in candidate or candidate in query:
        return True
    overlap = _token_overlap(query, candidate)
    if overlap >= 0.5 and score >= 0.72:
        return True
    if score >= 0.82:
        return True
    return False


def resolve_with_rag(
    column: ColumnProfile,
    dataset_meta: DatasetMeta,
    standard_terms: dict[str, StandardTerm],
    synonym_index: dict[str, str],
    example_index: dict[str, list[str]],
    top_k: int = 3,
) -> ColumnProfile:
    query = column.normalized_name
    candidates: list[tuple[float, str]] = []

    if query in synonym_index:
        canonical = synonym_index[query]
        candidates.append((1.0, canonical))

    for name, term in standard_terms.items():
        score = _score(query, name)
        if dataset_meta.keywords:
            joined_keywords = " ".join(dataset_meta.keywords)
            if any(keyword in (term.description + term.domain_name) for keyword in dataset_meta.keywords):
                score += 0.05
            elif _score(query + joined_keywords, name + term.description) > 0.55:
                score += 0.03
        if _is_candidate(query, name, score):
            candidates.append((score, name))

    candidates.sort(reverse=True)
    chosen: list[str] = []
    evidence: list[str] = []
    for score, name in candidates[:top_k]:
        if name not in chosen:
            chosen.append(name)
            evidence.append(f"standard_term:{name}:score={score:.2f}")

    for example in example_index.get(column.raw_name, [])[:2]:
        evidence.append(f"dataset_example:{example}")

    if chosen:
        column.standard_candidates = chosen
        column.rag_required = False
        column.routing_confidence = max(column.routing_confidence, min(0.92, candidates[0][0]))
    column.rag_evidence = evidence
    return column
