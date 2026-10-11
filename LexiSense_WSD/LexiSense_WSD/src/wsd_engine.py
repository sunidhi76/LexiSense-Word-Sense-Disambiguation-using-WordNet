from typing import Dict, List, Optional
from nltk.corpus import wordnet as wn
from .preprocessing import context_tokens, normalize_tokens


def _signature(synset, enhanced=False):
    parts = [synset.definition(), " ".join(l.name().replace('_', ' ') for l in synset.lemmas())]
    parts += synset.examples()
    if enhanced:
        parts += [h.definition() for h in synset.hypernyms()]
        parts += [h.definition() for h in synset.hyponyms()]
    return normalize_tokens(" ".join(parts))


def _score(context: List[str], synset, enhanced=False):
    sig = _signature(synset, enhanced)
    cset = set(context)
    matches = sorted(cset.intersection(sig))
    score = len(matches)
    if enhanced:
        # Give exact matches a small boost when they occur in examples/lemmas.
        lemma_set = set(normalize_tokens(" ".join(l.name() for l in synset.lemmas())))
        score += 0.5 * len(cset.intersection(lemma_set))
    return float(score), matches


def analyze(sentence: str, target: str, algorithm: str = "Enhanced Lesk") -> Dict:
    candidates = wn.synsets(target.lower())
    if not candidates:
        return {"ok": False, "error": "WordNet could not find a sense for this word. Try another target word."}
    enhanced = algorithm.lower().startswith("enhanced")
    context = context_tokens(sentence, target)
    scored = []
    for syn in candidates:
        score, matches = _score(context, syn, enhanced)
        scored.append({"synset": syn, "score": score, "matches": matches, "info": {
            "definition": syn.definition(),
            "synonyms": sorted(set(l.name().replace('_', ' ') for l in syn.lemmas())),
            "examples": syn.examples(),
            "hypernyms": sorted(set(l.name().replace('_', ' ') for h in syn.hypernyms() for l in h.lemmas())),
            "hyponyms": sorted(set(l.name().replace('_', ' ') for h in syn.hyponyms() for l in h.lemmas())),
        }})
    scored.sort(key=lambda x: (-x["score"], x["synset"]))
    best = scored[0]
    max_score = best["score"]
    # Confidence is a comparative score, not a statistical probability.
    total = sum(x["score"] for x in scored)
    confidence = (max_score / total) if total > 0 else (1 / len(scored))
    return {"ok": True, "target": target, "sentence": sentence, "algorithm": algorithm,
            "context": context, "best": best, "candidates": scored,
            "confidence": confidence, "score": max_score}


def suggest_ambiguous_words(sentence: str, limit=8):
    words = []
    seen = set()
    for token in normalize_tokens(sentence, remove_stopwords=False):
        if token in seen or len(token) < 3: continue
        senses = wn.synsets(token)
        if len(senses) > 1:
            words.append((token, len(senses))); seen.add(token)
    return [w for w, _ in sorted(words, key=lambda x: -x[1])[:limit]]
