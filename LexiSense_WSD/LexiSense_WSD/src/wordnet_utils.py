from functools import lru_cache
from nltk.corpus import wordnet as wn


def synsets_for_word(word: str):
    return wn.synsets(word.lower())


def synset_info(synset):
    lemmas = [l.name().replace('_', ' ') for l in synset.lemmas()]
    synonyms = sorted(set(lemmas))
    hypernyms = sorted({l.name().replace('_', ' ') for h in synset.hypernyms() for l in h.lemmas()})
    hyponyms = sorted({l.name().replace('_', ' ') for h in synset.hyponyms() for l in h.lemmas()})
    antonyms = sorted({a.name().replace('_', ' ') for l in synset.lemmas() for a in l.antonyms()})
    examples = synset.examples()
    return {
        "synset": synset.name(),
        "definition": synset.definition(),
        "pos": synset.pos(),
        "lemmas": synonyms,
        "synonyms": synonyms,
        "hypernyms": hypernyms,
        "hyponyms": hyponyms,
        "antonyms": antonyms,
        "examples": examples,
    }


def relation_graph_data(word: str):
    nodes, edges = [], []
    nodes.append({"id": word, "label": word, "group": "word"})
    seen = {word}
    for syn in synsets_for_word(word)[:5]:
        sid = syn.name()
        if sid not in seen:
            nodes.append({"id": sid, "label": sid, "group": "synset"}); edges.append((word, sid, "sense")); seen.add(sid)
        for label, rels in [("hypernym", syn.hypernyms()), ("hyponym", syn.hyponyms())]:
            for rel in rels[:3]:
                rid = rel.name()
                if rid not in seen:
                    nodes.append({"id": rid, "label": rid, "group": label}); seen.add(rid)
                edges.append((sid, rid, label))
    return nodes, edges
