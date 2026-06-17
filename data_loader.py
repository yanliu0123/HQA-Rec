"""Unified data loading and temporal split for citation recommendation."""

import json
import os
import pickle
from collections import defaultdict


def decode_abstract(indexed_abstract):
    """Decode inverted index abstract to plain text."""
    if not indexed_abstract:
        return ""
    length = indexed_abstract.get("IndexLength", 0)
    inv = indexed_abstract.get("InvertedIndex", {})
    words = [""] * length
    for word, positions in inv.items():
        for pos in positions:
            if pos < length:
                words[pos] = word
    return " ".join(words)


def load_dataset(path):
    """Load a processed dataset JSON, return list of paper dicts with 'text' field."""
    with open(path, "r") as f:
        papers = json.load(f)
    for p in papers:
        title = p.get("title", "") or ""
        abstract = decode_abstract(p.get("indexed_abstract"))
        p["text"] = (title + " " + abstract).strip()
    return papers


def temporal_split(papers, cutoff_year=2016):
    """Split papers by year: papers from cutoff_year onward as test set."""
    train_papers = [p for p in papers if p.get("year", 0) < cutoff_year]
    test_papers = [p for p in papers if p.get("year", 0) >= cutoff_year]

    train_ids = set(p["id"] for p in train_papers)

    # Only keep test papers with at least 1 reference in train set
    test_papers = [p for p in test_papers if
                   any(r in train_ids for r in p.get("references", []))]

    return train_papers, test_papers, train_ids, cutoff_year


def build_graph(papers):
    """Build adjacency list from citation references."""
    graph = defaultdict(set)
    id_set = set(p["id"] for p in papers)
    for p in papers:
        for r in p.get("references", []):
            if r in id_set:
                graph[p["id"]].add(r)
                graph[r].add(p["id"])
    return graph


def get_dataset_paths():
    """Return dict of dataset name -> path."""
    base = os.path.join(os.path.dirname(__file__), "data", "processed")
    return {
        "ML": os.path.join(base, "machine_learning.json"),
        "CV": os.path.join(base, "computer_vision.json"),
        "NLP": os.path.join(base, "natural_language_processing.json"),
    }


def load_and_split(name):
    """Load dataset by name, return (train, test, train_ids, cutoff_year)."""
    paths = get_dataset_paths()
    cache_path = paths[name].replace(".json", "_cache.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    papers = load_dataset(paths[name])
    result = temporal_split(papers)

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result
