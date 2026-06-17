"""Unified evaluation metrics for citation recommendation."""

import numpy as np
import torch
from tqdm import tqdm


def recall_at_k(ranked_list, ground_truth, k):
    return len(set(ranked_list[:k]) & set(ground_truth)) / len(ground_truth)


def precision_at_k(ranked_list, ground_truth, k):
    return len(set(ranked_list[:k]) & set(ground_truth)) / k


def mrr(ranked_list, ground_truth):
    for i, pid in enumerate(ranked_list):
        if pid in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_list, ground_truth, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, pid in enumerate(ranked_list[:k])
              if pid in ground_truth)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / ideal if ideal > 0 else 0.0


def ap_at_k(ranked_list, ground_truth, k):
    hits, score = 0, 0.0
    for i, pid in enumerate(ranked_list[:k]):
        if pid in ground_truth:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(ground_truth), k) if ground_truth else 0.0


def evaluate(ranked_list, ground_truth, ks=(10, 20)):
    """Compute all metrics, return dict."""
    gt = set(ground_truth)
    results = {}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(ranked_list, gt, k)
        results[f"NDCG@{k}"] = ndcg_at_k(ranked_list, gt, k)
        results[f"MAP@{k}"] = ap_at_k(ranked_list, gt, k)
    results["MRR"] = mrr(ranked_list, gt)
    return results


def evaluate_all(recommendations, ground_truths, ks=(10, 20)):
    """Average metrics over all queries."""
    all_metrics = {}
    for qid in recommendations:
        if qid not in ground_truths or not ground_truths[qid]:
            continue
        m = evaluate(recommendations[qid], ground_truths[qid], ks)
        for key, val in m.items():
            all_metrics.setdefault(key, []).append(val)

    avg = {k: np.mean(v) for k, v in all_metrics.items()}
    avg["num_queries"] = len(all_metrics.get("MRR", []))
    return avg


def recommend_by_cosine(test_vecs, train_vecs, test_papers, train_id_list, train_ids,
                        top_k=20, batch_size=1024):
    """Batch cosine similarity top-k on GPU. Inputs are torch tensors on device."""
    train_norm = torch.nn.functional.normalize(train_vecs, dim=-1)
    test_norm = torch.nn.functional.normalize(test_vecs, dim=-1)

    recommendations, ground_truths = {}, {}
    for start in tqdm(range(0, len(test_papers), batch_size), desc="  Recommending"):
        end = min(start + batch_size, len(test_papers))
        sims = test_norm[start:end] @ train_norm.T
        topk_indices = sims.topk(top_k, dim=-1).indices.cpu().tolist()

        for j, p in enumerate(test_papers[start:end]):
            gt = [r for r in p.get("references", []) if r in train_ids]
            if not gt:
                continue
            recommendations[p["id"]] = [train_id_list[k] for k in topk_indices[j]]
            ground_truths[p["id"]] = gt

    return recommendations, ground_truths
