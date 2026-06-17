"""
Step 3: Token Self-Attention Recommendation Model
===================================================
Candidate-independent attention -> neighbor side can be precomputed -> fast evaluation

  Level 1 (Token self-attn): 8 tokens per neighbor mutually attend
  Level 2 (Neighbor self-attn): K neighbors mutually attend
  Score: sum_k cosine(r'_k, candidate_zq)

Fast evaluation: precompute R = sum_k normalize(r'_k), then score = R @ normalize(candidate)
  -> all candidates scored in a single matrix multiplication

Usage:
  python step3_train_token_selfattn.py --device cuda:0 --dataset NLP
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_loader import load_and_split
from evaluate import evaluate_all
from step2_train_rqvae import load_model, extract_all, build_cite_graph


class TokenSelfAttnModel(nn.Module):
    """Two-level self-attention: candidate does not participate in attention,
    only used for final cosine scoring."""

    def __init__(self, n_levels, n_emb, d, codebook_weights, proj_dim=128):
        super().__init__()
        self.n_levels = n_levels
        self.d = d
        self.proj_dim = proj_dim

        # Frozen codebook embeddings
        self.token_embs = nn.ModuleList([
            nn.Embedding(n_emb, d) for _ in range(n_levels)
        ])
        for l, w in enumerate(codebook_weights):
            self.token_embs[l].weight.data.copy_(w)
            self.token_embs[l].weight.requires_grad = False

        # Level 1: Token-level self-attention Q/K
        self.q_mlp1 = nn.Sequential(
            nn.Linear(d, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.k_mlp1 = nn.Sequential(
            nn.Linear(d, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

        # Level 2: Neighbor-level self-attention Q/K
        self.q_mlp2 = nn.Sequential(
            nn.Linear(d, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.k_mlp2 = nn.Sequential(
            nn.Linear(d, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def encode_tokens(self, token_ids):
        """token_ids: [..., n_levels] -> [..., n_levels, d]"""
        embs = []
        for l in range(self.n_levels):
            embs.append(self.token_embs[l](token_ids[..., l]))
        return torch.stack(embs, dim=-2)

    def compute_neighbor_repr(self, nb_emb):
        """Candidate-independent: precompute neighbor representation.
        nb_emb: [B, K, 8, d] -> r_ctx: [B, K, d]
        """
        B, K, L, d = nb_emb.shape

        # Level 1: Token self-attention per neighbor
        nb_flat = nb_emb.reshape(B * K, L, d)
        q1 = self.q_mlp1(nb_flat)
        k1 = self.k_mlp1(nb_flat)
        attn1 = torch.bmm(q1, k1.transpose(-1, -2)) / (self.proj_dim ** 0.5)
        attn1 = F.softmax(attn1, dim=-1)
        out1 = torch.bmm(attn1, nb_flat)
        r = out1.sum(dim=-2).reshape(B, K, d)

        # Level 2: Neighbor self-attention
        q2 = self.q_mlp2(r)
        k2 = self.k_mlp2(r)
        attn2 = torch.bmm(q2, k2.transpose(-1, -2)) / (self.proj_dim ** 0.5)
        attn2 = F.softmax(attn2, dim=-1)
        r_ctx = torch.bmm(attn2, r)

        return r_ctx

    def compute_score(self, r_ctx, cand_emb):
        """
        r_ctx:    [B, K, d]  (precomputed neighbor repr)
        cand_emb: [B, 8, d]  (candidate token embeddings)
        return:   [B] scores
        """
        cand_repr = cand_emb.sum(dim=-2)  # [B, d] = zq
        r_norm = F.normalize(r_ctx, dim=-1)
        cand_norm = F.normalize(cand_repr, dim=-1)
        scores = (r_norm * cand_norm.unsqueeze(1)).sum(-1)
        return scores.sum(dim=-1)

    def compute_query_vec(self, r_ctx):
        """Fast eval: R = sum_k normalize(r'_k).
        r_ctx: [B, K, d] -> R: [B, d]
        Mathematically equivalent to sum_k cosine(r'_k, c) in compute_score.
        """
        return F.normalize(r_ctx, dim=-1).sum(dim=1)


class RecommendDataset(Dataset):
    def __init__(self, all_tokens, cite_set, specter_neighbors, K=20):
        self.all_tokens = all_tokens
        self.cite_set = cite_set
        self.specter_nb = specter_neighbors
        self.K = K
        self.N = len(all_tokens)
        self.anchors = [i for i in range(self.N) if i in cite_set and len(cite_set[i]) > 0]
        self.anchor_cites = {i: list(cite_set[i]) for i in self.anchors}

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        nb_indices = self.specter_nb[anchor, :self.K].long()
        nb_tokens = self.all_tokens[nb_indices]
        pos = random.choice(self.anchor_cites[anchor])
        pos_tokens = self.all_tokens[pos]
        return nb_tokens, pos_tokens


def encode_specter(papers, device, batch_size=256):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("allenai/specter")
    model = AutoModel.from_pretrained("allenai/specter").to(device).eval()
    vecs = []
    for i in tqdm(range(0, len(papers), batch_size), desc="SPECTER"):
        batch = papers[i:i + batch_size]
        texts = [p.get("title", "") + " " + p.get("text", "")[:256] for p in batch]
        enc = tok(texts, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state[:, 0]
        vecs.append(out.cpu())
    return torch.cat(vecs, 0)


def precompute_specter_neighbors(specter_vecs, K):
    spec_norm = F.normalize(specter_vecs, dim=-1)
    N = spec_norm.size(0)
    neighbors = torch.zeros(N, K, dtype=torch.long)
    for s in tqdm(range(0, N, 2048), desc="SPECTER neighbors"):
        e = min(s + 2048, N)
        sim = spec_norm[s:e] @ spec_norm.t()
        for j in range(e - s):
            sim[j, s + j] = -1e9
        neighbors[s:e] = sim.topk(K, dim=-1).indices
    return neighbors


def evaluate_recommendation(model, all_tokens, train_specter, test_specter,
                            test_papers, train_id_list, train_ids, device,
                            K=20, alpha=0.0):
    """Fast evaluation: precompute candidate repr + query vec, single matmul."""
    model.eval()

    with torch.no_grad():
        all_cand_repr = []
        for cs in range(0, len(all_tokens), 2048):
            ce = min(cs + 2048, len(all_tokens))
            emb = model.encode_tokens(all_tokens[cs:ce].to(device))
            repr_ = emb.sum(dim=-2)
            all_cand_repr.append(repr_.cpu())
        all_cand_repr = torch.cat(all_cand_repr, 0)
        all_cand_norm = F.normalize(all_cand_repr, dim=-1).to(device)

    train_spec_norm = F.normalize(train_specter, dim=-1)
    test_spec_norm = F.normalize(test_specter, dim=-1)
    all_tokens_dev = all_tokens.to(device)

    recommendations, ground_truths = {}, {}

    for start in tqdm(range(0, len(test_papers), 64), desc="Evaluating"):
        end = min(start + 64, len(test_papers))
        B_test = end - start
        batch_test_spec = test_spec_norm[start:end]
        spec_sim = batch_test_spec @ train_spec_norm.t()
        topk_nb = spec_sim.topk(K, dim=-1).indices

        nb_tokens_batch = all_tokens_dev[topk_nb.reshape(-1)].reshape(B_test, K, -1)
        with torch.no_grad():
            nb_emb = model.encode_tokens(nb_tokens_batch)
            r_ctx = model.compute_neighbor_repr(nb_emb)
            R = model.compute_query_vec(r_ctx)
            token_scores = (R @ all_cand_norm.T).cpu().numpy()

        for j, p in enumerate(test_papers[start:end]):
            gt = [r for r in p.get("references", []) if r in train_ids]
            if not gt:
                continue

            scores_j = token_scores[j]

            if alpha > 0:
                spec_scores = spec_sim[j].cpu().numpy()
                s_z = (spec_scores - spec_scores.mean()) / (spec_scores.std() + 1e-8)
                c_z = (scores_j - scores_j.mean()) / (scores_j.std() + 1e-8)
                hybrid = alpha * c_z + (1 - alpha) * s_z
                ranked = np.argsort(-hybrid)[:100]
            else:
                ranked = np.argsort(-scores_j)[:100]

            recommendations[p["id"]] = [train_id_list[idx] for idx in ranked]
            ground_truths[p["id"]] = gt

    return recommendations, ground_truths


def train_model(model, dataset, device, epochs=200, lr=1e-3, patience=3, n_neg=15,
                eval_every=20, eval_args=None):
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0,
                        drop_last=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    best_loss, pat, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        total_loss, cnt = 0, 0
        for nb_tok, pos_tok in loader:
            nb_tok = nb_tok.to(device)
            pos_tok = pos_tok.to(device)
            B = nb_tok.size(0)

            optimizer.zero_grad()

            nb_emb = model.encode_tokens(nb_tok)
            pos_emb = model.encode_tokens(pos_tok)

            # Neighbor repr computed once (candidate-independent)
            r_ctx = model.compute_neighbor_repr(nb_emb)

            # Positive + in-batch negatives (only candidate changes)
            scores_list = [model.compute_score(r_ctx, pos_emb)]
            for k in range(1, n_neg + 1):
                neg_emb = pos_emb.roll(k, dims=0)
                scores_list.append(model.compute_score(r_ctx, neg_emb))

            all_scores = torch.stack(scores_list, dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(all_scores, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            total_loss += loss.item()
            cnt += 1

        scheduler.step()
        avg_loss = total_loss / cnt
        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch+1:4d} | Loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

        # Mid-training evaluation
        if eval_args is not None and (epoch + 1) % eval_every == 0:
            recs, gts = evaluate_recommendation(
                model, eval_args['all_tokens'], eval_args['train_specter'],
                eval_args['test_specter'], eval_args['test_papers'],
                eval_args['train_id_list'], eval_args['train_ids'],
                device, K=eval_args['K'], alpha=0.0)
            m = evaluate_all(recs, gts)
            print(f"    [eval] Ep {epoch+1:4d} pure : R@10={m['Recall@10']:.4f}  R@20={m['Recall@20']:.4f}  "
                  f"NDCG@10={m['NDCG@10']:.4f}  NDCG@20={m['NDCG@20']:.4f}  "
                  f"MAP@10={m['MAP@10']:.4f}  MAP@20={m['MAP@20']:.4f}  MRR={m['MRR']:.4f}")
            recs, gts = evaluate_recommendation(
                model, eval_args['all_tokens'], eval_args['train_specter'],
                eval_args['test_specter'], eval_args['test_papers'],
                eval_args['train_id_list'], eval_args['train_ids'],
                device, K=eval_args['K'], alpha=0.1)
            m = evaluate_all(recs, gts)
            print(f"    [eval] Ep {epoch+1:4d} a=0.1: R@10={m['Recall@10']:.4f}  R@20={m['Recall@20']:.4f}  "
                  f"NDCG@10={m['NDCG@10']:.4f}  NDCG@20={m['NDCG@20']:.4f}  "
                  f"MAP@10={m['MAP@10']:.4f}  MAP@20={m['MAP@20']:.4f}  MRR={m['MRR']:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"  Best loss: {best_loss:.4f}")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="NLP")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_neg", type=int, default=15)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--eval_every", type=int, default=20)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(args.device)

    train_data, test_data, train_ids, _ = load_and_split(args.dataset)
    train_id_list = [p["id"] for p in train_data]
    _, cite_set, _ = build_cite_graph(args.dataset)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    print("\nLoading RQ-VAE...")
    rqvae_model, embeddings = load_model(device, dataset=args.dataset)
    _, _, all_tokens = extract_all(rqvae_model, embeddings, device)
    all_tokens = all_tokens.long()

    codebook_weights = [vq.embedding.weight.data.cpu().clone()
                        for vq in rqvae_model.rq.vq_layers]
    n_levels = len(codebook_weights)
    n_emb = codebook_weights[0].shape[0]
    d = codebook_weights[0].shape[1]
    del rqvae_model
    torch.cuda.empty_cache()
    print(f"Tokens: {all_tokens.shape}, Codebook: {n_levels}x{n_emb}x{d}")

    print("\nEncoding SPECTER (train)...")
    train_specter = encode_specter(train_data, device)

    print("Pre-computing SPECTER neighbors...")
    train_spec_nb = precompute_specter_neighbors(train_specter, args.K)
    print(f"  Done: {train_spec_nb.shape}")

    dataset = RecommendDataset(all_tokens, cite_set, train_spec_nb, K=args.K)
    print(f"Training samples: {len(dataset)}")

    model = TokenSelfAttnModel(n_levels, n_emb, d, codebook_weights,
                               proj_dim=args.proj_dim).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total:,} (trainable: {trainable:,})")

    print("\nEncoding SPECTER (test)...")
    test_specter = encode_specter(test_data, device)

    eval_args = {
        'all_tokens': all_tokens,
        'train_specter': train_specter,
        'test_specter': test_specter,
        'test_papers': test_data,
        'train_id_list': train_id_list,
        'train_ids': train_ids,
        'K': args.K,
    }

    print(f"\n{'='*60}")
    print("Training TokenSelfAttnModel (frozen codebook + self-attn Q/K MLP)")
    print(f"{'='*60}")
    model = train_model(model, dataset, device, epochs=args.epochs, lr=args.lr,
                        n_neg=args.n_neg, eval_every=args.eval_every,
                        eval_args=eval_args)

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'output', f'ckpt_token_selfattn_{args.dataset}')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
    print(f"Saved to {ckpt_dir}/best.pth")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final evaluation")
    print(f"{'='*60}")

    recs, gts = evaluate_recommendation(
        model, all_tokens, train_specter, test_specter,
        test_data, train_id_list, train_ids, device, K=args.K, alpha=0.0)
    m = evaluate_all(recs, gts)
    print(f"  SelfAttn pure   : R@10={m['Recall@10']:.4f}  R@20={m['Recall@20']:.4f}  "
          f"NDCG@10={m['NDCG@10']:.4f}  NDCG@20={m['NDCG@20']:.4f}  "
          f"MAP@10={m['MAP@10']:.4f}  MAP@20={m['MAP@20']:.4f}  MRR={m['MRR']:.4f}")

    for a in [0.1, 0.3, 0.5]:
        recs, gts = evaluate_recommendation(
            model, all_tokens, train_specter, test_specter,
            test_data, train_id_list, train_ids, device, K=args.K, alpha=a)
        m = evaluate_all(recs, gts)
        print(f"  SelfAttn a={a:.1f}  : R@10={m['Recall@10']:.4f}  R@20={m['Recall@20']:.4f}  "
              f"NDCG@10={m['NDCG@10']:.4f}  NDCG@20={m['NDCG@20']:.4f}  "
              f"MAP@10={m['MAP@10']:.4f}  MAP@20={m['MAP@20']:.4f}  MRR={m['MRR']:.4f}")


if __name__ == "__main__":
    main()
