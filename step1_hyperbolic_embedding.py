"""
Step 1: Two-Phase Hyperbolic Embedding
=======================================
Phase 1: Euclidean Collaborative Metric Learning (CML) with WARP hinge loss
Phase 2: Map to Lorentz manifold via exp_map0, then fine-tune with Riemannian BPR

Output: Lorentz coordinates (x0, x1, ..., xd), shape [N, d+1]
"""

import argparse
import random
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import load_and_split


# ==================== Hyperboloid Geometry ====================

class Hyperboloid:
    eps = {torch.float32: 1e-7, torch.float64: 1e-15}
    min_norm = 1e-15
    max_norm = 1e6

    @staticmethod
    def minkowski_dot(x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    @staticmethod
    def inner(u, v, keepdim=False, dim=-1):
        d = u.size(dim) - 1
        uv = u * v
        if not keepdim:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + \
                    uv.narrow(dim, 1, d).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)),
                             dim=dim).sum(dim=dim, keepdim=True)

    @staticmethod
    def sqdist(x, y, c):
        K = 1. / c
        prod = Hyperboloid.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + 1e-7)
        sqdist = K * torch.acosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    @staticmethod
    def proj(x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=-1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[..., 0] = 0
        vals = torch.zeros_like(x)
        vals[..., 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=1e-7))
        return vals + mask * x

    @staticmethod
    def proj_tan(u, x, c):
        d = x.size(-1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=-1, keepdim=True)
        mask = torch.ones_like(u)
        mask[..., 0] = 0
        vals = torch.zeros_like(u)
        vals[..., 0:1] = ux / torch.clamp(x[..., 0:1], min=1e-7)
        return vals + mask * u

    @staticmethod
    def minkowski_norm(u, keepdim=True):
        dot = Hyperboloid.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=1e-7))

    @staticmethod
    def expmap(u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = Hyperboloid.minkowski_norm(u)
        normu = torch.clamp(normu, max=1e6)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=1e-15)
        result = torch.cosh(theta) * x + torch.sinh(theta) * u / theta
        return Hyperboloid.proj(result, c)

    @staticmethod
    def logmap(x, y, c):
        K = 1. / c
        xy = torch.clamp(Hyperboloid.minkowski_dot(x, y) + K, max=-1e-7) - K
        u = y + xy * x * c
        normu = Hyperboloid.minkowski_norm(u)
        normu = torch.clamp(normu, min=1e-15)
        dist = Hyperboloid.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return Hyperboloid.proj_tan(result, x, c)

    @staticmethod
    def egrad2rgrad(x, grad, k, dim=-1):
        grad.narrow(-1, 0, 1).mul_(-1)
        grad = grad.addcmul(Hyperboloid.inner(x, grad, dim=dim, keepdim=True), x / k)
        return grad

    @staticmethod
    def ptransp(x, y, u, c):
        logxy = Hyperboloid.logmap(x, y, c)
        logyx = Hyperboloid.logmap(y, x, c)
        sqdist = torch.clamp(Hyperboloid.sqdist(x, y, c), min=1e-15)
        alpha = Hyperboloid.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return Hyperboloid.proj_tan(res, y, c)

    @staticmethod
    def dist_to_origin(x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        sqrtc = c ** 0.5
        return sqrtK * torch.acosh(torch.clamp(x[..., 0] * sqrtc, min=1.0 + 1e-7))


# ==================== Riemannian SGD ====================

class RiemannianSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.95, weight_decay=0.005,
                 dampening=0, c=1.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        dampening=dampening, c=c)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            lr = group['lr']
            c = group['c']
            for point in group['params']:
                grad = point.grad
                if grad is None:
                    continue
                state = self.state[point]
                if len(state) == 0:
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()
                if weight_decay > 0:
                    grad = grad.add(point, alpha=weight_decay)
                grad = Hyperboloid.egrad2rgrad(point, grad, c)
                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    grad = buf
                    new_point = Hyperboloid.expmap(-lr * grad, point, c)
                    components = new_point[:, 1:]
                    dim0 = torch.sqrt(torch.sum(components * components,
                                                dim=1, keepdim=True) + 1)
                    new_point = torch.cat([dim0, components], dim=1)
                    new_buf = Hyperboloid.ptransp(point, new_point, buf, c)
                    buf.copy_(new_buf)
                    point.data.copy_(new_point)
                else:
                    new_point = Hyperboloid.expmap(-lr * grad, point, c)
                    components = new_point[:, 1:]
                    dim0 = torch.sqrt(torch.sum(components * components,
                                                dim=1, keepdim=True) + 1)
                    new_point = torch.cat([dim0, components], dim=1)
                    point.data.copy_(new_point)


# ==================== exp_map0: Euclidean -> Lorentz ====================

def euclidean_to_lorentz(v):
    """Euclidean vector v in R^d -> Lorentz point x in H^d (d+1 dims).
    x_0 = cosh(||v||), x_spatial = sinh(||v||) * v / ||v||
    """
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-15)
    x0 = torch.cosh(norm)
    x_spatial = torch.sinh(norm) * v / norm
    return torch.cat([x0, x_spatial], dim=-1)


# ==================== Phase 1: Euclidean CML ====================

class EuclideanCML(nn.Module):
    """Collaborative Metric Learning: L2 distance + WARP hinge loss + unit ball projection."""
    def __init__(self, num_papers, embedding_dim=200,
                 margin=0.5, num_neg=50, lambda_c=1.0):
        super().__init__()
        self.embedding = nn.Embedding(num_papers, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        self.margin = margin
        self.num_neg = num_neg
        self.num_papers = num_papers
        self.lambda_c = lambda_c
        with torch.no_grad():
            self._project()

    def _project(self):
        """Project onto unit ball: y' = y / max(||y||, 1)."""
        w = self.embedding.weight.data
        norms = w.norm(dim=-1, keepdim=True)
        w.div_(norms.clamp(min=1.0))

    def get_all_embeddings(self):
        return self.embedding.weight

    def compute_loss(self, citer_idx, cited_idx):
        e_ci = self.embedding(citer_idx)
        e_cd = self.embedding(cited_idx)
        B = e_ci.size(0)

        # Positive pair L2 distance squared
        pos_dsq = ((e_ci - e_cd) ** 2).sum(-1)

        # Sample U negative samples
        neg_idx = torch.randint(0, self.num_papers, (B, self.num_neg),
                                device=e_ci.device)
        e_neg = self.embedding(neg_idx)
        neg_dsq = ((e_ci.unsqueeze(1) - e_neg) ** 2).sum(-1)

        # Hinge loss: [m + d_pos^2 - d_neg^2]+
        hinge = torch.clamp(self.margin + pos_dsq.unsqueeze(1) - neg_dsq, min=0)

        # WARP weighting: approximate rank
        n_impostor = (hinge > 0).float().sum(dim=1)
        approx_rank = (self.num_papers * n_impostor / self.num_neg).clamp(min=1)
        warp_w = torch.log(approx_rank + 1)

        # Hardest negative hinge
        max_hinge, _ = hinge.max(dim=1)
        loss = (warp_w * max_hinge).mean()

        # Covariance regularization (decorrelate dimensions)
        all_emb = torch.cat([e_ci, e_cd], dim=0)
        mu = all_emb.mean(0, keepdim=True)
        centered = all_emb - mu
        N = centered.size(0)
        cov = (centered.T @ centered) / N
        cov_reg = ((cov ** 2).sum() - (torch.diag(cov) ** 2).sum()) / N
        loss = loss + self.lambda_c * cov_reg

        acc = (pos_dsq.unsqueeze(1) < neg_dsq).float().mean().item()
        return loss, acc


# ==================== Phase 2: Lorentz BPR ====================

class LorentzBPR(nn.Module):
    def __init__(self, num_papers, embedding_dim=200, c=1.0):
        super().__init__()
        self.c = c
        self.num_papers = num_papers
        self.embedding = nn.Embedding(num_papers, embedding_dim + 1)

    def init_from_euclidean(self, euc_emb):
        """Initialize Lorentz coordinates from Euclidean embeddings via exp_map0."""
        with torch.no_grad():
            lorentz = euclidean_to_lorentz(euc_emb)
            self.embedding.weight.copy_(lorentz)

    def encode(self):
        return self.embedding.weight

    def compute_loss(self, embeddings, citer_idx, cited_idx):
        u_emb = embeddings[citer_idx]
        pos_emb = embeddings[cited_idx]
        neg_idx = torch.randint(0, self.num_papers, (citer_idx.size(0),),
                                device=citer_idx.device)
        neg_emb = embeddings[neg_idx]
        pos_dist = Hyperboloid.sqdist(u_emb, pos_emb, self.c)
        neg_dist = Hyperboloid.sqdist(u_emb, neg_emb, self.c)
        loss = -torch.log(torch.sigmoid(neg_dist - pos_dist) + 1e-8).mean()
        bpr_acc = (neg_dist > pos_dist).float().mean().item()
        return loss, bpr_acc


# ==================== Evaluation ====================

def evaluate_lorentz(emb, cite_pairs, num_papers, device):
    """Compute BPR acc + Hits@K + Poincare norm."""
    emb_spatial = emb[:, 1:]
    emb_time = emb[:, 0:1]
    poincare = emb_spatial / (1 + emb_time)
    pnorm = poincare.norm(dim=-1)

    eval_topk_n = min(5000, len(cite_pairs))
    eval_pairs = random.sample(cite_pairs, eval_topk_n)
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    chunk = 500
    for start in range(0, eval_topk_n, chunk):
        end = min(start + chunk, eval_topk_n)
        bp = eval_pairs[start:end]
        ci_b = torch.tensor([p[0] for p in bp], device=device)
        cd_b = torch.tensor([p[1] for p in bp], device=device)
        sd = emb[ci_b][:, 1:] @ emb_spatial.T
        td = emb[ci_b][:, 0:1] @ emb_time.T
        md = sd - td
        dist = torch.acosh(torch.clamp(-md, min=1.0 + 1e-7))
        dist[torch.arange(len(ci_b), device=device), ci_b] = float('inf')
        for k in hits:
            _, topk = dist.topk(k, dim=1, largest=False)
            hits[k] += (topk == cd_b.unsqueeze(1)).any(dim=1).sum().item()
    hits_rate = {k: hits[k] / eval_topk_n for k in hits}
    hits_str = " | ".join([f"H@{k}={hits_rate[k]:.4f}" for k in hits])

    eval_n = min(50000, len(cite_pairs))
    eval_pairs_bpr = random.sample(cite_pairs, eval_n)
    ci = torch.tensor([p[0] for p in eval_pairs_bpr], device=device)
    cd = torch.tensor([p[1] for p in eval_pairs_bpr], device=device)
    neg = torch.randint(0, num_papers, (eval_n,), device=device)
    pd = Hyperboloid.sqdist(emb[ci], emb[cd], 1.0).squeeze(-1)
    nd = Hyperboloid.sqdist(emb[ci], emb[neg], 1.0).squeeze(-1)
    rand_acc = (pd < nd).float().mean().item()

    return hits_str, hits_rate, rand_acc, pnorm.mean().item(), pnorm.std().item(), pnorm.max().item()


def evaluate_euclidean(emb, cite_pairs, num_papers, device):
    """Euclidean space: L2 distance BPR acc + Hits@K."""
    eval_n = min(50000, len(cite_pairs))
    ep = random.sample(cite_pairs, eval_n)
    ci = torch.tensor([p[0] for p in ep], device=device)
    cd = torch.tensor([p[1] for p in ep], device=device)
    neg = torch.randint(0, num_papers, (eval_n,), device=device)
    pos_dsq = ((emb[ci] - emb[cd]) ** 2).sum(-1)
    neg_dsq = ((emb[ci] - emb[neg]) ** 2).sum(-1)
    bpr = (pos_dsq < neg_dsq).float().mean().item()

    eval_topk_n = min(5000, len(cite_pairs))
    eval_pairs = random.sample(cite_pairs, eval_topk_n)
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    chunk = 500
    for start in range(0, eval_topk_n, chunk):
        end = min(start + chunk, eval_topk_n)
        bp = eval_pairs[start:end]
        ci_b = torch.tensor([p[0] for p in bp], device=device)
        cd_b = torch.tensor([p[1] for p in bp], device=device)
        dists = torch.cdist(emb[ci_b], emb)
        dists[torch.arange(len(ci_b), device=device), ci_b] = float('inf')
        for k in hits:
            _, topk = dists.topk(k, dim=1, largest=False)
            hits[k] += (topk == cd_b.unsqueeze(1)).any(dim=1).sum().item()

    hits_str = " | ".join([f"H@{k}={hits[k]/eval_topk_n:.4f}" for k in hits])
    return bpr, hits_str


# ==================== Training ====================

def build_citation_pairs(train_papers):
    id2idx = {p["id"]: i for i, p in enumerate(train_papers)}
    cite_pairs = []
    for p in train_papers:
        citer_idx = id2idx[p["id"]]
        for ref_id in p.get("references", []):
            if ref_id in id2idx:
                cite_pairs.append((citer_idx, id2idx[ref_id]))
    print(f"  Citation pairs: {len(cite_pairs)}")
    return cite_pairs


def train(args):
    train_papers, _, _, _ = load_and_split(args.dataset)
    num_papers = len(train_papers)
    print(f"Dataset: {args.dataset}, Papers: {num_papers}")

    cite_pairs = build_citation_pairs(train_papers)
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    base = os.path.splitext(args.save_path)[0]

    # ======================== Phase 1: Euclidean CML ========================
    print("\n" + "=" * 50)
    print("Phase 1: Euclidean CML")
    print("=" * 50)

    euc_model = EuclideanCML(num_papers, args.embedding_dim,
                              margin=args.margin, num_neg=args.num_neg,
                              lambda_c=args.lambda_c).to(device)
    euc_optimizer = torch.optim.Adam(euc_model.parameters(), lr=1e-3)

    num_batches = max(len(cite_pairs) // args.p1_batch_size, 1)
    best_bpr = 0.0
    patience_counter = 0

    pbar = tqdm(range(1, args.p1_epochs + 1), desc="Phase1", ncols=120)
    for epoch in pbar:
        euc_model.train()
        random.shuffle(cite_pairs)
        total_loss = 0
        total_bpr = 0

        for b in range(num_batches):
            batch = cite_pairs[b * args.p1_batch_size: (b + 1) * args.p1_batch_size]
            if len(batch) == 0:
                continue
            ci = torch.tensor([p[0] for p in batch], device=device)
            cd = torch.tensor([p[1] for p in batch], device=device)
            euc_optimizer.zero_grad()
            loss, bpr_acc = euc_model.compute_loss(ci, cd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(euc_model.parameters(), max_norm=5.0)
            euc_optimizer.step()
            with torch.no_grad():
                euc_model._project()
            total_loss += loss.item()
            total_bpr += bpr_acc

        avg_loss = total_loss / max(num_batches, 1)
        avg_bpr = total_bpr / max(num_batches, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", bpr=f"{avg_bpr:.4f}")

        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                emb = euc_model.get_all_embeddings()
                bpr, hits_str = evaluate_euclidean(emb, cite_pairs, num_papers, device)
                norms = emb.norm(dim=-1)
            print(f"\n  [P1] ep{epoch}: {hits_str} | BPR={bpr:.4f} | "
                  f"norm={norms.mean().item():.4f} std={norms.std().item():.4f}")

            if bpr > best_bpr:
                best_bpr = bpr
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.p1_patience:
                    print(f"  Phase 1 early stop at epoch {epoch}, best BPR={best_bpr:.4f}")
                    break

    # ======================== Mapping: Euclidean -> Lorentz ========================
    print("\n" + "=" * 50)
    print("Mapping: Euclidean -> Lorentz via exp_map0")
    print("=" * 50)

    with torch.no_grad():
        euc_emb = euc_model.get_all_embeddings().detach()
        euc_norms = euc_emb.norm(dim=-1)
        target_mean = 1.0
        scale = target_mean / euc_norms.mean()
        euc_emb_scaled = euc_emb * scale
        print(f"  Euclidean norm before: mean={euc_norms.mean().item():.4f} "
              f"std={euc_norms.std().item():.4f}")
        print(f"  Scale factor: {scale.item():.4f} -> target mean norm = {target_mean}")
        lorentz_init = euclidean_to_lorentz(euc_emb_scaled)
        spatial = lorentz_init[:, 1:]
        time_comp = lorentz_init[:, 0:1]
        poincare = spatial / (1 + time_comp)
        pnorm = poincare.norm(dim=-1)
        print(f"  After exp_map0: Poincare norm mean={pnorm.mean().item():.4f} "
              f"std={pnorm.std().item():.4f} max={pnorm.max().item():.4f}")

    # Save Euclidean CML embedding
    euc_path = f"{base}_euc.npy"
    np.save(euc_path, euc_emb.cpu().numpy())
    print(f"  Saved Euclidean CML embedding: {euc_path}")

    # Save Phase 1 Lorentz result
    ckpt_p1 = f"{base}_phase1.npy"
    np.save(ckpt_p1, lorentz_init.cpu().numpy())
    print(f"  Saved Phase 1 Lorentz: {ckpt_p1}")

    # ======================== Phase 2: Lorentz BPR ========================
    print("\n" + "=" * 50)
    print("Phase 2: Lorentz BPR")
    print("=" * 50)

    lor_model = LorentzBPR(num_papers, args.embedding_dim, c=1.0).to(device)
    lor_model.init_from_euclidean(euc_emb_scaled)

    lor_optimizer = RiemannianSGD(lor_model.parameters(),
                                  lr=args.p2_lr,
                                  momentum=args.p2_momentum,
                                  weight_decay=args.p2_weight_decay,
                                  c=1.0)

    del euc_model, euc_optimizer
    num_batches_p2 = max(len(cite_pairs) // args.p2_batch_size, 1)
    best_h20 = 0.0
    patience_counter = 0
    best_emb_np = None

    pbar = tqdm(range(1, args.p2_epochs + 1), desc="Phase2", ncols=120)
    for epoch in pbar:
        lor_model.train()
        random.shuffle(cite_pairs)
        total_loss = 0
        total_bpr = 0

        for b in range(num_batches_p2):
            batch = cite_pairs[b * args.p2_batch_size: (b + 1) * args.p2_batch_size]
            if len(batch) == 0:
                continue
            ci = torch.tensor([p[0] for p in batch], device=device)
            cd = torch.tensor([p[1] for p in batch], device=device)
            lor_optimizer.zero_grad()
            embeddings = lor_model.encode()
            loss, bpr_acc = lor_model.compute_loss(embeddings, ci, cd)
            if torch.isnan(loss):
                print(f"  NaN at epoch {epoch}, exiting!")
                return
            loss.backward()
            lor_optimizer.step()
            total_loss += loss.item()
            total_bpr += bpr_acc

        avg_loss = total_loss / max(num_batches_p2, 1)
        avg_bpr = total_bpr / max(num_batches_p2, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", bpr=f"{avg_bpr:.4f}")

        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                emb = lor_model.encode()
                hits_str, hits_rate, rand_acc, pn_mean, pn_std, pn_max = \
                    evaluate_lorentz(emb, cite_pairs, num_papers, device)
                all_emb_np = emb.cpu().numpy()

            print(f"\n  [P2] ep{epoch}: {hits_str} | BPR={rand_acc:.4f} | "
                  f"norm={pn_mean:.4f} std={pn_std:.4f} max={pn_max:.4f}")

            h20 = hits_rate[20]
            if h20 > best_h20:
                best_h20 = h20
                best_emb_np = all_emb_np.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.p2_patience:
                    print(f"  Phase 2 early stop at epoch {epoch}, best H@20={best_h20:.4f}")
                    break

    if best_emb_np is not None:
        all_emb = best_emb_np
        print(f"\n  Using best H@20={best_h20:.4f} checkpoint")
    else:
        with torch.no_grad():
            all_emb = lor_model.encode().cpu().numpy()
        print(f"\n  Using final model state (no eval was run in Phase 2)")
    np.save(args.save_path, all_emb)
    print(f"  Final saved to {args.save_path}, shape: {all_emb.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ML")
    p.add_argument("--embedding_dim", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save_path", type=str, default="./output/${DATASET}_embeddings_v3.npy")
    p.add_argument("--eval_freq", type=int, default=5)
    # Phase 1: Euclidean
    p.add_argument("--p1_epochs", type=int, default=50)
    p.add_argument("--p1_batch_size", type=int, default=32)
    p.add_argument("--p1_patience", type=int, default=3)
    # CML hyperparameters
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--num_neg", type=int, default=50)
    p.add_argument("--lambda_c", type=float, default=1.0)
    # Phase 2: Lorentz BPR
    p.add_argument("--p2_epochs", type=int, default=500)
    p.add_argument("--p2_batch_size", type=int, default=10000)
    p.add_argument("--p2_lr", type=float, default=0.001)
    p.add_argument("--p2_momentum", type=float, default=0.95)
    p.add_argument("--p2_weight_decay", type=float, default=0.005)
    p.add_argument("--p2_patience", type=int, default=3)
    args = p.parse_args()

    print("=" * 50)
    print("Step 1: Euclidean CML -> Lorentz BPR")
    print("=" * 50)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 50)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train(args)
