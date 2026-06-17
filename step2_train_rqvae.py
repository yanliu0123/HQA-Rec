"""
Step 2: Residual Quantized VAE (RQ-VAE)
========================================
Input:  Lorentz embeddings from Step 1
Output: RQ-VAE checkpoint (best.pth)

Two-phase training:
  Phase 1: MSE warmup (reconstruction)
  Phase 2: MSE + InfoNCE contrastive loss (preserve citation structure)

Validation: Train self-retrieval — Lorentz vs z vs zq on citation graph restoration

Usage:
  python step2_train_rqvae.py --device cuda:0 --dataset NLP
  python step2_train_rqvae.py --device cuda:0 --dataset ML --eval_only
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import load_and_split

# ======== Config ========
SCALE = 100.0
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# Default config (can be overridden via args)
E_DIM = 512
NUM_EMB = [4096] * 8
LAYERS = []  # encoder: Linear(769, 512)


def get_emb_path(dataset):
    return os.path.join(OUTPUT_DIR, f'{dataset}_embeddings_v3_dim768.npy')


def get_ckpt_dir(dataset, num_emb, e_dim):
    """Generate checkpoint directory name from config."""
    if len(set(num_emb)) == 1:
        tag = f"{len(num_emb)}x{num_emb[0]}_{e_dim}d"
    else:
        tag = f"{'_'.join(map(str, num_emb))}_{e_dim}d"
    return os.path.join(OUTPUT_DIR, f'ckpt_rqvae_{dataset}_{tag}')


# ======== Model ========

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, e_dim, ema_decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.e_dim = e_dim
        self.ema_decay = ema_decay
        self.eps = eps
        self.embedding = nn.Embedding(num_embeddings, e_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        self.register_buffer('epoch_usage', torch.zeros(num_embeddings))

    def init_codebook_from_data(self, data):
        flat = data.reshape(-1, self.e_dim)
        n = min(flat.size(0), self.num_embeddings)
        perm = torch.randperm(flat.size(0))[:n]
        self.embedding.weight.data[:n] = flat[perm].clone()
        if n < self.num_embeddings:
            self.embedding.weight.data[n:] = flat[torch.randint(0, flat.size(0),
                                                                 (self.num_embeddings - n,))].clone()
        self.embed_avg.data.copy_(self.embedding.weight.data)
        self.cluster_size.fill_(1.0)

    def forward(self, z):
        flat = z.reshape(-1, self.e_dim)
        d = flat.pow(2).sum(1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(1).unsqueeze(0) - \
            2 * flat @ self.embedding.weight.t()
        indices = d.argmin(-1)
        z_q = self.embedding(indices).view(z.shape)

        if self.training:
            onehot = F.one_hot(indices, self.num_embeddings).float()
            new_size = self.ema_decay * self.cluster_size + (1 - self.ema_decay) * onehot.sum(0)
            new_sum = self.ema_decay * self.embed_avg + (1 - self.ema_decay) * (onehot.t() @ flat)
            self.cluster_size.data.copy_(new_size)
            self.embed_avg.data.copy_(new_sum)
            n = new_size.sum()
            cluster_size = (new_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            self.embedding.weight.data.copy_(new_sum / cluster_size.unsqueeze(1))
            self.epoch_usage.data += onehot.sum(0)

        commitment_loss = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()
        return z_q, commitment_loss, indices.view(z.shape[:-1])


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_emb_list, e_dim, ema_decay=0.99):
        super().__init__()
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n, e_dim, ema_decay) for n in num_emb_list
        ])

    def forward(self, z):
        residual = z
        zq = torch.zeros_like(z)
        total_loss = 0
        all_indices = []
        for vq in self.vq_layers:
            zq_i, loss_i, idx_i = vq(residual)
            zq = zq + zq_i
            residual = z - zq
            total_loss = total_loss + loss_i
            all_indices.append(idx_i)
        return zq, total_loss, all_indices


class RQVAE(nn.Module):
    def __init__(self, in_dim, num_emb_list, e_dim=512, layers=None,
                 beta=0.25, ema_decay=0.99):
        super().__init__()
        self.beta = beta
        enc_layers = []
        prev = in_dim
        for h in (layers or []):
            enc_layers.extend([nn.Linear(prev, h, bias=False), nn.ReLU()])
            prev = h
        enc_layers.append(nn.Linear(prev, e_dim, bias=False))
        self.encoder = nn.Sequential(*enc_layers)
        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, ema_decay)
        self.decoder = nn.Sequential(nn.Linear(e_dim, in_dim, bias=False))

    def forward(self, x):
        z = self.encoder(x)
        zq, vq_loss, indices = self.rq(z)
        recon = self.decoder(zq)
        return recon, self.beta * vq_loss, indices

    def get_indices(self, x):
        z = self.encoder(x)
        residual = z.clone()
        zq = torch.zeros_like(z)
        idxs = []
        for vq in self.rq.vq_layers:
            _, _, idx = vq(residual)
            z_res = vq.embedding(idx.long())
            zq = zq + z_res
            residual = z - zq
            idxs.append(idx)
        return torch.stack(idxs, dim=-1)


# ======== Utilities ========

def build_cite_graph(dataset_name):
    train_papers, _, _, _ = load_and_split(dataset_name)
    id2idx = {p["id"]: i for i, p in enumerate(train_papers)}
    pairs, cite_set = [], {}
    for p in train_papers:
        ci = id2idx[p["id"]]
        for ref_id in p.get("references", []):
            if ref_id in id2idx:
                cd = id2idx[ref_id]
                pairs.append((ci, cd))
                cite_set.setdefault(ci, set()).add(cd)
                cite_set.setdefault(cd, set()).add(ci)
    return pairs, cite_set, len(train_papers)


def extract_all(model, embeddings_np, device):
    """Extract z (encoder output), zq (quantized), tok (token indices)."""
    model.eval()
    all_emb = torch.from_numpy(embeddings_np).float().to(device) * SCALE
    z_list, zq_list, tok_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(all_emb), 2048):
            batch = all_emb[i:i + 2048]
            z = model.encoder(batch)
            residual = z.clone()
            zq = torch.zeros_like(z)
            idxs = []
            for vq in model.rq.vq_layers:
                _, _, idx = vq(residual)
                z_res = vq.embedding(idx.long())
                zq = zq + z_res
                residual = z - zq
                idxs.append(idx)
            z_list.append(z.cpu())
            zq_list.append(zq.cpu())
            tok_list.append(torch.stack(idxs, dim=-1).cpu())
    return torch.cat(z_list, 0), torch.cat(zq_list, 0), torch.cat(tok_list, 0)


def eval_retrieval(emb, cite_set, tag="", ks=(10, 20, 50)):
    """Train self-retrieval: cosine top-K vs citation graph."""
    emb_norm = F.normalize(emb.float(), dim=-1)
    eval_papers = [p for p in cite_set if len(cite_set[p]) > 0]
    max_k = max(ks)

    recall_sums = {k: 0.0 for k in ks}
    count = 0
    for s in range(0, len(eval_papers), 512):
        bi = eval_papers[s:s + 512]
        sim = emb_norm[bi] @ emb_norm.t()
        for j, pi in enumerate(bi):
            sim[j, pi] = -1e9
        _, topk = sim.topk(max_k, dim=-1)
        topk = topk.numpy()
        for j, pi in enumerate(bi):
            gt = cite_set[pi]
            for k in ks:
                recall_sums[k] += len(set(topk[j, :k].tolist()) & gt) / len(gt)
            count += 1

    results = {k: recall_sums[k] / count for k in ks}
    print(f"  {tag:<25s}: R@10={results[10]:.4f}  R@20={results[20]:.4f}  R@50={results[50]:.4f}")
    return results


# ======== Training ========

def train_rqvae(embeddings, cite_pairs, cite_set, device,
                num_emb=None, e_dim=None, layers=None, ckpt_dir=None):
    """Two-phase RQ-VAE training with checkpoint saving."""
    num_emb = num_emb or NUM_EMB
    e_dim = e_dim or E_DIM
    layers = layers if layers is not None else LAYERS

    dataset = type('D', (), {
        'embeddings': embeddings, 'dim': embeddings.shape[-1],
        '__len__': lambda self: len(self.embeddings),
        '__getitem__': lambda self, i: (torch.FloatTensor(self.embeddings[i]), i),
    })()
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)

    model = RQVAE(in_dim=dataset.dim, num_emb_list=num_emb, e_dim=e_dim,
                  layers=layers, beta=0.25, ema_decay=0.99).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Encoder: {dataset.dim} -> {layers} -> {e_dim}")
    print(f"  Codebook: {num_emb} x {e_dim}d")

    # Pre-init codebooks from data
    all_data = torch.from_numpy(embeddings).float().to(device) * SCALE
    with torch.no_grad():
        z = model.encoder(all_data)
        residual = z.clone()
        zq = torch.zeros_like(z)
        for lvl, vq in enumerate(model.rq.vq_layers):
            vq.init_codebook_from_data(residual)
            flat = residual.view(-1, vq.e_dim)
            cb_sq = vq.embedding.weight.pow(2).sum(1).unsqueeze(0)
            idx_parts = []
            for ci in range(0, flat.size(0), 4096):
                chunk = flat[ci:ci + 4096]
                d_chunk = chunk.pow(2).sum(1, keepdim=True) + cb_sq - 2 * chunk @ vq.embedding.weight.t()
                idx_parts.append(d_chunk.argmin(-1))
            idx = torch.cat(idx_parts, 0)
            z_res = vq.embedding(idx)
            zq = zq + z_res.view(z.shape)
            residual = z - zq
            print(f"    Level {lvl}: residual MSE={residual.pow(2).sum(-1).mean().item():.2f}")

    # ======== Phase 1: MSE Warmup ========
    print(f"\n  Phase 1: MSE Warmup")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_rec, pat, best_state = float('inf'), 0, None

    for epoch in range(200):
        model.train()
        total_rec, cnt = 0, 0
        for batch in loader:
            x = batch[0].to(device) * SCALE
            opt.zero_grad()
            recon, vq_loss, _ = model(x)
            rec_loss = F.mse_loss(recon, x)
            loss = rec_loss + vq_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_rec += rec_loss.item()
            cnt += 1
        avg_rec = total_rec / cnt
        if (epoch + 1) % 20 == 0:
            print(f"    Ep {epoch+1:4d} | Rec={avg_rec:.6f}")
        if avg_rec < best_rec - 1e-5:
            best_rec = avg_rec
            pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= 3:
                print(f"    P1 early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    print(f"    P1 done, best rec={best_rec:.6f}")

    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "phase1.pth"))

    # ======== Phase 2: Contrastive ========
    print(f"\n  Phase 2: Contrastive")
    model.eval()
    all_tokens_list = []
    with torch.no_grad():
        for i in range(0, len(all_data), 2048):
            idx = model.get_indices(all_data[i:i + 2048])
            all_tokens_list.append(idx.cpu())
    all_tokens = torch.cat(all_tokens_list, 0).long().to(device)

    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_r20, pat, best_state = 0, 0, None
    pair_indices = list(range(len(cite_pairs)))

    for epoch in range(200):
        model.train()
        total_rec, total_cl, cnt = 0, 0, 0
        for batch in loader:
            x = batch[0].to(device) * SCALE
            opt2.zero_grad()
            recon, vq_loss, _ = model(x)
            rec_loss = F.mse_loss(recon, x)

            random.shuffle(pair_indices)
            cb = pair_indices[:4096]
            ci = torch.tensor([cite_pairs[i][0] for i in cb], device=device)
            cd = torch.tensor([cite_pairs[i][1] for i in cb], device=device)

            def get_zq(indices):
                bt = all_tokens[indices]
                zq_ = torch.zeros(len(indices), e_dim, device=bt.device)
                for l, vq in enumerate(model.rq.vq_layers):
                    zq_ = zq_ + vq.embedding(bt[:, l])
                return zq_

            a_zq = F.normalize(get_zq(ci), dim=-1)
            p_zq = F.normalize(get_zq(cd), dim=-1)
            cl_loss = F.cross_entropy(a_zq @ p_zq.t() / 0.1,
                                       torch.arange(a_zq.size(0), device=device))

            loss = rec_loss + vq_loss + cl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            total_rec += rec_loss.item()
            total_cl += cl_loss.item()
            cnt += 1

        if (epoch + 1) % 5 == 0:
            _, zq_all, _ = extract_all(model, embeddings, device)
            zq_norm = F.normalize(zq_all, dim=-1)
            eps = [p for p in cite_set if len(cite_set[p]) > 0]
            if len(eps) > 5000:
                eps = random.sample(eps, 5000)
            r20s, c = 0, 0
            for s in range(0, len(eps), 512):
                bi = eps[s:s + 512]
                sim = zq_norm[bi] @ zq_norm.t()
                for j, pi in enumerate(bi):
                    sim[j, pi] = -1e9
                _, topk = sim.topk(20, dim=-1)
                for j, pi in enumerate(bi):
                    r20s += len(set(topk[j].tolist()) & cite_set[pi]) / len(cite_set[pi])
                    c += 1
            r20 = r20s / c
            print(f"    P2 Ep {epoch+1:4d} | Rec={total_rec/cnt:.4f} | CL={total_cl/cnt:.4f} | zq R@20={r20:.4f}")

            if r20 > best_r20:
                best_r20 = r20
                pat = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                pat += 1
                if pat >= 3:
                    print(f"    P2 early stop at epoch {epoch+1}")
                    break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
    print(f"    P2 done, best zq R@20={best_r20:.4f}")
    print(f"    Saved to {ckpt_dir}/best.pth")
    return model


def load_model(device, dataset="NLP", num_emb=None, e_dim=None, layers=None):
    """Load a trained RQ-VAE checkpoint."""
    num_emb = num_emb or NUM_EMB
    e_dim = e_dim or E_DIM
    layers = layers if layers is not None else LAYERS
    ckpt_dir = get_ckpt_dir(dataset, num_emb, e_dim)
    ckpt_path = os.path.join(ckpt_dir, "best.pth")
    emb_path = get_emb_path(dataset)
    embeddings = np.load(emb_path)
    model = RQVAE(in_dim=embeddings.shape[-1], num_emb_list=num_emb,
                  e_dim=e_dim, layers=layers, beta=0.25, ema_decay=0.99).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False))
    print(f"Loaded checkpoint: {ckpt_path}")
    return model, embeddings


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="NLP")
    parser.add_argument("--eval_only", action="store_true", help="Skip training, only validate")
    parser.add_argument("--num_emb", nargs='+', type=int, default=None,
                        help="Codebook sizes per level, e.g. 4096 4096 4096 4096")
    parser.add_argument("--e_dim", type=int, default=None, help="Embedding dim")
    parser.add_argument("--layers", nargs='*', type=int, default=None,
                        help="Encoder hidden layers, e.g. 512 256")
    args = parser.parse_args()

    num_emb = args.num_emb or NUM_EMB
    e_dim = args.e_dim or E_DIM
    layers = args.layers if args.layers is not None else LAYERS
    ckpt_dir = get_ckpt_dir(args.dataset, num_emb, e_dim)
    emb_path = get_emb_path(args.dataset)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(args.device)

    cite_pairs, cite_set, num_papers = build_cite_graph(args.dataset)
    embeddings = np.load(emb_path)
    print(f"Dataset: {args.dataset}, Papers: {num_papers}, Citation pairs: {len(cite_pairs)}")
    print(f"Hyperbolic embeddings: {embeddings.shape}")
    print(f"Config: num_emb={num_emb}, e_dim={e_dim}, layers={layers}")

    # Train or load
    ckpt_path = os.path.join(ckpt_dir, "best.pth")
    if args.eval_only or os.path.exists(ckpt_path):
        if not os.path.exists(ckpt_path):
            print("ERROR: no checkpoint found, run without --eval_only first")
            return
        model, _ = load_model(device, args.dataset, num_emb, e_dim, layers)
    else:
        print(f"\n{'='*60}")
        print(f"Training RQ-VAE: {num_emb} x {e_dim}d")
        print(f"{'='*60}")
        t0 = time.time()
        model = train_rqvae(embeddings, cite_pairs, cite_set, device,
                            num_emb, e_dim, layers, ckpt_dir)
        print(f"Training took {time.time()-t0:.0f}s")

    # Validate: Lorentz vs z vs zq
    print(f"\n{'='*60}")
    print("Validation: Train Self-Retrieval (citation graph restoration)")
    print(f"{'='*60}")

    z, zq, tok = extract_all(model, embeddings, device)
    quant_mse = (z - zq).pow(2).sum(-1).mean().item()
    print(f"  z: {z.shape}, zq: {zq.shape}, tok: {tok.shape}")
    print(f"  Quantization MSE (z->zq): {quant_mse:.2f}\n")

    lorentz = torch.from_numpy(embeddings).float()
    eval_retrieval(lorentz, cite_set, tag="Lorentz 769d (upper bound)")
    eval_retrieval(z, cite_set, tag=f"z (encoder, {e_dim}d)")
    eval_retrieval(zq, cite_set, tag=f"zq (quantized, {e_dim}d)")


if __name__ == "__main__":
    main()
