"""
Compositional Memory Energy Network (CMEN)

The retrieval process IS the reasoning process.

Instead of scoring memories independently, CMEN optimizes a JOINT
memory configuration y = [y_1, ..., y_K] where y_i represents how
much memory i should be included in the answer context.

6 specialized energy functions compose via Product of Experts:
  E_total = E_relevance + λ₂E_temporal + λ₃E_conflict +
            λ₄E_speaker + λ₅E_sufficiency + λ₆E_composition

Inference: Parallel Energy Minimization (PEM) — gradient descent
through annealed energy landscapes with particle resampling.

Key references:
  - IRED (Du et al., ICML 2024): annealed energy landscapes for reasoning
  - Compositional Energy Minimization (Oarga & Du, NeurIPS 2025): subproblem composition
  - Modern Hopfield Networks (Ramsauer 2020): attention as energy minimization
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelevanceEnergy(nn.Module):
    """E_rel(h_q, M, y) — set-level relevance. Low energy = relevant set selected."""

    def __init__(self, emb_dim=384, hidden=128):
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, hidden)
        self.k_proj = nn.Linear(emb_dim, hidden)
        self.v_proj = nn.Linear(emb_dim, hidden)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden + emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_q, M, y):
        """h_q:[B,D], M:[B,K,D], y:[B,K] -> [B] energy"""
        Q = self.q_proj(h_q).unsqueeze(1)
        K_ = self.k_proj(M)
        V = self.v_proj(M)
        attn = torch.bmm(Q, K_.transpose(1, 2)) / math.sqrt(K_.shape[-1])
        attn = attn + torch.log(y.unsqueeze(1).clamp(min=1e-8))
        attn = F.softmax(attn, dim=-1)
        ctx = torch.bmm(attn, V).squeeze(1)
        return self.energy_head(torch.cat([ctx, h_q], -1)).squeeze(-1)


class TemporalEnergy(nn.Module):
    """E_temp(M, y, t) — penalizes selecting temporally conflicting memories."""

    def __init__(self, emb_dim=384, hidden=64):
        super().__init__()
        self.pair_net = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, M, y, timestamps):
        """M:[B,K,D], y:[B,K], timestamps:[B,K] -> [B] energy"""
        B, K, D = M.shape
        # Efficient: only check top-weighted pairs (avoid K^2)
        top_k = min(10, K)
        top_idx = y.topk(top_k, dim=1).indices  # [B, top_k]

        E = torch.zeros(B, device=M.device)
        for i in range(top_k):
            for j in range(i + 1, top_k):
                idx_i = top_idx[:, i]  # [B]
                idx_j = top_idx[:, j]  # [B]
                m_i = M[torch.arange(B), idx_i]  # [B, D]
                m_j = M[torch.arange(B), idx_j]  # [B, D]
                dt = (timestamps[torch.arange(B), idx_i] -
                      timestamps[torch.arange(B), idx_j]).unsqueeze(-1)
                pair = torch.cat([m_i, m_j, dt.abs()], dim=-1)
                conflict = self.pair_net(pair).squeeze(-1)  # [B]
                weight = y[torch.arange(B), idx_i] * y[torch.arange(B), idx_j]
                E = E + conflict * weight
        return E


class RecencyPreferenceEnergy(nn.Module):
    """E_recency(M, y, t) — penalizes selecting older fact when newer supersedes it.

    Unlike TemporalEnergy which detects CONFLICT (symmetric), this module
    learns directional preference: when two facts are about the same topic,
    give LOWER energy to configurations that weight the newer fact higher.

    Key: uses SIGNED time difference (not abs), so the network learns direction.
    """

    def __init__(self, emb_dim=384, hidden=64):
        super().__init__()
        # Pair scorer: sees (m_old, m_new, signed_dt, y_old, y_new)
        self.preference_net = nn.Sequential(
            nn.Linear(emb_dim * 2 + 3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, M, y, timestamps):
        """M:[B,K,D], y:[B,K], timestamps:[B,K] -> [B] energy

        High energy = older fact has too much weight relative to newer one.
        """
        B, K, D = M.shape
        top_k = min(8, K)
        top_idx = y.topk(top_k, dim=1).indices

        E = torch.zeros(B, device=M.device)
        for i in range(top_k):
            for j in range(i + 1, top_k):
                idx_i = top_idx[:, i]
                idx_j = top_idx[:, j]
                m_i = M[torch.arange(B), idx_i]
                m_j = M[torch.arange(B), idx_j]
                t_i = timestamps[torch.arange(B), idx_i]
                t_j = timestamps[torch.arange(B), idx_j]
                y_i = y[torch.arange(B), idx_i]
                y_j = y[torch.arange(B), idx_j]

                # Signed dt: positive = i is newer, negative = j is newer
                dt = (t_i - t_j).unsqueeze(-1)

                pair = torch.cat([m_i, m_j, dt, y_i.unsqueeze(-1), y_j.unsqueeze(-1)], dim=-1)
                penalty = self.preference_net(pair).squeeze(-1)
                # Weight by how much both are selected
                weight = y_i * y_j
                E = E + penalty * weight
        return E


class SufficiencyEnergy(nn.Module):
    """E_suff(h_q, M, y) — high energy = insufficient info to answer."""

    def __init__(self, emb_dim=384, hidden=128):
        super().__init__()
        self.agg = nn.Sequential(nn.Linear(emb_dim, hidden), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_q, M, y):
        weighted = (M * y.unsqueeze(-1)).sum(dim=1)
        agg = self.agg(weighted)
        return self.head(torch.cat([agg, h_q], -1)).squeeze(-1)


class CompositionEnergy(nn.Module):
    """E_comp(h_q, M, y) — multi-hop reasoning via Hopfield-style attention."""

    def __init__(self, emb_dim=384, hidden=128, n_hops=2):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden)
        self.hops = nn.ModuleList([
            nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
            for _ in range(n_hops)
        ])
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))

    def forward(self, h_q, M, y):
        H = self.proj(M)
        mask = (y < 0.05)  # ignore low-weight memories
        state = self.proj(h_q).unsqueeze(1)
        for hop in self.hops:
            state, _ = hop(state, H, H, key_padding_mask=mask)
        return self.head(state.squeeze(1)).squeeze(-1)


class CMEN(nn.Module):
    """
    Compositional Memory Energy Network.

    Composes 4 energy functions for joint memory configuration optimization.
    """

    def __init__(self, emb_dim=384, hidden=128):
        super().__init__()
        self.relevance = RelevanceEnergy(emb_dim, hidden)
        self.temporal = TemporalEnergy(emb_dim, hidden // 2)
        self.recency = RecencyPreferenceEnergy(emb_dim, hidden // 2)
        self.sufficiency = SufficiencyEnergy(emb_dim, hidden)
        self.composition = CompositionEnergy(emb_dim, hidden)

        # Learnable composition weights (5 modules)
        self.log_lambdas = nn.Parameter(torch.zeros(5))

    @property
    def lambdas(self):
        return F.softplus(self.log_lambdas)

    def total_energy(self, h_q, M, y, timestamps=None):
        """Compute composed energy over memory configuration."""
        lam = self.lambdas
        E = lam[0] * self.relevance(h_q, M, y)
        if timestamps is not None:
            E = E + lam[1] * self.temporal(M, y, timestamps)
            E = E + lam[2] * self.recency(M, y, timestamps)
        # Negate sufficiency: module outputs HIGH for sufficient, LOW for insufficient
        E = E - lam[3] * self.sufficiency(h_q, M, y)
        E = E + lam[4] * self.composition(h_q, M, y)
        return E

    def optimize_configuration(
        self, h_q, M, timestamps=None,
        n_particles=16, n_steps=15, n_landscapes=3, lr=0.1,
    ):
        """
        PEM: optimize memory configuration y to minimize total energy.

        Returns: y* [K] optimal inclusion weights, sufficiency energy
        """
        B, K, D = M.shape
        assert B == 1, "Single query optimization"

        # Initialize from cosine similarity prior
        cos = F.cosine_similarity(h_q.unsqueeze(1), M, dim=-1)  # [1, K]
        y = torch.sigmoid(cos.expand(n_particles, -1) + 0.3 * torch.randn(n_particles, K, device=M.device))

        sigmas = torch.linspace(0.5, 0.05, n_landscapes)

        for landscape in range(n_landscapes):
            for step in range(n_steps):
                y.requires_grad_(True)
                energies = []
                for p in range(n_particles):
                    y_noisy = y[p:p+1] + sigmas[landscape] * torch.randn(1, K, device=M.device)
                    y_noisy = y_noisy.clamp(0, 1)
                    E = self.total_energy(h_q, M, y_noisy, timestamps)
                    energies.append(E)
                energies = torch.stack(energies).squeeze(-1)

                total_E = energies.sum()
                grad = torch.autograd.grad(total_E, y, create_graph=False)[0]

                with torch.no_grad():
                    grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    grad = grad / grad_norm.clamp(max=1.0)
                    y = y.detach() - lr * grad
                    y = y.clamp(0, 1)

            # Resample particles (adaptive temperature)
            with torch.no_grad():
                e_det = energies.detach().clone()
                e_range = e_det.max() - e_det.min()
                temp = max(e_range.item() * 0.2, 0.1)
                log_w = (-e_det / temp).double()
                log_w = log_w - log_w.max()
                weights = torch.exp(log_w).float()
                weights = weights.abs() + 1e-6  # guarantee non-negative
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, n_particles, replacement=True)
                y = y[indices]
                if landscape < n_landscapes - 1:
                    y = y + sigmas[landscape + 1] * torch.randn_like(y)
                    y = y.clamp(0, 1)

        # Best particle
        with torch.no_grad():
            best = energies.argmin()
            y_star = y[best]
            suff_E = self.sufficiency(h_q, M, y_star.unsqueeze(0)).item()

        return y_star.detach(), suff_E


def _smoke_test():
    model = CMEN(384, 128)
    B, K = 1, 20
    h_q = torch.randn(B, 384)
    M = torch.randn(B, K, 384)
    timestamps = torch.arange(K).float().unsqueeze(0)

    # Forward pass
    y = torch.rand(B, K)
    E = model.total_energy(h_q, M, y, timestamps)
    print(f"Total energy: {E.item():.4f}")

    # PEM optimization
    y_star, suff = model.optimize_configuration(h_q, M, timestamps, n_particles=8, n_steps=5, n_landscapes=2)
    print(f"Optimized y: top-3 weights = {y_star.topk(3).values.tolist()}")
    print(f"Sufficiency energy: {suff:.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("PASSED")


if __name__ == "__main__":
    _smoke_test()
