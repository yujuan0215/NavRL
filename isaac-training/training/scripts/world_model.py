"""SuperMimic-style differentiable world model for NavRL (pure PyTorch, no Isaac imports).

Training loop (SuperMimic, arXiv 2610.28480, Sec. 3.1-3.3), adapted to NavRL:
  1. collect transitions with the current policy into a replay buffer      (existing collector)
  2. fit a deterministic world model on k-step autoregressive windows      (Eq. 1)  -> WMTrainer.fit_world_model
  3. improve the policy by back-propagating an H-step imagined return
     through the frozen model, bootstrapped by a critic                    (Eq. 2-3) -> WMTrainer.improve_policy

From the paper: k=8 window, H=32 imagination horizon, survival-weighted return, critic bootstrap,
GAE-style lambda returns for the critic, gradients never pass through the simulator.
My own choices (not from the paper): network sizes, learning rates, per-dimension state normalisation,
gradient clipping, and the lidar being held fixed as context during imagination (see imagination_interface
in ppo.py).
"""
from types import SimpleNamespace
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningNorm(nn.Module):
    """Running mean/std with a std floor (constant dims, e.g. out-of-range obstacle slots, must not blow up)."""

    def __init__(self, dim: int, min_std: float = 1e-2):
        super().__init__()
        self.min_std = min_std
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        bm, bv = x.mean(0), x.var(0, unbiased=False)
        delta = bm - self.mean
        tot = self.count + n
        m2 = self.var * self.count + bv * n + delta ** 2 * self.count * n / tot
        self.mean.copy_(self.mean + delta * n / tot)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def _std(self):
        return torch.sqrt(self.var).clamp_min(self.min_std)

    def normalize(self, x):
        return (x - self.mean) / self._std()

    def denormalize(self, x):
        return x * self._std() + self.mean


def mlp(inp: int, hidden: int, layers: int, out: int) -> nn.Sequential:
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


class WorldModel(nn.Module):
    """f_w(s, a) -> (s', r, termination logit). Deterministic, residual dynamics."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256, layers: int = 3, action_scale: float = 1.0):
        super().__init__()
        self.state_dim, self.action_dim, self.action_scale = state_dim, action_dim, action_scale
        self.s_norm = RunningNorm(state_dim)
        self.delta_norm = RunningNorm(state_dim)
        self.net = mlp(state_dim + action_dim, hidden, layers, state_dim + 2)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def step(self, s: torch.Tensor, a: torch.Tensor):
        out = self.net(torch.cat([self.s_norm.normalize(s), a / self.action_scale], dim=-1))
        d = self.state_dim
        s_next = s + self.delta_norm.denormalize(out[..., :d])
        return s_next, out[..., d], out[..., d + 1]

    def loss(self, b: Dict[str, torch.Tensor]):
        """Eq. 1: k-step autoregressive fit seeded at the logged state, rolled on logged actions.

        The window is cut after the first episode boundary (alive mask), because the env auto-resets there.
        """
        s, a = b["s0"], b["a"]
        B, k = a.shape[:2]
        alive = torch.ones(B, device=s.device)
        tot = {"state": 0., "reward": 0., "term": 0.}
        denom = 0.
        for j in range(k):
            s_hat, r_hat, d_logit = self.step(s, a[:, j])
            e_s = ((self.s_norm.normalize(s_hat) - self.s_norm.normalize(b["s_next"][:, j])) ** 2).sum(-1)
            e_r = (r_hat - b["r"][:, j]) ** 2
            e_d = F.binary_cross_entropy_with_logits(d_logit, b["term"][:, j], reduction="none")
            tot["state"] = tot["state"] + (alive * e_s).sum()
            tot["reward"] = tot["reward"] + (alive * e_r).sum()
            tot["term"] = tot["term"] + (alive * e_d).sum()
            denom = denom + alive.sum()
            alive = alive * (1. - b["done"][:, j])
            s = s_hat
        denom = denom.clamp_min(1.)
        parts = {k_: v / denom for k_, v in tot.items()}
        return sum(parts.values()), parts


class TransitionBuffer:
    """Ring buffer of (num_env, T) rollout chunks, so k-step windows stay contiguous in time."""

    def __init__(self, capacity_steps: int, device):
        self.capacity, self.device = int(capacity_steps), device
        self.store = None
        self.ptr = self.size = 0

    def add(self, roll: Dict[str, torch.Tensor]):
        E, T = roll["s"].shape[:2]
        if self.store is None:
            self.E, self.T = E, T
            self.n_slots = max(4, self.capacity // (E * T))
            self.store = {k: torch.zeros((self.n_slots,) + tuple(v.shape), dtype=v.dtype, device=self.device)
                          for k, v in roll.items()}
        assert (E, T) == (self.E, self.T), "rollout chunk shape changed"
        for k, v in roll.items():
            self.store[k][self.ptr] = v.to(self.device)
        self.ptr = (self.ptr + 1) % self.n_slots
        self.size = min(self.size + 1, self.n_slots)

    def __len__(self):
        return 0 if self.store is None else self.size * self.E * self.T

    def sample(self, B: int, k: int):
        k = min(k, self.T)
        slot = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.E, (B,), device=self.device)
        t0 = torch.randint(0, self.T - k + 1, (B,), device=self.device)
        idx_t = t0[:, None] + torch.arange(k, device=self.device)[None]
        win = lambda key: self.store[key][slot[:, None], env[:, None], idx_t]
        return {
            "s0": self.store["s"][slot, env, t0],
            "a": win("a"), "s_next": win("s_next"), "r": win("r"), "term": win("term"), "done": win("done"),
            "ctx": {k_[4:]: v[slot, env, t0] for k_, v in self.store.items() if k_.startswith("ctx_")},
        }


class WMTrainer:
    """Owns the world model, imagined critic, replay buffer and the two optimisers."""

    def __init__(self, cfg, state_dim: int, action_dim: int, action_limit: float, device,
                 policy_params: Iterable[nn.Parameter]):
        self.cfg, self.device = cfg, device
        self.model = WorldModel(state_dim, action_dim, cfg.hidden, cfg.layers, action_limit).to(device)
        self.critic = mlp(state_dim, cfg.hidden, 2, 1).to(device)
        self.buffer = TransitionBuffer(cfg.buffer_size, device)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=cfg.model_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.policy_params = list(policy_params)
        self.policy_opt = torch.optim.Adam(self.policy_params, lr=cfg.policy_lr)

    def value(self, s):
        return self.critic(self.model.s_norm.normalize(s)).squeeze(-1)

    def add(self, roll: Dict[str, torch.Tensor]):
        self.model.s_norm.update(roll["s"])
        self.model.delta_norm.update(roll["s_next"] - roll["s"])
        self.buffer.add(roll)

    def ready(self):
        return len(self.buffer) >= self.cfg.min_buffer

    def fit_world_model(self, n_iters: int = None) -> Dict[str, float]:
        if not self.ready():
            return {}
        acc = {"state": 0., "reward": 0., "term": 0.}
        n = n_iters or self.cfg.model_iters
        for _ in range(n):
            loss, parts = self.model.loss(self.buffer.sample(self.cfg.model_batch, self.cfg.k))
            self.model_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.model_opt.step()
            for k_ in acc:
                acc[k_] += parts[k_].item() / n
        return {"wm/state_err": acc["state"], "wm/reward_err": acc["reward"], "wm/term_bce": acc["term"]}

    def improve_policy(self, iface, n_iters: int = None) -> Dict[str, float]:
        """Eq. 2-3. iface.encode(ctx)->enc (once per rollout), iface.act(s, enc)->a (differentiable)."""
        if not self.ready():
            return {}
        H, gamma, lmbda = self.cfg.horizon, self.cfg.gamma, self.cfg.lmbda
        n = n_iters or self.cfg.policy_iters
        stats = {"imag/return": 0., "imag/survival": 0., "imag/critic_loss": 0.}
        frozen = list(self.model.parameters())
        for p in frozen:
            p.requires_grad_(False)  # model is held fixed inside each policy update
        try:
            for _ in range(n):
                b = self.buffer.sample(self.cfg.policy_batch, 1)
                enc = iface.encode(b["ctx"])
                s = b["s0"]
                c = torch.ones(s.shape[0], device=s.device)
                J = torch.zeros_like(c)
                states, rs, ds = [s], [], []
                for t in range(H):
                    a = iface.act(s, enc)
                    s, r, d_logit = self.model.step(s, a)
                    d = torch.sigmoid(d_logit)
                    J = J + (gamma ** t) * c * r          # sum_t gamma^t c_t r_{t+1}
                    c = c * (1. - d)                      # soft survival weight
                    states.append(s); rs.append(r); ds.append(d)
                J = J + (gamma ** H) * c * self.value(s)  # critic bootstraps beyond H
                policy_loss = -J.mean()
                if not torch.isfinite(policy_loss):
                    continue
                self.policy_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_params, 5.)
                self.policy_opt.step()

                # critic: lambda-returns on the imagined transitions, targets fixed per rollout (Eq. 3)
                with torch.no_grad():
                    vals = [self.value(x) for x in states]
                    R, targets = vals[H], [None] * H
                    for t in reversed(range(H)):
                        R = rs[t] + gamma * (1. - ds[t]) * ((1. - lmbda) * vals[t + 1] + lmbda * R)
                        targets[t] = R
                    targets = torch.stack(targets)
                pred = self.value(torch.stack([x.detach() for x in states[:H]]))
                critic_loss = F.mse_loss(pred, targets)
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 5.)
                self.critic_opt.step()

                stats["imag/return"] += J.mean().item() / n
                stats["imag/survival"] += c.mean().item() / n
                stats["imag/critic_loss"] += critic_loss.item() / n
        finally:
            for p in frozen:
                p.requires_grad_(True)
        return stats


def rollout_from_tensordict(data, action_limit: float) -> Dict[str, torch.Tensor]:
    """Pull one collector batch (num_env, T, ...) into flat tensors for the buffer.

    World-model state = [drone state (8) | closest-N dynamic-obstacle features (N*10)], all already in the
    goal frame. Action = the *local-frame* velocity command (before vec_to_world), so the model never needs
    the goal direction. Lidar and goal direction are stored as context only.
    """
    E, T = data.batch_size[:2]
    obs, nxt = data["agents", "observation"], data["next", "agents", "observation"]
    flat = lambda x: x.reshape(E, T, -1)
    state = lambda o: torch.cat([flat(o["state"]), flat(o["dynamic_obstacle"])], dim=-1).detach()
    a = 2. * flat(data["agents", "action_normalized"]) * action_limit - action_limit
    r = flat(data["next", "agents", "reward"])
    assert r.shape[-1] == 1, "expected a scalar reward per env"
    return {
        "s": state(obs), "s_next": state(nxt), "a": a.detach(), "r": r[..., 0].detach(),
        "term": flat(data["next", "terminated"]).float()[..., 0],
        "done": flat(data["next", "done"]).float()[..., 0],
        "ctx_lidar": flat(obs["lidar"]).detach(),
    }
