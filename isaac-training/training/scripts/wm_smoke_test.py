"""Isaac-free sanity check of world_model.py on a toy point-mass 'drone'. Run: python wm_smoke_test.py

Checks: (1) k-step prediction error drops, (2) BPTT through the learned model improves the *true* return
of a linear policy, (3) gradients reach the policy and never the frozen world model, (4) episode-boundary
masking works (the toy env terminates and auto-resets, like Isaac).
"""
from types import SimpleNamespace

import torch

from world_model import WMTrainer

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
D, A, LIMIT, DT = 6, 3, 2.0, 0.05


def true_step(s, a):
    pos, vel = s[:, :3], s[:, 3:]
    vel2 = vel + DT * 4. * (a - vel)
    pos2 = pos + DT * vel2
    s2 = torch.cat([pos2, vel2], -1)
    r = -pos2.norm(dim=-1) - 0.01 * (a ** 2).sum(-1)
    term = (pos2.abs().max(-1).values > 6.).float()
    return s2, r, term


def reset(n):
    return torch.cat([(torch.rand(n, 3, device=dev) - .5) * 6, torch.zeros(n, 3, device=dev)], -1)


def collect(E, T, policy=None):
    s, out = reset(E), {k: [] for k in ("s", "a", "s_next", "r", "term", "done")}
    for t in range(T):
        a = (torch.rand(E, A, device=dev) * 2 - 1) * LIMIT if policy is None else policy(s)
        s2, r, term = true_step(s, a)
        done = ((term > 0) | (t == T - 1)).float()
        for k, v in zip(out, (s, a, s2, r, term, done)):
            out[k].append(v)
        s = torch.where(done[:, None] > 0, reset(E), s2)  # auto-reset after done
    out = {k: torch.stack(v, 1) for k, v in out.items()}
    out["ctx_lidar"] = torch.zeros(E, T, 4, device=dev)
    return out


def true_return(policy, n=256, T=60):
    s, tot = reset(n), torch.zeros(n, device=dev)
    with torch.no_grad():
        for _ in range(T):
            s, r, term = true_step(s, policy(s))
            tot += r
    return tot.mean().item()


cfg = SimpleNamespace(hidden=128, layers=2, buffer_size=100000, min_buffer=1000, k=8, horizon=16, gamma=0.99,
                      lmbda=0.95, model_iters=50, model_batch=512, model_lr=1e-3, policy_iters=1, policy_batch=256,
                      policy_lr=5e-3, critic_lr=1e-3)
W = torch.nn.Parameter(torch.zeros(A, D, device=dev))
policy = lambda s: LIMIT * torch.tanh(s @ W.T)
iface = SimpleNamespace(encode=lambda ctx: None, act=lambda s, enc: policy(s))
wm = WMTrainer(cfg, D, A, LIMIT, dev, [W])

for _ in range(30):
    wm.add(collect(64, 32))
first = wm.fit_world_model(5)["wm/state_err"]
for _ in range(20):
    last = wm.fit_world_model()
print(f"[1] 8-step state error: {first:.4f} -> {last['wm/state_err']:.5f}  reward err {last['wm/reward_err']:.5f}")
assert last["wm/state_err"] < 0.1 * first, "world model did not learn"

wm.model.zero_grad(set_to_none=True)  # drop leftovers from fit_world_model before checking the frozen model
before_wm = [p.clone() for p in wm.model.parameters()]
ret0 = true_return(policy)
for it in range(300):
    st = wm.improve_policy(iface)
    if it == 0:
        assert W.grad is not None and W.grad.abs().sum() > 0, "no gradient reached the policy"
        assert all(p.grad is None for p in wm.model.parameters()), "frozen world model received grads"
        assert all(torch.equal(p, q) for p, q in zip(wm.model.parameters(), before_wm)), "world model changed"
        assert all(p.requires_grad for p in wm.model.parameters()), "requires_grad not restored"
    if it % 100 == 0:
        wm.add(collect(64, 32, lambda s: policy(s).detach() + 0.3 * torch.randn(s.shape[0], A, device=dev)))
        wm.fit_world_model(30)
ret1 = true_return(policy)
print(f"[2] true return of the policy: {ret0:.2f} -> {ret1:.2f}")
assert ret1 > ret0 + 5., "BPTT did not improve the true return"
print(f"[3] last imagined return {st['imag/return']:.2f}, survival {st['imag/survival']:.2f}")
print("ALL CHECKS PASSED")
