# SPDX-License-Identifier: Apache-2.0
"""Reference A2C trainer for the in-kernel Flappy Kernel agent.

Mirrors the CUDA co-tenant exactly (physics, observation, network, update)
so hyperparameters can be validated in PyTorch before porting.
"""
import argparse
import time

import torch

W, H, BASE_Y = 288, 512, 404
BIRD_X, BIRD_W, BIRD_H = 57, 34, 24
PIPE_W, GAP, PIPE_VX, PIPE_SPACING = 52, 100, 4, 160
GRAV, FLAP_V, MAX_VY = 1.0, -9.0, 10.0
NUM_PIPES = 3
OBS = 6
HID = 64


def gap_sample(n, g):
    lo = int(BASE_Y * 0.2)
    hi = int(BASE_Y * 0.6 - GAP) + lo
    return torch.randint(lo, hi, (n,), generator=g).float()


class Envs:
    def __init__(self, n, dev, seed):
        self.n, self.dev = n, dev
        self.g = torch.Generator().manual_seed(seed)
        self.y = torch.zeros(n, device=dev)
        self.vy = torch.zeros(n, device=dev)
        self.px = torch.zeros(n, NUM_PIPES, device=dev)
        self.gy = torch.zeros(n, NUM_PIPES, device=dev)
        self.score = torch.zeros(n, device=dev)
        self.reset(torch.ones(n, dtype=torch.bool, device=dev))

    def reset(self, m):
        k = int(m.sum())
        if k == 0:
            return
        self.y[m] = H * 0.35
        self.vy[m] = 0
        self.score[m] = 0
        for i in range(NUM_PIPES):
            self.px[m, i] = W + 60 + i * PIPE_SPACING
            self.gy[m, i] = gap_sample(k, self.g).to(self.dev)

    def obs(self):
        ahead = self.px + PIPE_W > BIRD_X
        dist = torch.where(ahead, self.px, torch.full_like(self.px, 1e9))
        order = dist.argsort(dim=1)
        nx = self.px.gather(1, order[:, :1]).squeeze(1)
        ng = self.gy.gather(1, order[:, :1]).squeeze(1)
        ng2 = self.gy.gather(1, order[:, 1:2]).squeeze(1)
        return torch.stack([
            self.y / H,
            self.vy / MAX_VY,
            (nx - BIRD_X) / W,
            (ng - self.y) / H,
            (ng + GAP - self.y) / H,
            (ng2 - self.y) / H,
        ], 1)

    def step(self, flap):
        self.vy = torch.where(flap, torch.full_like(self.vy, FLAP_V),
                              torch.clamp(self.vy + GRAV, max=MAX_VY))
        self.y = self.y + self.vy
        before = self.px + PIPE_W / 2
        self.px = self.px - PIPE_VX
        after = self.px + PIPE_W / 2
        passed = ((before >= BIRD_X + BIRD_W / 2) & (after < BIRD_X + BIRD_W / 2)).sum(1).float()
        wrap = self.px < -PIPE_W
        if wrap.any():
            self.px = torch.where(wrap, self.px + NUM_PIPES * PIPE_SPACING, self.px)
            new = gap_sample(int(wrap.sum()), self.g).to(self.dev)
            self.gy[wrap] = new
        overlap_x = (self.px < BIRD_X + BIRD_W) & (self.px + PIPE_W > BIRD_X)
        hit_pipe = overlap_x & ((self.y.unsqueeze(1) < self.gy) |
                                (self.y.unsqueeze(1) + BIRD_H > self.gy + GAP))
        dead = hit_pipe.any(1) | (self.y + BIRD_H >= BASE_Y) | (self.y < 0)
        self.score += passed
        rew = 0.1 + passed - 2.0 * dead.float()
        final_score = torch.where(dead, self.score, torch.full_like(self.score, -1))
        self.reset(dead)
        return rew, dead, final_score


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(OBS, HID)
        self.l2 = torch.nn.Linear(HID, HID)
        self.head = torch.nn.Linear(HID, 2)

    def forward(self, x):
        h = torch.relu(self.l2(torch.relu(self.l1(x))))
        o = self.head(h)
        return o[:, 0], o[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    env = Envs(a.envs, a.device, 0)
    net = Net().to(a.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    recent = []
    t0 = time.time()
    for it in range(a.steps):
        obs_b, act_b, rew_b, done_b = [], [], [], []
        for _ in range(a.T):
            o = env.obs()
            with torch.no_grad():
                logit, _ = net(o)
            act = torch.rand_like(logit) < torch.sigmoid(logit)
            rew, dead, fs = env.step(act)
            obs_b.append(o); act_b.append(act.float()); rew_b.append(rew); done_b.append(dead.float())
            recent.extend(fs[fs >= 0].tolist())
        with torch.no_grad():
            _, boot = net(env.obs())
        ret, R = [], boot
        for t in reversed(range(a.T)):
            R = rew_b[t] + a.gamma * R * (1 - done_b[t])
            ret.append(R)
        ret = torch.stack(ret[::-1]).reshape(-1)
        o = torch.stack(obs_b).reshape(-1, OBS)
        act = torch.stack(act_b).reshape(-1)
        logit, v = net(o)
        adv = (ret - v).detach()
        logp = -torch.nn.functional.binary_cross_entropy_with_logits(logit, act, reduction="none")
        p = torch.sigmoid(logit)
        ent = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        loss = -(logp * adv).mean() + a.vcoef * ((ret - v) ** 2).mean() - a.ent * ent.mean()
        opt.zero_grad()
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step()
        if it % 100 == 0:
            recent = recent[-2000:]
            ms = sum(recent) / max(1, len(recent))
            print(f"it {it} eps {len(recent)} mean_score {ms:.2f} max {max(recent, default=0):.0f} "
                  f"loss {loss.item():.3f} t {time.time()-t0:.1f}s", flush=True)
            recent = []


if __name__ == "__main__":
    main()
