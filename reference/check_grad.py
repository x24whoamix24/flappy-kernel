# SPDX-License-Identifier: Apache-2.0
"""Checks the co-tenant's hand-written A2C backward pass against autograd."""
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a2c_flappy import Net, OBS  # noqa: E402

ENT, VCOEF = 0.01, 0.5
WARP = 32  # the kernel pads the parameters to whole warps


def flat(net):
    return torch.cat([net.l1.weight.flatten(), net.l1.bias, net.l2.weight.flatten(),
                      net.l2.bias, net.head.weight.flatten(), net.head.bias])


def main(exe, workdir, n=1024):
    os.makedirs(workdir, exist_ok=True)
    torch.manual_seed(1)
    net = Net()
    obs = torch.randn(n, OBS) * 0.5
    act = (torch.rand(n) < 0.5).float()
    ret = torch.randn(n)
    logit, v = net(obs)
    adv = (ret - v).detach()
    logp = -torch.nn.functional.binary_cross_entropy_with_logits(logit, act, reduction="none")
    p = torch.sigmoid(logit)
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    loss = (-(logp * adv) + VCOEF * (ret - v) ** 2 - ENT * ent).sum()
    loss.backward()
    ref = torch.cat([net.l1.weight.grad.flatten(), net.l1.bias.grad, net.l2.weight.grad.flatten(),
                     net.l2.bias.grad, net.head.weight.grad.flatten(), net.head.bias.grad]).numpy()
    w = np.zeros((ref.size + WARP - 1) // WARP * WARP, np.float32)
    w[: ref.size] = flat(net).detach().numpy()
    w.tofile(os.path.join(workdir, "w.bin"))
    torch.cat([obs, act[:, None], ret[:, None]], 1).numpy().astype(np.float32).tofile(
        os.path.join(workdir, "samples.bin"))
    subprocess.run([exe, "grad", workdir], check=True)
    got = np.fromfile(os.path.join(workdir, "grad.bin"), np.float32)[: ref.size]
    err = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-12)
    print(f"max |grad| {np.abs(ref).max():.4f}  max rel err {err:.2e}")
    names = ["W1", "B1", "W2", "B2", "W3", "B3"]  # the order of flat()
    off = 0
    for name, size in zip(names, (p.numel() for p in net.parameters())):
        e = np.abs(got[off:off + size] - ref[off:off + size]).max()
        print(f"  {name}: max abs err {e:.2e} (ref scale {np.abs(ref[off:off + size]).max():.3e})")
        off += size
    sys.exit(0 if err < 1e-3 else 1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: check_grad.py <kernel/standalone> <work dir>")
    main(sys.argv[1], sys.argv[2])
