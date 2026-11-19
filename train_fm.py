import os, sys, time, copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import SSHData
from unet import UNet

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BATCH, LR, SIGMA_MIN, EMA = 32, 2e-4, 1e-4, 0.999
OUT = os.path.expanduser("~/prat/runs/fm_uncond")
os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda"))
torch.manual_seed(0)

data = SSHData("blocks")
xtrain = data.x[data.train].to(dev)
model = UNet().to(dev)
ema = copy.deepcopy(model).eval()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
log = open(f"{OUT}/log.txt", "a")


def say(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def cfm_loss(x1):
    """Conditional Flow Matching loss with the optimal transport path (Lipman et al., 2023)."""
    x0 = torch.randn_like(x1)
    t = torch.rand(len(x1), device=dev)
    tt = t[:, None, None, None]
    xt = (1 - (1 - SIGMA_MIN) * tt) * x0 + tt * x1
    target = x1 - (1 - SIGMA_MIN) * x0
    with torch.autocast(**amp):
        v = model(xt, t)
    return ((v.float() - target) ** 2).mean()


@torch.no_grad()
def sample(net, n, steps=50):
    """Euler integration of dx/dt = v(x, t) from t = 0 (noise) to t = 1 (data)."""
    x = torch.randn(n, 1, 128, 128, device=dev)
    for k in range(steps):
        t = torch.full((n,), k / steps, device=dev)
        with torch.autocast(**amp):
            v = net(x, t)
        x = x + v.float() / steps
    return x


def save_samples(step):
    gen = data.to_meters(sample(ema, 8).cpu().numpy()[:, 0])
    idx = np.random.default_rng(0).choice(np.where(data.test)[0], 8, replace=False)
    real = data.to_meters(data.x[idx].numpy()[:, 0])
    np.save(f"{OUT}/samples_{step:06d}.npy", gen)
    lo, hi = np.percentile(real, [1, 99])
    fig, axs = plt.subplots(2, 8, figsize=(20, 5.6), constrained_layout=True)
    for j in range(8):
        im = axs[0, j].imshow(gen[j], origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi)
        axs[1, j].imshow(real[j], origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi)
        axs[0, j].set_title("generated")
        axs[1, j].set_title(str(data.dates[idx[j]]))
        axs[0, j].axis("off")
        axs[1, j].axis("off")
    fig.colorbar(im, ax=axs, label="SSH [m]", shrink=0.8)
    fig.savefig(f"{OUT}/samples_{step:06d}.png", dpi=80)
    plt.close(fig)


say(f"train maps: {len(xtrain)}, parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, "
    f"device: {dev}, steps: {STEPS}")
t0 = time.time()
for step in range(1, STEPS + 1):
    for g in opt.param_groups:
        g["lr"] = LR * min(1.0, step / 500)
    x1 = xtrain[torch.randint(len(xtrain), (BATCH,), device=dev)]
    loss = cfm_loss(x1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.lerp_(p, 1 - EMA)
    if step % 200 == 0 or step == STEPS:
        say(f"step {step}/{STEPS}  loss {loss.item():.4f}  {time.time() - t0:.0f} s")
    if step % 5000 == 0 or step == STEPS:
        save_samples(step)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "step": step,
                    "mean": torch.from_numpy(data.mean), "std": data.std}, f"{OUT}/ckpt.pt")
        say(f"saved samples and checkpoint to {OUT}")
