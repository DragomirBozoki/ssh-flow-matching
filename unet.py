import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t, dim):
    """Sinusoidal embedding of the flow time t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = 1000 * t[:, None] * freqs[None, :]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection; the time embedding is added in between."""

    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb_dim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.e(emb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Single-head self-attention over all positions of a feature map."""

    def __init__(self, c):
        super().__init__()
        self.n, self.qkv, self.out = nn.GroupNorm(8, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).transpose(2, 3).unbind(1)
        h = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        return x + self.out(h.transpose(1, 2).reshape(B, C, H, W))


class UNet(nn.Module):
    """U-Net predicting the Flow Matching velocity v(x_t, t), optionally conditioned on a vector."""

    def __init__(self, ch=(64, 128, 256, 256), emb_dim=256, cond_dim=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.c_mlp = (nn.Sequential(nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
                      if cond_dim else None)
        self.inp = nn.Conv2d(1, ch[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        c = ch[0]
        for i, co in enumerate(ch):
            self.down.append(nn.ModuleList([ResBlock(c, co, emb_dim), ResBlock(co, co, emb_dim)]))
            self.pool.append(nn.Conv2d(co, co, 3, stride=2, padding=1) if i < len(ch) - 1 else nn.Identity())
            c = co
        self.mid1, self.attn, self.mid2 = ResBlock(c, c, emb_dim), Attention(c), ResBlock(c, c, emb_dim)
        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(ch))):
            self.up.append(nn.ModuleList([ResBlock(c + ch[i], ch[i], emb_dim), ResBlock(ch[i], ch[i], emb_dim)]))
            self.upsample.append(nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity())
            c = ch[i]
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t, cond=None):
        emb = self.t_mlp(time_embedding(t, self.emb_dim))
        if self.c_mlp is not None and cond is not None:
            emb = emb + self.c_mlp(cond)
        h, skips = self.inp(x), []
        for (b1, b2), pool in zip(self.down, self.pool):
            h = b2(b1(h, emb), emb)
            skips.append(h)
            h = pool(h)
        h = self.mid2(self.attn(self.mid1(h, emb)), emb)
        for (b1, b2), ups in zip(self.up, self.upsample):
            h = torch.cat([h, skips.pop()], dim=1)
            h = ups(b2(b1(h, emb), emb))
        return self.out(h)


if __name__ == "__main__":
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNet().to(dev)
    print(f"parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M, device: {dev}")
    x, t = torch.randn(32, 1, 128, 128, device=dev), torch.rand(32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    for i in range(12):
        if i == 2:
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            loss = net(x, t).float().pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / 10 * 1000
    print(f"output: {tuple(net(x[:2], t[:2]).shape)}, one training step (batch 32): {ms:.0f} ms, "
          f"20000 steps ~ {ms * 20000 / 60000:.0f} min")
