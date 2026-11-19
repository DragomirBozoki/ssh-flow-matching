import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DATA = sys.argv[1] if len(sys.argv) > 1 else f"/tmp/{os.environ['USER']}/data"
x = np.load(f"{DATA}/ssh_int16.npy", mmap_mode="r")
m = np.load(f"{DATA}/meta.npz")
scale, land, lat, lon, dates = float(m["scale"]), m["land"], m["lat"], m["lon"], m["dates"]
ocean = ~land
T, H, W = x.shape

s1, s2, daily = np.zeros((H, W)), np.zeros((H, W)), np.zeros(T)
for a in range(0, T, 100):
    z = x[a:a + 100].astype(np.float64) * scale
    z[:, land] = 0.0
    s1 += z.sum(0)
    s2 += (z ** 2).sum(0)
    daily[a:a + 100] = z[:, ocean].mean(1)
mean = s1 / T
std = np.sqrt(np.maximum(s2 / T - mean ** 2, 0.0))
mean[land], std[land] = np.nan, np.nan
np.savez(f"{DATA}/stats_all.npz", mean=mean, std=std, daily=daily)

k = np.nanargmax(std)
print(f"temporal std: ocean mean {np.nanmean(std) * 100:.1f} cm, "
      f"max {std.flat[k] * 100:.1f} cm at {lat[k // W]:.1f}N {-lon[k % W]:.1f}W")
years = (dates - dates[0]).astype(float) / 365.25
print(f"trend of the ocean-mean SSH: {np.polyfit(years, daily, 1)[0] * 1000:.1f} mm/year")
month = dates.astype("datetime64[M]").astype(int) % 12
clim = np.array([daily[month == i].mean() for i in range(12)])
print("seasonal cycle [cm], Jan..Dec:", np.round((clim - clim.mean()) * 100, 1))


def best_window(n, max_land=0.05):
    """n x n window with the highest mean temporal std and at most max_land land (integral images)."""
    S = np.pad(np.nan_to_num(std).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    O = np.pad(ocean.astype(float).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    box = lambda A: A[n:, n:] - A[:-n, n:] - A[n:, :-n] + A[:-n, :-n]
    s, o = box(S), box(O)
    land_frac = 1 - o / n ** 2
    score = np.where(land_frac <= max_land, s / np.maximum(o, 1), -1.0)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return i, j, score[i, j], land_frac[i, j]


wins = {}
for n in (128, 256):
    i, j, sc, lf = best_window(n)
    if sc < 0:
        print(f"{n}x{n}: no window with at most 5% land")
        continue
    wins[n] = (i, j)
    print(f"{n}x{n}: lat {lat[i]:.1f}-{lat[i + n - 1]:.1f}N, lon {-lon[j]:.1f}-{-lon[j + n - 1]:.1f}W, "
          f"mean std {sc * 100:.1f} cm, land {lf * 100:.1f}%, indices [{i}:{i + n}, {j}:{j + n}]")

ext = [lon[0], lon[-1], lat[0], lat[-1]]
fig = plt.figure(figsize=(14, 10))
ax1, ax2, ax3 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), fig.add_subplot(2, 1, 2)

cm1 = plt.get_cmap("RdBu_r").copy()
cm1.set_bad("0.6")
v = np.nanmax(np.abs(mean))
fig.colorbar(ax1.imshow(mean, origin="lower", extent=ext, cmap=cm1, vmin=-v, vmax=v), ax=ax1, label="m")
ax1.set_title("Mean SSH, 2007-2017")

cm2 = plt.get_cmap("Blues").copy()
cm2.set_bad("0.6")
fig.colorbar(ax2.imshow(std * 100, origin="lower", extent=ext, cmap=cm2), ax=ax2, label="cm")
ax2.set_title("SSH variability (standard deviation in time)")
for n, ls in ((128, "-"), (256, "--")):
    if n in wins:
        i, j = wins[n]
        ax2.add_patch(Rectangle((lon[j], lat[i]), n / 12, n / 12, fill=False, ec="k", lw=2, ls=ls))
        ax2.text(lon[j] + 0.4, lat[i + n - 1] - 0.4, f"{n}x{n}", va="top", fontsize=10,
                 bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
for ax in (ax1, ax2):
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")

ax3.plot(dates, daily * 100, lw=1.2, color="#2a6fb0")
ax3.set_title("SSH averaged over the ocean, daily")
ax3.set_ylabel("cm")
ax3.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.expanduser("~/prat/stats.png"), dpi=110)
print("saved: ~/prat/stats.png")
