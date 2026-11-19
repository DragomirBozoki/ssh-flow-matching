import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else f"/tmp/{os.environ['USER']}/data"
OUT = os.path.expanduser("~/prat/data")
os.makedirs(OUT, exist_ok=True)
I0, J0, N = 92, 116, 128    # 32.7-43.3N, 65.3-54.8W: the 33-43N, 65-55W Gulf Stream box, extended to 128

x = np.load(f"{DATA}/ssh_int16.npy", mmap_mode="r")
m = np.load(f"{DATA}/meta.npz")
crop = np.ascontiguousarray(x[:, I0:I0 + N, J0:J0 + N])
land = m["land"][I0:I0 + N, J0:J0 + N]
lat, lon, dates = m["lat"][I0:I0 + N], m["lon"][J0:J0 + N], m["dates"]
np.save(f"{OUT}/ssh_gs128_int16.npy", crop)
np.savez(f"{OUT}/meta_gs128.npz", dates=dates, lat=lat, lon=lon, land=land,
         scale=m["scale"], offset=m["offset"], fill=m["fill"], i0=I0, j0=J0)

z = crop.astype(np.float32) * float(m["scale"])
z[:, land] = np.nan
print(f"region: lat {lat[0]:.2f}-{lat[-1]:.2f}N, lon {-lon[0]:.2f}-{-lon[-1]:.2f}W, land {land.mean() * 100:.1f}%")
print(f"SSH [m]: min {np.nanmin(z):.3f}, max {np.nanmax(z):.3f}, mean {np.nanmean(z):.3f}, std {np.nanstd(z):.3f}")
print(f"mean temporal std per pixel: {z[:, ~land].std(axis=0).mean() * 100:.1f} cm")
print("saved:", OUT, f"({crop.nbytes / 1e6:.0f} MB)")

t0 = int(np.searchsorted(dates, np.datetime64("2010-06-01")))
idx = [t0, t0 + 5, t0 + 10, t0 + 15]
v = np.nanmax(np.abs(z[idx] - np.nanmean(z[idx])))
c = np.nanmean(z[idx])
ext = [lon[0], lon[-1], lat[0], lat[-1]]
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("0.6")
fig, axs = plt.subplots(1, 4, figsize=(16, 4.4), constrained_layout=True)
for ax, t in zip(axs, idx):
    im = ax.imshow(z[t], origin="lower", extent=ext, cmap=cmap, vmin=c - v, vmax=c + v)
    ax.set_title(str(dates[t]))
    ax.set_xlabel("longitude [deg]")
axs[0].set_ylabel("latitude [deg]")
fig.colorbar(im, ax=axs, label="SSH [m]", shrink=0.9)
fig.savefig(os.path.expanduser("~/prat/region.png"), dpi=110)
print("saved: ~/prat/region.png")
