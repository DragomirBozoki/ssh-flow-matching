import glob, gzip, os, sys, time
import numpy as np
import netCDF4

D = "/users/Enseignants/bereziat/NATL60"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"/tmp/{os.environ['USER']}/data"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(f"{D}/NATL_AN_*.nc.gz"))
dates = np.array([os.path.basename(f)[8:18] for f in files], dtype="datetime64[D]")


def read_zos(path):
    """Decompress one daily file in memory and return the raw int16 SSH field (lat, lon)."""
    with gzip.open(path) as f:
        data = f.read()
    nc = netCDF4.Dataset("mem", memory=data)
    nc.set_auto_maskandscale(False)
    v = nc["zos"]
    z = v[0, :, :]
    attrs = v.ncattrs()
    fill = v.getncattr("_FillValue" if "_FillValue" in attrs else "missing_value")
    info = dict(scale=v.scale_factor, offset=v.add_offset, fill=fill)
    lat, lon = nc["latitude"][:], nc["longitude"][:]
    nc.close()
    return z, info, lat, lon


z0, info, lat, lon = read_zos(files[0])
T, (H, W) = len(files), z0.shape
print(f"{T} days, grid {H} x {W}, encoding: {info}")

ssh = np.lib.format.open_memmap(f"{OUT}/ssh_int16.npy", mode="w+", dtype=np.int16, shape=(T, H, W))
land = z0 == info["fill"]
changed = 0
t0 = time.time()
for i, f in enumerate(files):
    z = read_zos(f)[0]
    ssh[i] = z
    changed += int(((z == info["fill"]) != land).any())
    if (i + 1) % 250 == 0 or i + 1 == T:
        el = time.time() - t0
        print(f"{i + 1}/{T}  {el:.0f} s, about {el / (i + 1) * (T - i - 1):.0f} s left", flush=True)
ssh.flush()

np.savez(f"{OUT}/meta.npz", dates=dates, lat=lat, lon=lon, land=land, **info)
print("days whose land mask differs from the first day:", changed)
print("done:", OUT, f"({ssh.nbytes / 1e9:.2f} GB)")
