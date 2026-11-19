import os
import numpy as np
import torch

DATA = os.path.expanduser("~/prat/data")


def date_features(dates, K=2):
    """Day of year -> sin/cos with periods of one year and half a year (2K values per day)."""
    doy = (dates - dates.astype("datetime64[Y]")).astype(int)
    a = 2 * np.pi * doy[:, None] * np.arange(1, K + 1)[None, :] / 365.25
    return np.concatenate([np.sin(a), np.cos(a)], axis=1).astype(np.float32)


def dist_to(days, ref):
    """Distance in days from each day to the closest day of ref (ref sorted)."""
    k = np.searchsorted(ref, days)
    left = np.abs(days - ref[np.clip(k - 1, 0, len(ref) - 1)])
    right = np.abs(days - ref[np.clip(k, 0, len(ref) - 1)])
    return np.minimum(left, right)


def split(dates, mode="blocks", gap=5):
    """Boolean train and test masks.

    blocks:    in every year and every season, one half of one month goes to the test set
               (about 16% of the days, the same share in each season, a different half every year)
    last_year: the last 365 days go to the test set
    Training days must be more than `gap` days away from any test day, because consecutive
    days are strongly correlated (otherwise information leaks from training to test).
    """
    days = (dates - dates[0]).astype(int)
    if mode == "blocks":
        year = dates.astype("datetime64[Y]").astype(int)
        month = dates.astype("datetime64[M]").astype(int) % 12
        season = ((month + 1) % 12) // 3
        half = 2 * ((month + 1) % 3) + ((dates - dates.astype("datetime64[M]")).astype(int) >= 15)
        test = half == (year + season) % 6
    elif mode == "last_year":
        test = days > days[-1] - 365
    else:
        raise ValueError(mode)
    train = ~test & (dist_to(days, days[test]) > gap)
    return train, test


class SSHData:
    """All days of the region in memory, normalized with statistics of the training days."""

    def __init__(self, mode="blocks"):
        x = np.load(f"{DATA}/ssh_gs128_int16.npy")
        m = np.load(f"{DATA}/meta_gs128.npz")
        ssh = x.astype(np.float32) * float(m["scale"]) + float(m["offset"])
        self.dates, self.lat, self.lon = m["dates"], m["lat"], m["lon"]
        self.train, self.test = split(self.dates, mode)
        self.mean = ssh[self.train].mean(axis=0)
        self.std = float((ssh[self.train] - self.mean).std())
        self.x = torch.from_numpy((ssh - self.mean) / self.std).unsqueeze(1)
        self.cond = torch.from_numpy(date_features(self.dates))

    def to_meters(self, xn):
        """Normalized map(s) -> SSH in meters."""
        return np.asarray(xn) * self.std + self.mean


if __name__ == "__main__":
    for mode in ("blocks", "last_year"):
        d = SSHData(mode)
        month = d.dates.astype("datetime64[M]").astype(int) % 12
        season = np.array(["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"])[month]
        n_tr, n_te = int(d.train.sum()), int(d.test.sum())
        print(f"\n{mode}: train {n_tr} days, test {n_te} days, removed by the gap {len(d.dates) - n_tr - n_te}")
        print("  test share per season:", {s: f"{d.test[season == s].mean() * 100:.0f}%" for s in ("DJF", "MAM", "JJA", "SON")})
        print(f"  normalization std: {d.std * 100:.1f} cm, data shape {tuple(d.x.shape)}")
        print(f"  mean of normalized data: train {d.x[d.train].mean():+.3f}, test {d.x[d.test].mean():+.3f}")
