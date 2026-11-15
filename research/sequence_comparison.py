"""Group-weighted diagnostics for the fixed B sequence contract, not C features."""
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


METRIC_GROUPS = {
    "projection": ("screen_span", "screen_path", "screen_speed_median"),
    "shape": ("straightness", "transverse_ratio", "turn_radians_mean", "reversal_fraction"),
    "timing": ("step_cv", "low_motion_fraction", "acf_lag3", "band_3_10_ratio"),
}
METRICS = tuple(name for names in METRIC_GROUPS.values() for name in names)


def sequence_metrics(x, scale, fps=30.):
    p = np.asarray(x[:2].T, dtype=float)
    delta = np.diff(p, axis=0)
    step = np.linalg.norm(delta, axis=1)
    path = float(step.sum())
    raw_step = step * scale
    centered = p - p.mean(axis=0)
    eigen = np.linalg.eigvalsh(centered.T @ centered / len(p))
    valid = (raw_step[:-1] * fps > .003) & (raw_step[1:] * fps > .003)
    cosine = np.sum(delta[:-1]*delta[1:], axis=1) / np.maximum(step[:-1]*step[1:], 1e-12)
    angle = np.arccos(np.clip(cosine[valid], -1, 1))
    fluctuation = detrend(delta, axis=0)
    denominator = float(np.sum(fluctuation**2))
    acf = float(np.sum(fluctuation[:-3]*fluctuation[3:])/denominator) if denominator > 1e-12 else 0.
    power = np.sum(np.abs(np.fft.rfft(fluctuation*np.hanning(len(delta))[:, None], axis=0))**2, axis=1)
    frequencies = np.fft.rfftfreq(len(delta), 1/fps)
    band = float(power[(frequencies >= 3) & (frequencies <= 10)].sum()/max(power[1:].sum(), 1e-12))
    return dict(screen_span=float(np.linalg.norm(np.ptp(p, axis=0))*scale),
                screen_path=path*scale, screen_speed_median=float(np.median(raw_step)*fps),
                straightness=float(np.linalg.norm(p[-1]-p[0])/max(path, 1e-12)),
                transverse_ratio=float(np.sqrt(max(eigen[0], 0)/max(eigen.sum(), 1e-12))),
                turn_radians_mean=float(angle.mean()) if len(angle) else 0.,
                reversal_fraction=float(np.mean(cosine[valid] < -.5)) if valid.any() else 0.,
                step_cv=float(step.std()/max(step.mean(), 1e-12)),
                low_motion_fraction=float(np.mean(raw_step*fps <= .003)),
                acf_lag3=acf, band_3_10_ratio=band,
                direction_support=float(valid.mean()))


def group_weights(table):
    weights = 1/table.groupby("group_id").group_id.transform("size").to_numpy(dtype=float)
    return weights/weights.sum()


def quantile(values, weights, levels):
    order = np.argsort(values)
    weights = np.asarray(weights)[order]
    # Weighted midpoint CDF; unlike W, interpolated quantiles can vary at ties.
    cdf = (np.cumsum(weights)-weights/2)/weights.sum()
    return np.interp(levels, cdf, np.asarray(values)[order])


def metric_scales(real_train):
    result = {}
    for label in ("bird", "drone"):
        part = real_train[real_train.label == label]
        if part.empty:
            raise ValueError("Training data requires both classes")
        weights = group_weights(part)
        result[label] = {}
        for name in METRICS:
            q05, q25, q75, q95 = quantile(part[name].to_numpy(), weights, [.05, .25, .75, .95])
            result[label][name] = max(float(q75-q25), .1*float(q95-q05), .01 if name in ("low_motion_fraction", "reversal_fraction") else .001)
    return result


def compare(real, sim, scales):
    rows = []
    for label in ("bird", "drone"):
        a, b = real[real.label == label], sim[sim.label == label]
        if a.empty or b.empty:
            raise ValueError("Cannot compare an empty class")
        aw, bw = group_weights(a), group_weights(b)
        for category, names in METRIC_GROUPS.items():
            for name in names:
                av, bv = a[name].to_numpy(), b[name].to_numpy()
                lo, hi = quantile(bv, bw, [.05, .95])
                distance = float(wasserstein_distance(av, bv, aw, bw))
                rows.append(dict(label=label, category=category, metric=name,
                                 distance=distance, train_real_scale=scales[label][name],
                                 normalized_distance=distance/scales[label][name],
                                 coverage=float(np.sum(aw*((av >= lo) & (av <= hi)))),
                                 real_median=float(quantile(av, aw, [.5])[0]),
                                 sim_median=float(quantile(bv, bw, [.5])[0])))
    return pd.DataFrame(rows)


def score(table):
    # Equal weight per class and category, rather than per number of metrics.
    per_class = table.groupby(["label", "category"]).normalized_distance.mean().groupby("label").mean()
    return float(per_class.mean())


def sequence_mmd(real_x, real_meta, sim_x, sim_meta, train_x, train_meta):
    """Biased RBF MMD^2, full four-channel windows with group weights; no p-value."""
    results = {}
    for label in ("bird", "drone"):
        train = train_x[train_meta.label.to_numpy() == label].astype(float)
        channel_scale = np.maximum(np.std(train, axis=(0, 2)), .01)
        flatten = lambda x: (x/channel_scale[None, :, None]).reshape(len(x), -1)
        train_flat = flatten(train)
        ds = cdist(train_flat, train_flat, "sqeuclidean")
        nonzero = ds[ds > 1e-10]
        bandwidth2 = float(np.median(nonzero)) if len(nonzero) else 1.
        ri, si = real_meta.label.to_numpy() == label, sim_meta.label.to_numpy() == label
        a, b = flatten(real_x[ri]), flatten(sim_x[si])
        aw, bw = group_weights(real_meta[ri]), group_weights(sim_meta[si])
        kernel = lambda u, v: np.exp(-cdist(u, v, "sqeuclidean")/(2*bandwidth2))
        value = aw@kernel(a, a)@aw + bw@kernel(b, b)@bw - 2*aw@kernel(a, b)@bw
        results[label] = dict(mmd2=float(max(value, 0)), bandwidth_squared=bandwidth2,
                              status="descriptive_biased_group_weighted_not_hypothesis_test")
    return results


def bootstrap_delta(real, before, after, scales, seed=73, draws=200):
    rng = np.random.default_rng(seed)
    differences = []
    def sample(table):
        groups = sorted(table.group_id.unique())
        pieces = []
        for i, group in enumerate(rng.choice(groups, len(groups), replace=True)):
            piece = table[table.group_id == group].copy()
            piece["group_id"] = f"bootstrap:{i}"
            pieces.append(piece)
        return pd.concat(pieces, ignore_index=True)
    for _ in range(draws):
        # Class-stratified group bootstrap keeps all correlated windows together.
        r = pd.concat([sample(real[real.label == c]) for c in ("bird", "drone")])
        b = pd.concat([sample(before[before.label == c]) for c in ("bird", "drone")])
        a = pd.concat([sample(after[after.label == c]) for c in ("bird", "drone")])
        differences.append(score(compare(r, a, scales))-score(compare(r, b, scales)))
    return dict(after_minus_before_q025_q50_q975=np.quantile(differences, [.025, .5, .975]).tolist(),
                draws=draws, unit="original real groups and synthetic flight groups",
                interpretation="negative favors after; fixed fitted settings; does not include calibration uncertainty")
