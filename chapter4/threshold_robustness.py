# -*- coding: utf-8 -*-
"""论文 4.1：阈值估计稳健性模拟。
单准则，4050 参数组合，比较三种阈值估计方法在缺失数据下的相对偏差。
"""

import itertools as its
import numpy as np

NOA = [20, 40, 60, 80, 100, 120]
QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
DEL_RATIO = [0.1, 0.2, 0.3]
DEL_PATTERN = ['random', 'largest', 'smallest']
RHO = [0.0, 0.2, 0.4, 0.6, 0.8]
MARGINALS = ['normal', 'lognormal', 'uniform']
B = 5000
REPS = 1000
SEED = 2026


def gen_evaluations(n, rho, marginal, rng):
    """Gaussian copula 生成 n 个方案的评价（单准则）。"""
    from scipy.stats import norm
    cov = np.fromfunction(lambda i, j: rho ** np.abs(i - j), (n, n))
    z = rng.multivariate_normal(np.zeros(n), cov)
    if marginal == 'normal':
        return z
    if marginal == 'lognormal':
        return np.exp(z)
    return norm.cdf(z)


def max_diff_threshold(x, q):
    return (x.max() - x.min()) * q


def quantile_threshold(x, q):
    d = np.abs(x[:, None] - x[None, :])
    d = d[np.triu_indices(len(x), 1)]
    d = d[d > 0]
    return np.quantile(d, q)


def bootstrap_threshold(x, q, rng, B=B):
    n = len(x)
    ii, jj = np.triu_indices(n, 1)
    idx = rng.integers(0, n, size=(B, n))
    samples = x[idx]
    d = np.abs(samples[:, jj] - samples[:, ii])
    ds = np.sort(d, axis=1)
    n0 = (d == 0).sum(axis=1)
    L = d.shape[1] - n0
    ok = L >= 2
    pos = n0 + (L - 1) * q
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    i1 = np.minimum(i0 + 1, d.shape[1] - 1)
    ar = np.arange(B)
    v0 = ds[ar, i0]
    v1 = ds[ar, i1]
    out = v0 * (1 - frac) + v1 * frac
    out[~ok] = np.nan
    qq = out[~np.isnan(out)]
    return qq.mean()


def delete_alternatives(x, pattern, ratio, rng):
    k = int(len(x) * ratio)
    if pattern == 'random':
        idx = rng.choice(len(x), size=k, replace=False)
    elif pattern == 'largest':
        idx = np.argsort(x)[-k:]
    else:
        idx = np.argsort(x)[:k]
    return np.delete(x, idx)


def main():
    rng = np.random.default_rng(SEED)

    results = {'maxdiff': [], 'quantile': [], 'bootstrap': []}
    combos = list(its.product(NOA, QUANTILES, QUANTILES, DEL_RATIO,
                              DEL_PATTERN, RHO, MARGINALS))
    print(f'组合数: {len(combos)}')

    for noa, q, pmax, ratio, pattern, rho, marginal in combos:
        devs = {'maxdiff': [], 'quantile': [], 'bootstrap': []}
        for _ in range(REPS):
            x = gen_evaluations(noa, rho, marginal, rng)
            t_m = max_diff_threshold(x, pmax)
            t_q = quantile_threshold(x, q)
            t_b = bootstrap_threshold(x, q, rng)
            xd = delete_alternatives(x, pattern, ratio, rng)
            devs['maxdiff'].append(abs(max_diff_threshold(xd, pmax) - t_m) / t_m)
            devs['quantile'].append(abs(quantile_threshold(xd, q) - t_q) / t_q)
            devs['bootstrap'].append(abs(bootstrap_threshold(xd, q, rng) - t_b) / t_b)
        for k in results:
            results[k].append(np.mean(devs[k]))

    print('Mean relative deviations:')
    for k in results:
        print(f'  {k:10s}: {np.mean(results[k]):.4f}')


if __name__ == '__main__':
    main()
