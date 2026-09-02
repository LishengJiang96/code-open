# -*- coding: utf-8 -*-
"""Core computations of ELECTRE-T2B (Section 3 of the paper)."""

import numpy as np

SEED = 2026
B_BOOT = 5000


def rankings(x):
    x = np.asarray(x)
    order = -np.sort(-x)
    r = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        r[i] = np.min(np.where(order == x[i])) + 1
    return r


def bootstrap_thresholds(col, qI, qP, qV, rng, B=B_BOOT):
    """Bootstrap threshold estimation on non-zero differences, returning (tau_I, tau_P, tau_V, cv)."""
    n = col.size
    ii, jj = np.triu_indices(n, 1)
    idx = rng.integers(0, n, size=(B, n))
    samples = col[idx]
    d = np.abs(samples[:, jj] - samples[:, ii])
    ds = np.sort(d, axis=1)
    n0 = (d == 0).sum(axis=1)
    L = d.shape[1] - n0
    ok = L >= 2
    qs = np.array([qI, qP, qV])
    pos = n0[:, None] + (L[:, None] - 1) * qs[None, :]
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    i1 = np.minimum(i0 + 1, d.shape[1] - 1)
    v0 = np.take_along_axis(ds, i0, axis=1)
    v1 = np.take_along_axis(ds, i1, axis=1)
    out = v0 * (1 - frac) + v1 * frac
    out[~ok] = np.nan
    qq = out[~np.isnan(out[:, 0])]
    taus = qq.mean(axis=0)
    cvs = qq.std(axis=0, ddof=0) / taus
    return taus, cvs


def weights_from_cvs(cvs_all):
    lam = 1.0 / cvs_all.sum(axis=1)
    return lam, lam / lam.sum()


def concordance(x, t_ind, t_pre, ctype):
    noa, noc = x.shape
    d_con = np.zeros((noc, noa, noa))
    for j in range(noc):
        alpha = 1 if ctype[j] == 'cost' else 0
        for i1 in range(noa):
            for i2 in range(noa):
                cha = (-1) ** alpha * (x[i1, j] - x[i2, j])
                if cha <= t_ind[j]:
                    d_con[j, i1, i2] = 0.0
                elif cha >= t_pre[j]:
                    d_con[j, i1, i2] = 1.0
                else:
                    d_con[j, i1, i2] = (cha - t_ind[j]) / (t_pre[j] - t_ind[j])
    return d_con


def discordance(x, t_pre, t_vot, ctype):
    noa, noc = x.shape
    d_dis = np.zeros((noc, noa, noa))
    for j in range(noc):
        alpha = 1 if ctype[j] == 'cost' else 0
        for i1 in range(noa):
            for i2 in range(noa):
                cha = (-1) ** alpha * (x[i2, j] - x[i1, j])
                if cha <= t_pre[j]:
                    d_dis[j, i1, i2] = 0.0
                elif cha >= t_vot[j]:
                    d_dis[j, i1, i2] = 1.0
                else:
                    d_dis[j, i1, i2] = (cha - t_pre[j]) / (t_vot[j] - t_pre[j])
    return d_dis


def wb_concordance(d_con, w, eta1=1.0, eta2=1.0):
    noa = d_con.shape[1]
    D = np.zeros((noa, noa))
    for i1 in range(noa):
        for i2 in range(noa):
            s = 0.0
            for j1 in range(len(w)):
                for j2 in range(len(w)):
                    if j1 != j2:
                        s += (w[j1] * w[j2]) / (1 - w[j1]) * \
                             d_con[j1, i1, i2] ** eta1 * \
                             d_con[j2, i1, i2] ** eta2
            D[i1, i2] = s ** (1.0 / (eta1 + eta2))
    return D


def credibility(D_con, d_dis):
    noa = D_con.shape[0]
    D_dis = np.ones((noa, noa))
    for i1 in range(noa):
        for i2 in range(noa):
            for j in range(d_dis.shape[0]):
                ratio = D_con[i1, i2] / (d_dis[j, i1, i2] + 1e-13)
                D_dis[i1, i2] *= min(ratio, 1.0)
    return D_con * D_dis


def comprehensive_index(c):
    c_plus = c.mean(axis=1)
    c_minus = c.mean(axis=0)
    close = c_plus / (c_minus + c_plus)
    rank_close = rankings(close)
    r = c.shape[0] - rank_close + 1
    score = (0.5 * close ** 2 + 0.5 * (r / c.shape[0]) ** 2) ** 0.5
    return close, score, rankings(score)


def electre_t2b(x, ctype, qI, qP, qV, eta1=1.0, eta2=1.0, rng=None):
    """Full pipeline: thresholds, weights, concordance, credibility, comprehensive index."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    taus = np.zeros((x.shape[1], 3))
    cvs = np.zeros((x.shape[1], 3))
    for j in range(x.shape[1]):
        taus[j], cvs[j] = bootstrap_thresholds(x[:, j], qI, qP, qV, rng)
    _, w = weights_from_cvs(cvs)
    d_con = concordance(x, taus[:, 0], taus[:, 1], ctype)
    D_con = wb_concordance(d_con, w, eta1, eta2)
    d_dis = discordance(x, taus[:, 1], taus[:, 2], ctype)
    c = credibility(D_con, d_dis)
    return comprehensive_index(c)
