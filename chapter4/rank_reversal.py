# -*- coding: utf-8 -*-
"""论文 4.2：秩反转模拟。
比较 ELECTRE-T2B 与 ELECTRE-III 在删除方案后的反转概率与反转比例。
972 参数组合，每组合 1000 次重复。
"""

import itertools as its
import numpy as np

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../chapter3'))
from electre_t2b import (bootstrap_thresholds, weights_from_cvs,
                         concordance, discordance, wb_concordance,
                         credibility, comprehensive_index, rankings)

NOA = NOC = [5, 10, 20]
Q_I = [0.1, 0.2, 0.3]
Q_P = [0.8, 0.9]
Q_V = [0.95, 0.99]
MODES = ['uniform', 'right-skewed', 'left-skewed']
ETA = [0.5, 1.0, 10.0]
B = 5000
REPS = 1000
SEED = 2026


def gen_beta(noa, noc, mode, rng):
    if mode == 'uniform':
        a = np.ones(noc)
        b = np.ones(noc)
    else:
        a = rng.uniform(0.1, 10, size=noc)
        b = rng.uniform(0.1, 10, size=noc)
        if mode == 'right-skewed':
            a, b = np.minimum(a, b), np.maximum(a, b)
        else:
            a, b = np.maximum(a, b), np.minimum(a, b)
    return rng.beta(a, b, size=(noa, noc))


def electre_iii(x, w, qI, qP, qV, ctype):
    """原版 ELECTRE III：阈值取非零差集分位数，蒸馏排序。"""
    noc = x.shape[1]
    t_ind = np.zeros(noc)
    t_pre = np.zeros(noc)
    t_vot = np.zeros(noc)
    for j in range(noc):
        d = np.abs(x[:, j][:, None] - x[:, j][None, :])
        d = d[np.triu_indices(x.shape[0], 1)]
        d = d[d > 0]
        t_ind[j], t_pre[j], t_vot[j] = np.quantile(d, [qI, qP, qV])

    noa = x.shape[0]
    c_con = np.zeros((noa, noa))
    c_dis = np.zeros((noa, noa))
    for i1 in range(noa):
        for i2 in range(noa):
            s = 0.0
            for j in range(noc):
                cha = x[i1, j] - x[i2, j]
                if cha <= t_ind[j]:
                    g = 0.0
                elif cha >= t_pre[j]:
                    g = 1.0
                else:
                    g = (cha - t_ind[j]) / (t_pre[j] - t_ind[j])
                s += w[j] * g
            c_con[i1, i2] = s
    for i1 in range(noa):
        for i2 in range(noa):
            tem = 1.0
            for j in range(noc):
                cha = x[i2, j] - x[i1, j]
                if cha <= t_pre[j]:
                    al = 0.0
                elif cha >= t_vot[j]:
                    al = 1.0
                else:
                    al = (cha - t_pre[j]) / (t_vot[j] - t_pre[j])
                tem *= min(1.0, (1 - al) / (1 - c_con[i1, i2] + 1e-10))
            c_dis[i1, i2] = tem
    cred = c_con * c_dis
    return distill_rank(cred)


def distill_rank(cred):
    noa = cred.shape[0]
    des = descending(cred.copy(), noa)
    asc = ascending(cred.copy(), noa)
    rank = np.zeros((noa, noa))
    for i in range(noa):
        for j in range(noa):
            if i == j:
                continue
            i1 = des[i]
            i2 = asc[i]
            j1 = des[j]
            j2 = asc[j]
            if i1 == j1 and i2 == j2:
                rank[i, j] = 0
            elif (i1 != j1 or i2 != j2) and i1 >= j1 and i2 >= j2:
                rank[i, j] = 1
            elif (i1 != j1 or i2 != j2) and i1 <= j1 and i2 <= j2:
                rank[i, j] = -1
            else:
                rank[i, j] = 0.5
    return rank


def descending(d, noa):
    level = {}
    rd = 1
    theta = np.max(d)
    while True:
        t = 1.1 * theta - 0.2
        if t < 0:
            level[rd] = list(np.where(d >= 0)[0])
            break
        a = np.unique(d)
        theta = max(a[a < t])
        score = np.zeros((noa, 1))
        for i in range(noa):
            for j in range(noa):
                if d[i, j] > theta:
                    score[i, 0] += 1
                    score[j, 0] -= 1
        level[rd] = list(np.where(score == np.max(score))[0])
        for i in level[rd]:
            d[i, :] = -10
            d[:, i] = -10
        if np.max(d) < 0:
            break
        rd += 1
    return np.array([k for k, v in level.items()
                     for _ in v], dtype=float)


def ascending(d, noa):
    level = {}
    rd = 1
    theta = np.max(d)
    while True:
        t = 1.1 * theta - 0.2
        if t < 0:
            level[rd] = list(np.where(d >= 0)[0])
            break
        a = np.unique(d)
        theta = max(a[a < t])
        score = np.zeros((noa, 1))
        for i in range(noa):
            for j in range(noa):
                if d[i, j] > theta:
                    score[i, 0] += 1
                    score[j, 0] -= 1
        level[rd] = list(np.where(score == np.min(score))[0])
        for i in level[rd]:
            d[i, :] = -10
            d[:, i] = -10
        if np.max(d) < 0:
            break
        rd += 1
    return np.array([k for k, v in level.items()
                     for _ in v], dtype=float)


def reversal_ratio_t2b(r1, r2):
    r1, r2 = np.asarray(r1), np.asarray(r2)
    comp1 = np.sign(r1[:, None] - r1)
    comp2 = np.sign(r2[:, None] - r2)
    mask = (comp1 != comp2) & (comp1 != 0) & (comp2 != 0)
    return np.sum(np.triu(mask, 1))


def reversal_ratio_iii(m1, m2):
    mask = (m1 != m2) & (m1 != 0) & (m2 != 0)
    return np.sum(np.triu(mask, 1))


def main():
    rng = np.random.default_rng(SEED)
    combos = list(its.product(NOA, NOC, Q_I, Q_P, Q_V, MODES, ETA))
    print(f'组合数: {len(combos)}')

    t2b_prob = []
    t2b_ratio = []
    iii_prob = []
    iii_ratio = []

    for noa, noc, qI, qP, qV, mode, eta in combos:
        ctype = ['benefit'] * noc
        p2 = r2 = p3 = r3 = 0.0
        for _ in range(REPS):
            x = gen_beta(noa, noc, mode, rng)
            # T2B 初始
            taus = np.zeros((noc, 3))
            cvs = np.zeros((noc, 3))
            for j in range(noc):
                taus[j], cvs[j] = bootstrap_thresholds(
                    x[:, j], qI, qP, qV, rng, B=B)
            _, w = weights_from_cvs(cvs)
            d_con = concordance(x, taus[:, 0], taus[:, 1], ctype)
            D_con = wb_concordance(d_con, w, eta, eta)
            d_dis = discordance(x, taus[:, 1], taus[:, 2], ctype)
            c = credibility(D_con, d_dis)
            _, s1, rank1 = comprehensive_index(c)
            # III 初始
            m1 = electre_iii(x, w, qI, qP, qV, ctype)
            # 删除一个方案
            k = rng.integers(noa)
            keep = np.ones(noa, bool)
            keep[k] = False
            xd = x[keep]
            # T2B 重算
            taus2 = np.zeros((noc, 3))
            cvs2 = np.zeros((noc, 3))
            for j in range(noc):
                taus2[j], cvs2[j] = bootstrap_thresholds(
                    xd[:, j], qI, qP, qV, rng, B=B)
            _, w2 = weights_from_cvs(cvs2)
            d_con2 = concordance(xd, taus2[:, 0], taus2[:, 1], ctype)
            D_con2 = wb_concordance(d_con2, w2, eta, eta)
            d_dis2 = discordance(xd, taus2[:, 1], taus2[:, 2], ctype)
            c2 = credibility(D_con2, d_dis2)
            _, s2, rank2 = comprehensive_index(c2)
            # III 重算
            m2 = electre_iii(xd, w2, qI, qP, qV, ctype)

            n_pairs = noa * (noa - 1)
            rt = reversal_ratio_t2b(rank1[keep], rank2) / n_pairs
            rm = reversal_ratio_iii(m1[np.ix_(keep, keep)], m2) / n_pairs
            p2 += (rt > 0)
            r2 += rt
            p3 += (rm > 0)
            r3 += rm
        t2b_prob.append(p2 / REPS)
        t2b_ratio.append(r2 / REPS)
        iii_prob.append(p3 / REPS)
        iii_ratio.append(r3 / REPS)

    print('Reversal probability (mean over combinations):')
    print(f'  T2B: {np.mean(t2b_prob):.4f}   III: {np.mean(iii_prob):.4f}')
    print('Reversal ratio (mean over combinations):')
    print(f'  T2B: {np.mean(t2b_ratio):.4f}   III: {np.mean(iii_ratio):.4f}')


if __name__ == '__main__':
    main()
