# -*- coding: utf-8 -*-
"""Section 5.5 comparative analysis: Table 9 (SWM, TOPSIS, four PROMETHEE variants) and Spearman tests."""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../chapter3'))
from electre_t2b import (SEED, bootstrap_thresholds, weights_from_cvs,
                         concordance, discordance, wb_concordance,
                         credibility, comprehensive_index, rankings)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/data_sc.xlsx')
Q_I, Q_P, Q_V = 0.1, 0.9, 0.95
CTYPE = ['cost'] * 7 + ['benefit'] * 8
CITIES = ['Chengdu', 'Zigong', 'Panzhihua', 'Luzhou', 'Deyang', 'Mianyang',
          'Guangyuan', 'Suining', 'Neijiang', 'Leshan', 'Nanchong', 'Meishan',
          'Yibin', "Guang'an", 'Dazhou', "Ya'an", 'Bazhong', 'Ziyang']


def minmax(x):
    for j in range(x.shape[1]):
        lo, hi = x[:, j].min(), x[:, j].max()
        x[:, j] = (x[:, j] - lo) / (hi - lo) if CTYPE[j] == 'benefit' \
            else 1 - (x[:, j] - lo) / (hi - lo)
    return x


def topsis(x, w):
    x = x / np.sum(x ** 2, axis=0) ** 0.5
    xw = x * w
    pos = np.zeros(x.shape[1])
    neg = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        if CTYPE[j] == 'cost':
            pos[j], neg[j] = xw[:, j].min(), xw[:, j].max()
        else:
            pos[j], neg[j] = xw[:, j].max(), xw[:, j].min()
    d_pos = np.sum((xw - pos) ** 2, axis=1) ** 0.5
    d_neg = np.sum((xw - neg) ** 2, axis=1) ** 0.5
    return d_neg / (d_neg + d_pos)


def promethee(x, w, func):
    noa, noc = x.shape
    ti, tp = 0.1, 0.9
    pdm = np.zeros((noc, noa, noa))
    for j in range(noc):
        for i1 in range(noa):
            for i2 in range(noa):
                d = x[i1, j] - x[i2, j]
                if d <= 0:
                    pdm[j, i1, i2] = 0.0
                elif func == 'U':
                    pdm[j, i1, i2] = 1.0
                elif func == 'Q':
                    pdm[j, i1, i2] = 1.0 if d > ti else 0.0
                elif func == 'LP':
                    pdm[j, i1, i2] = 0.0 if d <= ti else (
                        1.0 if d >= tp else (d - ti) / (tp - ti))
                else:
                    pdm[j, i1, i2] = 0.0 if d <= ti else (
                        1.0 if d >= tp else 0.5)
    agg = np.zeros((noa, noa))
    for j in range(noc):
        agg += pdm[j] * w[j]
    return agg.sum(axis=1) / noa - agg.sum(axis=0) / noa


def main():
    df = pd.read_excel(DATA, sheet_name='2023')
    x = df.iloc[:, 2:].to_numpy()
    rng = np.random.default_rng(SEED)

    taus = np.zeros((15, 3))
    cvs = np.zeros((15, 3))
    for j in range(15):
        taus[j], cvs[j] = bootstrap_thresholds(x[:, j], Q_I, Q_P, Q_V, rng)
    _, w = weights_from_cvs(cvs)

    d_con = concordance(x, taus[:, 0], taus[:, 1], CTYPE)
    D_con = wb_concordance(d_con, w)
    d_dis = discordance(x, taus[:, 1], taus[:, 2], CTYPE)
    c = credibility(D_con, d_dis)
    _, score_t2b, rank_t2b = comprehensive_index(c)

    xs = minmax(x.copy())
    methods = ['SWM', 'TOPSIS', 'PROMETHEE_U', 'PROMETHEE_Q',
               'PROMETHEE_LP', 'PROMETHEE_L']
    scores = {'SWM': xs @ w, 'TOPSIS': topsis(x, w)}
    for f in ['U', 'Q', 'LP', 'L']:
        scores[f'PROMETHEE_{f}'] = promethee(xs, w, f)

    print('Table 9: comprehensive indices and ranks')
    for i in range(18):
        line = f'{CITIES[i]:12s} T2B={score_t2b[i]:6.3f}'
        for m in methods:
            line += f' {m}={scores[m][i]:8.4f}'
        print(line)

    print('\nSpearman correlation (with T2B ranks):')
    for m in methods:
        rho, p = spearmanr(rank_t2b, rankings(scores[m]))
        print(f'  {m:12s} rho={rho:.4f} p={p:.4g}')


if __name__ == '__main__':
    main()
