# -*- coding: utf-8 -*-
"""Section 5.4 sensitivity analysis: Table 8 (standard deviations of CI and ranks under 9 interaction parameter sets and 8 quantile sets)."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../chapter3'))
from electre_t2b import (SEED, bootstrap_thresholds, weights_from_cvs,
                         concordance, discordance, wb_concordance,
                         credibility, comprehensive_index)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/data_sc.xlsx')
ETA_PAIRS = [(0.5, 0.5), (0.5, 1), (0.5, 10), (1, 0.5), (1, 1),
             (1, 10), (10, 0.5), (10, 1), (10, 10)]
QUAN_SETS = [(0.1, 0.9, 0.95), (0.1, 0.6, 0.95), (0.1, 0.6, 0.99),
             (0.1, 0.9, 0.99), (0.5, 0.6, 0.95), (0.5, 0.6, 0.99),
             (0.5, 0.9, 0.95), (0.5, 0.9, 0.99)]
CTYPE = ['cost'] * 7 + ['benefit'] * 8
CITIES = ['Chengdu', 'Zigong', 'Panzhihua', 'Luzhou', 'Deyang', 'Mianyang',
          'Guangyuan', 'Suining', 'Neijiang', 'Leshan', 'Nanchong', 'Meishan',
          'Yibin', "Guang'an", 'Dazhou', "Ya'an", 'Bazhong', 'Ziyang']


def run(x, taus, w, eta1, eta2):
    d_con = concordance(x, taus[:, 0], taus[:, 1], CTYPE)
    D_con = wb_concordance(d_con, w, eta1, eta2)
    d_dis = discordance(x, taus[:, 1], taus[:, 2], CTYPE)
    c = credibility(D_con, d_dis)
    _, score, rank = comprehensive_index(c)
    return score, rank


def main():
    df = pd.read_excel(DATA, sheet_name='2023')
    x = df.iloc[:, 2:].to_numpy()
    rng = np.random.default_rng(SEED)

    quan_ci = np.zeros((18, len(QUAN_SETS)))
    quan_rk = np.zeros((18, len(QUAN_SETS)))
    for gi, (qI, qP, qV) in enumerate(QUAN_SETS):
        taus = np.zeros((15, 3))
        cvs = np.zeros((15, 3))
        for j in range(15):
            taus[j], cvs[j] = bootstrap_thresholds(x[:, j], qI, qP, qV, rng)
        _, w = weights_from_cvs(cvs)
        score, rank = run(x, taus, w, 1.0, 1.0)
        quan_ci[:, gi] = score
        quan_rk[:, gi] = rank
        if gi == 0:
            taus_main, w_main = taus, w

    eta_ci = np.zeros((18, len(ETA_PAIRS)))
    eta_rk = np.zeros((18, len(ETA_PAIRS)))
    for gi, (e1, e2) in enumerate(ETA_PAIRS):
        score, rank = run(x, taus_main, w_main, e1, e2)
        eta_ci[:, gi] = score
        eta_rk[:, gi] = rank

    print('Table 8: standard deviations of CI and ranks')
    for i in range(18):
        print(f'{CITIES[i]:12s} eta_CI={eta_ci.std(axis=1)[i]:6.3f} '
              f'eta_Rank={eta_rk.std(axis=1)[i]:6.3f} '
              f'q_CI={quan_ci.std(axis=1)[i]:6.3f} '
              f'q_Rank={quan_rk.std(axis=1)[i]:6.3f}')


if __name__ == '__main__':
    main()
