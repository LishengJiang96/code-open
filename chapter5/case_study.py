# -*- coding: utf-8 -*-
"""Case study in Chapter 5: Table 4 (thresholds, comprehensive variation coefficients, weights) and Table 7 (closeness indices, comprehensive indices, ranks)."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../chapter3'))
from electre_t2b import (bootstrap_thresholds, weights_from_cvs, concordance,
                         discordance, wb_concordance, credibility,
                         comprehensive_index)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/data_sc.xlsx')
Q_I, Q_P, Q_V = 0.1, 0.9, 0.95
ETA1 = ETA2 = 1.0

CRITERIA = [
    'Per capita carbon dioxide emissions', 'Carbon dioxide emission intensity',
    'Energy consumption intensity', 'Industrial smoke and dust emissions intensity',
    'Industrial SO2 emission intensity', 'Industrial wastewater emission intensity',
    'Fertilizer application intensity', 'Green patents intensity',
    'Green coverage rate of built-up areas', 'Per capita park green space area',
    'Per capita GDP', 'PER GDP Rate', 'Financial development level',
    'Social consumption level', 'Urbanization Level',
]
CTYPE = ['cost'] * 7 + ['benefit'] * 8
CITIES = ['Chengdu', 'Zigong', 'Panzhihua', 'Luzhou', 'Deyang', 'Mianyang',
          'Guangyuan', 'Suining', 'Neijiang', 'Leshan', 'Nanchong', 'Meishan',
          'Yibin', "Guang'an", 'Dazhou', "Ya'an", 'Bazhong', 'Ziyang']


def main():
    df = pd.read_excel(DATA, sheet_name='2023')
    x = df[CRITERIA].to_numpy()

    from electre_t2b import SEED
    rng = np.random.default_rng(SEED)
    taus = np.zeros((15, 3))
    cvs = np.zeros((15, 3))
    for j in range(15):
        taus[j], cvs[j] = bootstrap_thresholds(
            x[:, j], Q_I, Q_P, Q_V, rng)
    lam, w = weights_from_cvs(cvs)

    print('Table 4: thresholds, comprehensive variation coefficients, weights')
    for j in range(15):
        print(f'{CRITERIA[j]:38s} '
              f'Q_I={taus[j, 0]:9.4f} Q_P={taus[j, 1]:9.4f} '
              f'Q_V={taus[j, 2]:9.4f} lam={lam[j]:7.4f} w={w[j]:7.4f}')

    d_con = concordance(x, taus[:, 0], taus[:, 1], CTYPE)
    D_con = wb_concordance(d_con, w, ETA1, ETA2)
    d_dis = discordance(x, taus[:, 1], taus[:, 2], CTYPE)
    c = credibility(D_con, d_dis)
    close, score, rank = comprehensive_index(c)

    print('\nTable 7: closeness indices, comprehensive indices, ranks')
    for i in range(18):
        print(f'{CITIES[i]:12s} rho+={c.mean(axis=1)[i]:6.3f} '
              f'rho-={c.mean(axis=0)[i]:6.3f} psi={close[i]:7.3f} '
              f'beta={score[i]:7.3f} rank={rank[i]:2d}')


if __name__ == '__main__':
    main()
