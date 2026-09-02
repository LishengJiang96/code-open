# -*- coding: utf-8 -*-
"""Section 6.2 discussion: indifference and preference threshold ratios (Q_I/(2*mean), Q_P/(2*mean)) for the quadrant analysis."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../chapter3'))
from electre_t2b import SEED, bootstrap_thresholds

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/data_sc.xlsx')
CRITERIA = [
    'Per capita carbon dioxide emissions', 'Carbon dioxide emission intensity',
    'Energy consumption intensity', 'Industrial smoke and dust emissions intensity',
    'Industrial SO2 emission intensity', 'Industrial wastewater emission intensity',
    'Fertilizer application intensity', 'Green patents intensity',
    'Green coverage rate of built-up areas', 'Per capita park green space area',
    'Per capita GDP', 'PER GDP Rate', 'Financial development level',
    'Social consumption level', 'Urbanization Level',
]


def main():
    df = pd.read_excel(DATA, sheet_name='2023')
    x = df[CRITERIA].to_numpy()
    rng = np.random.default_rng(SEED)

    print('Criterion           Q_I/(2*mean)   Q_P/(2*mean)')
    for j in range(15):
        taus, _ = bootstrap_thresholds(x[:, j], 0.1, 0.9, 0.95, rng)
        m = x[:, j].mean()
        print(f'{CRITERIA[j]:38s} {taus[0] / (2 * m):10.4f} '
              f'{taus[1] / (2 * m):10.4f}')


if __name__ == '__main__':
    main()
