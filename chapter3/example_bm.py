# -*- coding: utf-8 -*-
"""Example 2 in the paper: interaction parameters of the weighted Bonferroni mean (Section 3)."""

import numpy as np

from electre_t2b import wb_concordance

w = np.array([0.5, 0.5])

cases = [
    (2, 2, np.array([0.1, 0.4])),
    (2, 2, np.array([0.5, 0.4])),
    (8, 2, np.array([0.5, 0.4])),
]

for eta1, eta2, gamma in cases:
    d_con = np.zeros((2, 2, 2))
    d_con[0, 0, 1] = gamma[0]
    d_con[1, 0, 1] = gamma[1]
    d_con[:, 1, 0] = d_con[:, 0, 1]
    Gamma = wb_concordance(d_con, w, eta1, eta2)[0, 1]
    Gamma_w = gamma @ w
    print(f'eta=({eta1},{eta2}) gamma={gamma}  '
          f'Gamma_w={Gamma_w:.4f}  Gamma={Gamma:.4f}')
