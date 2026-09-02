# -*- coding: utf-8 -*-
"""Example 1 in the paper: thresholds and weights for a single criterion (Section 3)."""

import numpy as np

from electre_t2b import bootstrap_thresholds, weights_from_cvs

B = 10000
Q_I, Q_P, Q_V = 0.1, 0.8, 0.9
E = np.array([-104, 90, -38, 119, 99])

rng = np.random.default_rng(2026)

taus, cvs = bootstrap_thresholds(E, Q_I, Q_P, Q_V, rng, B=B)
tau_I, tau_P, tau_V = taus
cv_I, cv_P, cv_V = cvs

lam, _ = weights_from_cvs(cvs.reshape(1, 3))

print(f'tau_I = {tau_I:.4f}, CV_I = {cv_I:.4f}')
print(f'tau_P = {tau_P:.4f}, CV_P = {cv_P:.4f}')
print(f'tau_V = {tau_V:.4f}, CV_V = {cv_V:.4f}')
print(f'lambda = {lam[0]:.4f}')
