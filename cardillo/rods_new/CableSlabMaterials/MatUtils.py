# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

from numpy import dot, array
from math import sqrt


def vonMisesStress(s):

    smises = 0.0

    if len(s) == 3:
        return sqrt(s[0] * s[0] + s[1] * s[1] - s[0] * s[1] + 3.0 * s[2] * s[2])
    elif len(s) == 6:
        smises = (
            (s[0] - s[1]) * (s[0] - s[1])
            + (s[1] - s[2]) * (s[1] - s[2])
            + (s[2] - s[0]) * (s[2] - s[0])
        )

        smises += 6.0 * dot(s[3:], s[3:])
        return sqrt(0.5 * smises)


def hydrostaticStress(s):

    return 0.333333333333333 * sum(s[:3])


def transform2To3(s):
    return array([s[0], s[1], 0.0, 0.0, 0.0, s[2]])


def transform3To2(s, t):
    return array([s[0], s[1], s[5]]), array(
        [
            (t[0, 0], t[0, 1], t[0, 5]),
            (t[1, 0], t[1, 1], t[1, 5]),
            (t[5, 0], t[5, 1], t[5, 5]),
        ]
    )
