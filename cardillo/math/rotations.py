import numpy as np
from cardillo.math import ax2skew, ax2skew_a, ax2skew_squared, skew2ax_A

# All functions accept either a single argument (e.g. psi of shape (3,)) or a
# stack of arguments (e.g. psi of shape (n, 3)) and return the result with
# the corresponding shape (e.g. (3, 3) or (n, 3, 3)).
#
# Most of the SO(3) and SE(3) equations contain coefficients that are of the
# form 0 / 0 for angle -> 0. They are evaluated using np.sinc where possible.
# All other coefficients are evaluated by their Taylor series for small angles,
# which avoids both, divisions by zero and cancellation errors.

pi_inv = 1 / np.pi

eye3 = np.eye(3, dtype=float)
_ax2skew_psi = ax2skew_a()
_skew2ax_A = skew2ax_A()

# Taylor series are used for angle**2 < _angle2_series
_angle2_series = 0.25


def _stack(x: np.ndarray, ndim: int):
    """Returns x as a stack of ndim-dimensional arrays and whether x was a
    single ndim-dimensional array."""
    x = np.asarray(x)
    single = x.ndim == ndim
    return (x[None] if single else x), single


def _angle2(psi: np.ndarray) -> np.ndarray:
    """Squared rotation angle of a stack of rotation vectors (n, 3). In
    contrast to np.linalg.norm, this is compatible with complex steps."""
    return np.sum(psi * psi, axis=-1)


def _series_or_exact(angle2, coefficients, exact):
    """Evaluates the Taylor series sum_k coefficients[k] * angle2**k for small
    angles and exact(angle2) otherwise."""
    small = np.real(angle2) < _angle2_series
    series = np.full_like(angle2, coefficients[-1], dtype=np.result_type(angle2, 1.0))
    for c in coefficients[-2::-1]:
        series = series * angle2 + c
    return np.where(small, series, exact(np.where(small, 1.0, angle2)))


######################################
# coefficients of SO(3) and SE(3) maps
######################################
# with sinc(x) = sin(x) / x and np.sinc(x / pi) = sinc(x), see numpy documentation
# The suffix _d denotes the operator (1 / angle) * d / d angle, such that
# d f(angle) / d psi = f_d(angle) * psi.
def _alpha(angle2):
    """alpha = sin(angle) / angle"""
    return np.sinc(np.sqrt(angle2) * pi_inv)


def _beta2(angle2):
    """beta2 = (1 - cos(angle)) / angle**2"""
    return 0.5 * np.sinc(0.5 * np.sqrt(angle2) * pi_inv) ** 2


def _delta(angle2):
    """delta = (angle - sin(angle)) / angle**3 = (1 - alpha) / angle**2"""
    # fmt: off
    return _series_or_exact(
        angle2,
        (1 / 6, -1 / 120, 1 / 5040, -1 / 362880, 1 / 39916800,
         -1 / 6227020800, 1 / 1307674368000, -1 / 355687428096000),
        lambda angle2: (1.0 - _alpha(angle2)) / angle2,
    )
    # fmt: on


def _gamma(angle2):
    """gamma = 0.5 * angle / tan(0.5 * angle)"""
    half_angle = 0.5 * np.sqrt(angle2)
    return np.cos(half_angle) / np.sinc(half_angle * pi_inv)


def _eta(angle2):
    """eta = (1 - gamma) / angle**2"""
    # fmt: off
    return _series_or_exact(
        angle2,
        (1 / 12, 1 / 720, 1 / 30240, 1 / 1209600, 1 / 47900160,
         691 / 1307674368000, 1 / 74724249600, 3617 / 10670622842880000),
        lambda angle2: (1.0 - _gamma(angle2)) / angle2,
    )
    # fmt: on


def _alpha_d(angle2):
    """alpha_d = (angle * cos(angle) - sin(angle)) / angle**3 = delta - beta2"""
    return _delta(angle2) - _beta2(angle2)


def _beta2_d(angle2):
    """beta2_d = (alpha - 2 * beta2) / angle**2"""
    # fmt: off
    return _series_or_exact(
        angle2,
        (-1 / 12, 1 / 180, -1 / 6720, 1 / 453600, -1 / 47900160,
         1 / 7264857600, -1 / 1494484992000, 1 / 400148356608000),
        lambda angle2: (_alpha(angle2) - 2.0 * _beta2(angle2)) / angle2,
    )
    # fmt: on


def _delta_d(angle2):
    """delta_d = (beta2 - 3 * delta) / angle**2"""
    # fmt: off
    return _series_or_exact(
        angle2,
        (-1 / 60, 1 / 1260, -1 / 60480, 1 / 4989600, -1 / 622702080,
         1 / 108972864000, -1 / 25406244864000, 1 / 7602818775552000),
        lambda angle2: (_beta2(angle2) - 3.0 * _delta(angle2)) / angle2,
    )
    # fmt: on


def _eta_d(angle2):
    """eta_d = -(2 * eta + gamma_d) / angle**2 with gamma_d = gamma * eta - 1 / 4"""
    # fmt: off
    def exact(angle2):
        eta = _eta(angle2)
        gamma_d = _gamma(angle2) * eta - 0.25
        return -(2.0 * eta + gamma_d) / angle2

    return _series_or_exact(
        angle2,
        (1 / 360, 1 / 7560, 1 / 201600, 1 / 5987520, 691 / 130767436800,
         1 / 6227020800, 3617 / 762187345920000, 43867 / 319318388573184000),
        exact,
    )
    # fmt: on


def _ax2skew_squared_psi(psi: np.ndarray) -> np.ndarray:
    """Derivative of ax2skew_squared(psi) = psi psi^T - (psi @ psi) eye3 for a
    stack of vectors psi (n, 3)."""
    return (
        np.einsum("ik,nj->nijk", eye3, psi)
        + np.einsum("ni,jk->nijk", psi, eye3)
        - 2.0 * np.einsum("ij,nk->nijk", eye3, psi)
    )


def _outer3(M: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Outer product of stacked matrices (n, 3, 3) and vectors (n, 3)."""
    return M[:, :, :, None] * psi[:, None, None, :]


########
# Exp_SO3
########
def Exp_SO3(psi: np.ndarray) -> np.ndarray:
    """SO(3) exponential function, see Crisfield1999 above (4.1) and
    Park2005 (11).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Park2005: https://doi.org/10.1109/TRO.2005.852253
    """
    psi, single = _stack(psi, 1)
    angle2 = _angle2(psi)[:, None, None]
    A = eye3 + _alpha(angle2) * ax2skew(psi) + _beta2(angle2) * ax2skew_squared(psi)
    return A[0] if single else A


def Exp_SO3_psi(psi: np.ndarray) -> np.ndarray:
    """Derivative of the axis-angle rotation found in Crisfield1999 above (4.1).
    Derivations and final results are given in Gallego2015 (9).

    References
    ----------
    Crisfield1999: https://doi.org/10.1098/rspa.1999.0352 \\
    Gallego2015: https://doi.org/10.1007/s10851-014-0528-x
    """
    psi, single = _stack(psi, 1)
    angle2 = _angle2(psi)[:, None, None, None]

    # derivative of alpha * psi_tilde + beta2 * psi_tilde @ psi_tilde
    A_psi = (
        _alpha(angle2) * _ax2skew_psi
        + _alpha_d(angle2) * _outer3(ax2skew(psi), psi)
        + _beta2(angle2) * _ax2skew_squared_psi(psi)
        + _beta2_d(angle2) * _outer3(ax2skew_squared(psi), psi)
    )
    return A_psi[0] if single else A_psi


def Log_SO3(A: np.ndarray) -> np.ndarray:
    A, single = _stack(A, 2)

    ca = 0.5 * (np.trace(A, axis1=1, axis2=2) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)

    # psi = angle / sin(angle) * skew2ax(A)
    psi = np.einsum("ijk,njk->ni", _skew2ax_A, A) / np.sinc(angle * pi_inv)[:, None]
    return psi[0] if single else psi


def Log_SO3_A(A: np.ndarray) -> np.ndarray:
    """Derivative of the SO(3) Log map. See Blanco-Claraco2010 (10.11)

    References:
    ===========
    Blanco-Claraco2010: https://doi.org/10.48550/arXiv.2103.15980
    """
    A, single = _stack(A, 2)

    ca = 0.5 * (np.trace(A, axis1=1, axis2=2) - 1.0)
    ca = np.clip(ca, -1, 1)  # clip to [-1, 1] for arccos!
    angle = np.arccos(ca)
    sinc = np.sinc(angle * pi_inv)

    # a = (angle * cos(angle) - sin(angle)) / (2 * sin(angle)**3) * skew2ax(A)
    a = (_alpha_d(angle * angle) / (2.0 * sinc**3))[:, None] * np.einsum(
        "ijk,njk->ni", _skew2ax_A, A
    )
    # b = angle / sin(angle)
    b = 1.0 / sinc

    psi_A = a[:, :, None, None] * eye3 + b[:, None, None, None] * _skew2ax_A
    return psi_A[0] if single else psi_A


def T_SO3(psi: np.ndarray) -> np.ndarray:
    # Park2005 (19), actually its the transposed!
    psi, single = _stack(psi, 1)
    angle2 = _angle2(psi)[:, None, None]
    T = eye3 - _beta2(angle2) * ax2skew(psi) + _delta(angle2) * ax2skew_squared(psi)
    return T[0] if single else T


def T_SO3_psi(psi: np.ndarray) -> np.ndarray:
    psi, single = _stack(psi, 1)
    angle2 = _angle2(psi)[:, None, None, None]

    # derivative of -beta2 * psi_tilde + delta * psi_tilde @ psi_tilde
    T_psi = (
        -_beta2(angle2) * _ax2skew_psi
        - _beta2_d(angle2) * _outer3(ax2skew(psi), psi)
        + _delta(angle2) * _ax2skew_squared_psi(psi)
        + _delta_d(angle2) * _outer3(ax2skew_squared(psi), psi)
    )
    return T_psi[0] if single else T_psi


def T_SO3_dot(psi: np.ndarray, psi_dot: np.ndarray) -> np.ndarray:
    """Derivative of tangent map w.r.t. scalar argument of rotation vector, see
    Ibrahimbegović1995 (71). Actually in Ibrahimbegovic1995 (28) T_s^{T}
    is shown!

    References
    ----------
    Ibrahimbegovic1995: https://doi.org/10.1002/nme.1620382107
    """
    psi, single = _stack(psi, 1)
    psi_dot, _ = _stack(psi_dot, 1)
    angle2 = _angle2(psi)[:, None, None]
    psi_psi_dot = np.sum(psi * psi_dot, axis=-1)[:, None, None]
    delta = _delta(angle2)

    # transpose of Ibrahimbegović1995 (71)
    T_dot = (
        _alpha_d(angle2) * np.einsum("ni,nj->nij", psi, psi_dot)
        + _beta2_d(angle2) * np.einsum("ni,nj->nij", psi, np.cross(psi, psi_dot))
        + _delta_d(angle2) * psi_psi_dot * np.einsum("ni,nj->nij", psi, psi)
        + _beta2(angle2) * ax2skew(psi_dot)
        + delta * psi_psi_dot * eye3
        + delta * np.einsum("ni,nj->nij", psi_dot, psi)
    )
    return T_dot[0] if single else T_dot


def T_SO3_inv(psi: np.ndarray) -> np.ndarray:
    # Park2005 (19), actually its the transposed!
    psi, single = _stack(psi, 1)
    angle2 = _angle2(psi)[:, None, None]
    T_inv = eye3 + 0.5 * ax2skew(psi) + _eta(angle2) * ax2skew_squared(psi)
    return T_inv[0] if single else T_inv


def T_SO3_inv_psi(psi: np.ndarray) -> np.ndarray:
    psi, single = _stack(psi, 1)
    angle2 = _angle2(psi)[:, None, None, None]

    # derivative of 0.5 * psi_tilde + eta * psi_tilde @ psi_tilde
    T_inv_psi = (
        0.5 * _ax2skew_psi
        + _eta(angle2) * _ax2skew_squared_psi(psi)
        + _eta_d(angle2) * _outer3(ax2skew_squared(psi), psi)
    )
    return T_inv_psi[0] if single else T_inv_psi


########
# SE(3)
########
def SE3(A_IB: np.ndarray, r_OP: np.ndarray) -> np.ndarray:
    A_IB, single = _stack(A_IB, 2)
    r_OP, _ = _stack(r_OP, 1)

    H = np.zeros((A_IB.shape[0], 4, 4), dtype=np.result_type(A_IB, r_OP, 1.0))
    H[:, :3, :3] = A_IB
    H[:, :3, 3] = r_OP
    H[:, 3, 3] = 1.0
    return H[0] if single else H


def SE3inv(H: np.ndarray) -> np.ndarray:
    H, single = _stack(H, 2)
    assert H.shape[1:] == (4, 4)

    A_BI = np.swapaxes(H[:, :3, :3], 1, 2)
    r_OP = H[:, :3, 3]

    H_inv = np.zeros_like(H, dtype=np.result_type(H, 1.0))
    H_inv[:, :3, :3] = A_BI
    H_inv[:, :3, 3] = -np.einsum("nij,nj->ni", A_BI, r_OP)
    H_inv[:, 3, 3] = 1.0

    return H_inv[0] if single else H_inv


def Exp_SE3(h: np.ndarray) -> np.ndarray:
    h, single = _stack(h, 1)
    assert h.shape[1] == 6

    r = h[:, :3]
    psi = h[:, 3:]

    H = np.zeros((h.shape[0], 4, 4), dtype=np.result_type(h, 1.0))
    H[:, :3, :3] = Exp_SO3(psi)
    H[:, :3, 3] = np.einsum("nji,nj->ni", T_SO3(psi), r)
    H[:, 3, 3] = 1.0

    return H[0] if single else H


def Exp_SE3_h(h: np.ndarray) -> np.ndarray:
    h, single = _stack(h, 1)
    assert h.shape[1] == 6

    r = h[:, :3]
    psi = h[:, 3:]

    H_h = np.zeros((h.shape[0], 4, 4, 6), dtype=np.result_type(h, 1.0))
    H_h[:, :3, :3, 3:] = Exp_SO3_psi(psi)
    H_h[:, :3, 3, 3:] = np.einsum("nl,nlik->nik", r, T_SO3_psi(psi))
    H_h[:, :3, 3, :3] = np.swapaxes(T_SO3(psi), 1, 2)

    return H_h[0] if single else H_h


def Log_SE3(H: np.ndarray) -> np.ndarray:
    H, single = _stack(H, 2)
    assert H.shape[1:] == (4, 4)

    A = H[:, :3, :3]
    r = H[:, :3, 3]
    psi = Log_SO3(A)

    h = np.zeros((H.shape[0], 6), dtype=np.result_type(H, 1.0))
    h[:, :3] = np.einsum("nji,nj->ni", T_SO3_inv(psi), r)
    h[:, 3:] = psi

    return h[0] if single else h


def Log_SE3_H(H: np.ndarray) -> np.ndarray:
    H, single = _stack(H, 2)
    assert H.shape[1:] == (4, 4)

    A = H[:, :3, :3]
    r = H[:, :3, 3]
    psi = Log_SO3(A)
    psi_A = Log_SO3_A(A)

    h_H = np.zeros((H.shape[0], 6, 4, 4), dtype=np.result_type(H, 1.0))
    h_H[:, :3, :3, :3] = np.einsum("nl,nlim,nmjk->nijk", r, T_SO3_inv_psi(psi), psi_A)
    h_H[:, :3, :3, 3] = np.swapaxes(T_SO3_inv(psi), 1, 2)
    h_H[:, 3:, :3, :3] = psi_A

    return h_H[0] if single else h_H


def U(a, b):
    """Off-diagonal block of the SE(3) tangent map, see Sonneville2014 (A.12),
    expressed with the SO(3) coefficients:
    U = -beta2 a~ + delta (a~ b~ + b~ a~) + (b . a) (-beta2_d b~ + delta_d b~ b~)
    """
    a, single = _stack(a, 1)
    b, _ = _stack(b, 1)
    angle2 = _angle2(b)[:, None, None]
    a_tilde = ax2skew(a)
    b_tilde = ax2skew(b)
    ba = np.sum(b * a, axis=-1)[:, None, None]

    U = (
        -_beta2(angle2) * a_tilde
        + _delta(angle2) * (a_tilde @ b_tilde + b_tilde @ a_tilde)
        + ba * (-_beta2_d(angle2) * b_tilde + _delta_d(angle2) * b_tilde @ b_tilde)
    )
    return U[0] if single else U


def T_SE3(h: np.ndarray) -> np.ndarray:
    """Tangent map of SE(3) with h = (r, psi), i.e.,
    Exp_SE3(h)^-1 @ d Exp_SE3(h) = hat(T_SE3(h) @ dh)."""
    h, single = _stack(h, 1)
    assert h.shape[1] == 6

    T = np.zeros((h.shape[0], 6, 6), dtype=np.result_type(h, 1.0))
    T[:, :3, :3] = T[:, 3:, 3:] = T_SO3(h[:, 3:])
    T[:, :3, 3:] = U(h[:, :3], h[:, 3:])
    return T[0] if single else T


def T_SE3_inv(h: np.ndarray) -> np.ndarray:
    """Inverse of the tangent map of SE(3)."""
    h, single = _stack(h, 1)
    assert h.shape[1] == 6

    T_inv_SO3 = T_SO3_inv(h[:, 3:])
    T_inv = np.zeros((h.shape[0], 6, 6), dtype=np.result_type(h, 1.0))
    T_inv[:, :3, :3] = T_inv[:, 3:, 3:] = T_inv_SO3
    T_inv[:, :3, 3:] = -T_inv_SO3 @ U(h[:, :3], h[:, 3:]) @ T_inv_SO3
    return T_inv[0] if single else T_inv


def Ad_SE3(H: np.ndarray) -> np.ndarray:
    """Adjoint of H = (A, r) acting on twists (v, omega), i.e.,
    hat(Ad_SE3(H) @ s) = H @ hat(s) @ H^-1."""
    H, single = _stack(H, 2)
    assert H.shape[1:] == (4, 4)

    A = H[:, :3, :3]
    Ad = np.zeros((H.shape[0], 6, 6), dtype=np.result_type(H, 1.0))
    Ad[:, :3, :3] = Ad[:, 3:, 3:] = A
    Ad[:, :3, 3:] = ax2skew(H[:, :3, 3]) @ A
    return Ad[0] if single else Ad


class A_IB_basic:
    """Basic rotations in Euclidean space."""

    def __init__(self, phi: float):
        self.phi = phi
        self.sp = np.sin(phi)
        self.cp = np.cos(phi)

    @property
    def x(self) -> np.ndarray:
        """Rotation around x-axis."""
        # fmt: off
        return np.array([[1,       0,        0],
                         [0, self.cp, -self.sp],
                         [0, self.sp,  self.cp]])
        # fmt: on

    @property
    def dx(self) -> np.ndarray:
        """Derivative of Rotation around x-axis."""
        # fmt: off
        return np.array([[0,        0,        0],
                         [0, -self.sp, -self.cp],
                         [0,  self.cp, -self.sp]])
        # fmt: on

    @property
    def y(self) -> np.ndarray:
        """Rotation around y-axis."""
        # fmt: off
        return np.array([[ self.cp, 0, self.sp],
                         [       0, 1,       0],
                         [-self.sp, 0, self.cp]])
        # fmt: on

    @property
    def dy(self) -> np.ndarray:
        """Derivative of Rotation around y-axis."""
        # fmt: off
        return np.array([[-self.sp, 0,  self.cp],
                         [       0, 0,        0],
                         [-self.cp, 0, -self.sp]])
        # fmt: on

    @property
    def z(self) -> np.ndarray:
        """Rotation around z-axis."""
        # fmt: off
        return np.array([[self.cp, -self.sp, 0],
                         [self.sp,  self.cp, 0],
                         [      0,        0, 1]])
        # fmt: on

    @property
    def dz(self) -> np.ndarray:
        """Derivative of Rotation around z-axis."""
        # fmt: off
        return np.array([[-self.sp,  -self.cp, 0],
                         [  self.cp, -self.sp, 0],
                         [        0,        0, 0]])
        # fmt: on


def Spurrier(R: np.ndarray) -> np.ndarray:
    """
    Spurrier's algorithm to extract the unit quaternion from a given rotation
    matrix, see Spurrier19978, Simo1986 Table 12 and Crisfield1997 Section 16.10.

    References
    ----------
    Spurrier19978: https://arc.aiaa.org/doi/10.2514/3.57311 \\
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4 \\
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    R, single = _stack(R, 2)

    diag = np.diagonal(R, axis1=1, axis2=2)
    trace = np.sum(diag, axis=1)
    decision = np.concatenate((diag, trace[:, None]), axis=1)
    case = np.argmax(np.real(decision), axis=1)

    quat = np.zeros((R.shape[0], 4), dtype=np.result_type(R, 1.0))

    # trace is the largest element
    mask = case == 3
    Rm = R[mask]
    q0 = 0.5 * np.sqrt(1 + trace[mask])
    quat[mask, 0] = q0
    quat[mask, 1] = (Rm[:, 2, 1] - Rm[:, 1, 2]) / (4 * q0)
    quat[mask, 2] = (Rm[:, 0, 2] - Rm[:, 2, 0]) / (4 * q0)
    quat[mask, 3] = (Rm[:, 1, 0] - Rm[:, 0, 1]) / (4 * q0)

    # i-th diagonal element is the largest element
    for i in range(3):
        j = (i + 1) % 3
        k = (j + 1) % 3

        mask = case == i
        Rm = R[mask]
        qi = np.sqrt(0.5 * Rm[:, i, i] + 0.25 * (1 - trace[mask]))
        quat[mask, i + 1] = qi
        quat[mask, 0] = (Rm[:, k, j] - Rm[:, j, k]) / (4 * qi)
        quat[mask, j + 1] = (Rm[:, j, i] + Rm[:, i, j]) / (4 * qi)
        quat[mask, k + 1] = (Rm[:, k, i] + Rm[:, i, k]) / (4 * qi)

    return quat[0] if single else quat


def quat2axis_angle(Q: np.ndarray) -> np.ndarray:
    """Extract the rotation vector psi for a given quaterion Q = [q0, q] in
    accordance with Wiki2021.

    References
    ----------
    Wiki2021: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Recovering_the_axis-angle_representation
    """
    Q, single = _stack(Q, 1)

    q0, vq = Q[:, 0], Q[:, 1:]
    q = np.sqrt(_angle2(vq))
    half_angle = np.arctan2(q, q0)

    # psi = angle * vq / ||vq||. Using ||vq|| = ||Q|| * sin(0.5 * angle) and
    # sinc is ill-conditioned for angle -> 2 pi, hence we only guard q = 0,
    # where vq = 0 and thus psi = 0.
    psi = (2.0 * half_angle / np.where(q > 0, q, 1.0))[:, None] * vq
    return psi[0] if single else psi


def smallest_rotation(
    J_a: np.ndarray, J_b: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Compute the transformation matrix A_JK in accordance with
    {}_J a = A_JK {}_J b. This is sometimes referred to 'smallest rotation',
    see Crisield1996 Section 16.13. Both vectors are normalized if
    normalize=True is used.

    This tranformation has a singularity for {}_J b = -{}_J a. This can be
    overcome using a singular value decomposition that determines the rotation
    axis, see Eigen3.

    References
    ----------
    Crisfield1996: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf \\
    Eigen3: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Geometry/Quaternion.h#L633-669
    """
    J_a, single = _stack(J_a, 1)
    J_b, _ = _stack(J_b, 1)
    J_a, J_b = np.broadcast_arrays(J_a, J_b)

    if normalize:
        J_a = J_a / np.sqrt(_angle2(J_a))[:, None]
        J_b = J_b / np.sqrt(_angle2(J_b))[:, None]

    cos_psi = np.sum(J_a * J_b, axis=1)
    denom = 1.0 + cos_psi

    # TODO: What is a good singular value her?
    singular = denom <= 1e-6

    e = np.cross(J_a, J_b)
    A = (
        cos_psi[:, None, None] * eye3
        + ax2skew(e)
        + np.einsum("ni,nj->nij", e, e) / np.where(singular, 1.0, denom)[:, None, None]
    )

    if np.any(singular):
        M = np.stack((J_a[singular], J_b[singular]), axis=1)
        _, _, Vh = np.linalg.svd(M)
        axis = Vh[:, 2]
        psi = np.arccos(np.clip(cos_psi[singular], -1, 1))
        A[singular] = Exp_SO3(psi[:, None] * axis)

    return A[0] if single else A


def Exp_SO3_quat(P: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.163), Nuetzi2016 (3.31) and Rucker2018 (13).

    normalize = True : normalizes P
    normalize = False: assumes always ||P|| = 1

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    P = np.asarray(P)
    was_1d = P.ndim == 1
    P = np.atleast_2d(P)
    assert P.shape[1] == 4

    p0, p = P[:, 0], P[:, 1:]
    matrix = 2.0 * (p0[:, None, None] * ax2skew(p) + ax2skew_squared(p))
    if normalize:
        matrix /= np.sum(P * P, axis=1)[:, None, None]

    result = eye3[None, :, :] + matrix
    return result[0] if was_1d else result


def Exp_SO3_quat_P(P_IB, normalize=True):
    """Derivative of Exp_SO3_quat with respect to P."""
    P_IB = np.asarray(P_IB)
    was_1d = P_IB.ndim == 1
    P_IB = np.atleast_2d(P_IB)
    assert P_IB.shape[1] == 4

    n = P_IB.shape[0]
    p0, p = P_IB[:, 0], P_IB[:, 1:]
    p_tilde = ax2skew(p)
    p_tilde_p = ax2skew_a()
    matrix_P = np.zeros((n, 3, 3, 4), dtype=np.result_type(P_IB, 1.0))
    # deriv. w.r.t. p0
    matrix_P[:, :, :, 0] = 2.0 * p_tilde
    # deriv. w.r.t. p: d(p0 * p_tilde + p_tilde @ p_tilde)/dp
    term1 = p0[:, None, None, None] * p_tilde_p[None, :, :, :]
    term2 = np.einsum("ilk,nlj->nijk", p_tilde_p, p_tilde)
    term3 = np.einsum("nil,ljk->nijk", p_tilde, p_tilde_p)
    matrix_P[:, :, :, 1:] = 2.0 * (term1 + term2 + term3)

    if normalize:
        P2_inv = 1 / np.sum(P_IB * P_IB, axis=1)
        matrix_P *= P2_inv[:, None, None, None]
        # inner derivative due to normalization
        matrix = 2.0 * (p0[:, None, None] * p_tilde + ax2skew_squared(p))
        matrix_P += (
            matrix[:, :, :, None]
            * (-2.0 * (P2_inv**2)[:, None] * P_IB)[:, None, None, :]
        )
    return matrix_P[0] if was_1d else matrix_P


Log_SO3_quat = Spurrier


def T_SO3_quat(P_IB: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Tangent map for unit quaternion. See Egeland2002 (6.327).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    P_IB = np.asarray(P_IB)
    was_1d = P_IB.ndim == 1
    P_IB = np.atleast_2d(P_IB)
    assert P_IB.shape[1] == 4

    p0, p = P_IB[:, 0], P_IB[:, 1:]
    result = np.empty((P_IB.shape[0], 3, 4), dtype=np.result_type(P_IB, 1.0))
    result[:, :, 0] = -2.0 * p
    result[:, :, 1:] = 2.0 * (p0[:, None, None] * eye3 - ax2skew(p))
    if normalize:
        result /= np.sum(P_IB * P_IB, axis=1)[:, None, None]
    return result[0] if was_1d else result


def T_SO3_inv_quat(P_IB, normalize=True):
    P_IB = np.asarray(P_IB)
    was_1d = P_IB.ndim == 1
    P_IB = np.atleast_2d(P_IB)
    assert P_IB.shape[1] == 4

    p0, p = P_IB[:, 0], P_IB[:, 1:]
    result = np.empty((P_IB.shape[0], 4, 3), dtype=np.result_type(P_IB, 1.0))
    result[:, 0] = -p
    result[:, 1:] = p0[:, None, None] * eye3 + ax2skew(p)
    result *= 0.5

    return result[0] if was_1d else result


def T_SO3_inv_quat_(P, normalize=True):
    """Inverse tangent map for unit quaternion. See Egeland2002 (6.329) and
    (6.330), Nuetzi2016 (3.11) and (4.19) as well as Rucker2018 (21) 
    and (22).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165 \\
    Rucker2018: https://ieeexplore.ieee.org/document/8392463
    """
    p0, p = P[0], P[1:]
    return np.vstack((-p, p0 * eye3 + ax2skew(p))) / 2


def T_SO3_quat_P(P_IB, normalize=True):
    P_IB = np.asarray(P_IB)
    was_1d = P_IB.ndim == 1
    P_IB = np.atleast_2d(P_IB)
    assert P_IB.shape[1] == 4

    n = P_IB.shape[0]
    p0, p = P_IB[:, 0], P_IB[:, 1:]
    result = np.zeros((n, 3, 4, 4), dtype=np.result_type(P_IB, 1.0))
    result[:, :, 0, 1:] = -2.0 * eye3
    result[:, :, 1:, 0] = 2.0 * eye3
    result[:, :, 1:, 1:] = -2.0 * ax2skew_a()

    if normalize:
        P2_inv = 1 / np.sum(P_IB * P_IB, axis=1)
        result *= P2_inv[:, None, None, None]

        # matrix = 2 * [ -p , p0*I - skew(p) ]
        matrix = np.empty((n, 3, 4), dtype=result.dtype)
        matrix[:, :, 0] = -2.0 * p
        matrix[:, :, 1:] = 2.0 * (p0[:, None, None] * eye3 - ax2skew(p))

        result += (
            matrix[:, :, :, None]
            * (-2.0 * P2_inv**2)[:, None, None, None]
            * P_IB[:, None, None, :]
        )
    return result[0] if was_1d else result


# fmt: off
T_inv_P = np.zeros((4, 3, 4), dtype=float)
T_inv_P[0, :, 1:] = -0.5 * eye3
T_inv_P[1:, :, 0] = 0.5 * eye3
T_inv_P[1:, :, 1:] = 0.5 * ax2skew_a()
def T_SO3_inv_quat_P(P, normalize=True):
    return T_inv_P
# fmt: on


def Exp_SO3_R9(R9):
    R9 = np.asarray(R9)
    was_1d = R9.ndim == 1
    R9 = np.atleast_2d(R9)
    assert R9.shape[1] == 9

    A_IB = R9.reshape(-1, 3, 3, order="F")
    return A_IB[0] if was_1d else A_IB


# fmt: off
A_IB_R9 = np.zeros((3, 3, 9), dtype=float)
A_IB_R9[:, 0, :3] = eye3
A_IB_R9[:, 1, 3:6] = eye3
A_IB_R9[:, 2, 6:] = eye3
def Exp_SO3_R9_R9(R9):
    R9 = np.asarray(R9)
    was_1d = R9.ndim == 1
    R9 = np.atleast_2d(R9)
    assert R9.shape[1] == 9
    return A_IB_R9 if was_1d else np.broadcast_to(A_IB_R9, (R9.shape[0], 3, 3, 9))
# fmt: on


def Log_SO3_R9(A_IB):
    A_IB, single = _stack(A_IB, 2)
    R9 = np.swapaxes(A_IB, 1, 2).reshape(-1, 9)
    return R9[0] if single else R9


def T_SO3_R9(R9):
    R9 = np.asarray(R9)
    was_1d = R9.ndim == 1
    R9 = np.atleast_2d(R9)
    assert R9.shape[1] == 9

    T_IB = np.empty((R9.shape[0], 3, 9), dtype=np.result_type(R9, 1.0))
    T_IB[:, 0, :3] = 0.0
    T_IB[:, 0, 3:6] = 0.5 * R9[:, 6:]
    T_IB[:, 0, 6:] = -0.5 * R9[:, 3:6]
    T_IB[:, 1, :3] = -0.5 * R9[:, 6:]
    T_IB[:, 1, 3:6] = 0.0
    T_IB[:, 1, 6:] = 0.5 * R9[:, :3]
    T_IB[:, 2, :3] = 0.5 * R9[:, 3:6]
    T_IB[:, 2, 3:6] = -0.5 * R9[:, :3]
    T_IB[:, 2, 6:] = 0.0
    return T_IB[0] if was_1d else T_IB


# fmt: off
T_IB_R9 = np.zeros((3, 9, 9), dtype=float)
T_IB_R9[0, 3:6, 6:] = 0.5 * eye3
T_IB_R9[0, 6:, 3:6] = -0.5 * eye3
T_IB_R9[1, :3, 6:] = -0.5 * eye3
T_IB_R9[1, 6:, :3] = 0.5 * eye3
T_IB_R9[2, :3, 3:6] = 0.5 * eye3
T_IB_R9[2, 3:6, :3] = -0.5 * eye3
def T_SO3_R9_R9(R9):
    R9 = np.asarray(R9)
    was_1d = R9.ndim == 1
    R9 = np.atleast_2d(R9)
    assert R9.shape[1] == 9
    return T_IB_R9 if was_1d else np.broadcast_to(T_IB_R9, (R9.shape[0], 3, 9, 9))
# fmt: on


def T_SO3_inv_R9(R9):
    # is this 4 * T_SO3_R9.T?
    R9 = np.asarray(R9)
    was_1d = R9.ndim == 1
    R9 = np.atleast_2d(R9)
    assert R9.shape[1] == 9

    T_IB_inv = np.empty((R9.shape[0], 9, 3), dtype=np.result_type(R9, 1.0))
    T_IB_inv[:, :3, 0] = 0.0
    T_IB_inv[:, :3, 1] = -R9[:, 6:]
    T_IB_inv[:, :3, 2] = R9[:, 3:6]

    T_IB_inv[:, 3:6, 0] = R9[:, 6:]
    T_IB_inv[:, 3:6, 1] = 0.0
    T_IB_inv[:, 3:6, 2] = -R9[:, :3]

    T_IB_inv[:, 6:, 0] = -R9[:, 3:6]
    T_IB_inv[:, 6:, 1] = R9[:, :3]
    T_IB_inv[:, 6:, 2] = 0.0
    return T_IB_inv[0] if was_1d else T_IB_inv


# fmt: off
T_IB_inv_R9 = np.zeros((9, 3, 9), dtype=float)
T_IB_inv_R9[:3, 1, 6:] = -eye3
T_IB_inv_R9[:3, 2, 3:6] = eye3
T_IB_inv_R9[3:6, 0, 6:] = eye3
T_IB_inv_R9[3:6, 2, :3] = -eye3
T_IB_inv_R9[6:, 0, 3:6] = -eye3
T_IB_inv_R9[6:, 1, :3] = eye3
def T_SO3_inv_R9_R9(R9=None):
    # constant derivative: R9=None returns the single (9, 3, 9) array
    if R9 is None:
        return T_IB_inv_R9
    R9, single = _stack(R9, 1)
    assert R9.shape[1] == 9
    return T_IB_inv_R9 if single else np.broadcast_to(T_IB_inv_R9, (R9.shape[0], 9, 3, 9))
# fmt: on


def quatprod(P_IJ, P_JB):
    """Quaternion product, see Egeland2002 (6.190).

    References:
    -----------
    Egeland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf
    """
    P_IJ = np.asarray(P_IJ)
    P_JB = np.asarray(P_JB)

    was_1d = P_IJ.ndim == 1
    P_IJ = np.atleast_2d(P_IJ)
    P_JB = np.atleast_2d(P_JB)

    assert P_IJ.shape == P_JB.shape
    assert P_IJ.shape[1] == 4

    p0, p = P_IJ[:, 0], P_IJ[:, 1:]
    q0, q = P_JB[:, 0], P_JB[:, 1:]

    # scalar part
    z0 = p0 * q0 - np.sum(p * q, axis=1)

    # vectorial part
    z = p0[:, None] * q + q0[:, None] * p + np.cross(p, q)

    result = np.concatenate((z0[:, None], z), axis=1)
    return result[0] if was_1d else result


def axis_angle2quat(axis: np.ndarray, angle) -> np.ndarray:
    axis, single = _stack(axis, 1)
    angle = np.broadcast_to(angle, axis.shape[:1])

    n = axis / np.sqrt(_angle2(axis))[:, None]
    result = np.concatenate(
        (np.cos(0.5 * angle)[:, None], np.sin(0.5 * angle)[:, None] * n), axis=1
    )
    return result[0] if single else result
