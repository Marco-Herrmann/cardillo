import numpy as np

from cardillo.discrete import RigidBody


# https://link.springer.com/article/10.1007/s11071-020-06069-5
if __name__ == "__main__":
    # values from table 1
    w = 1.02
    c = 0.08
    la = np.pi / 10
    x_B = 0.3
    z_B = 0.9
    x_H = 0.9
    z_H = 0.7

    # rear body and frame assembly (B)
    m_B = 85.0
    I_B = np.array([[9.2, 0.0, -2.4], [0.0, 11.0, 0.0], [-2.4, 0.0, 2.8]])
    # front handlebar and fork assembly (H)
    m_H = 4.0
    I_H = np.array(
        [[0.05841, 0.0, -0.00912], [0.0, 0.06, 0.0], [-0.00912, 0.0, 0.00759]]
    )
    # rear wheel (R)
    rho_R = 0.3
    m_R = 2.0
    I_R = np.diag([0.0603, 0.12, 0.0603])
    # front wheel (F)
    rho_F = 0.35
    m_F = 3.0
    I_F = np.diag([0.1405, 0.28, 0.148])

    # auxiliary variables (8)
    x_I = w + c - x_B - z_B * np.tan(la)
    a = (x_H - w - c) * np.cos(la) + z_H * np.sin(la)
    b = (
        1
        / (2 * np.cos(la))
        * ((x_H - w - c) * np.sin(2 * la) - z_H * np.cos(2 * la) + 2 * z_B - z_H)
    )
    d = rho_F * np.sin(la) - c * np.cos(la)
    e = (
        1
        / (2 * np.cos(la))
        * (2 * z_B - rho_F - c * np.sin(2 * la) - rho_F * np.cos(2 * la))
    )

    # create rigid bodies
    r_OB0 = np.array([x_B, 0.0, z_B])
    q0_B = RigidBody.pose2q(r_OB0, np.eye(3, dtype=float))
    body_B = RigidBody(m_B, I_B, q0_B)
