import numpy as np

import matplotlib.pyplot as plt


p_deg = np.arange(5) + 1
n_red = p_deg
n_full_Masterthesis = 2 * p_deg - 1
n_full_curently = np.ceil((p_deg + 1) ** 2 / 2)
n_full_new = np.ceil((5 * p_deg - 1) / 2)
n_exact_masse = p_deg + 1
n_exact_fgyr = np.ceil((3 * p_deg + 1) / 2)


fig, ax = plt.subplots(2)

ax[0].plot(p_deg, n_red, ":o", label="RI")
ax[0].plot(p_deg, n_full_curently, ":o", label="FI ceil[(p+1)^2 / 2] (currently)")
ax[0].plot(p_deg, n_full_new, ":o", label="FI ceil[(5p-1)/2] (my counting)")
ax[0].plot(
    p_deg, n_full_Masterthesis, ":o", label="FI 2p-1 (KirchhoffLove rod/Masterthesis)"
)

ax[0].grid()
ax[0].legend()

ax[1].plot(p_deg, n_red, ":o", label="RI")
ax[1].plot(p_deg, n_full_curently, ":o", label="FI ceil[(p+1)^2 / 2] (currently)")
ax[1].plot(p_deg, n_exact_masse, ":o", label="Exact for M p+1")
ax[1].plot(p_deg, n_exact_fgyr, ":o", label="Exact for f^gyr ceil[(3p+1)/2]")

ax[1].grid()
ax[1].legend()

# # comparison for different r values
# r = 0.5 * p_deg

# n_f = np.ceil(r + 2 * p_deg - 1/2)
# n_W = np.ceil(r/2 + 2*p_deg - 1/2)
# n_K = np.ceil(p_deg - 0.5)
# n_l = np.ceil(r/2 + 3/2 * p_deg - 1/2)

# n_DB = n_f
# n_MX = np.max([n_W, n_K, n_l], axis=0)

# ax[1].plot(p_deg, n_DB, ":o", label="DB")
# ax[1].plot(p_deg, n_MX, ":o", label="MX")
# ax[1].grid()
# ax[1].legend()

plt.show()
