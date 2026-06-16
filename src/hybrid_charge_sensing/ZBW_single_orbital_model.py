"""Charge-related simulations and numerical calculations."""

#%%
import numpy as np

def ZBW_single_orbital_hamiltonian(U, Delta, t, mu, E_Z=0.0):
    """Finite-gap ZBW Hamiltonian in the basis
    |0;0>, |0;up>, |0;down>, |0;2>,
    |up;0>, ..., |2;2>.

    Convention:
        epsilon_up   = mu + E_Z
        epsilon_down = mu - E_Z

    Thus E_Z is half of the total Zeeman splitting.
    The superconducting ZBW level is fixed at xi = 0.
    """
    I4 = np.eye(4, dtype=complex)
    P = np.diag([1, -1, -1, 1]).astype(complex)

    f_up = np.array([[0, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]], dtype=complex)

    f_down = np.array([[0, 0, 1, 0],
                       [0, 0, 0, -1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]], dtype=complex)

    d_up, d_down = np.kron(f_up, I4), np.kron(f_down, I4)
    c_up, c_down = np.kron(P, f_up), np.kron(P, f_down)

    n_du, n_dd = d_up.conj().T @ d_up, d_down.conj().T @ d_down

    epsilon_up, epsilon_down = mu + E_Z, mu - E_Z

    H_dot = epsilon_up * n_du + epsilon_down * n_dd + U * (n_du @ n_dd)

    H_sc = -Delta * (c_up.conj().T @ c_down.conj().T + c_down @ c_up)

    H_t = t * (d_up.conj().T @ c_up + c_up.conj().T @ d_up
               + d_down.conj().T @ c_down + c_down.conj().T @ d_down)

    return H_dot + H_sc + H_t, n_du, n_dd


#%%
import matplotlib.pyplot as plt

# Parameters (energies in units of Delta)
U, Delta, E_Z = 3.0, 1.0, 0.0
mu_vals = np.linspace(-4.5, 1.5, 400)

# Doublet GS: small t, pairing insufficient to quench the spin
# Singlet GS: large t, SC correlations pull ground state into even parity
cases = [
    ("Doublet GS", 0.5),
    ("Singlet GS", 1.2),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for row, (label, t) in enumerate(cases):
    energies, charges = [], []
    for mu in mu_vals:
        H, n_du, n_dd = ZBW_single_orbital_hamiltonian(U, Delta, t, mu, E_Z)
        evals, evecs = np.linalg.eigh(H)
        gs = evecs[:, 0]
        charges.append((gs.conj() @ (n_du + n_dd) @ gs).real)
        energies.append(evals)

    energies = np.array(energies)
    excitations = energies - energies[:, [0]]   # E_i - E_GS at each mu
    charges = np.array(charges)

    ax_e, ax_n = axes[row, 0], axes[row, 1]

    ax_e.plot(mu_vals, excitations[:, 2], color="C0", lw=1.5)
    ax_e.plot(mu_vals, -1 * excitations[:, 2], color="C0", lw=1.5)
    ax_e.set_xlabel(r"$\mu\;/\;\Delta$")
    ax_e.set_ylabel(r"$E_1 - E_0\;/\;\Delta$")
    ax_e.set_title(rf"{label}  ($t = {t}\,\Delta$, $U = {U}\,\Delta$): excitation spectrum")

    ax_n.plot(mu_vals, charges, color="C0", lw=1.8)
    ax_n.set_xlabel(r"$\mu\;/\;\Delta$")
    ax_n.set_ylabel(r"$\langle \hat{n} \rangle$")
    ax_n.set_title(rf"{label}  ($t = {t}\,\Delta$): total charge")
    ax_n.set_ylim(-0.1, 2.1)
    ax_n.set_yticks([0, 1, 2])

fig.tight_layout()
plt.show()

#%%
t_vals = np.linspace(0.2, 1.2, 100)

gap_map    = np.zeros((len(t_vals), len(mu_vals)))
charge_map = np.zeros((len(t_vals), len(mu_vals)))

for i, t in enumerate(t_vals):
    for j, mu in enumerate(mu_vals):
        H, n_du, n_dd = ZBW_single_orbital_hamiltonian(U, Delta, t, mu, E_Z)
        evals, evecs = np.linalg.eigh(H)
        gs = evecs[:, 0]
        gap_map[i, j]    = evals[1] - evals[0]
        charge_map[i, j] = (gs.conj() @ (n_du + n_dd) @ gs).real

fig, (ax_gap, ax_n) = plt.subplots(1, 2, figsize=(12, 5))

ax_gap.contour(mu_vals, t_vals, gap_map, levels=[0.25 * Delta], colors="C0")
ax_gap.set_xlabel(r"$\mu\;/\;\Delta$")
ax_gap.set_ylabel(r"$t\;/\;\Delta$")
ax_gap.set_title(r"$E_1 - E_0 = 0.25\,\Delta$  contour")

pcm = ax_n.pcolormesh(mu_vals, t_vals, charge_map, shading="auto", cmap="RdBu_r")
fig.colorbar(pcm, ax=ax_n, label=r"$\langle \hat{n} \rangle$")
ax_n.set_xlabel(r"$\mu\;/\;\Delta$")
ax_n.set_ylabel(r"$t\;/\;\Delta$")
ax_n.set_title(r"Total charge $\langle \hat{n} \rangle$")

fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
cut_indices = np.linspace(0, len(t_vals) - 1, 10, dtype=int)
cmap = plt.get_cmap("viridis")
for k, idx in enumerate(cut_indices):
    color = cmap(k / (len(cut_indices) - 1))
    ax.plot(mu_vals, charge_map[idx], color=color, lw=1.5, label=rf"$t={t_vals[idx]:.2f}\,\Delta$")
ax.set_xlabel(r"$\mu\;/\;\Delta$")
ax.set_ylabel(r"$\langle \hat{n} \rangle$")
ax.set_title(r"Total charge cuts over $t$")
ax.set_ylim(-0.1, 2.1)
ax.set_yticks([0, 1, 2])
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
plt.show()
