import numpy as np
import sys
from scipy.linalg import eigh
import module as mod
import plotting_module as plot

def main():
    # =================Constants============================================
    # check_flag = False
    # delta = 4.0
    # t = 0.1
    # us = 16.0
    # uhl = 6.0
    # first_time = True
    #
    # # =================Initialize our system================================
    # nsiti = 4
    # nso = 2 * nsiti
    # basis, dimension = mod.generate_basis(nsiti)
    #
    # # =========================Generate some useful operators===============
    # sz, s2 = mod.spin_matrices(basis, nso)
    #
    # number_operator = np.zeros((dimension, dimension, nso), dtype=complex)
    # for i in range(dimension):
    #     for j in range(nso):
    #         if mod.btest(basis[i], j):
    #             number_operator[i, i, j] = 1.0
    #
    # # site parameters
    # esite = np.zeros(nsiti, dtype=float)
    # u = np.zeros(nsiti, dtype=float)
    # esite[0] = 0.0
    # esite[3] = 0.0
    # esite[1] = -delta
    # esite[2] = +delta
    # u[0] = us
    # u[3] = us
    # u[1] = uhl
    # u[2] = uhl
    #
    # # scan ranges
    # jsl_value = np.linspace(-1,0.01, 20)
    # jhl_value = np.linspace(0.01, 1.0, 20)
    #
    # # open results file once (we will append inside loop with proper mode)
    # results_file = "results.txt"
    #
    # # total expected points
    # n_jsl = len(jsl_value)
    # n_jhl = len(jhl_value)
    #
    # # main scan
    # for idx_jsl in range(n_jsl):
    #     for idx_jhl in range(n_jhl):
    #         jsl = jsl_value[idx_jsl]
    #         jhl = jhl_value[idx_jhl]
    #
    #         # build J-matrix (symmetric)
    #         jmat = np.zeros((nsiti, nsiti), dtype=float)
    #         jmat[0, 2] = jsl
    #         jmat[2, 3] = jsl
    #         jmat[1, 2] = jhl
    #         jmat = jmat + jmat.T - np.diag(np.diag(jmat))
    #
    #         # =================Generate Hamiltonian================================
    #         hamiltonian = mod.hubbard_diagonal(nsiti, dimension, basis, esite, u)
    #
    #         # =================Hopping term========================================
    #         hop_mat = np.zeros((nsiti, nsiti), dtype=complex)
    #         hop_mat[0, 2] = t
    #         hop_mat[2, 3] = t
    #         # make Hermitian
    #         hop_mat = np.triu(hop_mat) + np.triu(hop_mat, 1).T
    #
    #         # lift to nso (spin-orbitals)
    #         hopping_nso = np.zeros((nso, nso), dtype=complex)
    #         for i in range(nso):
    #             for j in range(nso):
    #                 if (i % 2) == (j % 2):  # same spin index parity
    #                     hopping_nso[i, j] = hop_mat[i // 2, j // 2]
    #                 else:
    #                     hopping_nso[i, j] = 0.0 + 0.0j
    #
    #         hopping = mod.tb_to_rs(dimension, nso, basis, hopping_nso)
    #         hamiltonian -= hopping
    #
    #         # =================Exchange terms======================================
    #         # S·S block in local spin-pair basis [aa,ab,ba,bb]
    #         S0dotS1 = np.zeros((2, 2, 2, 2), dtype=float)
    #         S0dotS1[0, 0, 0, 0] = 0.25
    #         S0dotS1[0, 1, 0, 1] = -0.25
    #         S0dotS1[1, 0, 1, 0] = -0.25
    #         S0dotS1[1, 1, 1, 1] = 0.25
    #         S0dotS1[0, 1, 1, 0] = 0.50
    #         S0dotS1[1, 0, 0, 1] = 0.50
    #
    #         scambio_totale = mod.build_exchange_tensor(nso, jmat, S0dotS1)
    #         exchange = mod.bielectron(basis, nso, scambio_totale)
    #         hamiltonian += exchange
    #
    #         # =================Diagonalize Hamiltonian==============================
    #         if check_flag:
    #             plot.plot_heatmap_real(np.real(hopping), 'Hopping RS')
    #             plot.plot_heatmap_cplx(hamiltonian, 'Hamiltonian')
    #
    #         if mod.is_hermitian(hamiltonian, 1e-8):
    #             pass
    #         else:
    #             print('Hamiltonian is not Hermitian. Blocking the program')
    #             sys.exit()
    #
    #         eigenvalue, eigenvectors = eigh(hamiltonian)
    #         eigenvalue -= eigenvalue[0]  # set ground state to zero
    #
    #         sz_rot = mod.rotate_matrix(sz, eigenvectors, 1, dimension)
    #         s2_rot = mod.rotate_matrix(s2, eigenvectors, 1, dimension)
    #
    #         # =================Extract multiplets==================================
    #         E_sing, E_trip, E_quint = mod.extract_multiplet_energies(sz_rot, s2_rot, eigenvalue)
    #
    #         # --- apertura file (w la prima volta, a le altre) ---
    #         mode = "w" if first_time else "a"
    #         with open(results_file, mode) as f:
    #             if first_time:
    #                 f.write("jhl   jsl   E_sing   E_trip   E_quint\n")
    #             f.write(f"{jhl:6.3f}  {jsl:6.3f}  ")
    #             f.write(f"{E_sing:10.6f}  " if E_sing is not None else f"{'NA':>10}  ")
    #             f.write(f"{E_trip:10.6f}  " if E_trip is not None else f"{'NA':>10}  ")
    #             f.write(f"{E_quint:10.6f}\n" if E_quint is not None else f"{'NA':>10}\n")
    #
    #         first_time = False

    # =================Post-process results and build heatmaps===================
    data = np.loadtxt('results.txt', skiprows=1)

    # columns: jhl, jsl, E_sing, E_trip, E_quint
    jhl_all = data[:, 0]
    jsl_all = data[:, 1]
    E_sing_all = data[:, 2]
    E_trip_all = data[:, 3]
    E_quint_all = data[:, 4]

    st = E_sing_all - E_trip_all
    tq = E_trip_all - E_quint_all
    sq = E_sing_all - E_quint_all
    # Stack energies in shape (3, N)
    E = np.vstack([E_sing_all, E_trip_all, E_quint_all])  # shape (3, N)

    # Ranking per colonna: 1=min, 2=medio, 3=max
    rank_order = np.argsort(np.argsort(E, axis=0), axis=0) + 1

    # Estrai i tre vettori (ognuno lungo N)
    s = rank_order[0, :]  # ranking singlet
    t = rank_order[1, :]  # ranking triplet
    q = rank_order[2, :]  # ranking quintet
    with open('order.txt', 'w') as f:
        for i in range(len(s)):
         #   f.write(f"{jhl_all[i]:6.3f}  {jsl_all[i]:6.3f}  {E_sing_all[i]:6.3f} {E_trip_all[i]:6.3f} {E_quint_all[i]:6.3f}\n")
         f.write(
             f"{jhl_all[i]:6.3f}  {jsl_all[i]:6.3f}  {s[i]:6.3f} {t[i]:6.3f} {q[i]:6.3f}\n")

    # ---- prepara data_list CORRETTO ----
    # heatmaps_multiple vuole triplette (x, y, z)

    titles = ['Singlet', 'Triplet', 'Quintet']
    titles2 =['Gap Singlet-Triplet', 'Gap Triplet-Quintet', 'Gap Singlet-Quintet']

    plot.plot_three_heatmaps_from_points(jhl_all, jsl_all, s, t, q, titles=titles)
    list = [(jhl_all, jsl_all, st), (jhl_all, jsl_all, tq), (jhl_all, jsl_all, sq)]
    label = [('J HOMO-LUMO', 'J SOMO-LUMO'), ('J HOMO-LUMO', 'J SOMO-LUMO'), ('J HOMO-LUMO', 'J SOMO-LUMO')]
    plot.heatmaps_multiple(list, titles=titles2, labels=label, ncols=3)
if __name__ == "__main__":
    main()
