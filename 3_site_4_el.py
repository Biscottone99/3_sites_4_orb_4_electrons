import numpy as np
import sys
from scipy.linalg import eigh
import module as mod
import plotting_module as plot
import subprocess
if __name__ == "__main__":
    #================Constants============================================

    check_flag = False
    #================Initialize our system===================================================
    input = mod.read_input("input.inp", "output.txt")
    basis, dimension = mod.generate_basis(input["nsiti"])
    coords = np.zeros((input["nsiti"], 3))
    coords[0,0] = -input["length"]
    coords[3,0] = input["length"]
    with open('output.txt', 'a') as f:
        f.write("\n")
        f.write('\nAtomic Coordinates:\n')
        for i in range(input["nsiti"]):
            f.write(f'Atom {i + 1}: x={coords[i, 0]:.6f} y={coords[i, 1]:.6f} z={coords[i, 2]:.6f}\n')
        f.write("\n")
    nso = 2 * input["nsiti"]
    # =========================Generate some usefull operators================================
    sz, s2 = mod.spin_matrices(basis, nso)
    number_operator = np.zeros((dimension, dimension, nso), dtype=complex)
    for i in range(dimension):
        for j in range(nso):
            if mod.btest(basis[i], j):
                number_operator[i, i, j] = 1.0
    dipole_moment = np.zeros((dimension, dimension, 3), dtype=complex)
    for i in range(dimension):
        for j in range(0, nso - 2, 2):
            sito = j // 2
            for k in range(3):
                dipole_moment[i, i, k] += coords[sito, k] * (
                            input["nz"][sito] - (number_operator[i, i, j] + number_operator[i, i, j + 1]))

    #================Generate Hamiltonian===================================================
    hamiltonian = np.zeros((dimension,dimension), dtype = complex)
    hamiltonian = mod.hubbard_diagonal(input["nsiti"], dimension, basis, input["esite"], input["u"])

    hop_mat = np.zeros((input["nsiti"], input["nsiti"]), dtype = complex)
    hop_mat[0,1] = -input["t"]
    hop_mat[0,2] = -input["t"]
    hop_mat[1,3] = -input["t"]
    hop_mat[2,3] = -input["t"]

    hop_mat[1,2] = -input["J"]
    #hop_mat[0,3] = -input["J"]/2
    hop_mat = np.triu(hop_mat) + np.triu(hop_mat, 1).T
    hopping_nso = np.zeros((nso,nso))
    for i in range(nso):
        for j in range(nso):
            if i % 2 == j % 2:  # stessi "spin" o sottolivelli
                hopping_nso[i, j] = hop_mat[i // 2, j // 2]
            else:
                hopping_nso[i, j] = 0.0
    hopping = mod.tb_to_rs(dimension, nso, basis, hopping_nso)
    hamiltonian += hopping

    V_term = mod.compute_V_term(number_operator, coords, input, dimension)

    #hamiltonian += V_term

    with open('output.txt', 'a') as f:
        f.write('Hopping matrix between sites (eV):\n')
        for i in range(input["nsiti"]):
            for j in range(input["nsiti"]):
                f.write(f'{np.real(hop_mat[i,j]):.6f} ')
            f.write('\n')
        f.write('\n')


    #========================Diagonalize Hamiltonian=========================================
    #Let's check if is hermitian
    if check_flag:
        plot.plot_heatmap_real(np.real(hopping), 'Hopping RS')
        plot.plot_heatmap_cplx(hamiltonian,'Hamiltonian')
    bool = mod.is_hermitian(hamiltonian, 1e-8)
    if(bool):
        print('Hamiltonian is Hermitian')
        plot.plot_heatmap_cplx(hamiltonian,'Hamiltonian')
    else:
        print('Hamiltonian is not Hermitian')
        print('Blocking the program')
        sys.exit()

    eigenvalue, eigenvectors = eigh(hamiltonian)
    eigenvalue -= eigenvalue[0]
    D = np.diag(eigenvalue)
    res = eigenvectors.conj().T @ hamiltonian @ eigenvectors - D
    print("Residuo max:", np.max(np.abs(res)))
    sz_rot = mod.rotate_matrix(sz,eigenvectors,1,dimension)
    s2_rot = mod.rotate_matrix(s2,eigenvectors,1,dimension)
    number_operator_rot = mod.rotate_matrix(number_operator,eigenvectors,nso,dimension)
    rotated_dipole = mod.rotate_matrix(dipole_moment,eigenvectors,3,dimension)
    #========================Print some results=============================================
    with open('output.txt', 'a') as f:
        f.write('Eigenvalues (eV):\n')
        for i, (energy, sz, s2) in enumerate(zip(eigenvalue,
                                                 np.real(np.diag(sz_rot)),
                                                 np.real(np.diag(s2_rot))), start=1):
            f.write(f'{i:3d}) Energy: {energy:.6f} eV, <Sz>: {sz:.6f}, <S2>: {s2:.6f}\n')
        f.write('\n')

    with open('output.txt', 'a') as f:
        f.write('Number operator:\n')
        for i, row in enumerate(np.real(number_operator_rot[:20, :20, :]), start=1):
            row_str = " ".join(f"{x:.6f}" for x in row[i - 1, :])
            f.write(f"{i:3d}) State: {i - 1}, <n>: {row_str}\n")
        f.write('\n')

#========================Dynamic part===========================================================
sys.exit()
psi0 = rotated_dipole[0,:,0] #prima riga del vettore dipolo di transizione lungo z
psi0 /= np.linalg.norm(psi0)
density_matrix = np.outer(np.conj(psi0), psi0)
density_matrix.tofile('rho.bin')
spindensity = 0.5 * ( (number_operator_rot[:,:,nso-2] - number_operator_rot[:,:,nso-1]) - (number_operator_rot[:,:,0] - number_operator_rot[:,:,1]) )
spindensity.tofile('spindensity.bin')
eigenvalue.tofile('eigenvalues.bin')


rho_file = "rho.bin"
eigen_file = "eigenvalues.bin"
props_files = ["spindensity.bin"]  # lista dei file dei tensori di proprietà
n_prop = len(props_files)

subprocess.run(['ifx', 'Unitaria.f90', '-o', 'unitary.e', '-qmkl', '-qopenmp'], check=True)

program_input = "\n".join([
    str(dimension),
    rho_file,
    eigen_file,
    str(n_prop),
] + props_files + [
    str(input["deltat"]),
    str(input["points"])
]) + "\n"

subprocess.run(['./unitary.e'], input=program_input, text=True, check=True)
subprocess.run(['mv', 'prop1.dat', 'spinpol_evolution.dat'], check=True)
#========================End of unitary evolution===========================================
# Now we can plot some results from the unitary evolution
data = np.loadtxt('spinpol_evolution.dat', usecols=(0, 1))

time = data[:, 0] * 1e-3     # converto in ns
spinpol = data[:, 1]  * 50  # convert in percentual
plot.plot_curve(time, spinpol, 'Spin Polarization Evolution', 'Time (ns)', 'Spin Polarization (%)')
