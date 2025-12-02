import numpy as np
import sys
from scipy.linalg import eigh
import module as mod
import plotting_module as plot
import subprocess
if __name__ == "__main__":
    #================Constants============================================
    pauli_matrix = np.zeros((2, 2, 3), dtype=complex)
    pauli_matrix[0, 1, 0] = 1.0
    pauli_matrix[1, 0, 0] = 1.0

    pauli_matrix[0, 1, 1] = -1j
    pauli_matrix[1, 0, 1] = 1j

    pauli_matrix[0, 0, 2] = 1.0
    pauli_matrix[1, 1, 2] = -1.0
    magnetic_field = 0.0
    check_flag = False

    #================Initialize our system===================================================
    input = mod.read_input("input.inp", "output.txt")
    basis, dimension = mod.generate_basis(input["nsiti"])
    coords = np.zeros((input["nsiti"], 3))
    coords[0,0] = -input["length"]
    coords[3,0] = input["length"]
    with open('output.txt', 'a') as f:
        f.write("\n")
        f.write('Atomic Coordinates\n')
        f.write("----------------------------------------------------|\n")
        f.write("Idx |       x       |       y       |       z       |\n")
        f.write("----------------------------------------------------|\n")

        for i in range(input["nsiti"]):
            f.write(
                f"{i + 1:3d} | "
                f"{coords[i, 0]:12.6f}  | "
                f"{coords[i, 1]:12.6f}  | "
                f"{coords[i, 2]:12.6f}  |\n"
            )
        f.write("----------------------------------------------------|\n")

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

    #================Generate Hamiltonian============================================================#
    hamiltonian = np.zeros((dimension,dimension), dtype = complex)                                   #
    hamiltonian = mod.hubbard_diagonal(input["nsiti"], dimension, basis, input["esite"], input["u"]) #
    #=================Hopping term===================================================================#
    hop_mat = np.zeros((input["nsiti"], input["nsiti"]), dtype = complex)
    #hop_mat[0,1] = +input["t"]
    hop_mat[0,2] = +input["t"]
    #hop_mat[1,3] = +input["t"]
    hop_mat[2,3] = +input["t"]

    hop_mat = np.triu(hop_mat) + np.triu(hop_mat, 1).T
    hopping_nso = np.zeros((nso,nso))
    for i in range(nso):
        for j in range(nso):
            if i % 2 == j % 2:  # stessi "spin" o sottolivelli
                hopping_nso[i, j] = hop_mat[i // 2, j // 2]
            else:
                hopping_nso[i, j] = 0.0
    hopping = mod.tb_to_rs(dimension, nso, basis, hopping_nso)
    hamiltonian -= hopping
    #====================V term ===================================================================#
    # V_term = mod.compute_V_term(number_operator, coords, input, dimension)                         #
    #hamiltonian += V_term                                                                          #
    #=============================Magnetic Field===================================================#
    for i in range(dimension):
        hamiltonian[i,i] += -magnetic_field * np.sqrt(s2[i,i])
    #===================Exchange terms=============================================================#
    S0dotS1 = np.zeros((2,2,2,2), dtype = float)
    S0dotS1[0,0,0,0] = 0.25
    S0dotS1[0,1,0,1] = -0.25
    S0dotS1[1,0,1,0] = -0.25
    S0dotS1[1,1,1,1] = 0.25
    S0dotS1[0,1,1,0] = 0.50
    S0dotS1[1,0,0,1] = 0.50
    scambio_totale = mod.build_exchange_tensor(nso, input["Jmat"], S0dotS1)

    exchange = mod.bielectron(basis,nso,scambio_totale)
    hamiltonian += exchange
    # ===============================SOC==============================================================#
    if input["soc_flag"] == 1:
        soc_so = np.zeros((nso, nso), dtype=complex)
        for i in range(0, nso - 2, 2):
            if i != 2:
                soc_so[i, i + 3] = -input["t"] * 1j * 3.94e-4
                soc_so[i + 1, i + 2] = -input["t"] * 1j * 3.94e-4
        soc_so = np.triu(soc_so) + np.conj(np.triu(soc_so, 1).T)
        if check_flag:
            plot.plot_heatmap_cplx(soc_so, 'SOC')
        soc_mono = mod.tb_to_rs(dimension, nso, basis, soc_so)
        hamiltonian += soc_mono*1
     #===========================Check=============================================================
    with open('check.txt', 'w') as w:
        for i in range(dimension-1):
            for j in range(i+1,dimension):
                if np.abs(np.real(hamiltonian[i,j]))>1e-8:
                    w.write(f'{i} {j} {np.real(hamiltonian[i,j]):.6f}\n')

    #==============================================================================================#
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
    sz_rot = mod.rotate_matrix(sz,eigenvectors,1,dimension)
    s2_rot = mod.rotate_matrix(s2,eigenvectors,1,dimension)
    number_operator_rot = mod.rotate_matrix(number_operator,eigenvectors,nso,dimension)
    rotated_dipole = mod.rotate_matrix(dipole_moment,eigenvectors,3,dimension)
    #========================Print some results=============================================
    with open('output.txt', 'a') as f:
        header = ( "Idx|   Energy (eV)      <Sz>        <S2>\n")
        f.write(header)
        f.write('-------------------------------------------\n')
        for i, (energy, sz, s2) in enumerate(zip(eigenvalue[:20],
                                                 np.real(np.diag(sz_rot[:20,:20])),
                                                 np.real(np.diag(s2_rot[:20,:20]))), start=1):
            f.write(f'{i:3d}) {energy:12.6f} {sz:12.6f} {s2:12.6f}\n')
        f.write('\n')

    with open('output.txt', 'a') as f:
        f.write('Number operator:\n')

        # Header
        header = (
            "Idx |"
            "    α SOMO      β SOMO  |"
            "    α HOMO      β HOMO  |"
            "    α LUM0      β LUMO  |"
            "    α SOMO      β SOMO  |\n"
        )
        f.write(header)
        f.write('_________________________________________________________________________________________________________\n')
        # Righe
        for i in range(min(20, number_operator_rot.shape[0])):
            vals = np.real(number_operator_rot[i, i, :])  # 8 valori αβαβαβαβ
            # Formattazione 4 blocchi da 2 numeri
            blocks = []
            for b in range(0, 8, 2):
                blocks.append(f"{vals[b]:10.6f}  {vals[b + 1]:10.6f}")
            line = f"{i + 1:3d} | " + " | ".join(blocks) + " |\n"
            f.write(line)

        f.write('\n')

    # Calcolo delle cariche per ogni stato e per ogni sito
    cariche = np.zeros((dimension, input["nsiti"]-1))

    for i in range(dimension):
        # Sito 0 → orbitali 0,1
        cariche[i, 0] = input["nz"][0] - np.real((number_operator_rot[i, i, 0] +
                                          number_operator_rot[i, i, 1]))

        # Sito 1 → orbitali 2,3,4,5
        cariche[i, 1] = (input["nz"][1]+input["nz"][2]) - np.real((number_operator_rot[i, i, 2] +
                                          number_operator_rot[i, i, 3] +
                                          number_operator_rot[i, i, 4] +
                                          number_operator_rot[i, i, 5]))

        # Sito 2 → orbitali 6,7
        cariche[i, 2] = input["nz"][3] - np.real((number_operator_rot[i, i, 6] +
                                          number_operator_rot[i, i, 7]))

    # Scrittura del file
    with open('output.txt', 'a') as f:
        f.write('Charges\n')

        # Header
        header = (
            "Idx |"
            "      SOMO    |"
            "     BRIDGE   |"
            "      SOMO    |\n"
        )
        f.write(header)
        f.write('__________________________________________________\n')
        # Righe
        for idx_state in range(min(20, dimension)):
            vals = np.real(cariche[idx_state, :])  # array di 4 valori
            # blocchi ben allineati
            blocks = []
            for v in vals:
                blocks.append(f"{v:12.6f}")
            line = f"{idx_state + 1:3d} | " + " | ".join(blocks) + " |\n"
            f.write(line)

        f.write('\n')

    with open('output.txt', 'a') as f:
        f.write('Transition dipole moments\n')
        f.write('Idx   <Sz>        <S2>    Dipole(S0→State)\n')
        f.write('-------------------------------------------\n')

        for idx_state in range(min(20, dimension)):
            sz_val = np.real(sz_rot[idx_state, idx_state])
            s2_val = np.real(s2_rot[idx_state, idx_state])
            dip_val = (np.real(rotated_dipole[0, idx_state,0]))

            f.write(f"{idx_state + 1:3d})  {sz_val:7.3f}   {s2_val:7.3f}   {dip_val:12.6f}\n")

        f.write('\n')

#========================Dynamic part===========================================================
if input["dynamic_flag"] == 2:
    sys.exit()
if input["dynamic_flag"] == 0:
    psi0 = rotated_dipole[0,:,0] #prima riga del vettore dipolo di transizione lungo x
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
if ["dynamic_flag"] == 1:
    if input["dynamic_flag"] == 1:
        treshold = 0.02
    for i in range(dimension):
        if np.real(rotated_dipole[0,i,0])>treshold:
            dimension_reduced = i
    psi0 = rotated_dipole[0,:dimension_reduced,0]
    psi0 = psi0/np.linalg.norm(psi0)
    hopping_rotated = mod.rotate_matrix(-hopping,eigenvectors,1,dimension)
    spindensity = 0.5 * ( (number_operator_rot[:,:,nso-2] - number_operator_rot[:,:,nso-1]) - (number_operator_rot[:,:,0] - number_operator_rot[:,:,1]) )
    with open('input-red/system_input.dat', 'w') as f:
        f.write(f"{dimension_reduced}\n")
        f.write(f"{input["nsiti"]}\n")
    eigenvalue[:dimension_reduced].tofile('input-red/eigenvalues.bin')
    psi0.tofile('input-red/psi0.bin')
    spindensity[:dimension_reduced,:dimension_reduced].tofile('input-red/spindensity.bin')
    hopping_rotated[:dimension_reduced,:dimension_reduced].tofile('input-red/op1.bin')
    subprocess.run(['ifx', 'red.f90', '-o', 'red.e', '-qmkl', '-qopenmp'], check=True)
    subprocess.run(['./red.e'])
#========================End of unitary evolution===========================================
# Now we can plot some results from the unitary evolution
if input["dynamic_flag"] == 0:
    data = np.loadtxt('spinpol_evolution.dat', usecols=(0, 1))
    time = data[:, 0] * 1e-3     # converto in ns
    spinpol = data[:, 1] * 100
    plot.plot_curve(time, spinpol, 'Spin Polarization Evolution', 'Time (ns)', 'Spin Polarization (%)')
if input["dynamic_flag"] == 1:
    data = np.loadtxt('dynamic-results/properties.dat')
    time = data[:, 0]
    energy = data[:,1]
    spinpol = data[:, 2] * 100  # convert in percentual
    plot.plot_curve(time, energy, 'Energy Evolution', 'Time (ns)', 'Energy (eV)')
    plot.plot_curve(time, spinpol, 'Spin Polarization Evolution', 'Time (ns)', 'Spin Polarization (%)')
