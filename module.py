import numpy as np
import numpy as np
import os

import plotting_module as plot

imag = 1j
me = 9.1093837015e-31
gs = 2.00231930436256
e = 1.602176634e-19
e0 = 8.8541878128e-12
cl = 299792458
pf = ((gs * e ** 2) / (8 * np.pi * e0 * me * cl ** 2)) * 10.0e10
pauli_matrix = np.zeros((2, 2, 3), dtype=complex)
pauli_matrix[0, 1, 0] = 1.0
pauli_matrix[1, 0, 0] = 1.0

pauli_matrix[0, 1, 1] = -imag
pauli_matrix[1, 0, 1] = imag

pauli_matrix[0, 0, 2] = 1.0
pauli_matrix[1, 1, 2] = -1.0


def generate_basis(electrons):
    """
    Generate basis states in bit representation, arranged in order of increasing spin.

    Args:
        electrons (int): Number of electrons.

    Returns:
        configs (np.array): configurazioni in forma intera
        spins (np.array): spin associati
        nf (int): numero di configurazioni generate
    """
    # ==== Parametri ====
    nso = electrons * 2  # numero di spinorbitali

    # ==== Limiti del ciclo ====
    n_max = sum(2 ** i for i in range(nso - electrons, nso))
    n_min = sum(2 ** i for i in range(electrons))

    nf = 0
    usefull = []

    # ==== Loop principale ====
    for n in range(n_min, n_max + 1):
        count = 0
        a = 0  # spin-up
        b = 0  # spin-down
        array = ['0'] * nso

        for i in range(nso):
            if (n >> i) & 1:
                array[i] = '1'
                count += 1
                if i % 2 == 0:
                    a += 1
                else:
                    b += 1

        spin = 0.5 * (a - b)

        if count == electrons:
            config = sum(2 ** i for i in range(nso) if array[i] == '1')
            usefull.append((config, spin, ''.join(array)))
            nf += 1

    print(f"Numero totale di configurazioni valide: {nf}")

    # ==== Ordinamento per spin crescente ====
    usefull_sorted = sorted(usefull, key=lambda x: x[1])

    # ==== Creazione cartella se manca ====
    folder = '../hubbard_soc'
    os.makedirs(folder, exist_ok=True)

    # ==== Stampa su file ====
    with open(os.path.join(folder, 'configurations.txt'), 'w') as f3:
        for config, spin, array in usefull_sorted:
            f3.write(f"{array}  spin={spin:+.1f}  config={config}\n")

    # ==== Array NumPy da restituire ====
    configs = np.array([u[0] for u in usefull_sorted])

    return configs, nf
def btest(n, i):
    return ((n >> i) & 1) == 1
def ibset(n, i):
    return n | (1 << i)
def ibclr(n, i):
    return n & ~(1 << i)
def linear_search(arr, val):
    """
    Cerca un valore in una lista non ordinata.

    Args:
        arr (list): Lista in cui cercare.
        val: Valore da cercare.

    Returns:
        int: indice del valore se trovato, -1 altrimenti.
    """
    for i, v in enumerate(arr):
        if v == val:
            return i
    return -1
def tb_to_rs(dim, nso, basis, op_tb):
    op = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):  # indice di colonna
        for iso in range(nso):  # creation operator index
            for jso in range(nso):  # annihilation operator index
                jstate = basis[j]
                if btest(jstate, jso):
                    istate = ibclr(jstate, jso)
                    if not btest(istate, iso):
                        istate = ibset(istate, iso)
                        i = linear_search(basis, istate)
                        if i != -1:
                            # Calcolo fase
                            if jso != iso:
                                step = 1 if iso > jso else -1
                                conta = 0
                                for a in range(jso + step, iso, step):
                                    if btest(istate, a):
                                        conta += 1
                                phase = 1 if conta % 2 == 0 else -1
                            else:
                                phase = 1
                            # Aggiornamento matrice
                            op[i, j] += phase * op_tb[iso, jso]
    return op
def read_input(filename="input.inp", outputfile="output.txt"):
    """
    Formato atteso:

      nsiti
      length
      t
      (righe exchange: i j Jij)
      deltat
      points
      (righe per-sito: U  epsilon  nz)

    Le righe possono contenere commenti tipo: 4   #nsiti
    """

    def clean(line):
        """Rimuove eventuali commenti (#...) e spazi."""
        return line.split('#')[0].strip()

    # === Leggi file eliminando righe vuote/commentate ===
    with open(filename, "r") as f:
        raw = f.readlines()

    lines = [clean(line) for line in raw if clean(line)]

    # === Letture iniziali ===
    nsiti  = int(lines[0])
    length = float(lines[1])
    t      = float(lines[2])

    # === Lettura automatica righe exchange ===
    exchange_lines = []
    idx = 3   # dopo t iniziano le righe J

    while idx < len(lines):
        parts = lines[idx].split()

        # Le righe exchange hanno esattamente 3 campi numerici
        if len(parts) != 3:
            break

        # Verifica che siano numerici
        try:
            int(parts[0]); int(parts[1]); float(parts[2])
        except:
            break

        exchange_lines.append(parts)
        idx += 1

    # === Crea matrice simmetrica J ===
    Jmat = np.zeros((nsiti, nsiti))

    for line in exchange_lines:
        i, j, Jij = int(line[0]), int(line[1]), float(line[2])
        Jmat[i, j] = Jij
        Jmat[j, i] = Jij

    # === Continua con i parametri scalari ===
    deltat = int(lines[idx]); idx += 1
    points = int(lines[idx]); idx += 1

    # === Lettura per-sito ===
    u     = np.zeros(nsiti)
    esite = np.zeros(nsiti)
    nz    = np.zeros(nsiti, dtype=int)

    for s in range(nsiti):
        parts = lines[idx + s].split()
        u[s]     = float(parts[0])
        esite[s] = float(parts[1])
        nz[s]    = int(parts[2])
    soc_flag = float(lines[14])
    dynamic_flag = float(lines[15])
    # === Scrivi output ===
    with open(outputfile, "w") as out:
        out.write(f"Numero siti: {nsiti}\n")
        out.write(f"Lunghezza (Å): {length}\n")
        out.write(f"t (eV): {t}\n")
        out.write(f"Δt: {deltat}\n")
        out.write(f"Punti: {points}\n\n")
        out.write("=== Matrice di exchange J_ij (eV) ===\n")

        # stampa tabellata
        for row in Jmat:
            out.write("  ".join(f"{val:10.6f}" for val in row) + "\n")

        out.write("\n")

        out.write("=== Parametri per ogni sito ===\n")
        out.write('__________________________________________|\n')
        out.write("Idx |    U (eV)    |    ε (eV)    |   Z   |\n")
        out.write('----!--------------|--------------|-------|\n')

        for i in range(nsiti):
            out.write(
                f"{i + 1:3d} | "
                f"{u[i]:7.3f}      | "
                f"{esite[i]:7.3f}      | "
                f"{nz[i]:3d}   |\n"
            )
        out.write('__________________________________________|\n')

    # === Ritorna dati ===
    return {
        "nsiti": nsiti,
        "length": length,
        "t": t,
        "deltat": deltat,
        "points": points,
        "u": u,
        "esite": esite,
        "nz": nz,
        "Jmat": Jmat,
        "soc_flag": soc_flag,
        "dynamic_flag": dynamic_flag
    }
def hubbard_diagonal(nsiti, dimension, basis, esite, u):
    """Generate the diagonal part of the Hubbard Hamiltonian.
               Args:
                nso (int): Number of spin orbitals.
                dimension (int): Dimension of the basis.
                basis (array): Array of basis states in bit representation.
                esite (array): On-site energies.
                u (array): Hubbard U values for each site.

            Returns:
                H (array): Hamiltonian 2D array but with only diagonal elements filled.
            """
    H = np.zeros((dimension, dimension), dtype=complex)
    for i in range(dimension):
        occupazioni = np.zeros(nsiti)
        for j in range(nsiti):
            bool_bit_up = (basis[i] >> (2 * j)) & 1
            occupazioni[j] += bool_bit_up
            bool_bit_down = (basis[i] >> (2 * j + 1)) & 1
            occupazioni[j] += bool_bit_down

        # Inter-site Coulomb interactions
        H[i, i] += occupazioni @ esite
        for j in range(nsiti):
            if occupazioni[j] == 2:
                H[i, i] += u[j]
    return H
def hopping_matrix(nsiti, t, length, hop_flag, geom):
    """
    Generate the hopping matrix for the system.

    Args:
        nsiti (int): Number of sites.
        t (float): Hopping parameter.
        length (float): Length of the system.
        hop_flag (int): Flag to determine hopping type.
        geom (ndarray): Array of atomic coordinates, shape (nsiti, 3).

    Returns:
        hopping (ndarray): (2*nsiti, 2*nsiti) hopping matrix.
    """
    hop = np.zeros((nsiti, nsiti))
    distances = np.zeros((nsiti, nsiti))

    # --- Calcolo delle distanze ---
    for i in range(nsiti):
        for j in range(nsiti):
            if i != j:
                dist = np.linalg.norm(geom[i] - geom[j])
                distances[i, j] = dist
    # --- Costruzione matrice hop in base al flag ---
    if hop_flag == 1:
        for i in range(nsiti):
            for j in range(nsiti):
                if distances[i, j] <= 1.001 * length and i != j:
                    hop[i, j] = t

    elif hop_flag == 2:
        for i in range(nsiti):
            for j in range(nsiti):
                if distances[i, j] <= length and i != j:
                    hop[i, j] = t * np.exp(length - distances[i, j])

    elif hop_flag == 3:
        for i in range(nsiti):
            for j in range(nsiti):
                if (i == 2 and j == 3) or (i == 3 and j == 2):
                    hop[i, j] = t
                else:
                    hop[i, j] = 0.1 * t

    elif hop_flag == 4:
        for i in range(nsiti):
            for j in range(nsiti):
                if (i == 2 and j == 3) or (i == 3 and j == 2):
                    hop[i, j] = t * np.exp(length - distances[i, j])
                else:
                    hop[i, j] = 0.1 * t * np.exp(length - distances[i, j])
    # --- Espansione nella matrice completa 2*nsiti x 2*nsiti ---
    hopping = np.zeros((2 * nsiti, 2 * nsiti))
    for i in range(2 * nsiti):
        for j in range(2 * nsiti):
            if i % 2 == j % 2:  # stessi "spin" o sottolivelli
                hopping[i, j] = hop[i // 2, j // 2]
            else:
                hopping[i, j] = 0.0

    return hopping
def is_hermitian(H, tol=1e-12):
    return np.allclose(H, H.conj().T, atol=tol)
def hopping_siti_matrix(coord, nsiti, hop_flag, length, t):
    hop = np.zeros((nsiti, nsiti))
    distances = np.zeros((nsiti, nsiti))

    # --- Calcolo delle distanze ---
    for i in range(nsiti):
        for j in range(nsiti):
            if i != j:
                dist = np.linalg.norm(coord[i] - coord[j])
                distances[i, j] = dist
    # --- Costruzione matrice hop in base al flag ---
    if hop_flag == 1:
        for i in range(nsiti):
            for j in range(nsiti):
                if distances[i, j] <= 1.001 * length and i != j:
                    hop[i, j] = t

    elif hop_flag == 2:
        for i in range(nsiti):
            for j in range(nsiti):
                if distances[i, j] <= length and i != j:
                    hop[i, j] = t * np.exp(length - distances[i, j])

    elif hop_flag == 3:
        for i in range(nsiti):
            for j in range(nsiti):
                if i == 1  or j == 1 or i == nsiti or j == nsiti:
                    hop[i, j] = 0.1 * t
                else:
                    hop[i, j] = t

    elif hop_flag == 4:
        for i in range(nsiti):
            for j in range(nsiti):
                if i == 1  or j == 1 or i == nsiti or j == nsiti:
                    hop[i, j] = 0.1 * t * np.exp(length - distances[i, j])
                else:
                    hop[i, j] = t * np.exp(length - distances[i, j])

    return hop
def rotate_matrix(op, eigenvectors, strati, dimension):
    """
    Rotates operator 'op' in the eigenvector basis.

    Args:
        op: (dim, dim) operator matrix
        eigenvectors: (dim, dim) eigenvectors
        strati: number of layers (int)
        dimension: dimension of the system (int)

    Returns:
        op_rotated: (dim, dim, strati) rotated operator
    """
    # Crea array 3D per il risultato
    if strati != 1:
        op_rotated = np.zeros((dimension, dimension, strati), dtype=complex)

        # Rotazione: op_rotated[:,:,s] = V^dagger @ op @ V
        for s in range(strati):
            op_rotated[:, :, s] = np.conj(eigenvectors.T) @ op[:, :, s] @ eigenvectors
    else:
        op_rotated = np.zeros((dimension, dimension), dtype=complex)
        op_rotated = np.conj(eigenvectors.T) @ op @ eigenvectors

    return op_rotated
def compute_soc_mono(basis, coord, nsiti, hop_flag, length, t):
    hop = hopping_siti_matrix(coord, nsiti, hop_flag, length, t)
    mom = np.zeros((nsiti, nsiti, 3), dtype=complex)
    for i in range(nsiti):
        for j in range(nsiti):
            for k in range(3):
                dr = coord[i, k] - coord[j, k]
                mom[i, j, k] = imag * dr * hop[i, j]

    soc_mono = np.zeros((2 * nsiti, 2 * nsiti), dtype=complex)

    for i in range(2 * nsiti):
        sitoi = i // 2
        for j in range(2 * nsiti):
            sitoj = j // 2
            for atomo in range(nsiti):
                if (atomo != sitoj) and (atomo != sitoi):
                    term1 = np.cross(coord[sitoi] - coord[atomo], mom[sitoi, sitoj]) / np.linalg.norm(
                        coord[sitoi] - coord[atomo]) ** 3
                    term2 = np.cross(coord[sitoj] - coord[atomo], mom[sitoi, sitoj]) / np.linalg.norm(
                        coord[sitoj] - coord[atomo]) ** 3
                    if i % 2 == 0:
                        si = 0
                    else:
                        si = 1
                    if j % 2 == 0:
                        sj = 0
                    else:
                        sj = 1
                    soc_mono[i, j] = pf * 0.5 * (term1 + term2) @ pauli_matrix[si, sj, :]

    one_electron_soc = tb_to_rs(len(basis), 2 * nsiti, basis, soc_mono)
    return one_electron_soc
def bielectron(basis, nso, op):
    """
    Costruisce la matrice dell'operatore bielettronico su una base determinante.
    basis : lista di determinanti (bitstring interi)
    nso   : numero di spinorbitali
    op    : tensore bielettronico (a,b,c,d) con convenzione ⟨ab|cd⟩
    """
    dim = len(basis)
    op_rs = np.zeros((dim, dim), dtype=complex)

    for n, det_n in enumerate(basis):
        # ciclo su indici dell'operatore a_a† a_b† a_c a_d
        for a in range(nso):
            for b in range(nso):
                for c in range(nso):
                    for d in range(nso):
                        # controlla se c e d sono occupati nel determinante di partenza
                        if btest(det_n, d) and btest(det_n, c):
                            # rimuovi d, c
                            temp = ibclr(ibclr(det_n, d), c)

                            # controlla se a e b sono vuoti
                            if not btest(temp, a) and not btest(temp, b):
                                # aggiungi b, a
                                temp2 = ibset(ibset(temp, b), a)

                                # calcola la fase da permutazione fermionica
                                perm = 0
                                for i in range(min(a, b) + 1, max(a, b)):
                                    if btest(temp, i):
                                        perm += 1
                                for i in range(min(c, d) + 1, max(c, d)):
                                    if btest(det_n, i):
                                        perm += 1
                                phase = 1 if perm % 2 == 0 else -1

                                # trova l'indice m del nuovo determinante
                                m = linear_search(basis, temp2)
                                if m != -1:
                                    op_rs[m, n] += phase * op[a, b, c, d]

    return op_rs
def compute_sso(basis, coord, nsiti, hop_flag, length, t):
    hop = hopping_siti_matrix(coord, nsiti, hop_flag, length, t)
    mom = np.zeros((nsiti, nsiti, 3), dtype=complex)
    for i in range(nsiti):
        for j in range(nsiti):
            for k in range(3):
                dr = coord[i, k] - coord[j, k]
                mom[i, j, k] = imag * dr * hop[i, j]

    soc_bi = np.zeros((2 * nsiti, 2 * nsiti, 2 * nsiti, 2 * nsiti), dtype=complex)

    for a in range(2 * nsiti):
        for b in range(2 * nsiti):
            for c in range(2 * nsiti):
                asito = a // 2
                bsito = b // 2
                csito = c // 2
                if asito != bsito and bsito != csito:
                    term1 = np.cross(coord[asito] - coord[bsito], mom[asito, csito]) / np.linalg.norm(
                        coord[asito] - coord[bsito]) ** 3
                    term2 = np.cross(coord[csito] - coord[bsito], mom[asito, csito]) / np.linalg.norm(
                        coord[csito] - coord[bsito]) ** 3
                    if a % 2 == 0:
                        sa = 0
                    else:
                        sa = 1
                    if c % 2 == 0:
                        sc = 0
                    else:
                        sc = 1
                    soc_bi[a, b, c, b] += pf * 0.5 * (term1 + term2) @ pauli_matrix[sa, sc, :]
    two_electron_soc = bielectron(basis, 2 * nsiti, soc_bi)
    return two_electron_soc
def compute_soo(basis, coord, nsiti, hop_flag, length, t):
    hop = hopping_siti_matrix(coord, nsiti, hop_flag, length, t)
    mom = np.zeros((nsiti, nsiti, 3), dtype=complex)
    for i in range(nsiti):
        for j in range(nsiti):
            for k in range(3):
                dr = coord[i, k] - coord[j, k]
                mom[i, j, k] = imag * dr * hop[i, j]

    soc_bi = np.zeros((2 * nsiti, 2 * nsiti, 2 * nsiti, 2 * nsiti), dtype=complex)

    for a in range(2 * nsiti):
        for b in range(2 * nsiti):
            for c in range(2 * nsiti):
                for d in range(2 * nsiti):
                    asito = a // 2
                    bsito = b // 2
                    csito = c // 2
                    dsito = d // 2
                    if a % 2 == 0:
                        sa = 0
                    else:
                        sa = 1
                    if b % 2 == 0:
                        sb = 0
                    else:
                        sb = 1
                    if c % 2 == 0:
                        sc = 0
                    else:
                        sc = 1
                    if d % 2 == 0:
                        sd = 0
                    else:
                        sd = 1
                    if sa == sc and bsito == dsito:
                        if asito != bsito and csito != bsito:
                            term1 = np.cross(coord[asito] - coord[bsito], mom[asito, csito]) / np.linalg.norm(coord[asito] - coord[bsito]) ** 3
                            term2 = np.cross(coord[csito] - coord[bsito], mom[asito, csito]) / np.linalg.norm(coord[csito] - coord[bsito]) ** 3
                            soc_bi[a, b, c, d] += pf * (term1 + term2) @ pauli_matrix[sb, sd, :]

    two_electron_soc = bielectron(basis, 2 * nsiti, soc_bi)
    return two_electron_soc
def apply_annihilation(state, i):
    """Applica c_i sul determinante. Ritorna (new_state, phase) oppure (None, 0) se annulla."""
    if not btest(state, i):
        return None, 0  # stato vuoto → annulla
    # segno = (-1)^(numero elettroni prima del sito i)
    phase = (-1)**(bin(state & ((1 << i)-1)).count("1"))
    new_state = ibclr(state, i)
    return new_state, phase
def apply_creation(state, i):
    """Applica c_i^† sul determinante. Ritorna (new_state, phase) oppure (None, 0)."""
    if btest(state, i):
        return None, 0  # già occupato → annulla
    phase = (-1)**(bin(state & ((1 << i)-1)).count("1"))
    new_state = ibset(state, i)
    return new_state, phase
def spin_matrices(configs, nso):
    """
    Costruisce S_z, S^+, S^-, S^2 usando operatori scaletta in base di configurazioni.
    """
    # 🔧 fix: configs deve essere una lista per usare .index()
    configs = list(configs)

    dim = len(configs)

    Sz = np.zeros((dim, dim), complex)
    Sp = np.zeros((dim, dim), complex)
    Sm = np.zeros((dim, dim), complex)
    S2 = np.zeros((dim, dim), complex)

    # ----------------------------
    #     S_z (diagonale)
    # ----------------------------
    for a, state in enumerate(configs):
        n_up = 0
        n_down = 0
        for i in range(0, nso, 2):
            if btest(state, i):     n_up += 1
            if btest(state, i+1):   n_down += 1
        Sz[a, a] = 0.5 * (n_up - n_down)

    # ----------------------------
    #     S^+ = sum_i c†_{i↑} c_{i↓}
    # ----------------------------
    for a, state in enumerate(configs):
        for site in range(nso // 2):
            down = 2*site + 1
            up   = 2*site

            state2, ph1 = apply_annihilation(state, down)
            if state2 is None:
                continue

            state3, ph2 = apply_creation(state2, up)
            if state3 is None:
                continue

            try:
                b = configs.index(state3)
                Sp[b, a] += ph1 * ph2
            except ValueError:
                pass

    # ----------------------------
    #     S^- = sum_i c†_{i↓} c_{i↑}
    # ----------------------------
    for a, state in enumerate(configs):
        for site in range(nso // 2):
            down = 2*site + 1
            up   = 2*site

            state2, ph1 = apply_annihilation(state, up)
            if state2 is None:
                continue

            state3, ph2 = apply_creation(state2, down)
            if state3 is None:
                continue

            try:
                b = configs.index(state3)
                Sm[b, a] += ph1 * ph2
            except ValueError:
                pass

    # ----------------------------
    #     S^2 = S^- S^+ + S_z(S_z + 1)
    # ----------------------------
    S2 = Sm @ Sp + Sz @ (Sz + np.identity(dim))

    return Sz, S2
def compute_V_term(number_operator, coords, input_data, dimension):
    """
    Calcola V_term per una configurazione elettronica.
    number_operator : array (dimension, dimension, n_orb)
    coords          : array (nsiti, 3)
    input_data      : dict con chiavi: nsiti, nz, u
    dimension       : dimensione della matrice di Hamiltoniana
    """

    # ----- Occupation numbers -----
    occupation = np.zeros((dimension, 3), dtype=complex)

    occupation[:, 0] = (
        number_operator[:, :, 0].diagonal() +
        number_operator[:, :, 1].diagonal()
    )

    occupation[:, 1] = (
        number_operator[:, :, 2].diagonal() +
        number_operator[:, :, 3].diagonal() +
        number_operator[:, :, 4].diagonal() +
        number_operator[:, :, 5].diagonal()
    )

    occupation[:, 2] = (
        number_operator[:, :, 6].diagonal() +
        number_operator[:, :, 7].diagonal()
    )

    # ----- Reduced coords -----
    reduced_coord = np.zeros((input_data["nsiti"] - 1, 3))
    reduced_coord[0, :] = coords[0, :]
    reduced_coord[1, :] = coords[1, :]
    reduced_coord[2, :] = coords[3, :]

    # ----- Reduced nz -----
    nz = np.zeros(input_data["nsiti"] - 1)
    nz[0] = input_data["nz"][0]
    nz[1] = input_data["nz"][1]
    nz[2] = input_data["nz"][3]

    # ----- V_term -----
    V_term = np.zeros((dimension, dimension), dtype=complex)

    for i in range(dimension):
        for j in range(input_data["nsiti"] - 1):
            for k in range(input_data["nsiti"] - 1):
                if k != j:

                    dist = np.linalg.norm(reduced_coord[k] - reduced_coord[j])
                    vjk = 14.397 / dist  # eV·Å

                    ppp = (
                        28.794 / (input_data["u"][k] + input_data["u"][j]) ** 2
                    ) * (
                        (input_data["u"][j] - occupation[i, j]) *
                        (input_data["u"][k] - occupation[i, k])
                    )

                    V_term[i, i] += 0.5 * (ppp + vjk)

    return V_term
def spin(i):
    return i % 2  # mapping spinorbitale → spin (0=↑,1=↓)
def build_exchange_tensor(nso, Jmat, S0dotS1):
    """
    Costruisce il tensore di scambio tra siti nearest neighbor.

    nso       : numero di spinorbitali (2 * nsiti)
    Jmat      :Matrice dei coupling
    S0dotS1   : matrice 2x2x2x2 dei prodotti di spin
    nn_pairs  : lista di coppie di siti nearest-neighbor, es. [(0,1),(1,2),(2,3)]

    Ritorna: scambio_totale (nso,nso,nso,nso)
    """
    # Trova gli indici (i, j) del triangolo superiore con valore J ≠ 0
    nn_pairs = [(i, j) for i, j in zip(*np.triu_indices(nso//2, k=1)) if Jmat[i, j] != 0]
    scambio_totale = np.zeros((nso, nso, nso, nso), dtype=float)

    for (a, b) in nn_pairs:  # loop sui siti nearest-neighbor
        for i in range(2 * a, 2 * a + 2):  # spinorbitali del sito a
            si = spin(i)
            for j in range(2 * b, 2 * b + 2):  # spinorbitali del sito b
                sj = spin(j)
                for k in range(2 * a, 2 * a + 2):  # "annihilation" sul sito a
                    sk = spin(k)
                    for d in range(2 * b, 2 * b + 2):  # "annihilation" sul sito b
                        sd = spin(d)

                        # contributo di scambio
                        scambio_totale[i, j, k, d] += Jmat[a,b]* S0dotS1[si, sj, sk, sd]

                # simmetria a <-> b per garantire hermiticità
                for k in range(2 * b, 2 * b + 2):
                    sk = spin(k)
                    for d in range(2 * a, 2 * a + 2):
                        sd = spin(d)
                        scambio_totale[i, j, k, d] += Jmat[a,b] * S0dotS1[si, sj, sk, sd]

    return scambio_totale
def extract_multiplet_energies(sz_rot, s2_rot, eigenvalue):
    """Ritorna le energie del 2° singoletto Sz=0, 2° tripletto Sz=0, 1° quintetto Sz=0."""

    diag_sz = np.diagonal(sz_rot)
    diag_s2 = np.diagonal(s2_rot)

    # maschere
    sing_mask  = np.isclose(diag_sz, 0.0) & np.isclose(diag_s2, 0.0)
    trip_mask  = np.isclose(diag_sz, 0.0) & np.isclose(diag_s2, 2.0)
    quint_mask = np.isclose(diag_sz, 0.0) & np.isclose(diag_s2, 6.0)

    # indici
    idx_sing  = np.where(sing_mask)[0]
    idx_trip  = np.where(trip_mask)[0]
    idx_quint = np.where(quint_mask)[0]

    # energie (None se non esistono abbastanza stati)
    E_sing  = eigenvalue[idx_sing[1]]  if len(idx_sing)  > 1 else None  # 2° singoletto
    E_trip  = eigenvalue[idx_trip[1]]  if len(idx_trip)  > 1 else None  # 2° tripletto
    E_quint = eigenvalue[idx_quint[0]] if len(idx_quint) > 0 else None  # 1° quintetto

    return E_sing, E_trip, E_quint
