import numpy as np
import plotting_module as plot
import module as mod
from scipy.linalg import block_diag
import math

from theodore.units import energy


def costruisci_hamiltoniana_gerade(e_H, e_L, U_H, U_L, U_S, J_HL, J_SL, t):
    """
    Costruisce la matrice Hamiltoniana singoletto gerade 12x12.
    """

    # 1. Definiamo le energie diagonali (D1 ... D12)
    D1 = 2 * e_H + U_H
    D2 = e_H + e_L - (3 / 4) * J_HL
    D3 = e_H + U_S
    D4 = 2 * e_H + e_L + U_H - (3 / 4) * J_SL
    D5 = 2 * e_H + U_H + U_S
    D6 = 2 * e_L + U_L
    D7 = e_L + U_S - (3 / 4) * J_SL
    D8 = 2 * U_S
    D9 = e_H + 2 * e_L + U_L
    D10 = e_H + e_L + U_S - (3 / 4) * J_HL
    D11 = 2 * e_H + 2 * e_L + U_H + U_L
    D12 = 2 * e_L + U_L + U_S

    # 2. Inizializziamo la matrice 12x12 con zeri
    H = np.zeros((12, 12))

    # Costante utile
    sqrt2 = np.sqrt(2)

    # 3. Assegniamo gli elementi sulla diagonale principale
    H[0, 0] = D1
    H[1, 1] = D2
    H[2, 2] = D3
    H[3, 3] = D4
    H[4, 4] = D5
    H[5, 5] = D6
    H[6, 6] = D7
    H[7, 7] = D8
    H[8, 8] = D9
    H[9, 9] = D10
    H[10, 10] = D11
    H[11, 11] = D12

    # 4. Assegniamo gli elementi fuori diagonale (sfruttando la simmetria della matrice)

    # Riga 1 (Indice 0)
    H[0, 3] = -sqrt2 * t;
    H[3, 0] = H[0, 3]

    # Riga 2 (Indice 1)
    H[1, 2] = t;
    H[2, 1] = H[1, 2]
    H[1, 8] = t;
    H[8, 1] = H[1, 8]

    # Riga 3 (Indice 2)
    H[2, 9] = -t;
    H[9, 2] = H[2, 9]

    # Riga 4 (Indice 3)
    H[3, 4] = -sqrt2 * t;
    H[4, 3] = H[3, 4]
    H[3, 10] = -2 * t;
    H[10, 3] = H[3, 10]

    # Riga 6 (Indice 5)
    H[5, 6] = sqrt2 * t;
    H[6, 5] = H[5, 6]

    # Riga 7 (Indice 6)
    H[6, 7] = -2 * t;
    H[7, 6] = H[6, 7]
    H[6, 11] = -sqrt2 * t;
    H[11, 6] = H[6, 11]

    # Riga 9 (Indice 8)
    H[8, 9] = t;
    H[9, 8] = H[8, 9]

    return H
def costruisci_hamiltoniana_ungerade(e_H, e_L, U_H, U_L, U_S, J_HL, J_SL, t):
    """
    Costruisce la matrice Hamiltoniana singoletto ungerade 8x8.
    """

    # 1. Definiamo le energie diagonali (F1 ... F8)
    F1 = e_H + U_S
    F2 = 2 * e_H + e_L + U_H - (3 / 4) * J_SL
    F3 = 2 * e_H + U_H + U_S
    F4 = e_H + e_L + (1 / 4) * J_HL - J_SL
    F5 = e_L + U_S - (3 / 4) * J_SL
    F6 = e_H + 2 * e_L + U_L
    F7 = e_H + e_L + U_S - (3 / 4) * J_HL
    F8 = 2 * e_L + U_L + U_S

    # 2. Inizializziamo la matrice 8x8 con zeri
    H = np.zeros((8, 8))

    # Costanti utili
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)

    # 3. Assegniamo gli elementi sulla diagonale principale
    H[0, 0] = F1
    H[1, 1] = F2
    H[2, 2] = F3
    H[3, 3] = F4
    H[4, 4] = F5
    H[5, 5] = F6
    H[6, 6] = F7
    H[7, 7] = F8

    # 4. Assegniamo gli elementi fuori diagonale (sfruttando la simmetria della matrice)

    # Riga 1 (Indice 0)
    H[0, 3] = sqrt3 * t;
    H[3, 0] = H[0, 3]
    H[0, 6] = -t;
    H[6, 0] = H[0, 6]

    # Riga 2 (Indice 1)
    H[1, 2] = -sqrt2 * t;
    H[2, 1] = H[1, 2]

    # Riga 4 (Indice 3) (H[3, 0] già assegnato)
    H[3, 5] = -sqrt3 * t;
    H[5, 3] = H[3, 5]

    # Riga 5 (Indice 4)
    H[4, 7] = -sqrt2 * t;
    H[7, 4] = H[4, 7]

    # Riga 6 (Indice 5) (H[5, 3] già assegnato)
    H[5, 6] = t;
    H[6, 5] = H[5, 6]

    return H
def costruisci_hamiltoniana_tripletto_gerade(e_H, e_L, U_H, U_L, U_S, J_HL, J_SL, t):
    """
    Costruisce la matrice Hamiltoniana tripletto gerade 5x5.
    (Nota: alcuni parametri come U_L potrebbero non essere usati in questo blocco,
    ma li manteniamo nella firma per coerenza con le altre funzioni).
    """

    # 1. Definiamo le energie base (EG, EA, EB, EC)
    E_G = 2 * e_H + U_H
    E_A = e_H + U_S
    E_B = 2 * e_H + e_L + U_H
    E_C = e_H + e_L

    # 2. Inizializziamo la matrice 5x5 con zeri
    H = np.zeros((5, 5))

    # Costanti utili
    sqrt2 = np.sqrt(2)

    # 3. Assegniamo gli elementi sulla diagonale principale
    H[0, 0] = E_G
    H[1, 1] = E_B + (1 / 4) * J_SL
    H[2, 2] = E_A
    H[3, 3] = E_C - (3 / 4) * J_HL
    H[4, 4] = E_C + (1 / 4) * J_HL - (1 / 2) * J_SL

    # 4. Assegniamo gli elementi fuori diagonale (sfruttando la simmetria)

    # Blocco 2x2 in alto a sinistra
    H[0, 1] = -sqrt2 * t;
    H[1, 0] = H[0, 1]

    # Blocco 3x3 in basso a destra
    H[2, 3] = -t;
    H[3, 2] = H[2, 3]
    H[2, 4] = sqrt2 * t;
    H[4, 2] = H[2, 4]

    H[3, 4] = (1 / sqrt2) * J_SL;
    H[4, 3] = H[3, 4]

    return H
def ordina_matrice_per_diagonale(H):
    """
    Riordina una matrice quadrata in modo che gli elementi sulla
    diagonale principale siano in ordine crescente.
    """

    # 1. Estraiamo la diagonale
    diagonale = np.diag(H)

    # 2. Otteniamo gli indici che ordinerebbero la diagonale
    # np.argsort restituisce un array di indici posizionali
    indici_ordinati = np.argsort(diagonale)

    # 3. Applichiamo il riordinamento a righe e colonne
    # Prima ordiniamo le righe [indici_ordinati, :],
    # poi le colonne [:, indici_ordinati]
    H_ordinata = H[indici_ordinati, :][:, indici_ordinati]

    return H_ordinata, indici_ordinati

# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Parametri di test arbitrari (sostituiscili con i tuoi valori fisici)
    parametri = {
        "e_H": -6.0,
        "e_L": 6.0,
        "U_H": 8.0,
        "U_L": 6.0,
        "U_S": 10.0,
        "J_HL": -0.0,
        "J_SL": -1.0,
        "t": 1.0
    }


    #===================================================================================================================
    # ==========================INIZIO GERADE DI SINGOLETTO=============================================================
    #===================================================================================================================

    H_matrix_G = costruisci_hamiltoniana_gerade(**parametri)
    singlet_gerade, trash = ordina_matrice_per_diagonale(H_matrix_G)

    # print("\nMatrice Hamiltoniana Singoletto gerade creata con successo!")
    # print("Shape della matrice:", singlet_gerade.shape)
    # print(f"Elemento H[0,0] (F1): {singlet_gerade[0, 0]:.4f}")
    # # Nota: L'elemento 0,3 nell'ungerade è sqrt(3)*t o 0 a seconda dell'ordinamento
    # print(f"Diagonale: {np.diag(singlet_gerade)}")
    # print(f"Indici ordinati: {trash}")
    print("=====================SINGLET GERADE======================")
    """
    G1 = GS
    G4 = 1111
    """
    # =============COUPLING G1-G4-G5-G11================================================================
    # 1. Definizioni indici
    idx1 = np.where(trash == 0)[0][0]
    idx2 = np.where(trash == 3)[0][0]
    idx3 = np.where(trash == 4)[0][0]
    idx4 = np.where(trash == 10)[0][0]

    # 2. Costruzione e diagonalizzazione del blocco 2x2
    mini_H_g1g4 = np.array([
        [singlet_gerade[idx1, idx1], singlet_gerade[idx1, idx2]],
        [singlet_gerade[idx2, idx1], singlet_gerade[idx2, idx2]]
    ])

    w, vec = np.linalg.eigh(mini_H_g1g4)

    # 3. Energie di base degli stati perturbanti
    E_idx3 = singlet_gerade[idx3, idx3]
    E_idx4 = singlet_gerade[idx4, idx4]

    # 4. Accoppiamenti "nudi" (Solo idx2 si accoppia. idx1 ha coupling 0)
    V_2_idx3 = singlet_gerade[idx2, idx3]
    V_2_idx4 = singlet_gerade[idx2, idx4]

    # ================== CORREZIONE STATO MINUS (w[0]) ==================
    # Autovettore: vec[:, 0].
    # Il coefficiente di G1 è vec[0,0], il coefficiente di G4 è vec[1,0].
    # Poiché G4 non si accoppia, il coupling efficace dipende solo da vec[1,0].

    coupling_minus_3 = vec[1, 0] * V_2_idx3
    coupling_minus_4 = vec[1, 0] * V_2_idx4

    # Sommiamo le due correzioni perturbative indipendenti
    energy_s_gerade_GS_minus = w[0] + (coupling_minus_3 ** 2) / (w[0] - E_idx3) + (coupling_minus_4 ** 2) / (w[0] - E_idx4)
    peso_g1 = np.zeros((2,1))
    peso_g1[0] = vec[0, 0] ** 2  # Salvo il peso del GS nello stato perturbato minus
    peso_g1[1] = vec[0, 1] ** 2  # Salvo il peso del GS nello stato perturbato plus

    # ================== CORREZIONE STATO PLUS (w[1]) ==================
    # Autovettore: vec[:, 1].
    # Il coefficiente di G1 è vec[0,1], il coefficiente di G4 è vec[1,1].
    # Poiché G1 non si accoppia, il coupling efficace dipende solo da vec[1,1].

    coupling_plus_3 = vec[1, 1] * V_2_idx3
    coupling_plus_4 = vec[1, 1] * V_2_idx4

    energy_s_gerade_GS_plus = w[1] + (coupling_plus_3 ** 2) / (w[1] - E_idx3) + (coupling_plus_4 ** 2) / (w[1] - E_idx4)
    print("Energy Singlet Gerade GS minus:", energy_s_gerade_GS_minus, "Peso GS:", peso_g1[0])
    print("Energy Singlet Gerade GS plus:", energy_s_gerade_GS_plus, "Peso GS:", peso_g1[1])
    # =============COUPLING G2-G3-G9-G10================================================================
    idx1 = np.where(trash == 1)[0][0]
    idx2 = np.where(trash == 2)[0][0]    # 1. Costruzione e diagonalizzazione del blocco 2x2 (Indici 2 e 3)
    mini_H_g2g3 = np.array([
        [singlet_gerade[idx1, idx1], singlet_gerade[idx1, idx2]],
        [singlet_gerade[idx2, idx1], singlet_gerade[idx2, idx2]]
    ])

    w1, v1 = np.linalg.eigh(mini_H_g2g3)
    idx3=np.where(trash == 8)[0][0]
    idx4 = np.where(trash == 9)[0][0]  # 2. Costruzione e diagonalizzazione del blocco 2x2 (Indici 9 e 10)

    mini_H_g9g10 = np.array([
        [singlet_gerade[idx3, idx3], singlet_gerade[idx3, idx4]],
        [singlet_gerade[idx3, idx4], singlet_gerade[idx4, idx4]]
    ])
    w2, v2 = np.linalg.eigh(mini_H_g9g10)
    v_tot = block_diag(v1,v2)          # 3. Unisco i due blocchi di autovettori in una matrice diagonale a blocchi
    w_tot = np.concatenate((w1, w2))
    mini_H_vcoup = np.zeros((4,4))

    mini_H_vcoup[0,2] = singlet_gerade[idx1,idx3]
    mini_H_vcoup[2,0] = mini_H_vcoup[0,2]
    mini_H_vcoup[1,3] = singlet_gerade[idx2,idx4]
    mini_H_vcoup[3,1] = mini_H_vcoup[1,3]  # 4. costruzione della matrice di accoppiamento tra i due blocchi 2x2

    v_coup = mod.rotate_matrix(mini_H_vcoup,v_tot,1,4)   # 5. Ruoto sulla nuova base degli autovettori
    energy_s_gerade_minus = w_tot[0] + (v_coup[0,2]**2/(w_tot[0]-w_tot[2])) + (v_coup[0,3]**2/(w_tot[0]-w_tot[3]))
    energy_s_gerade_plus = w_tot[1] + (v_coup[1,2]**2/(w_tot[1]-w_tot[2])) + (v_coup[1,3]**2/(w_tot[1]-w_tot[3]))

    peso_g2 = np.zeros((2,1))
    peso_g2[0] = v1[0, 0] ** 2  #Peso G2 in s ungerade minus
    peso_g2[1] = v1[0, 1] ** 2  #Peso G2 in s ungerade plus
    print("Energy Singlet Gerade minus:", energy_s_gerade_minus, "Peso G2:", peso_g2[0])
    print("Energy Singlet Gerade plus:", energy_s_gerade_plus, "Peso G2:", peso_g2[1])


    #===================================================================================================================
    # ==========================INIZIO UNGERADE DI SINGOLETTO===========================================================
    #=========================== U4 = 1111==============================================================================
    #===================================================================================================================
    H_matrix_u = costruisci_hamiltoniana_ungerade(**parametri)
    singlet_ungerade, trash = ordina_matrice_per_diagonale(H_matrix_u)
    print("=====================SINGLET UNGERADE======================")

    # print(f"Diagonale: {np.diag(singlet_ungerade)}")
    # print(f"Indici ordinati: {trash}")

    # =============COUPLING U1-U4-U7-U6================================================================

    idx1 = np.where(trash == 0)[0][0]
    idx2 = np.where(trash == 3)[0][0]    # 1. Costruzione e diagonalizzazione del blocco 2x2 (Indici 1 e 4)
    mini_H_u1u4 = np.array([
        [singlet_ungerade[idx1, idx1], singlet_ungerade[idx1, idx2]],
        [singlet_ungerade[idx2, idx1], singlet_ungerade[idx2, idx2]]
    ])
    w1, v1 = np.linalg.eigh(mini_H_u1u4)


    idx3=np.where(trash == 6)[0][0]
    idx4 = np.where(trash == 5)[0][0]   # 2. Costruzione e diagonalizzazione del blocco 2x2 (Indici 7 e 6)

    mini_H_u7u6 =np.array([
        [singlet_ungerade[idx3, idx3], singlet_ungerade[idx3, idx4]],
        [singlet_ungerade[idx3, idx4], singlet_ungerade[idx4, idx4]]
    ])

    w2, v2 = np.linalg.eigh(mini_H_u7u6)

    # 3. Creazione autovettori a blocchi (v_tot è 4x4)
    v_tot = block_diag(v1, v2)

    # Unione delle energie (w_tot diventa un array 1D di 4 elementi)
    w_tot = np.concatenate((w1, w2))

    mini_H_vcoup = np.zeros((4, 4))
    mini_H_vcoup[0, 2] = singlet_ungerade[idx1, idx3]
    mini_H_vcoup[2, 0] = mini_H_vcoup[0, 2]
    mini_H_vcoup[1, 3] = singlet_ungerade[idx2, idx4]
    mini_H_vcoup[3, 1] = mini_H_vcoup[1, 3]

    # 4. RUOTIAMO USANDO v_tot (GLI AUTOVETTORI), NON w_tot!
    v_coup = mod.rotate_matrix(mini_H_vcoup, v_tot, 1, 4)

    # 5. Ora v_coup è 4x4 e w_tot è un vettore lungo 4, gli indici non daranno più errore.
    energy_s_ungerade_minus = w_tot[0] + (v_coup[0, 2] ** 2 / (w_tot[0] - w_tot[2])) + (v_coup[0, 3] ** 2 / (w_tot[0] - w_tot[3]))
    energy_s_ungerade_plus = w_tot[1] + (v_coup[1, 2] ** 2 / (w_tot[1] - w_tot[2])) + (v_coup[1, 3] ** 2 / (w_tot[1] - w_tot[3]))
    peso_u4 = np.zeros((2,1))
    peso_u4[0] = v1[1, 0] ** 2
    peso_u4[1] = v1[1, 1] ** 2
    print("Energy Singlet ungerade minus:", energy_s_ungerade_minus, "Peso U4:", peso_u4[0])
    print("Energy Singlet ungerade minus:", energy_s_ungerade_plus, "Peso U4:", peso_u4[1])
    # =============COUPLING U2-U3================================================================

    idx1 = np.where(trash == 1)[0][0]
    idx2 = np.where(trash == 2)[0][0]
    mini_H_u2u3 = np.array([
        [singlet_ungerade[idx1, idx1], singlet_ungerade[idx1, idx2]],
        [singlet_ungerade[idx2, idx1], singlet_ungerade[idx2, idx2]]
    ])
    w1, v1 = np.linalg.eigh(mini_H_u2u3)
    energy_s3_ungerade = w1[0]
    energy_s4_ungerade = w1[1]
    print("Energy Singlet Ungerade minus (U2-U3):", energy_s3_ungerade)
    print("Energy Singlet Ungerade plus (U2-U3):", energy_s4_ungerade)
    # ==================================================================================================================
    #========================TRIPLETTI GERADE===========================================================================
    #===================================================================================================================
    H_matrix = costruisci_hamiltoniana_tripletto_gerade(**parametri)
    triplet_gerade, trash = ordina_matrice_per_diagonale(H_matrix)
    # print(f"Diagonale: {np.diag(triplet_gerade)}")
    # print(f"Indici ordinati: {trash}")
    print("=====================TRIPLETTI GERADE======================")

    #======================TG 4 - TG 5 -TG 3==========================================================
    idx1 = np.where(trash == 4)[0][0]
    idx2 = np.where(trash == 3)[0][0]
    mini_H = np.zeros((2,2))
    mini_H[0,0] = triplet_gerade[idx1, idx1]
    mini_H[1,1] = triplet_gerade[idx2, idx2]
    mini_H[0,1] = triplet_gerade[idx1, idx2]
    mini_H[1,0] = triplet_gerade[idx2, idx1]
    w, vec = np.linalg.eigh(mini_H)


    idx = np.where(trash == 2)[0][0]

    # 1. Energia dello stato perturbante (Ordine 0)
    E_idx = triplet_gerade[idx, idx]

    # 2. Elementi di accoppiamento "nudi" (V_13 e V_23) tra la base originale e lo stato idx
    V_1_idx = triplet_gerade[idx1, idx]
    V_2_idx = triplet_gerade[idx2, idx]

    # --- CALCOLO PER LO STATO MINUS (w[0]) ---
    # Autovettore associato: vec[:, 0]
    # L'accoppiamento totale è la somma lineare pesata dai coefficienti dell'autovettore
    coupling_minus = (vec[0, 0] * V_1_idx) + (vec[1, 0] * V_2_idx)

    # Teoria delle perturbazioni al 2° ordine: E_new = E_old + |V|^2 / (E_old - E_idx)
    energy_t_gerade_minus = w[0] + (coupling_minus ** 2) / (w[0] - E_idx)
    peso_t5 = np.zeros((2,1))
    peso_t5[0] = vec[0, 0]**2
    peso_t5[1] = vec[0, 1]**2
    pes_t4 = np.zeros((2,1))
    pes_t4[0] = vec[1, 0]**2
    pes_t4[1] = vec[1, 1]**2
    # --- CALCOLO PER LO STATO PLUS (w[1]) ---
    # Autovettore associato: vec[:, 1]
    coupling_plus = (vec[0, 1] * V_1_idx) + (vec[1, 1] * V_2_idx)

    energy_t_gerade_plus = w[1] + (coupling_plus ** 2) / (w[1] - E_idx)
    print("Energy Triplet gerade minus:", energy_t_gerade_minus, "Peso T5:", peso_t5[0], "Peso T4:", pes_t4[0])
    print("Energy Triplet gerade plus:", energy_t_gerade_plus, "Peso U4:", peso_t5[1], "Peso U4:", pes_t4[1])


    #=========================TRIPLETTO GS==============================================================================
    mini_H_GS = np.zeros((2,2))
    idx1 = np.where(trash == 0)[0][0]
    idx2 = np.where(trash == 1)[0][0]
    mini_H_GS[0,0] = triplet_gerade[idx1, idx1]
    mini_H_GS[1,1] = triplet_gerade[idx2, idx2]
    mini_H_GS[0,1] = triplet_gerade[idx1, idx2]
    mini_H_GS[1,0] = triplet_gerade[idx2, idx1]
    w_t_gs, vec_t_gs = np.linalg.eigh(mini_H_GS)
    print("Energy Triplet GS minus:", w_t_gs[0])
    print("Energy Triplet GS plus:", w_t_gs[1])
    # ==================================================================================================================
    # ========================TRIPLETTI UNGERADE========================================================================
    # ==================================================================================================================
    print("=====================TRIPLET UNGERADE======================")

    mini_H_tu = np.zeros((2,2))
    mini_H_tu[0,0] = parametri["e_H"] + parametri["U_S"]
    mini_H_tu[1,1] = parametri["e_H"] + parametri["e_L"] + 0.25 * parametri["J_HL"]
    mini_H_tu[0,1] = parametri["t"]
    mini_H_tu[1,0] = parametri["t"]
    print(mini_H_tu[0,0], mini_H_tu[1,1])
    w_tu, v_tu = np.linalg.eigh(mini_H_tu)
    energy_t_ungerade_minus = w_tu[0]
    energy_t_ungerade_plus = w_tu[1]
    peso_t = np.zeros((2,1))
    peso_t[0] = vec[1, 0] ** 2
    peso_t[1] = vec[1, 1] ** 2
    print("Energy Triplet ungerade minus:", energy_t_ungerade_minus, "Peso T:", peso_t[0])
    print("Energy Triplet ungerade plus:", energy_t_ungerade_plus, "Peso T:", peso_t[1])
    # ==================================================================================================================
    # ========================QUINTETTO========================================================================
    # ==================================================================================================================
    print("=====================QUINTETTO======================")
    E_quintetto = parametri["e_H"] + parametri["e_L"] + 0.25 * parametri["J_HL"] + 0.5 * parametri["J_SL"]
    print("Energy Quintetto:", E_quintetto)
