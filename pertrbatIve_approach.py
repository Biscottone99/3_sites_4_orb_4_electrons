import numpy as np
import plotting_module as plot
import math
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
        "J_HL": -1.0,
        "J_SL": -1.0,
        "t": 1.0
    }

    # =======================SINGOLETTI GERADE============================================
    H_matrix = costruisci_hamiltoniana_gerade(**parametri)
    singlet_gerade, trash = ordina_matrice_per_diagonale(H_matrix)
    # plot.plot_heatmap_real( singlet_gerade, "Reordered Matrix")
    # Stampa un paio di elementi per verifica
    print("Matrice Hamiltoniana creata con successo!")
    print("Shape della matrice:", singlet_gerade.shape)
    print(f"Elemento H[0,0] (D1): {singlet_gerade[0, 0]}")
    print(f"Elemento H[0,3] (-sqrt(2)*t): {singlet_gerade[0, 3]:.3f}")
    print(f"Diagonale: {np.diag(singlet_gerade)}")
    print(f"Indici ordinati: {trash}")

    # =============COUPLING G1-G4================================================================
    print(f"=================G1-G4==================")
    idx1 = np.where(trash == 0)[0][0]
    idx2 = np.where(trash == 3)[0][0]
    # 1. Costruzione e diagonalizzazione del blocco 2x2 (Indici 0 e 2)
    mini_H_g1g4 = np.array([
        [singlet_gerade[idx1, idx1], singlet_gerade[idx1, idx2]],
        [singlet_gerade[idx2, idx1], singlet_gerade[idx2, idx2]]
    ])

    w_g1g4, vec_g1g4 = np.linalg.eigh(mini_H_g1g4)
    E_psi_minus_g1g4 = w_g1g4[0]
    E_psi_plus_g1g4 = w_g1g4[1]

    # Calcolo dell'angolo (mantenuto per puro log/stampa)
    deltae_g1g4 = singlet_gerade[idx1, idx1] - singlet_gerade[idx2, idx2]
    argomento_g1g4 = (2 * singlet_gerade[idx1, idx2]) / deltae_g1g4 if deltae_g1g4 != 0 else 0
    angolo_def_g1g4 = math.degrees(np.arctan(argomento_g1g4) / 2)

    print(f"angolo G1-G4 (gradi): {angolo_def_g1g4:.2f}")
    print(f"E_psi_minus esatta: {E_psi_minus_g1g4:.4f}")
    print(f"E_psi_plus esatta: {E_psi_plus_g1g4:.4f}")
    # (Non hai applicato PT2 qui nel tuo script originale)
    GS_singoletto = E_psi_minus_g1g4
    # =============================================================================================

    # =============COUPLING G2-G3================================================================
    print(f"=================G2-G3==================")
    idx1 = np.where(trash == 1)[0][0]
    idx2 = np.where(trash == 2)[0][0]
    # 1. Costruzione e diagonalizzazione del blocco 2x2 (Indici 1 e 3)
    mini_H_g2g3 = np.array([
        [singlet_gerade[idx1, idx1], singlet_gerade[idx1, idx2]],
        [singlet_gerade[idx2, idx1], singlet_gerade[idx2, idx2]]
    ])

    w_g2g3, vec_g2g3 = np.linalg.eigh(mini_H_g2g3)
    E_psi_minus_g2g3 = w_g2g3[0]
    E_psi_plus_g2g3 = w_g2g3[1]

    deltae_g2g3 = singlet_gerade[idx1, idx1] - singlet_gerade[idx2, idx2]
    argomento_g2g3 = (2 * singlet_gerade[idx1, idx2]) / deltae_g2g3 if deltae_g2g3 != 0 else 0
    angolo_def_g2g3 = math.degrees(np.arctan(argomento_g2g3) / 2)

    print(f"angolo G2-G3 (gradi): {angolo_def_g2g3:.2f}")
    print(f"E_psi_minus esatta: {E_psi_minus_g2g3:.4f}")
    print(f"E_psi_plus esatta: {E_psi_plus_g2g3:.4f}")

    # 2. Applicazione PT2 con lo stato originale 8
    idx_pt2_g2g3 = np.where(trash == 8)[0][0]
    E_distante_g2g3 = singlet_gerade[idx_pt2_g2g3, idx_pt2_g2g3]

    # vec_g2g3[:, 0] è v_minus, vec_g2g3[:, 1] è v_plus
    V_eff_minus_g2g3 = vec_g2g3[0, 0] * singlet_gerade[1, idx_pt2_g2g3] + vec_g2g3[1, 0] * singlet_gerade[
        3, idx_pt2_g2g3]
    V_eff_plus_g2g3 = vec_g2g3[0, 1] * singlet_gerade[1, idx_pt2_g2g3] + vec_g2g3[1, 1] * singlet_gerade[
        3, idx_pt2_g2g3]

    pt2_minus_g2g3 = (V_eff_minus_g2g3 ** 2) / (E_psi_minus_g2g3 - E_distante_g2g3)
    pt2_plus_g2g3 = (V_eff_plus_g2g3 ** 2) / (E_psi_plus_g2g3 - E_distante_g2g3)

    E_psi_minus_corretta_g2g3 = E_psi_minus_g2g3 + pt2_minus_g2g3
    E_psi_plus_corretta_g2g3 = E_psi_plus_g2g3 + pt2_plus_g2g3

    print(f"--- After PT2 (Stato 8) ---")
    print(f"Correzione PT2 su minus: {pt2_minus_g2g3:.6f}")
    print(f"E_psi_minus_corretta: {E_psi_minus_corretta_g2g3:.4f}")
    print(f"Correzione PT2 su plus: {pt2_plus_g2g3:.6f}")
    print(f"E_psi_plus_corretta: {E_psi_plus_corretta_g2g3:.4f}")
    s_gerade = E_psi_minus_corretta_g2g3
    # =============================================================================================

    # ==========================INIZIO UNGERADE DI SINGOLETTO======================================
    H_matrix_u = costruisci_hamiltoniana_ungerade(**parametri)
    singlet_ungerade, trash = ordina_matrice_per_diagonale(H_matrix_u)

    print("\nMatrice Hamiltoniana Singoletto Ungerade creata con successo!")
    print("Shape della matrice:", singlet_ungerade.shape)
    print(f"Elemento H[0,0] (F1): {singlet_ungerade[0, 0]:.4f}")
    # Nota: L'elemento 0,3 nell'ungerade è sqrt(3)*t o 0 a seconda dell'ordinamento
    print(f"Diagonale: {np.diag(singlet_ungerade)}")
    print(f"Indici ordinati: {trash}")

    # =============COUPLING U1-U4================================================================
    print(f"=================U1-U4==================")
    idx1 = np.where(trash == 0)[0][0]
    idx2 = np.where(trash == 3)[0][0]    # 1. Costruzione e diagonalizzazione del blocco 2x2 (Indici 0 e 2)
    mini_H_u = np.array([
        [singlet_ungerade[idx1, idx1], singlet_ungerade[idx1, idx2]],
        [singlet_ungerade[idx2, idx1], singlet_ungerade[idx2, idx2]]
    ])

    w_u, vec_u = np.linalg.eigh(mini_H_u)
    E_psi_minus_u = w_u[0]
    E_psi_plus_u = w_u[1]

    deltae_u = singlet_ungerade[idx1, idx1] - singlet_ungerade[idx2, idx2]
    argomento_u = (2 * singlet_ungerade[idx1, idx2]) / deltae_u if deltae_u != 0 else 0
    angolo_def_u = math.degrees(np.arctan(argomento_u) / 2)

    print(f"angolo U1-U4 (gradi): {angolo_def_u:.2f}")
    print(f"E_psi_minus esatta: {E_psi_minus_u:.4f}")
    print(f"E_psi_plus esatta: {E_psi_plus_u:.4f}")

    # 2. Applicazione PT2 con lo stato originale 5
    idx_pt2_u = np.where(trash == 5)[0][0]
    E_distante_u = singlet_ungerade[idx_pt2_u, idx_pt2_u]

    V_eff_minus_u = vec_u[0, 0] * singlet_ungerade[0, idx_pt2_u] + vec_u[1, 0] * singlet_ungerade[2, idx_pt2_u]
    V_eff_plus_u = vec_u[0, 1] * singlet_ungerade[0, idx_pt2_u] + vec_u[1, 1] * singlet_ungerade[2, idx_pt2_u]

    pt2_minus_u = (V_eff_minus_u ** 2) / (E_psi_minus_u - E_distante_u)
    pt2_plus_u = (V_eff_plus_u ** 2) / (E_psi_plus_u - E_distante_u)

    E_psi_minus_corretta_u = E_psi_minus_u + pt2_minus_u
    E_psi_plus_corretta_u = E_psi_plus_u + pt2_plus_u

    print(f"--- After PT2 (Stato 5) ---")
    print(f"Correzione PT2 su minus: {pt2_minus_u:.6f}")
    print(f"E_psi_minus_corretta: {E_psi_minus_corretta_u:.4f}")
    print(f"Correzione PT2 su plus: {pt2_plus_u:.6f}")
    print(f"E_psi_plus_corretta: {E_psi_plus_corretta_u:.4f}")
    s_ungerade = E_psi_minus_corretta_u
    # =============================================================================================

    #========================TRIPLETTI GERADE==================================================
    H_matrix = costruisci_hamiltoniana_tripletto_gerade(**parametri)
    triplet_gerade, trash = ordina_matrice_per_diagonale(H_matrix)
    print(f"Diagonale: {np.diag(triplet_gerade)}")
    print(f"Indici ordinati: {trash}")
    #======================TG1 - TG2==========================================================
    idx1 = np.where(trash == 0)[0][0]
    idx2 = np.where(trash == 1)[0][0]
    mini_H_t1t2 = np.zeros((2,2))
    mini_H_t1t2[0,0] = triplet_gerade[idx1, idx1]
    mini_H_t1t2[1,1] = triplet_gerade[idx2, idx2]
    mini_H_t1t2[0,1] = triplet_gerade[idx1, idx2]
    mini_H_t1t2[1,0] = triplet_gerade[idx2, idx1]
    w_t1t2, vec_t1t2 = np.linalg.eigh(mini_H_t1t2)
    E_GS_trip = w_t1t2[0]


    #======================TG 4 - TG 5==========================================================
    idx1 = np.where(trash == 4)[0][0]
    idx2 = np.where(trash == 3)[0][0]
    coupling = triplet_gerade[idx1, idx2]
    deltae = triplet_gerade[idx1, idx1]-triplet_gerade[idx2,idx2]
    argomento = np.abs((2*coupling) / (triplet_gerade[idx1, idx1]-triplet_gerade[idx2,idx2]))
    angolo = np.arctan(argomento)
    print(f"=================TG 4 - TG 5==================")
    angolo_def = math.degrees(angolo/2)
    print(f"angolo TG 4 - TG 5: {angolo_def}")
    mini_H = np.zeros((2,2))
    mini_H[0,0] = triplet_gerade[idx1, idx1]
    mini_H[1,1] = triplet_gerade[idx2, idx2]
    mini_H[0,1] = triplet_gerade[idx1, idx2]
    mini_H[1,0] = triplet_gerade[idx2, idx1]
    w, vec = np.linalg.eigh(mini_H)
    E_psi_minus = w[0]
    E_psi_plus = w[1]
    v_minus = vec[:, 0]
    v_plus = vec[:, 1]

    print(f"E_psi_minus: {E_psi_minus}")
    print(f"E_psi_plus: {E_psi_plus}")

    idx = np.where(trash == 2)[0][0]
    E_distante = triplet_gerade[idx, idx]

    V_eff_minus = v_minus[0] * triplet_gerade[idx1, idx] + v_minus[1] * triplet_gerade[idx2, idx]
    V_eff_plus = v_plus[0] * triplet_gerade[idx1, idx] + v_plus[1] * triplet_gerade[idx2, idx]

    # 2. Applichiamo la formula di perturbazione del secondo ordine: |V|^2 / (E - E0)
    pt2_minus = (V_eff_minus ** 2) / (E_psi_minus - E_distante)
    pt2_plus = (V_eff_plus ** 2) / (E_psi_plus - E_distante)

    # 3. Applichiamo le correzioni
    E_psi_minus_corretta = E_psi_minus + pt2_minus
    E_psi_plus_corretta = E_psi_plus + pt2_plus
    print(f"After PT2")
    print(f"E_psi_minus: {E_psi_minus}")
    print(f"E_psi_plus: {E_psi_plus}")
    print(f"E_psi_minus_corretta: {E_psi_minus_corretta}")
    print(f"E_psi_plus_corretta: {E_psi_plus_corretta}")
    t_ungerade1 = E_psi_minus_corretta
    t_ungerade2 = E_psi_plus_corretta
    print(f"GS Singlet, G_triplet, S_gerade, S_ungerade, T_ungerade1: {GS_singoletto}, {E_GS_trip}, {s_gerade}, {s_ungerade}, {t_ungerade1}, {t_ungerade2}")