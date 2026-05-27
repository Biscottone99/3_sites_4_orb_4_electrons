import os
import re
import subprocess
import numpy as np
from pathlib import Path

# --- CONFIGURAZIONI ---
MULTIWFN_PATH = "/mnt/c/Users/Matteo/OneDrive - Università degli Studi di Parma/Desktop/Multiwfn_3.8_dev_bin_Win64/Multiwfn.exe"
MOLDEN_FILE = "/home/lince/radicals/ab-initio/diallile-quadratino_cas66.molden"
FOCK_FILE = "/home/lince/radicals/ab-initio/Fock.txt"

# Soglia numerica per considerare due orbitali "co-localizzati"
SOGLIA_GCI = 0.01 

def identifica_orbitali(molden_path):
    occupations = []
    with open(molden_path, 'r') as f:
        for line in f:
            if "Occup=" in line:
                val_str = line.split('=')[1].strip()
                occupations.append(float(val_str))
                
    n_orbitali = len(occupations)
    valid_occs = [(i + 1, occ) for i, occ in enumerate(occupations)]
    somo1 = next((i for i, occ in valid_occs if 0.5 < occ < 1.5), 1)
    
    return n_orbitali, somo1-1, somo1, somo1+1, somo1+2

def estrai_coefficienti_mo(molden_path, indici_orbitali):
    """
    Legge il file Molden ed estrae i coefficienti LCAO per gli orbitali richiesti.
    Restituisce un dizionario {indice_orbitale: [array di coefficienti]}.
    """
    coeff = {i: [] for i in indici_orbitali}
    in_mo = False
    curr_orb = 0
    
    with open(molden_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith('[MO]'):
                in_mo = True
                continue
            if in_mo:
                if line.startswith('['): # Fine della sezione MO
                    break
                if "Ene=" in line or "Spin=" in line or "Sym=" in line:
                    continue
                if "Occup=" in line:
                    curr_orb += 1
                    continue
                
                # Se siamo in un orbitale di interesse, salviamo il coefficiente
                if curr_orb in indici_orbitali:
                    parts = line.split()
                    if len(parts) >= 2:
                        val = float(parts[1].replace('D', 'E').replace('d', 'e'))
                        coeff[curr_orb].append(val)
    return coeff

def main():
    n_orbitali, homo, somo1, somo2, lumo = identifica_orbitali(MOLDEN_FILE)
    
    print(f"[SUBROUTINE] Generazione Matrice di Fock per {n_orbitali} orbitali...")
    
    comandi = f"{MOLDEN_FILE}\n100\n17\n{FOCK_FILE}\n0\n"
    
    process = subprocess.run(["wine", MULTIWFN_PATH] if "wine" in os.environ else [MULTIWFN_PATH], 
                             input=comandi, text=True, capture_output=True)
    
    if os.path.exists(FOCK_FILE):
        with open(FOCK_FILE, 'r') as f:
            contenuto = f.read().replace('D', 'e').replace('d', 'e')
        
        valori = [float(x) for x in contenuto.split()]
        matrice = np.zeros((n_orbitali, n_orbitali))
        idx = 0
        for i in range(n_orbitali):
            for j in range(i + 1):
                matrice[i, j] = matrice[j, i] = valori[idx]
                idx += 1
        print("[SUBROUTINE] Matrice di Fock ricostruita correttamente.")

    # Indici per gli array (0-based)
    idx_h  = homo - 1
    idx_s1 = somo1 - 1
    idx_s2 = somo2 - 1
    idx_l  = lumo - 1
    
    # Caricamento matrice cinetica (orbint.txt)
    data = []
    with open("orbint.txt", "r") as l:
        for line in l:
            i, j, val = line.split()
            data.append((int(i), int(j), float(val)))
            
    n = max(max(i, j) for i, j, _ in data) + 1
    kinetic = np.zeros((n, n))
    for i, j, val in data:
        kinetic[i, j] = val

    print(f"Hopping t (da orbint): {kinetic[idx_h, idx_l]:.6f}, {kinetic[idx_s1, idx_l]:.6f}, {kinetic[idx_s2, idx_l]:.6f}")   
    
    # Valori di K hardcoded (come da tua richiesta)
    khl = 0.013819
    ks1l = 0.021312
    ks2l = 0.124117
    
    # --- CALCOLO GCI (Filtro Spaziale) ---
    print("\n[SUBROUTINE] Estrazione coefficienti e calcolo GCI...")
    coeff = estrai_coefficienti_mo(MOLDEN_FILE, [homo, somo1, somo2, lumo])
    
    # Formule GCI: somma dei quadrati dei coefficienti moltiplicati tra loro
    gci_hl  = sum(c_h**2 * c_l**2  for c_h, c_l  in zip(coeff[homo], coeff[lumo]))
    gci_s1l = sum(c_s1**2 * c_l**2 for c_s1, c_l in zip(coeff[somo1], coeff[lumo]))
    gci_s2l = sum(c_s2**2 * c_l**2 for c_s2, c_l in zip(coeff[somo2], coeff[lumo]))
    
    print(f"GCI HOMO-LUMO:  {gci_hl:.6f}")
    print(f"GCI SOMO1-LUMO: {gci_s1l:.6f}")
    print(f"GCI SOMO2-LUMO: {gci_s2l:.6f}\n")

    # --- CALCOLO PREDITTORI CON FILTRO GCI ---
    
    # Coppia HOMO-LUMO
    gap_hl = abs(matrice[idx_h, idx_h] - matrice[idx_l, idx_l])
    if gci_hl > SOGLIA_GCI:
        se_hl = 0.0
        lambda_hl = float('inf')
    else:
        se_hl = (kinetic[idx_h, idx_l]**2) / gap_hl if gap_hl > 1e-9 else float('inf')
        lambda_hl = abs(2 * khl) / se_hl if se_hl > 1e-9 else float('inf')

    # Coppia SOMO1-LUMO
    gap_s1l = abs(matrice[idx_s1, idx_s1] - matrice[idx_l, idx_l])
    if gci_s1l > SOGLIA_GCI:
        se_s1l = 0.0
        lambda_s1l = float('inf')
    else:
        se_s1l = (kinetic[idx_s1, idx_l]**2) / gap_s1l if gap_s1l > 1e-9 else float('inf')
        lambda_s1l = abs(2 * ks1l) / se_s1l if se_s1l > 1e-9 else float('inf')

    # Coppia SOMO2-LUMO
    gap_s2l = abs(matrice[idx_s2, idx_s2] - matrice[idx_l, idx_l])
    if gci_s2l > SOGLIA_GCI:
        se_s2l = 0.0
        lambda_s2l = float('inf')
    else:
        se_s2l = (kinetic[idx_s2, idx_l]**2) / gap_s2l if gap_s2l > 1e-9 else float('inf')
        lambda_s2l = abs(2 * ks2l) / se_s2l if se_s2l > 1e-9 else float('inf')

    
    # Scrittura output aggiornata con i valori GCI
    with open("descrittori_2.txt", "w") as f:
        # intestazione
        f.write(f"{'Stato':<12}{'Fock_11':>15}{'Fock_22':>15}{'GCI':>15}"
            f"{'F_12^2':>15}{'Exchange':>15}{'Super-Ex':>15}{'Lambda':>15}\n")

        f.write("-" * 114 + "\n")

        # HOMO-LUMO
        f.write(f"{'HOMO-LUMO':<12}"
            f"{matrice[idx_h, idx_h]:15.8f}"
            f"{matrice[idx_l, idx_l]:15.8f}"
            f"{gci_hl:15.8f}"
            f"{kinetic[idx_h, idx_l]**2:15.8f}"
            f"{khl:15.8f}"
            f"{se_hl:15.8f}"
            f"{lambda_hl:15.8f}\n")

        # SOMO1-LUMO
        f.write(f"{'SOMO1-LUMO':<12}"
            f"{matrice[idx_s1, idx_s1]:15.8f}"
            f"{matrice[idx_l, idx_l]:15.8f}"
            f"{gci_s1l:15.8f}"
            f"{kinetic[idx_s1, idx_l]**2:15.8f}"
            f"{ks1l:15.8f}"
            f"{se_s1l:15.8f}"
            f"{lambda_s1l:15.8f}\n")

        # SOMO2-LUMO
        f.write(f"{'SOMO2-LUMO':<12}"
            f"{matrice[idx_s2, idx_s2]:15.8f}"
            f"{matrice[idx_l, idx_l]:15.8f}"
            f"{gci_s2l:15.8f}"
            f"{kinetic[idx_s2, idx_l]**2:15.8f}"
            f"{ks2l:15.8f}"
            f"{se_s2l:15.8f}"
            f"{lambda_s2l:15.8f}\n")
            
if __name__ == "__main__":
    main()
