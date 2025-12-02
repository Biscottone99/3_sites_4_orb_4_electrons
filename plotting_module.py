import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap, BoundaryNorm
def plot_heatmap_real(matrice, name):
    plt.imshow(matrice, cmap='viridis', origin='upper')
    plt.colorbar(label='Valore')
    plt.title(name)
    plt.xlim(-0.5, matrice.shape[1] - 0.5)
    plt.ylim(matrice.shape[0] - 0.5, -0.5)  # inverti asse y per mantenere "origin='upper'"
    plt.xlabel('Colonna')
    plt.ylabel('Riga')
    plt.show()

def plot_heatmap_cplx(matrice, nome):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    A = np.real(matrice)
    B = np.imag(matrice)
    im1 = axes[0].imshow(A, cmap='viridis', origin='upper')
    axes[0].set_title("Real")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Seconda heatmap
    im2 = axes[1].imshow(B, cmap='plasma', origin='upper')
    axes[1].set_title("Imaginary")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(nome)
    # Layout ordinato
    plt.tight_layout()
    plt.show()
def plot_geom(coords, charges):
    """
    Visualizza sfere 3D colorate in base alla carica e collegate in sequenza.

    Parameters
    ----------
    coords : array-like, shape (N, 3)
        Coordinate cartesiane dei punti.
    charges : array-like, shape (N,)
        Valori di carica (usati per il colore).
    """
    coords = np.asarray(coords)
    charges = np.asarray(charges)
    n = len(charges)

    # Colori in base alla carica
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=np.min(charges), vmax=np.max(charges))
    colors = cmap(norm(charges))

    # Figura 3D
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Sfere (scatter 3D)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               s=800, c=colors, edgecolor='k', alpha=0.9)

    # Etichette (numero d’ordine dentro la sfera)
    for i, (x, y, z) in enumerate(coords):
        ax.text(x, y, z, str(i + 1), color='black',
                ha='center', va='center', fontsize=10, weight='bold')

    # Collega solo la sfera i con i+1
    for i in range(n - 1):
        x_line = [coords[i, 0], coords[i + 1, 0]]
        y_line = [coords[i, 1], coords[i + 1, 1]]
        z_line = [coords[i, 2], coords[i + 1, 2]]
        ax.plot(x_line, y_line, z_line, color='gray', linewidth=2)

    # Impostazioni grafiche
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])  # rapporti uguali
    plt.tight_layout()
    plt.show()

def plot_curve(x, y, name, xlabel, ylabel):
    """
    Plotta una curva 2D con etichette e titolo.

    Parameters
    ----------
    x : array-like
        Dati per l'asse x.
    y : array-like
        Dati per l'asse y.
    name : str
        Titolo del grafico.
    xlabel : str
        Etichetta dell'asse x.
    ylabel : str
        Etichetta dell'asse y.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linestyle='-', color='b', linewidth=3)
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(np.min(x), np.max(x))
    plt.show()

def heatmaps_multiple(data_list, titles=None, labels=None, cmap='viridis', grid_res=100, ncols=2):
    """
    Plotta multiple heatmap da liste di vettori x, y, z.

    Parameters
    ----------
    data_list : list of tuples
        Lista di (x, y, z) da plottare.
    titles : list of str
        Titoli dei subplot.
    labels : list of tuples
        Lista di (xlabel, ylabel) per ogni subplot.
    cmap : str
        Colormap.
    grid_res : int
        Risoluzione griglia di interpolazione.
    ncols : int
        Numero di colonne dei subplot.
    """
    n = len(data_list)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)  # semplifica l'indicizzazione

    for i, (x, y, z) in enumerate(data_list):
        ax = axes[i]
        xi = np.linspace(np.min(x), np.max(x), grid_res)
        yi = np.linspace(np.min(y), np.max(y), grid_res)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x, y), z, (X, Y), method='cubic')

        pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
        #ax.scatter(x, y, c=z, edgecolor='k', cmap=cmap, s=30)
        fig.colorbar(pcm, ax=ax, label='Energy (eV)')

        if titles: ax.set_title(titles[i])
        if labels: ax.set_xlabel(labels[i][0]); ax.set_ylabel(labels[i][1])
        ax.grid(True, alpha=0.3)

        # y_mid = 0.5 * (np.min(y) + np.max(y))
        # ax.plot([np.min(x), np.max(x)],
        #         [y_mid, y_mid],
        #         '--', color='white', linewidth=1.5)
        # x_mid = 0.5 * (np.min(x) + np.max(x))
        # ax.plot([x_mid, x_mid],
        #         [np.min(y), np.max(y)],
        #         '--', color='white', linewidth=1.5)
    # Rimuove eventuali subplot vuoti
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def heatmaps_multiple_masked(data_list, threshold, titles=None, labels=None,
                      cmap='viridis', grid_res=100, ncols=2):
    """
    Plotta multiple heatmap da liste di vettori x, y, z.
    La prima heatmap filtra i punti con |z| < threshold.
    Le mappe successive usano solo quei punti filtrati.

    Parameters
    ----------
    data_list : list of tuples
        Lista di (x, y, z) da plottare.
    threshold : float
        Valore soglia su |z| per filtrare i punti della prima mappa.
    titles : list of str
        Titoli dei subplot.
    labels : list of tuples
        Lista di (xlabel, ylabel) per ogni subplot.
    cmap : str
        Colormap.
    grid_res : int
        Risoluzione della griglia per l'interpolazione.
    ncols : int
        Numero di colonne dei subplot.
    """
    # ===== FILTRO SULLA PRIMA HEATMAP =====
    x0, y0, z0 = data_list[0]

    mask = np.abs(z0) < threshold
    xf, yf, zf = x0[mask], y0[mask], z0[mask]

    print(f"[INFO] Punti che soddisfano |z| < {threshold}: {np.sum(mask)} / {len(z0)}")

    # Rimpiazza la prima tripletta nella lista dei dati
    data_list_filtered = [(xf, yf, zf)] + data_list[1:]

    # ===== PLOT =====
    n = len(data_list_filtered)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (x, y, z) in enumerate(data_list_filtered):
        ax = axes[i]

        xi = np.linspace(np.min(x), np.max(x), grid_res)
        yi = np.linspace(np.min(y), np.max(y), grid_res)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x, y), z, (X, Y), method='cubic')

        pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
        fig.colorbar(pcm, ax=ax, label='Z')

        if titles:
            ax.set_title(titles[i])
        if labels:
            ax.set_xlabel(labels[i][0])
            ax.set_ylabel(labels[i][1])

        ax.grid(True, alpha=0.3)

    # Rimuovi subplot extra
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_three_heatmaps_from_points(x, y, z1, z2, z3,
                                    titles=("Map 1", "Map 2", "Map 3"),
                                    grid_res=100):
    # --- griglia regolare ---
    xi = np.linspace(np.min(x), np.max(x), grid_res)
    yi = np.linspace(np.min(y), np.max(y), grid_res)
    X, Y = np.meshgrid(xi, yi)

    # --- interpolazione ---
    Z1 = griddata((x, y), z1, (X, Y), method='nearest')
    Z2 = griddata((x, y), z2, (X, Y), method='nearest')
    Z3 = griddata((x, y), z3, (X, Y), method='nearest')

    maps = [Z1, Z2, Z3]

    # --- colormap discreta per valori 1,2,3 ---
    cmap = ListedColormap(["blue", "green", "red"])
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, Z, title in zip(axes, maps, titles):
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='auto')
        ax.set_title(title)
        ax.set_xlabel(r"$J_{H-L}$")
        ax.set_ylabel("$J_{S-L}$")

    #fig.colorbar(im, ax=axes.tolist(), ticks=[1,2,3], label="Rank")
    plt.tight_layout()
    plt.show()
