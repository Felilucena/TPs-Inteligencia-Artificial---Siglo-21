# *********************************************************
# Nombre: Felipe Lucena
# Legajo: VINF016997
# *********************************************************

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin ventana (guarda figuras a archivo)
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import os


# --- Conversión y métricas ---

def to_pm1(matrix01: np.ndarray) -> np.ndarray:
    """Convierte matriz binaria {0,1} a vector bipolar {±1} columna."""
    mat = matrix01.astype(np.int8)
    mat_pm1 = np.where(mat > 0, 1, -1).astype(np.int8)
    return mat_pm1.reshape(-1, 1)

def to_01(vector_pm1: np.ndarray) -> np.ndarray:
    """Convierte vector bipolar {±1} a matriz 10x10 binaria {0,1}."""
    vec = vector_pm1.reshape(10, 10)
    return np.where(vec > 0, 1, 0).astype(np.int8)

def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Cuenta píxeles distintos entre dos patrones."""
    return int(np.sum(x.reshape(-1) != y.reshape(-1)))

def add_noise_pm1(x: np.ndarray, flip_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Invierte un porcentaje de bits del patrón (±1)."""
    n = x.size
    k = int(round(flip_ratio * n))
    idx = rng.choice(n, size=k, replace=False)
    noisy = x.copy().reshape(-1)
    noisy[idx] *= -1
    return noisy.reshape(-1, 1)


# --- Generadores de patrones 10x10 ---

def pattern_ring(radius: float = 3.0, thickness: float = 1.0) -> np.ndarray:
    """Patrón: anillo centrado."""
    g = np.zeros((10, 10), dtype=np.int8)
    cx, cy = 4.5, 4.5
    for i in range(10):
        for j in range(10):
            r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
            if abs(r - radius) <= thickness / 2.0:
                g[i, j] = 1
    return g

def pattern_plus() -> np.ndarray:
    """Patrón: cruz (+)."""
    g = np.zeros((10, 10), dtype=np.int8)
    g[:, 5] = 1
    g[5, :] = 1
    return g

def pattern_L() -> np.ndarray:
    """Patrón: forma de L."""
    g = np.zeros((10, 10), dtype=np.int8)
    g[7:, 1] = 1
    g[9, 1:7] = 1
    return g


# --- Red de Hopfield ---

@dataclass
class Hopfield:
    """Red de Hopfield: n neuronas y matriz de pesos W."""
    n: int
    W: np.ndarray

    @staticmethod
    def train_hebb(patterns_pm1: List[np.ndarray]) -> "Hopfield":
        """Entrena W con la regla de Hebb (suma de x·xᵀ, diagonal en 0)."""
        n = patterns_pm1[0].size
        W = np.zeros((n, n), dtype=np.float64)
        I = np.eye(n, dtype=np.float64)
        for x in patterns_pm1:
            x = x.reshape(n, 1).astype(np.float64)
            W += x @ x.T - I
        np.fill_diagonal(W, 0.0)  # Sin auto-conexiones
        return Hopfield(n=n, W=W)

    @staticmethod
    def train_pseudoinverse(patterns_pm1: List[np.ndarray]) -> "Hopfield":
        """Entrena W con la regla de la pseudoinversa (proyección)."""
        n = patterns_pm1[0].size
        U = np.hstack([p.reshape(n, 1).astype(np.float64) for p in patterns_pm1])  # n x q
        G = U.T @ U
        G_inv = np.linalg.pinv(G)
        W = U @ G_inv @ U.T
        np.fill_diagonal(W, 0.0)
        return Hopfield(n=n, W=W)

    def energy(self, x: np.ndarray) -> float:
        """Energía de Lyapunov: E = -½ xᵀ W x."""
        x = x.reshape(-1).astype(np.float64)
        return float(-0.5 * (x @ self.W @ x))

    def recall(
        self,
        x0: np.ndarray,
        max_iters: int = 50,
        asynchronous: bool = True,
        rng: Optional[np.random.Generator] = None,
        track_energy: bool = False
    ) -> Tuple[np.ndarray, int, Optional[List[float]]]:
        """Recupera un patrón desde x0 (asíncrono o síncrono) hasta estabilizar."""
        x = x0.reshape(self.n, 1).astype(np.float64).copy()
        energy_list = [self.energy(x)] if track_energy else None
        if asynchronous:
            # Actualiza una neurona a la vez (orden aleatorio)
            if rng is None:
                rng = np.random.default_rng(0)
            for it in range(max_iters):
                order = rng.permutation(self.n)
                changed = False
                for i in order:
                    s = float(self.W[i, :] @ x.reshape(-1))
                    new_state = 1.0 if s >= 0 else -1.0
                    if new_state != x[i, 0]:
                        x[i, 0] = new_state
                        changed = True
                if track_energy:
                    energy_list.append(self.energy(x))
                if not changed:
                    return x.astype(np.int8), it + 1, energy_list
            return x.astype(np.int8), max_iters, energy_list
        else:
            # Actualiza todas las neuronas a la vez
            for it in range(max_iters):
                x_new = np.sign(self.W @ x)
                x_new[x_new == 0] = 1.0
                if np.allclose(x_new, x):
                    if track_energy:
                        energy_list.append(self.energy(x_new))
                    return x_new.astype(np.int8), it + 1, energy_list
                x = x_new
                if track_energy:
                    energy_list.append(self.energy(x))
            return x.astype(np.int8), max_iters, energy_list


# --- Visualización ---

def visualize_triplets(triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], titles: List[str], out_path: str):
    """Guarda grilla: objetivo | entrada ruidosa | patrón recuperado."""
    rows = len(triplets)
    fig, axes = plt.subplots(rows, 3, figsize=(6, 2 * rows))
    if rows == 1:
        axes = np.array([axes])
    for r, (target_pm1, noisy_pm1, recalled_pm1) in enumerate(triplets):
        mats = [to_01(target_pm1), to_01(noisy_pm1), to_01(recalled_pm1)]
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(mats[c], interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(["Target", "Noisy input", "Recalled"][c])
        axes[r, 0].set_ylabel(titles[r], rotation=90, va='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_energy_curves(energies: Dict[str, List[float]], out_prefix: str):
    """Guarda una curva de energía por cada experimento."""
    for label, E in energies.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(range(len(E)), E)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.set_title(f"Energy convergence: {label}")
        plt.tight_layout()
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
        out_i = f"{out_prefix}_{safe_label}.png"
        plt.savefig(out_i, dpi=150, bbox_inches='tight')
        plt.close(fig)


# --- Demo principal ---

def main(seed: int = 123):
    """Entrena, prueba con ruido, reporta métricas y guarda figuras."""
    rng = np.random.default_rng(seed)

    # Patrones a memorizar
    p_ring = pattern_ring(radius=3.0, thickness=1.2)
    p_plus = pattern_plus()
    p_L = pattern_L()
    X_targets = [to_pm1(p) for p in (p_ring, p_plus, p_L)]
    names = ["Ring", "Plus", "L-shape"]

    # Dos reglas de aprendizaje
    net_hebb = Hopfield.train_hebb(X_targets)
    net_pinv = Hopfield.train_pseudoinverse(X_targets)

    noise_levels = [0.10, 0.30, 0.50]
    results_grid = []
    row_titles = []
    energy_curves: Dict[str, List[float]] = {}
    report_lines = []
    report_lines.append("Hopfield demo (10x10)")
    report_lines.append(f"Stored patterns: {', '.join(names)}")
    report_lines.append(f"Noise levels tested (bit flip ratios): {noise_levels}")
    report_lines.append("")

    # Prueba cada red × patrón × nivel de ruido
    for rule_name, net in [("Hebb", net_hebb), ("Pseudoinverse", net_pinv)]:
        report_lines.append(f"== {rule_name} training ==")
        for name, x_true in zip(names, X_targets):
            for nl in noise_levels:
                noisy = add_noise_pm1(x_true, flip_ratio=nl, rng=rng)
                x_rec, iters, E = net.recall(noisy, max_iters=60, asynchronous=True, rng=rng, track_energy=True)
                d = hamming_distance(x_rec, x_true)
                acc = 1.0 - d / x_true.size
                report_lines.append(f"[{rule_name}] {name} | noise={int(nl*100)}% | iters={iters:2d} | "
                                    f"hamming={d:3d} / {x_true.size} | acc={acc:.3f}")
                if nl in (0.10, 0.30):
                    results_grid.append((x_true, noisy, x_rec))
                    row_titles.append(f"{name} ({rule_name}, {int(nl*100)}% noise)")
                label = f"{rule_name}_{name}_{int(nl*100)}"
                energy_curves[label] = E if E is not None else []
        report_lines.append("")

    # Salidas: imagen, curvas y reporte
    img_grid_path = "hopfield_demo_results.png"
    visualize_triplets(results_grid, row_titles, img_grid_path)
    energy_prefix = "hopfield_energy"
    plot_energy_curves(energy_curves, energy_prefix)
    report_text = "\n".join(report_lines)
    with open("hopfield_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"Saved: {img_grid_path}, energy curves with prefix {energy_prefix}, report hopfield_report.txt")

if __name__ == "__main__":
    main()
