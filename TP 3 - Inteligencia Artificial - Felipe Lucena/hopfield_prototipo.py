# *********************************************************
# Nombre: Felipe Lucena
# Legajo: VINF016997
# *********************************************************
"""
Prototipo de red de Hopfield 10x10 (TP3 - Inteligencia Artificial).

Experimento A: memoria autoasociativa de 3 figuras (anillo, cruz, L) con ruido
               creciente, comparando aprendizaje de Hebb y matriz pseudoinversa.
Experimento B: localización del aro "C". Se memoriza el aro en 9 posiciones
               posibles (desplazamientos de -1/0/+1 píxel en X e Y) junto con una
               escuadra de referencia fija. Ante una imagen degradada (ruido,
               elemento extraño y tramo faltante) la red converge al atractor más
               cercano, que indica el desplazamiento; luego se calculan las
               coordenadas X, Y del centro "A" respecto de la escuadra.

Salidas: hopfield_report.txt (con las imágenes sucesivas en caracteres),
         hopfield_demo_results.png y hopfield_energy_*.png
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin ventana (guarda figuras a archivo)
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


# --- Conversión, métricas y representación en caracteres ---

def to_pm1(matrix01: np.ndarray) -> np.ndarray:
    """Convierte matriz binaria {0,1} a vector bipolar {±1} columna."""
    return np.where(matrix01 > 0, 1, -1).astype(np.int8).reshape(-1, 1)

def to_01(vector_pm1: np.ndarray) -> np.ndarray:
    """Convierte vector bipolar {±1} a matriz 10x10 binaria {0,1}."""
    return np.where(vector_pm1.reshape(10, 10) > 0, 1, 0).astype(np.int8)

def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Cuenta píxeles distintos entre dos patrones."""
    return int(np.sum(x.reshape(-1) != y.reshape(-1)))

def add_noise_pm1(x: np.ndarray, flip_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Invierte un porcentaje de bits del patrón (±1)."""
    k = int(round(flip_ratio * x.size))
    idx = rng.choice(x.size, size=k, replace=False)
    noisy = x.copy().reshape(-1)
    noisy[idx] *= -1
    return noisy.reshape(-1, 1)

def ascii_side_by_side(images: List[np.ndarray], titles: List[str]) -> str:
    """Dibuja varias imágenes 10x10 lado a lado con caracteres (# = 1, . = 0)."""
    mats = [to_01(im) if im.size == 100 and im.ndim == 2 and im.shape[1] == 1 else im for im in images]
    out = ["   ".join(t.ljust(10)[:10] for t in titles)]
    for r in range(10):
        out.append("   ".join("".join("#" if m[r, c] else "." for c in range(10)) for m in mats))
    return "\n".join(out)


# --- Generadores de patrones 10x10 ---

def add_reference(g: np.ndarray) -> np.ndarray:
    """Escuadra fija en el ángulo inferior izquierdo (elemento de referencia)."""
    g = g.copy()
    g[6:10, 0] = 1
    g[9, 0:4] = 1
    return g

def pattern_ring(cx: float = 4.5, cy: float = 4.5, radius: float = 3.0, thickness: float = 1.0) -> np.ndarray:
    """Patrón: anillo (aro "C") de centro (cx, cy) en coordenadas de columna/fila."""
    g = np.zeros((10, 10), dtype=np.int8)
    for i in range(10):
        for j in range(10):
            if abs(np.hypot(i - cy, j - cx) - radius) <= thickness / 2.0:
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
        """Regla de Hebb: W = Σ x·xᵀ con diagonal en 0 (sin auto-conexiones)."""
        n = patterns_pm1[0].size
        W = np.zeros((n, n), dtype=np.float64)
        for x in patterns_pm1:
            x = x.reshape(n, 1).astype(np.float64)
            W += x @ x.T
        np.fill_diagonal(W, 0.0)
        return Hopfield(n=n, W=W / n)

    @staticmethod
    def train_pseudoinverse(patterns_pm1: List[np.ndarray]) -> "Hopfield":
        """Regla de la pseudoinversa (proyección): W = U (UᵀU)⁻¹ Uᵀ, diagonal en 0."""
        n = patterns_pm1[0].size
        U = np.hstack([p.reshape(n, 1).astype(np.float64) for p in patterns_pm1])  # n x q
        W = U @ np.linalg.pinv(U.T @ U) @ U.T
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
        track_energy: bool = False,
        track_states: bool = False,
    ) -> Tuple[np.ndarray, int, Optional[List[float]], Optional[List[np.ndarray]]]:
        """Recupera un patrón desde x0 hasta estabilizar.
        Devuelve (estado final, ciclos, energías por ciclo, estados por ciclo)."""
        x = x0.reshape(self.n, 1).astype(np.float64).copy()
        energy_list = [self.energy(x)] if track_energy else None
        states = [x.astype(np.int8).copy()] if track_states else None
        if rng is None:
            rng = np.random.default_rng(0)
        for it in range(max_iters):
            if asynchronous:
                # Actualiza una neurona a la vez (orden aleatorio): garantiza convergencia
                changed = False
                for i in rng.permutation(self.n):
                    new_state = 1.0 if float(self.W[i, :] @ x[:, 0]) >= 0 else -1.0
                    if new_state != x[i, 0]:
                        x[i, 0] = new_state
                        changed = True
            else:
                # Actualiza todas las neuronas a la vez (puede oscilar en ciclos de 2)
                x_new = np.sign(self.W @ x)
                x_new[x_new == 0] = 1.0
                changed = not np.array_equal(x_new, x)
                x = x_new
            if track_energy:
                energy_list.append(self.energy(x))
            if track_states and changed:
                states.append(x.astype(np.int8).copy())
            if not changed:
                return x.astype(np.int8), it + 1, energy_list, states
        return x.astype(np.int8), max_iters, energy_list, states


# --- Localización del centro "A" ---

def ring_center(img01: np.ndarray) -> Tuple[float, float]:
    """Centro del aro (centroide de sus píxeles, sin la escuadra) en coordenadas
    relativas a la escuadra: X hacia la derecha desde la columna 0, Y hacia
    arriba desde la fila 9."""
    g = img01.copy()
    ref = add_reference(np.zeros((10, 10), dtype=np.int8))
    g[ref == 1] = 0
    rows, cols = np.nonzero(g)
    return float(cols.mean()), float(9 - rows.mean())

def closest_pattern(x: np.ndarray, patterns: List[np.ndarray]) -> Tuple[int, int]:
    """Índice del patrón memorizado más cercano y su distancia de Hamming."""
    d = [hamming_distance(x, p) for p in patterns]
    k = int(np.argmin(d))
    return k, d[k]


# --- Visualización ---

def visualize_triplets(triplets, titles: List[str], out_path: str):
    """Guarda grilla: objetivo | entrada degradada | patrón recuperado."""
    rows = len(triplets)
    fig, axes = plt.subplots(rows, 3, figsize=(6, 2 * rows))
    if rows == 1:
        axes = np.array([axes])
    for r, trip in enumerate(triplets):
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(to_01(trip[c]), cmap="gray_r", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(["Objetivo", "Entrada", "Recuperada"][c])
        axes[r, 0].set_ylabel(titles[r], rotation=90, va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_energy_curves(energies: Dict[str, List[float]], out_prefix: str):
    """Guarda una curva de energía por cada experimento."""
    for label, E in energies.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(range(len(E)), E, marker="o")
        ax.set_xlabel("Ciclo")
        ax.set_ylabel("Energía")
        ax.set_title(f"Convergencia: {label}", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# --- Experimentos ---

def experiment_a(rng, report, grid, grid_titles, energy_curves):
    """Tres figuras con ruido creciente, Hebb vs pseudoinversa."""
    names = ["Anillo", "Cruz", "L"]
    shapes = [pattern_ring(radius=3.0, thickness=1.2), pattern_plus(), pattern_L()]
    X = [to_pm1(add_reference(s)) for s in shapes]
    nets = [("Hebb", Hopfield.train_hebb(X)), ("Pseudoinversa", Hopfield.train_pseudoinverse(X))]

    report.append("=" * 70)
    report.append("EXPERIMENTO A - Figuras memorizadas: " + ", ".join(names) + " (+ escuadra fija)")
    report.append("=" * 70)
    report.append("Solapamiento entre patrones (xᵢ·xⱼ / n; 0 = ortogonales, 1 = iguales):")
    for i in range(3):
        report.append("  " + names[i].ljust(7) + " ".join(f"{int(X[i].astype(int).ravel() @ X[j].astype(int).ravel()) / 100:5.2f}" for j in range(3)))
    report.append("")

    for rule, net in nets:
        report.append(f"-- Regla de {rule} --")
        for name, x in zip(names, X):
            for nl in (0.10, 0.20, 0.30, 0.40):
                ok, dist = 0, []
                for t in range(20):  # 20 repeticiones por nivel de ruido
                    noisy = add_noise_pm1(x, nl, rng)
                    xr, it, E, _ = net.recall(noisy, rng=rng, track_energy=True)
                    d = hamming_distance(xr, x)
                    ok += d == 0
                    dist.append(d)
                    if t == 0 and nl in (0.10, 0.30):
                        grid.append((x, noisy, xr))
                        grid_titles.append(f"{name} {rule} {int(nl*100)}%")
                    if t == 0 and nl == 0.30:
                        energy_curves[f"A_{rule}_{name}_30"] = E
                report.append(f"  {name:7s} ruido={int(nl*100):2d}% | recuperación exacta {ok:2d}/20 "
                              f"| Hamming medio={np.mean(dist):5.1f}")
        report.append("")
    report.append("Nota: con 50% de bits invertidos la entrada deja de estar correlacionada con el")
    report.append("patrón original (es ruido puro), por lo que cualquier resultado es azar. Por eso")
    report.append("no se usa como nivel de prueba. Con más de 50% la red tiende al patrón inverso")
    report.append("(-x), que también es un mínimo de energía (estado espurio simétrico).")
    report.append("")

def experiment_b(rng, report, grid, grid_titles, energy_curves):
    """Localización del aro con desplazamientos, elemento extraño y tramo faltante."""
    offsets = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    base_cx, base_cy = 4.5, 4.0
    P01 = [add_reference(pattern_ring(base_cx + dx, base_cy + dy, radius=2.3)) for dx, dy in offsets]
    X = [to_pm1(p) for p in P01]
    nets = [("Hebb", Hopfield.train_hebb(X)), ("Pseudoinversa", Hopfield.train_pseudoinverse(X))]

    report.append("=" * 70)
    report.append("EXPERIMENTO B - Localización del aro 'C' y de su centro 'A'")
    report.append("=" * 70)
    report.append("Se memoriza el aro en 9 posiciones (desplazamientos de -1/0/+1 píxel en X e Y) junto con")
    report.append("la escuadra fija. Posición nominal = desplazamiento (0, 0).")
    report.append("")

    # Caso de prueba: aro desplazado (+1, -1), con ruido, mancha extraña y tramo faltante
    true_k = offsets.index((1, -1))
    img = P01[true_k].copy()
    img[0:2, 8:10] = 1          # elemento extraño (mancha / reflejo)
    rr, cc = np.nonzero(pattern_ring(base_cx + 1, base_cy - 1, radius=2.3))
    img[rr[:3], cc[:3]] = 0     # tramo del aro faltante
    test = add_noise_pm1(to_pm1(img), 0.08, rng)

    for rule, net in nets:
        xr, it, E, states = net.recall(test, rng=np.random.default_rng(7), track_energy=True, track_states=True)
        k, d = closest_pattern(xr, X)
        report.append(f"-- Regla de {rule} --")
        report.append("Imágenes sucesivas (entrada -> ciclos -> referencia memorizada):")
        imgs = [to_01(s) for s in states] + [P01[true_k]]
        titles = ["Entrada"] + [f"Ciclo {i}" for i in range(1, len(states))] + ["Referencia"]
        report.append(ascii_side_by_side(imgs, titles))
        report.append(f"Ciclos hasta estabilizar: {it} | energía: " + " -> ".join(f"{e:.1f}" for e in E))
        if d == 0:
            dx, dy = offsets[k]
            X_c, Y_c = ring_center(to_01(xr))
            X_0, Y_0 = ring_center(P01[offsets.index((0, 0))])
            report.append(f"Atractor alcanzado: aro desplazado ΔX={dx:+d}, ΔY={-dy:+d} píxeles "
                          f"[real: ΔX={offsets[true_k][0]:+d}, ΔY={-offsets[true_k][1]:+d}]")
            report.append(f"Centro 'A' respecto de la escuadra: X={X_c:.2f}, Y={Y_c:.2f} "
                          f"(nominal X={X_0:.2f}, Y={Y_0:.2f}) -> corrección para el robot: "
                          f"dX={X_c - X_0:+.2f}, dY={Y_c - Y_0:+.2f} píxeles")
        else:
            report.append(f"No converge a un patrón memorizado (estado espurio; Hamming {d} "
                          f"al más cercano) -> no se puede calcular el centro")
        report.append("")
        grid.append((X[true_k], test, xr))
        grid_titles.append(f"Aro desplazado {rule}")
        energy_curves[f"B_{rule}"] = E

    # Robustez estadística: cada posición, 20 pruebas con 10% de ruido
    report.append("Robustez (9 posiciones x 20 pruebas, 10% de ruido): posición identificada correctamente")
    for rule, net in nets:
        stable = sum(hamming_distance(net.recall(x, rng=rng)[0], x) == 0 for x in X)
        ok = 0
        for x in X:
            for _ in range(20):
                xr = net.recall(add_noise_pm1(x, 0.10, rng), rng=rng)[0]
                ok += hamming_distance(xr, x) == 0
        report.append(f"  {rule:13s}: patrones estables sin ruido {stable}/9 | aciertos con ruido {ok}/180")
    report.append("Los 9 aros comparten la escuadra y casi todo el fondo: son patrones muy correlacionados.")
    report.append("Hebb sufre interferencia cruzada (crosstalk) y ninguno queda estable; la pseudoinversa")
    report.append("los vuelve puntos fijos exactos y recupera la posición correcta.")
    report.append("")


def main(seed: int = 123):
    rng = np.random.default_rng(seed)
    report = ["Prototipo Hopfield 10x10 - TP3 Inteligencia Artificial", "(# = píxel activo, . = fondo)", ""]
    grid, grid_titles, energy_curves = [], [], {}
    experiment_a(rng, report, grid, grid_titles, energy_curves)
    experiment_b(rng, report, grid, grid_titles, energy_curves)

    visualize_triplets(grid, grid_titles, "hopfield_demo_results.png")
    plot_energy_curves(energy_curves, "hopfield_energy")
    text = "\n".join(report)
    with open("hopfield_report.txt", "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows (cp1252) no soporta ᵢ, Δ
    print(text)


if __name__ == "__main__":
    main()
