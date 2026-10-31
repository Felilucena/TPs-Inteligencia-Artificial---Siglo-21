# *********************************************************
# Nombre: Felipe Lucena
# Legajo: VINF016997
# *********************************************************
"""
Motor de la transformada de Hough (rectas y circunferencias)
------------------------------------------------------------
Idea central: cada píxel de borde "vota" por todas las figuras que
podrían pasar por él. Las figuras reales acumulan muchos votos y
aparecen como picos en el espacio de parámetros.

  - Rectas:          x * cos(theta) + y * sin(theta) = rho
  - Circunferencias: (x - xo)^2 + (y - yo)^2 = R^2

Modos (--modo):
  validar   : imágenes sintéticas con resultado conocido (rectas y círculo).
  rectas    : rectas sobre una foto; el ángulo theta dominante estima la
              rotación del block sobre el eje Z.
  circulos  : centro A = (X, Y) y radio del aro sobre una foto.

Pipeline sobre la foto (todo implementado con NumPy, sin OpenCV):
  load_image -> recorte -> mask_annotations -> to_gray -> gaussian_blur
  -> sobel -> umbral -> Hough

Salidas (carpeta salidas/): <modo>_preproceso.png, <modo>_resultado.png
y <modo>_reporte.txt.

Ejemplos:
  python hough_motor.py --modo validar
  python hough_motor.py --modo rectas --imagen foto.jpg --recorte 820,120,1260,560
  python hough_motor.py --modo circulos --imagen foto.jpg --recorte 820,120,1260,560 --r-min 60 --r-max 90
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# 1. Entrada
# =========================================================

def generate_test_image(width=200, height=200):
    """
    Genera una imagen binaria de prueba con dos líneas blancas
    sobre fondo negro: una horizontal y una diagonal.

    Sirve para validar la transformada sin depender de una
    imagen externa ni de un detector de bordes previo.
    """
    # Fondo negro (valor 0)
    img = np.zeros((height, width), dtype=np.uint8)

    # Línea horizontal en el centro (fila fija, columnas 20..179)
    img[height // 2, 20:180] = 255

    # Línea diagonal: recorre x=i e y=height-i-1 (de abajo-izq a arriba-der)
    for i in range(40, 160):
        y = height - i - 1
        x = i
        if 0 <= y < height and 0 <= x < width:
            img[y, x] = 255

    return img


def generate_test_circle(width=200, height=200, center=(110, 90), radius=40):
    """
    Genera una imagen RGB de prueba: un disco claro sobre fondo oscuro,
    con ruido gaussiano leve. A diferencia de generate_test_image, no es
    binaria: pasa por todo el preprocesamiento (suavizado, Sobel, umbral),
    así que valida también esa etapa.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    disk = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    gray = np.where(disk, 200.0, 60.0)
    # Semilla fija: la validación tiene que ser reproducible
    rng = np.random.default_rng(0)
    gray += rng.normal(0, 8, gray.shape)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def load_image(path):
    """Lee una imagen con Pillow y la devuelve como array RGB (alto, ancho, 3)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def crop(rgb, box):
    """
    Recorta la región box = (x0, y0, x1, y1) en píxeles.
    Devuelve el recorte y su desplazamiento (x0, y0), necesario para
    expresar los resultados en coordenadas de la imagen completa.
    """
    if box is None:
        return rgb, (0, 0)
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = box
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"El recorte {box} queda vacío en una imagen de {width}x{height}")
    return rgb[y0:y1, x0:x1], (x0, y0)


# =========================================================
# 2. Preprocesamiento (bajo nivel)
# =========================================================

def to_gray(rgb):
    """Luminancia (ITU-R BT.601): pondera más el verde, al que el ojo es más sensible."""
    rgb_f = rgb.astype(np.float64)
    return 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]


def dilate(mask, radius):
    """
    Dilatación binaria: un píxel queda marcado si algún vecino a distancia
    <= radius (ventana cuadrada) está marcado. Se implementa sumando la
    máscara desplazada en todas las direcciones.
    """
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant")
    height, width = mask.shape
    out = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + height, dx:dx + width]
    return out


def mask_annotations(rgb, sat_threshold=0.35, grow=2):
    """
    Elimina las anotaciones de color (trazos amarillos, flechas, etc.).

    Saturación por píxel: S = (max - min) / max sobre los canales RGB.
    La pieza y el fondo son casi grises (S baja); una anotación de color
    tiene S alta. Los píxeles con S > sat_threshold se reemplazan por un
    gris neutro: la mediana de gris de la escena sin anotar, para que el
    parche genere el menor salto de intensidad posible (= menos bordes falsos).

    grow: radio (px) de dilatación de la máscara, para tapar el halo de
    compresión que rodea a cada trazo de color.

    Retorna (rgb_limpio, mascara).
    """
    rgb_f = rgb.astype(np.float64) / 255.0
    c_max = rgb_f.max(axis=2)
    c_min = rgb_f.min(axis=2)
    saturation = np.where(c_max > 0, (c_max - c_min) / np.maximum(c_max, 1e-12), 0.0)
    # La compresión JPEG deja un "halo" de color pálido (poca saturación)
    # alrededor de cada trazo: se dilata la máscara para cubrirlo también.
    mask = dilate(saturation > sat_threshold, grow)

    if mask.any() and (~mask).any():
        neutral = np.median(to_gray(rgb)[~mask])
    else:
        neutral = 128.0

    clean = rgb.copy()
    clean[mask] = np.uint8(round(neutral))
    return clean, mask


def convolve2d(img, kernel):
    """
    Desliza el núcleo sobre la imagen y suma los productos (correlación).
    En vez de recorrer píxel por píxel, recorre las celdas del núcleo y
    suma la imagen desplazada completa: |núcleo| operaciones vectorizadas.

    Para núcleos simétricos (gaussiano) coincide con la convolución; en
    Sobel solo cambiaría el signo, y los núcleos de abajo están definidos
    para que Gx > 0 cuando la intensidad crece hacia la derecha.
    Los bordes de la imagen se rellenan por reflexión.
    """
    k_h, k_w = kernel.shape
    pad_y, pad_x = k_h // 2, k_w // 2
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    height, width = img.shape
    out = np.zeros((height, width), dtype=np.float64)
    for i in range(k_h):
        for j in range(k_w):
            if kernel[i, j] != 0:
                out += kernel[i, j] * padded[i:i + height, j:j + width]
    return out


def gaussian_blur(img, sigma):
    """
    Suavizado gaussiano para atenuar el ruido antes de derivar.
    El gaussiano es separable: dos pasadas 1D (filas y columnas) dan el
    mismo resultado que el núcleo 2D con mucho menos cálculo.
    """
    if sigma <= 0:
        return img.astype(np.float64)
    radius = int(np.ceil(3 * sigma))  # ±3σ contiene el 99,7% del peso
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    out = convolve2d(img, kernel[np.newaxis, :])
    return convolve2d(out, kernel[:, np.newaxis])


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def sobel(img):
    """
    Gradiente de Sobel.
    Retorna (magnitud, direccion): la dirección (radianes) es la normal
    al borde, que apunta hacia donde la intensidad aumenta.
    """
    gx = convolve2d(img, SOBEL_X)
    gy = convolve2d(img, SOBEL_Y)
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def non_max_suppression(magnitude, direction):
    """
    Adelgaza los bordes a 1 px (la idea central del detector de Canny).
    Un píxel se conserva solo si su magnitud es mayor o igual que la de sus
    dos vecinos en la dirección del gradiente (perpendicular al borde).
    La dirección se cuantiza en 4 sectores: 0°, 45°, 90° y 135°.
    Sin este paso, cada borde real tiene 3-5 px de ancho y una textura
    rugosa (fundición) aporta miles de píxeles que votan rectas falsas.
    """
    angle = (np.rad2deg(direction) + 180.0) % 180.0
    sector = (np.round(angle / 45.0).astype(int)) % 4
    # Desplazamientos (dy, dx) del vecino para cada sector
    neighbours = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    padded = np.pad(magnitude, 1, mode="constant")
    height, width = magnitude.shape
    keep = np.zeros(magnitude.shape, dtype=bool)
    for sec, (dy, dx) in neighbours.items():
        before = padded[1 - dy:1 - dy + height, 1 - dx:1 - dx + width]
        after = padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        local_max = (magnitude >= before) & (magnitude >= after)
        keep |= (sector == sec) & local_max
    return np.where(keep, magnitude, 0.0)


def threshold_edges(magnitude, rel_threshold):
    """Binariza: es borde todo píxel con magnitud > rel_threshold * máximo."""
    max_val = magnitude.max()
    if max_val == 0:
        return np.zeros(magnitude.shape, dtype=bool)
    return magnitude > rel_threshold * max_val


def preprocess(rgb, sigma, sat_threshold, edge_threshold, thin=True, grow=2):
    """Encadena todas las etapas y devuelve cada resultado intermedio (para el informe)."""
    clean, mask = mask_annotations(rgb, sat_threshold, grow)
    gray = to_gray(clean)
    smooth = gaussian_blur(gray, sigma)
    magnitude, direction = sobel(smooth)
    thinned = non_max_suppression(magnitude, direction) if thin else magnitude
    edges = threshold_edges(thinned, edge_threshold)
    return {"rgb": rgb, "clean": clean, "mask": mask, "gray": gray, "magnitude": magnitude,
            "direction": direction, "edges": edges}


# =========================================================
# 3. Hough para rectas
# =========================================================

def hough_lines(binary_img, theta_res=1, rho_res=1, chunk_size=20000):
    """
    Transformada de Hough para rectas.

    Cada píxel de borde (x, y) vota, para cada theta, por el
    rho = x*cos(theta) + y*sin(theta). La votación está vectorizada:
    se calcula la matriz (píxeles x thetas) de rhos de una sola vez y
    np.add.at suma los votos (a diferencia de acc[idx] += 1, acumula
    correctamente los índices repetidos). Se procesa por bloques de
    chunk_size píxeles para acotar la memoria.

    Parámetros
    ----------
    binary_img : np.ndarray (2D)
        Imagen binaria donde los bordes son píxeles no nulos.
    theta_res : float
        Paso angular en grados (menor = más precisión, más lento).
    rho_res : float
        Paso de rho en píxeles (menor = más precisión, más lento).

    Retorna
    -------
    accumulator : np.ndarray  (filas = rho, columnas = theta)
    thetas : np.ndarray       ángulos muestreados (radianes)
    rhos : np.ndarray         valores de rho muestreados (píxeles)
    """
    height, width = binary_img.shape

    # theta en [-90°, 90°): -90° y 90° son la misma familia de rectas
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_res))

    # Cualquier recta que cruce la imagen tiene rho en [-diag, +diag]
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    num_rhos = int(np.ceil(2 * diag_len / rho_res)) + 1
    rhos = -diag_len + np.arange(num_rhos) * rho_res

    accumulator = np.zeros((num_rhos, len(thetas)), dtype=np.int64)

    y_idxs, x_idxs = np.nonzero(binary_img)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for start in range(0, len(x_idxs), chunk_size):
        xs = x_idxs[start:start + chunk_size, np.newaxis]
        ys = y_idxs[start:start + chunk_size, np.newaxis]
        rho = xs * cos_t + ys * sin_t  # (píxeles, thetas)
        # Índice de celda: desplaza rho a >= 0 y lo divide por el paso
        rho_idx = np.round((rho + diag_len) / rho_res).astype(int)
        theta_idx = np.broadcast_to(np.arange(len(thetas)), rho_idx.shape)
        np.add.at(accumulator, (rho_idx, theta_idx), 1)

    return accumulator, thetas, rhos


def find_peaks(accumulator, thetas, rhos, threshold, min_distance=10, max_peaks=None):
    """
    Extrae las rectas más votadas del acumulador de Hough.

    Estrategia:
    1. Tomar el máximo global del acumulador.
    2. Si supera el umbral, guardarlo como pico (rho, theta).
    3. Anular una ventana alrededor del pico (supresión de no-máximos)
       para no detectar la misma recta varias veces.
    4. Repetir hasta que el máximo restante quede bajo el umbral
       (o hasta juntar max_peaks picos).

    Retorna
    -------
    list of (rho, theta, votes)
        Picos ordenados de mayor a menor cantidad de votos.
    """
    peaks = []
    num_rhos, num_thetas = accumulator.shape
    acc_copy = accumulator.copy()

    while max_peaks is None or len(peaks) < max_peaks:
        max_val = acc_copy.max()
        if max_val < threshold or max_val == 0:
            break

        rho_idx, theta_idx = np.unravel_index(acc_copy.argmax(), acc_copy.shape)
        peaks.append((rhos[rho_idx], thetas[theta_idx], int(max_val)))

        rho_min = max(0, rho_idx - min_distance)
        rho_max = min(num_rhos, rho_idx + min_distance + 1)
        theta_min = max(0, theta_idx - min_distance)
        theta_max = min(num_thetas, theta_idx + min_distance + 1)
        acc_copy[rho_min:rho_max, theta_min:theta_max] = 0

    return peaks


def estimate_z_rotation(peaks):
    """
    Estima la rotación del block sobre Z a partir de las rectas detectadas.

    Los lados de un block son paralelos o perpendiculares entre sí, así que
    theta y theta + 90° indican la misma rotación: el ángulo tiene período
    90°. Se promedia como ángulo circular sobre 4*theta (que tiene período
    360°), ponderando por votos, y se divide por 4. Así 44° y -44° promedian
    cerca de ±45° y no en 0°, como pasaría con un promedio común.

    Retorna la rotación en grados, en (-45°, 45°], o None si no hay rectas.
    """
    if not peaks:
        return None
    thetas = np.array([theta for _, theta, _ in peaks])
    weights = np.array([votes for _, _, votes in peaks], dtype=np.float64)
    s = np.sum(weights * np.sin(4 * thetas))
    c = np.sum(weights * np.cos(4 * thetas))
    return float(np.rad2deg(np.arctan2(s, c) / 4))


# =========================================================
# 5. Hough para circunferencias
# =========================================================

def hough_circles(edges, direction, r_min, r_max, r_step=1,
                  use_gradient=True, angle_tol_deg=3.0):
    """
    Transformada de Hough para circunferencias, acumulador 3D (R, yo, xo).

    El rango de radios es corto a propósito: por la perspectiva el aro se
    ve como una elipse, y un rango acotado alrededor de su radio aparente
    evita picos falsos con otros radios.

    Votación:
    - use_gradient=True: el centro de una circunferencia está sobre la
      normal al borde, que es la dirección del gradiente de Sobel. Cada
      píxel vota solo a distancia R en esa dirección (en ambos sentidos,
      porque no sabemos si el interior es más claro u oscuro), con una
      tolerancia de ±angle_tol_deg para absorber el ruido del gradiente
      y la deformación elíptica. Son 6 votos por píxel y radio en lugar
      de 180: más rápido y con menos picos falsos.
    - use_gradient=False: votación clásica sobre toda la circunferencia
      (cada 2°), para comparar en el informe.

    Retorna (accumulator, radii).
    """
    height, width = edges.shape
    radii = np.arange(r_min, r_max + 1, r_step)
    accumulator = np.zeros((len(radii), height, width), dtype=np.int32)

    y_idxs, x_idxs = np.nonzero(edges)
    # angles: matriz (píxeles, votos por píxel) con las direcciones en que vota cada uno
    if use_gradient:
        phi = direction[y_idxs, x_idxs, np.newaxis]
        offsets = np.deg2rad(np.unique([-angle_tol_deg, 0.0, angle_tol_deg]))
        angles = np.concatenate([phi + offsets, phi + offsets + np.pi], axis=1)
    else:
        circle_angles = np.deg2rad(np.arange(0, 360, 2))
        angles = np.tile(circle_angles, (len(x_idxs), 1))
    xs = np.repeat(x_idxs, angles.shape[1])
    ys = np.repeat(y_idxs, angles.shape[1])
    angles = angles.ravel()

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    for r_idx, radius in enumerate(radii):
        xc = np.round(xs + radius * cos_a).astype(int)
        yc = np.round(ys + radius * sin_a).astype(int)
        inside = (xc >= 0) & (xc < width) & (yc >= 0) & (yc < height)
        np.add.at(accumulator[r_idx], (yc[inside], xc[inside]), 1)

    return accumulator, radii


def find_circles(accumulator, radii, n_circles=1, min_distance=10, smooth_sigma=1.0):
    """
    Extrae las circunferencias más votadas.

    - Cada capa de radio se suaviza con un gaussiano: en una elipse los
      votos no caen en un único píxel sino en una mancha alrededor del
      centro; el suavizado los concentra en un pico.
    - El puntaje se normaliza por el perímetro 2*pi*R: una circunferencia
      grande tiene más píxeles de borde y, sin normalizar, siempre ganaría.
    - Con una elipse, ningún R encaja en todo el contorno: los votos del
      eje menor caen de un lado del centro y los del eje mayor del otro,
      y el pico se parte en dos lóbulos simétricos respecto del centro
      real. Por eso el centro no es el argmax (que elige un lóbulo) sino
      el centroide ponderado de las celdas con al menos la mitad del pico,
      en una ventana de semiancho R alrededor del máximo (subpíxel).

    Retorna list of (xo, yo, R, votes, score), score = votos / perímetro.
    """
    score = np.empty(accumulator.shape, dtype=np.float64)
    for r_idx, radius in enumerate(radii):
        score[r_idx] = gaussian_blur(accumulator[r_idx].astype(np.float64), smooth_sigma)
        score[r_idx] /= 2 * np.pi * radius

    circles = []
    _, height, width = accumulator.shape
    for _ in range(n_circles):
        r_idx, y_max, x_max = np.unravel_index(score.argmax(), score.shape)
        peak = score[r_idx, y_max, x_max]
        if peak <= 0:
            break
        radius = int(radii[r_idx])

        half = max(min_distance, radius)
        y0, y1 = max(0, y_max - half), min(height, y_max + half + 1)
        x0, x1 = max(0, x_max - half), min(width, x_max + half + 1)
        patch = score[r_idx, y0:y1, x0:x1]
        weights = np.where(patch >= 0.5 * peak, patch, 0.0)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        xo = float((weights * xx).sum() / weights.sum())
        yo = float((weights * yy).sum() / weights.sum())

        circles.append((xo, yo, radius, int(accumulator[r_idx, y_max, x_max]), float(peak)))
        # Supresión de no-máximos en la ventana (ambos lóbulos) para todos los radios
        score[:, y0:y1, x0:x1] = 0
    return circles


# =========================================================
# 4. Visualización y reporte
# =========================================================

def _show_image(ax, img):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)


def plot_preprocessing(stages, out_path):
    """Guarda las etapas del preprocesamiento lado a lado."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [(stages["rgb"], "Entrada (recorte)"),
              (stages["clean"], "Anotaciones enmascaradas"),
              (stages["magnitude"], "Magnitud de Sobel"),
              (stages["edges"], f"Bordes ({int(stages['edges'].sum())} px)")]
    for ax, (img, title) in zip(axes, panels):
        _show_image(ax, img)
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lines(img, accumulator, thetas, rhos, peaks, out_path):
    """
    Izquierda: imagen con las rectas detectadas superpuestas.
    Derecha: acumulador de Hough con los picos marcados.
    """
    fig, (ax_img, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    _show_image(ax_img, img)
    ax_img.set_title(f"{len(peaks)} rectas detectadas")
    ax_img.set_axis_off()

    height, width = img.shape[:2]
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(peaks), 1)))
    reach = 2 * (height + width)  # largo suficiente para cruzar toda la imagen

    for idx, (rho, theta, votes) in enumerate(peaks):
        # Punto de la recta más cercano al origen y dirección (-sin θ, cos θ)
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        ax_img.plot((x0 + reach * (-b), x0 - reach * (-b)),
                    (y0 + reach * a, y0 - reach * a),
                    linewidth=2, color=colors[idx],
                    label=f"L{idx+1}: ρ={rho:.1f}, θ={np.rad2deg(theta):.1f}°, votos={votes}")

    # Las rectas se dibujan muy largas: sin esto matplotlib agranda los ejes
    # para mostrarlas completas y la imagen queda chica en una esquina
    ax_img.set_xlim(0, width)
    ax_img.set_ylim(height, 0)

    if peaks:
        ax_img.legend(loc="upper right", fontsize=8)

    extent = [np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]]
    ax_acc.imshow(accumulator, cmap="hot", aspect="auto", extent=extent)
    ax_acc.set_title("Acumulador de Hough")
    ax_acc.set_xlabel("θ (grados)")
    ax_acc.set_ylabel("ρ (píxeles)")
    for rho, theta, _ in peaks:
        ax_acc.plot(np.rad2deg(theta), rho, "cx", markersize=10, markeredgewidth=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_circles(img, accumulator, radii, circles, out_path):
    """
    Izquierda: imagen con las circunferencias y el centro A marcados.
    Derecha: capa del acumulador correspondiente al radio ganador.
    """
    fig, (ax_img, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    _show_image(ax_img, img)
    ax_img.set_title(f"{len(circles)} circunferencia(s) detectada(s)")
    ax_img.set_axis_off()
    for idx, (xo, yo, radius, _, _) in enumerate(circles):
        ax_img.add_patch(plt.Circle((xo, yo), radius, fill=False, color="cyan", linewidth=2))
        ax_img.plot(xo, yo, "r+", markersize=14, markeredgewidth=2,
                    label=f"A{idx+1} = ({xo:.1f}, {yo:.1f}), R={radius}")
    height, width = img.shape[:2]
    ax_img.set_xlim(0, width)
    ax_img.set_ylim(height, 0)
    if circles:
        ax_img.legend(loc="upper right", fontsize=8)

    best_r_idx = int(np.argmax(radii == circles[0][2])) if circles else 0
    ax_acc.imshow(accumulator[best_r_idx], cmap="hot")
    ax_acc.set_title(f"Acumulador de Hough, capa R={radii[best_r_idx]}")
    ax_acc.set_xlabel("xo (píxeles)")
    ax_acc.set_ylabel("yo (píxeles)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(lines, out_path):
    """Guarda el reporte en .txt y lo devuelve como texto (para imprimirlo)."""
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return text


# =========================================================
# 6. Modos de ejecución
# =========================================================

def _line_report(peaks):
    report = [f"Rectas detectadas: {len(peaks)}"]
    for idx, (rho, theta, votes) in enumerate(peaks):
        report.append(f"  L{idx+1}: ρ = {rho:8.2f} px   θ = {np.rad2deg(theta):7.2f}°   votos = {votes}")
    return report


def _circle_report(circles, offset):
    report = [f"Circunferencias detectadas: {len(circles)}"]
    for idx, (xo, yo, radius, votes, score) in enumerate(circles):
        report.append(f"  A{idx+1} = (X, Y) = ({xo:.1f}, {yo:.1f}) px en el recorte"
                      f"  ->  ({xo + offset[0]:.1f}, {yo + offset[1]:.1f}) px en la imagen completa")
        report.append(f"       R = {radius} px   votos = {votes}   puntaje (votos/perímetro) = {score:.3f}")
    return report


def _load_input(args):
    rgb = load_image(args.imagen)
    cropped, offset = crop(rgb, args.recorte)
    report = [f"Imagen: {args.imagen} ({rgb.shape[1]}x{rgb.shape[0]} px)",
              f"Recorte: {args.recorte if args.recorte else 'imagen completa'} "
              f"-> {cropped.shape[1]}x{cropped.shape[0]} px, desplazamiento {offset}"]
    return cropped, offset, report


def _preprocess_report(args, stages):
    return [f"Preprocesamiento: saturación > {args.umbral_saturacion} = anotación "
            f"({int(stages['mask'].sum())} px enmascarados), σ = {args.sigma}, "
            f"adelgazamiento = {'no' if args.sin_adelgazar else 'sí'}, "
            f"umbral de borde = {args.umbral_borde} x máx. -> {int(stages['edges'].sum())} px de borde"]


def run_validation(args, out_dir):
    report = ["=== MODO VALIDAR ===", "",
              "1) Rectas sobre imagen binaria sintética (generate_test_image)",
              "   Esperado: horizontal y = 100 (θ = -90°, ρ = -100)"
              " y diagonal x + y = 199 (θ = 45°, ρ ≈ 140,7)"]
    img = generate_test_image()
    accumulator, thetas, rhos = hough_lines(img, args.theta_res, args.rho_res)
    threshold = args.umbral_votos if args.umbral_votos is not None else 80
    peaks = find_peaks(accumulator, thetas, rhos, threshold, args.min_distancia, args.max_rectas)
    report += ["   " + line for line in _line_report(peaks)]
    plot_lines(img, accumulator, thetas, rhos, peaks,
               os.path.join(out_dir, "validar_rectas_resultado.png"))

    center, radius = (110, 90), 40
    report += ["", "2) Circunferencia sobre imagen sintética con ruido (generate_test_circle)",
               f"   Esperado: centro A = {center}, R = {radius}"]
    rgb = generate_test_circle(center=center, radius=radius)
    stages = preprocess(rgb, args.sigma, args.umbral_saturacion, args.umbral_borde, not args.sin_adelgazar, args.dilatar_mascara)
    plot_preprocessing(stages, os.path.join(out_dir, "validar_circulo_preproceso.png"))
    r_min = args.r_min if args.r_min is not None else radius - 10
    r_max = args.r_max if args.r_max is not None else radius + 10
    accumulator, radii = hough_circles(stages["edges"], stages["direction"], r_min, r_max,
                                       args.r_paso, not args.votar_circunferencia,
                                       args.tolerancia_angular)
    circles = find_circles(accumulator, radii, args.n_circulos, args.min_distancia)
    report += ["   " + line for line in _circle_report(circles, (0, 0))]
    plot_circles(rgb, accumulator, radii, circles,
                 os.path.join(out_dir, "validar_circulo_resultado.png"))
    return report


def run_lines(args, out_dir):
    img, _, report = _load_input(args)
    report = ["=== MODO RECTAS ===", ""] + report
    stages = preprocess(img, args.sigma, args.umbral_saturacion, args.umbral_borde, not args.sin_adelgazar, args.dilatar_mascara)
    report += _preprocess_report(args, stages)
    plot_preprocessing(stages, os.path.join(out_dir, "rectas_preproceso.png"))

    accumulator, thetas, rhos = hough_lines(stages["edges"], args.theta_res, args.rho_res)
    # Sin umbral explícito: la mitad del pico máximo (se adapta al tamaño de la foto)
    threshold = args.umbral_votos if args.umbral_votos is not None else 0.5 * accumulator.max()
    peaks = find_peaks(accumulator, thetas, rhos, threshold, args.min_distancia, args.max_rectas)
    report += [f"Hough: Δθ = {args.theta_res}°, Δρ = {args.rho_res} px, "
               f"umbral = {threshold:.0f} votos", ""]
    report += _line_report(peaks)

    rotation = estimate_z_rotation(peaks)
    report.append("")
    if rotation is None:
        report.append("Rotación sobre Z: sin rectas, no se puede estimar.")
    else:
        dominant = np.rad2deg(peaks[0][1])
        # θ es el ángulo de la normal: la recta forma θ - 90° con la horizontal
        incline = dominant - 90.0 if dominant > 0 else dominant + 90.0
        report += [f"θ del borde dominante (L1): {dominant:.2f}°  ->  "
                   f"el borde forma {incline:.2f}° con la horizontal de la imagen",
                   "  (con el eje y hacia abajo, un ángulo negativo sube hacia la derecha)",
                   f"Rotación sobre Z (θ mod 90°, promedio ponderado): {rotation:.2f}°",
                   "  Válida solo con cámara perpendicular a la cara (vista en planta): en una foto",
                   "  en perspectiva los lados paralelos no se ven paralelos y el promedio mezcla",
                   "  familias de rectas distintas. En ese caso usar la inclinación del borde",
                   "  dominante y compararla con la de un block de referencia bien posicionado."]
    plot_lines(img, accumulator, thetas, rhos, peaks, os.path.join(out_dir, "rectas_resultado.png"))
    return report


def run_circles(args, out_dir):
    img, offset, report = _load_input(args)
    report = ["=== MODO CIRCULOS ===", ""] + report
    stages = preprocess(img, args.sigma, args.umbral_saturacion, args.umbral_borde, not args.sin_adelgazar, args.dilatar_mascara)
    report += _preprocess_report(args, stages)
    plot_preprocessing(stages, os.path.join(out_dir, "circulos_preproceso.png"))

    accumulator, radii = hough_circles(stages["edges"], stages["direction"], args.r_min, args.r_max,
                                       args.r_paso, not args.votar_circunferencia,
                                       args.tolerancia_angular)
    voting = ("toda la circunferencia" if args.votar_circunferencia
              else f"dirección del gradiente ±{args.tolerancia_angular}°")
    report += [f"Hough: R en [{args.r_min}, {args.r_max}] paso {args.r_paso}, votación por {voting}", ""]
    circles = find_circles(accumulator, radii, args.n_circulos, args.min_distancia)
    report += _circle_report(circles, offset)
    plot_circles(img, accumulator, radii, circles, os.path.join(out_dir, "circulos_resultado.png"))
    return report


def parse_box(text):
    try:
        values = [int(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != 4:
        raise argparse.ArgumentTypeError("el recorte debe ser x0,y0,x1,y1 (enteros, en píxeles)")
    return tuple(values)


def build_parser():
    parser = argparse.ArgumentParser(description="Transformada de Hough (rectas y circunferencias) con NumPy.")
    parser.add_argument("--modo", choices=["validar", "rectas", "circulos"], default="validar")
    parser.add_argument("--imagen", help="ruta de la foto (obligatoria en rectas y circulos)")
    parser.add_argument("--recorte", type=parse_box, help="región x0,y0,x1,y1 en píxeles (p. ej. el recuadro ampliado)")
    parser.add_argument("--salidas", default="salidas", help="carpeta de salida (relativa al script)")

    pre = parser.add_argument_group("preprocesamiento")
    pre.add_argument("--umbral-saturacion", type=float, default=0.35,
                     help="saturación a partir de la cual un píxel es anotación (default 0.35)")
    pre.add_argument("--dilatar-mascara", type=int, default=2,
                     help="radio en px para agrandar la máscara de anotaciones (default 2)")
    pre.add_argument("--sin-adelgazar", action="store_true",
                     help="no aplicar supresión de no-máximos a los bordes (para comparar)")
    pre.add_argument("--sigma", type=float, default=1.5, help="σ del suavizado gaussiano (default 1.5)")
    pre.add_argument("--umbral-borde", type=float, default=0.25,
                     help="fracción de la magnitud máxima de Sobel para ser borde (default 0.25)")

    lines = parser.add_argument_group("rectas")
    lines.add_argument("--theta-res", type=float, default=1.0, help="paso de θ en grados (default 1)")
    lines.add_argument("--rho-res", type=float, default=1.0, help="paso de ρ en píxeles (default 1)")
    lines.add_argument("--umbral-votos", type=float,
                       help="votos mínimos por recta (default: 80 en validar, 50%% del máximo en rectas)")
    lines.add_argument("--max-rectas", type=int, default=8, help="cantidad máxima de rectas (default 8)")
    parser.add_argument("--min-distancia", type=int, default=10,
                        help="ventana de supresión de no-máximos, en celdas (default 10)")

    circles = parser.add_argument_group("circulos")
    circles.add_argument("--r-min", type=int, help="radio mínimo en píxeles")
    circles.add_argument("--r-max", type=int, help="radio máximo en píxeles")
    circles.add_argument("--r-paso", type=int, default=1, help="paso de radio (default 1)")
    circles.add_argument("--tolerancia-angular", type=float, default=3.0,
                         help="± grados alrededor del gradiente al votar (default 3)")
    circles.add_argument("--votar-circunferencia", action="store_true",
                         help="votar en toda la circunferencia en vez de en la dirección del gradiente")
    circles.add_argument("--n-circulos", type=int, default=1, help="cantidad de circunferencias (default 1)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.modo in ("rectas", "circulos") and not args.imagen:
        parser.error(f"--imagen es obligatoria en el modo {args.modo}")
    if args.modo == "circulos":
        if args.r_min is None or args.r_max is None:
            parser.error("el modo circulos necesita --r-min y --r-max")
        if not 0 < args.r_min <= args.r_max:
            parser.error("se requiere 0 < --r-min <= --r-max")

    out_dir = args.salidas if os.path.isabs(args.salidas) else os.path.join(SCRIPT_DIR, args.salidas)
    os.makedirs(out_dir, exist_ok=True)

    runners = {"validar": run_validation, "rectas": run_lines, "circulos": run_circles}
    report = runners[args.modo](args, out_dir)
    text = write_report(report, os.path.join(out_dir, f"{args.modo}_reporte.txt"))

    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows (cp1252) no soporta θ, ρ, σ
    print(text)
    print(f"\nSalidas guardadas en: {out_dir}")


if __name__ == "__main__":
    main()
