# *********************************************************
# Nombre: Felipe Lucena
# Legajo: VINF016997
# *********************************************************
#
# Transformada de Hough para detección de rectas
# -----------------------------------------------
# Idea central: cada píxel de borde "vota" por todas las rectas
# que podrían pasar por él. Las rectas reales acumulan muchos
# votos y aparecen como picos en el espacio (rho, theta).
#
# Parametrización de una recta:
#   x * cos(theta) + y * sin(theta) = rho
#   - rho   : distancia del origen al punto más cercano de la recta
#   - theta : ángulo de la normal a la recta (en radianes)

import numpy as np
import matplotlib.pyplot as plt


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


def hough_lines(binary_img, theta_res=1, rho_res=1):
    """
    Implementación de la transformada de Hough para rectas.

    Recorre cada píxel de borde y, para cada ángulo theta posible,
    calcula el rho correspondiente y suma un voto en el acumulador.

    Parámetros
    ----------
    binary_img : np.ndarray (2D)
        Imagen binaria donde los bordes son píxeles no nulos (> 0).
    theta_res : float
        Paso angular en grados (menor = más precisión, más lento).
    rho_res : float
        Paso de rho en píxeles (menor = más precisión, más lento).

    Retorna
    -------
    accumulator : np.ndarray
        Matriz de votos. Filas = rho, columnas = theta.
    thetas : np.ndarray
        Ángulos theta muestreados (en radianes).
    rhos : np.ndarray
        Valores de rho muestreados (en píxeles).
    """
    height, width = binary_img.shape

    # Discretización de theta: de -90° a 90° (excluye 90° para no duplicar
    # la misma orientación que -90°, ya que representan la misma familia de rectas)
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_res))

    # rho máximo posible = longitud de la diagonal de la imagen
    # (cualquier recta que cruce la imagen queda dentro de [-diag, +diag])
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)

    # Acumulador vacío: cada celda (rho_idx, theta_idx) cuenta votos
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    # Coordenadas (y, x) de todos los píxeles de borde (valor != 0)
    y_idxs, x_idxs = np.nonzero(binary_img)

    # Votación: cada píxel (x, y) contribuye a una curva en el espacio de Hough.
    # Para un theta fijo, hay un único rho = x*cos(theta) + y*sin(theta).
    for x, y in zip(x_idxs, y_idxs):
        for theta_idx, theta in enumerate(thetas):
            # Rho correspondiente a este píxel con el ángulo actual
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            # Offset +diag_len: convierte rho negativo en índice de fila >= 0
            rho_idx = rho + diag_len
            accumulator[rho_idx, theta_idx] += 1

    return accumulator, thetas, rhos


def find_peaks(accumulator, thetas, rhos, threshold, min_distance=10):
    """
    Extrae las rectas más votadas del acumulador de Hough.

    Estrategia:
    1. Tomar el máximo global del acumulador.
    2. Si supera el umbral, guardarlo como pico (rho, theta).
    3. Anular una ventana alrededor del pico (supresión de no-máximos)
       para no detectar la misma recta varias veces.
    4. Repetir hasta que el máximo restante quede bajo el umbral.

    Parámetros
    ----------
    accumulator : np.ndarray
        Matriz de votos de Hough.
    thetas, rhos : np.ndarray
        Ejes del espacio de parámetros (para convertir índices a valores).
    threshold : float
        Votos mínimos para considerar un pico como línea válida.
    min_distance : int
        Radio (en celdas del acumulador) de la zona a suprimir alrededor
        de cada pico detectado.

    Retorna
    -------
    list of (rho, theta, votes)
        Picos ordenados de mayor a menor cantidad de votos.
    """
    peaks = []
    num_rhos, num_thetas = accumulator.shape

    # Copia de trabajo: al detectar un pico, se pone a 0 su vecindario
    acc_copy = accumulator.copy()

    while True:
        # Máximo de votos restante en el acumulador
        max_val = acc_copy.max()

        # Si nadie supera el umbral, no hay más líneas relevantes
        if max_val < threshold:
            break

        # Índice (fila rho, columna theta) del máximo
        max_pos = np.unravel_index(acc_copy.argmax(), acc_copy.shape)
        rho_idx, theta_idx = max_pos

        # Convertir índices a parámetros reales de la recta
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        peaks.append((rho, theta, max_val))

        # Supresión de no-máximos: limpia el entorno del pico
        # (evita contar la misma recta o líneas casi idénticas)
        rho_min = max(0, rho_idx - min_distance)
        rho_max = min(num_rhos, rho_idx + min_distance + 1)
        theta_min = max(0, theta_idx - min_distance)
        theta_max = min(num_thetas, theta_idx + min_distance + 1)

        acc_copy[rho_min:rho_max, theta_min:theta_max] = 0

    return peaks


def plot_results(img, accumulator, thetas, rhos, peaks):
    """
    Visualiza el resultado en dos paneles:
    - Izquierda: imagen original con las rectas detectadas superpuestas.
    - Derecha: acumulador de Hough (mapa de calor) con los picos marcados.
    """
    fig, (ax_img, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: imagen + rectas ---
    ax_img.imshow(img, cmap='gray')
    ax_img.set_title(f"Imagen original - {len(peaks)} líneas detectadas")
    ax_img.set_axis_off()

    height, width = img.shape
    # Un color distinto por cada línea detectada
    colors = plt.cm.rainbow(np.linspace(0, 1, len(peaks)))

    for idx, (rho, theta, votes) in enumerate(peaks):
        # Punto sobre la recta más cercano al origen: (rho*cos θ, rho*sin θ)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Dirección perpendicular a la normal (a, b): vector (-b, a).
        # Se alarga ±1000 px para dibujar la recta atravesando toda la imagen.
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        ax_img.plot((x1, x2), (y1, y2), linewidth=2, color=colors[idx],
                   label=f'L{idx+1}: ρ={rho:.1f}, θ={np.rad2deg(theta):.1f}°, votos={votes}')

    if peaks:
        ax_img.legend(loc='upper right', fontsize=8)

    # --- Panel 2: espacio de Hough ---
    # extent mapea índices de la matriz a grados (θ) y píxeles (ρ)
    extent = [np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]]
    ax_acc.imshow(accumulator, cmap='hot', aspect='auto', extent=extent)
    ax_acc.set_title("Acumulador de Hough")
    ax_acc.set_xlabel("θ (grados)")
    ax_acc.set_ylabel("ρ (píxeles)")

    # Cruz cian sobre cada pico encontrado
    for rho, theta, votes in peaks:
        ax_acc.plot(np.rad2deg(theta), rho, 'cx', markersize=10, markeredgewidth=2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Pipeline completo de detección de líneas

    # 1) Imagen sintética con dos rectas conocidas
    img = generate_test_image()

    # 2) Ya es binaria (bordes = blanco); en un caso real aquí iría Canny u otro detector
    binary_img = img

    # 3) Votación en el espacio (rho, theta)
    print("Aplicando transformada de Hough...")
    accumulator, thetas, rhos = hough_lines(binary_img,
                                            theta_res=1,
                                            rho_res=1)

    # 4) Extraer picos: threshold=80 exige al menos 80 votos por línea;
    #    min_distance=10 separa picos cercanos en el acumulador
    print("Buscando líneas en el acumulador...")
    peaks = find_peaks(accumulator, thetas, rhos, threshold=80, min_distance=10)

    print(f"\nSe detectaron {len(peaks)} líneas:")
    for idx, (rho, theta, votes) in enumerate(peaks):
        print(f"  Línea {idx+1}: ρ={rho:.2f} píxeles, θ={np.rad2deg(theta):.2f}°, votos={votes}")

    # 5) Mostrar imagen + acumulador
    plot_results(img, accumulator, thetas, rhos, peaks)
