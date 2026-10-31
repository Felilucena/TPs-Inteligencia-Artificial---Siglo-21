# TP4 - Transformada de Hough (rectas y circunferencias)

Inteligencia Artificial - Universidad Siglo 21 - Felipe Lucena

Prototipo que aplica la transformada de Hough sobre la foto del block del caso de la industria de motores:

- **Rectas**: detecta los bordes rectos del block y su inclinación en la imagen (referencia para la rotación sobre Z).
- **Circunferencias**: detecta el aro "C" y devuelve las coordenadas del centro **A = (X, Y)**.

Todo el procesamiento de imagen está implementado con NumPy, sin OpenCV ni otras "cajas negras". Pillow solo se usa para leer la imagen y Matplotlib para las figuras.

## Archivos

| Archivo | Contenido |
|---|---|
| `hough_motor.py` | Prototipo completo (validación, rectas, circunferencias) |
| `block_canvas.png` | Foto de entrada: Figura 2 de la situación problemática (750×395 px) |
| `salidas/` | Figuras y reportes generados por cada corrida |
| `hough-rectas.py` | Primera versión (solo rectas sobre imagen sintética), se conserva como antecedente |

## Instalación

```bash
python -m venv .venv
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Linux / macOS:        source .venv/bin/activate
pip install numpy matplotlib pillow
```

## Uso

```bash
# 1) Validación con imágenes sintéticas de resultado conocido
python hough_motor.py --modo validar

# 2) Rectas sobre el block (se excluye el recuadro ampliado de la derecha)
python hough_motor.py --modo rectas --imagen block_canvas.png --recorte 0,0,440,395 --max-rectas 6 --umbral-borde 0.3

# 3) Circunferencia: aro "C" y centro A
python hough_motor.py --modo circulos --imagen block_canvas.png --recorte 240,150,360,250 --r-min 17 --r-max 26
```

`python hough_motor.py -h` lista todos los parámetros. Cada corrida guarda en `salidas/`:

- `<modo>_preproceso.png`: entrada, anotaciones enmascaradas, magnitud de Sobel y bordes
- `<modo>_resultado.png`: figuras detectadas sobre la imagen + acumulador de Hough
- `<modo>_reporte.txt`: el mismo reporte que se imprime en consola

## Pipeline

Sigue las etapas de bajo nivel de visión por computador (adquisición, preprocesamiento, extracción de características) y luego la transformada:

1. `load_image` y `crop`: lectura y recorte de la región de interés.
2. `mask_annotations`: la foto del caso tiene ejes, flechas y puntos de color dibujados encima. El metal es casi gris, así que todo píxel con saturación `(max−min)/max > 0,35` se considera anotación y se reemplaza por un gris neutro. La máscara se dilata 2 px para cubrir el halo pálido que deja la compresión JPEG alrededor de cada trazo.
3. `to_gray` → `gaussian_blur` (convolución separable) → `sobel` (magnitud y dirección del gradiente).
4. `non_max_suppression`: adelgaza los bordes a 1 px conservando solo los máximos en la dirección del gradiente (idea central de Canny).
5. `threshold_edges`: imagen binaria de bordes.
6. Hough de rectas o de circunferencias.

## Hough de rectas

- Parametrización polar: `ρ = x·cos θ + y·sen θ`, donde **θ es el ángulo de la normal** a la recta y ρ su distancia al origen. Se usa en lugar de `y = mx + b` porque la pendiente se indetermina en rectas verticales.
- θ en [−90°, 90°) y ρ en [−diagonal, +diagonal].
- Votación vectorizada con `np.add.at` (cada píxel de borde vota por todos los θ).
- `find_peaks`: toma el máximo, lo guarda si supera el umbral y anula su vecindario (supresión de no-máximos), hasta `--max-rectas`.
- Una recta con normal θ forma `θ − 90°` con la horizontal. El reporte informa esa inclinación para el borde dominante.
- `estimate_z_rotation`: promedio circular de θ módulo 90°, ponderado por votos. Solo es válido con una cámara perpendicular a la cara del block; en una foto en perspectiva los lados paralelos no se ven paralelos.

## Hough de circunferencias

- `(x − xo)² + (y − yo)² = R²` → acumulador 3D (R, yo, xo) con un rango corto de radios, porque por la perspectiva el aro se ve como elipse.
- **Votación por gradiente** (default): el centro está sobre la normal al borde, así que cada píxel vota solo en esa dirección (±3°, en ambos sentidos), en lugar de en toda la circunferencia. Con `--votar-circunferencia` se usa la votación clásica, para comparar.
- El puntaje se normaliza por el perímetro 2πR para no favorecer radios grandes.
- **Centro de una elipse**: ningún R encaja en todo el contorno y el pico se parte en dos lóbulos sobre el eje mayor. El centro A se calcula como el centroide ponderado de las celdas con al menos la mitad del pico (precisión subpíxel).

## Resultados sobre la foto del caso

| Modo | Resultado |
|---|---|
| Validar | Rectas: ρ = −100, θ = −90° y ρ = 141, θ = 45° (esperado). Círculo: A = (110, 90), R = 40 (exacto). |
| Rectas | Borde dominante θ = 75° (el lateral largo del block sube 15° hacia la derecha). Las familias de ≈ ±75° corresponden a las aristas longitudinales y transversales del block en perspectiva. |
| Circunferencias | A = (299,4 ; 201,3) px en la imagen completa, R = 19 px (borde interior del aro). |

## Limitaciones conocidas

- La foto es una vista en perspectiva de 750×395 px: el aro mide unos 20 px de radio y se ve como elipse. En la planta la cámara iría fija y perpendicular a la cara del block.
- Las anotaciones enmascaradas dejan "huecos" en los bordes reales que cruzaban.
- Algunas rectas a ±45° pueden ser artefactos: muchos bordes cortos de la textura de fundición se alinean en diagonal por la discretización.
- Los umbrales (`--umbral-borde`, `--umbral-votos`) y el rango de radios se ajustan a mano para cada imagen.
