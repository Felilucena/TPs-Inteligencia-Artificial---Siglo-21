# TP4 - Transformada de Hough para Detección de Rectas

## Descripción

Implementación desde cero de la **Transformada de Hough** para la detección de rectas en imágenes binarias. Este prototipo didáctico demuestra cómo identificar patrones lineales mediante la concentración de votos en el espacio de parámetros.

## Representación de las Rectas

Utilizamos la formulación habitual de Hough en coordenadas polares:

```
ρ = x·cos(θ) + y·sin(θ)
```

Donde:
- **x, y**: coordenadas del píxel de borde
- **θ**: ángulo de la recta respecto del eje X
- **ρ**: distancia de la recta al origen

Cada punto de borde `(x,y)` genera una curva sinusoidal en el espacio de parámetros `(θ,ρ)`. Donde muchas curvas se cruzan (máxima acumulación), tenemos una recta real de la imagen.

## Pasos del Algoritmo

### 1. Generar la imagen de prueba
- Creamos una matriz `height × width` de ceros (fondo negro)
- Dibujamos:
  - Una recta horizontal (píxeles a 255 en una fila concreta)
  - Una recta diagonal (píxeles a 255 siguiendo una diagonal)

### 2. Seleccionar los píxeles de borde
- En este prototipo, la imagen ya es binaria: 0 = fondo, 255 = borde
- Se toman todos los píxeles con valor distinto de cero usando `np.nonzero()`

### 3. Definir el espacio de parámetros
- **θ**: de -90° a +90° (convertido a radianes)
- **ρ**: de -diag a +diag, donde diag es la diagonal de la imagen

### 4. Construir el acumulador
- Creamos una matriz `accumulator[rho_index, theta_index]` inicializada en cero
- Para cada píxel de borde y para cada θ:
  - Calculamos ρ = x·cos(θ) + y·sin(θ)
  - Sumamos 1 en la celda correspondiente del acumulador

### 5. Detección de picos
- Recorremos el acumulador buscando valores por encima de un umbral (`threshold`)
- Aplicamos **supresión de no-máximos** para evitar detecciones duplicadas
- Cada pico se interpreta como una recta detectada (par (ρ, θ))
- Los picos se ordenan por número de votos (de mayor a menor)

### 6. Visualización
- Dibujamos las rectas detectadas sobre la imagen original usando la ecuación x·cos(θ) + y·sin(θ) = ρ
- Mostramos el acumulador como imagen 2D (ρ vs θ) con los picos marcados
- Cada línea se visualiza con color diferente y su leyenda correspondiente

## Instalación

### 1. Crear entorno virtual

**Windows (PowerShell / CMD):**
```powershell
python -m venv .venv
```

**Linux / macOS:**
```bash
python3 -m venv .venv
```

### 2. Activar entorno virtual

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

Si aparece un error de política de ejecución:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

Con el entorno virtual activado (igual en ambos sistemas):

```bash
pip install numpy matplotlib
```

## Uso

Desde la carpeta del proyecto, con el entorno virtual activado:

**Windows (PowerShell / CMD):**
```powershell
python hough-rectas.py
```

**Linux / macOS:**
```bash
python3 hough-rectas.py
```

> Si el entorno virtual ya está activado, también suele funcionar `python hough-rectas.py` en Linux/macOS.

El script generará:
1. Una ventana con dos gráficos:
   - **Izquierda**: Imagen original con las rectas detectadas superpuestas
   - **Derecha**: Acumulador de Hough mostrando el espacio de parámetros
2. Salida en consola con información de las líneas detectadas

## Parámetros Configurables

En la función `hough_lines()`:
- `theta_res`: Resolución angular en grados (default: 1°)
- `rho_res`: Resolución en píxeles para ρ (default: 1)

En la función `find_peaks()`:
- `threshold`: Umbral mínimo de votos para detectar una línea (default: 80)
- `min_distance`: Distancia mínima entre picos en el espacio de Hough (default: 10)

## Características

### Características del prototipo
- ✅ Implementa la transformada de Hough para rectas desde cero (sin usar OpenCV)
- ✅ Trabaja sobre una imagen binaria simple
- ✅ Usa resolución configurable de θ (por defecto 1°)
- ✅ Resolución de ρ en píxeles
- ✅ Permite:
  - Obtener el acumulador de Hough
  - Extraer las rectas detectadas a partir de un umbral
  - Visualizar el resultado con información detallada
  - Supresión de no-máximos para evitar duplicados

### Ventajas
- **Didáctico**: Se ve claramente cómo:
  - Se construye el espacio de parámetros
  - Se llenan los votos en el acumulador
  - Se relacionan los picos con las rectas de la imagen
  
- **Totalmente controlable y modificable**:
  - Se pueden cambiar fácilmente las resoluciones
  - Probar distintas imágenes sintéticas
  - Integrar luego detección de bordes más realista

- **Reproduce el concepto teórico**: A partir de los bordes se identifican patrones (rectas) gracias a la concentración de votos en el espacio de parámetros

### Limitaciones
- **Costo computacional elevado** para imágenes grandes:
  - Doble bucle sobre píxeles de borde y ángulos θ
  - Complejidad O(n·m) donde n = píxeles de borde, m = ángulos

- **No incluye**:
  - Detección de bordes realista (Canny, Sobel, etc.)
  - Optimizaciones avanzadas de rendimiento

- **Entornos reales**: Para aplicaciones industriales habría que:
  - Optimizar la implementación (vectorización, GPU)
  - Mejorar el preprocesamiento (ruido, iluminación)
  - Limitar el rango de θ según la aplicación

## Dificultades

Aunque el código funciona correctamente, conceptualmente aparecen algunas dificultades clásicas:

### 1. Elección del umbral
- Umbral muy alto → se pierden rectas
- Umbral muy bajo → aparecen muchas rectas falsas
- **Solución implementada**: Supresión de no-máximos + búsqueda ordenada por votos

### 2. Resolución de θ y ρ
- Más resolución = mejor precisión, pero más tiempo de cómputo
- Hay que balancear precisión vs. rendimiento según la aplicación

### 3. Visualización de rectas infinitas
- Hay que convertir (ρ, θ) a segmentos finitos para dibujar sobre la imagen
- **Solución**: Extender la recta con puntos muy alejados y dejar que matplotlib recorte

### 4. Escalado a casos reales
- En imágenes reales, el ruido y los bordes incompletos hacen más difícil elegir parámetros "buenos"
- Requiere preprocesamiento cuidadoso y ajuste fino de parámetros

## Ejemplo de Salida

```
Aplicando transformada de Hough...
Buscando líneas en el acumulador...

Se detectaron 2 líneas:
  Línea 1: ρ=-100.00 píxeles, θ=-90.00°, votos=160
  Línea 2: ρ=141.00 píxeles, θ=45.00°, votos=121
```

## Estructura del Código

```python
generate_test_image()     # Genera imagen binaria de prueba
hough_lines()             # Implementa la transformada de Hough
find_peaks()              # Detecta picos en el acumulador
plot_results()            # Visualiza resultados
```