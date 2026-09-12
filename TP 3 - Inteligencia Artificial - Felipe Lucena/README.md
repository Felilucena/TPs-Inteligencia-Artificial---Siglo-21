# Hopfield 10×10 – Prototipo de Memoria Autoasociativa (Python)

Prototipo educativo de una **red de Hopfield** para **almacenamiento y recuperación de patrones binarios** (10×10 píxeles), probado con **ruido** y entrenado con **Hebb** y **Pseudoinversa**. Ideal para acompañar un informe académico o mostrar un PoC de visión artificial clásica.

---

## 🧠 ¿Qué hace?

- Genera **patrones 10×10** (anillo, cruz y “L”) como matrices binarias.
- Convierte a representación **bipolar** (±1) y **entrena** la red:
  - **Hebb**: simple y robusto para ruido moderado.
  - **Pseudoinversa**: mayor exactitud con patrones no ortogonales.
- **Inyecta ruido** (10%, 30%, 50%) y **recupera** los patrones.
- Guarda **grillas de resultados** (objetivo / ruidosa / recuperada) y **curvas de energía** (convergencia).

---

## 📂 Archivos

```
.
├─ hopfield_prototipo.py          # Script ejecutable principal
├─ hopfield_report.txt            # Resumen de experimentos y métricas
├─ hopfield_demo_results.png      # Grilla de objetivos, entradas ruidosas y salidas recuperadas
└─ README.md                      # Este archivo
```

---

## ⚙️ Requisitos

- **Python 3.9+** (probado con 3.10+)
- Dependencias: `numpy`, `matplotlib` (ver `requirements.txt`)

### 1. Instalar Python

**Windows**

1. Descargá el instalador desde [python.org/downloads](https://www.python.org/downloads/).
2. Durante la instalación, marcá **“Add python.exe to PATH”**.
3. Abrí **PowerShell** o **CMD** y verificá:

```powershell
python --version
pip --version
```

Si `python` no se reconoce, probá `py --version` (launcher de Windows).

**Linux**

En la mayoría de distros Python ya viene instalado. Si no:

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora
sudo dnf install python3 python3-pip

# Arch
sudo pacman -S python python-pip
```

Verificá la instalación:

```bash
python3 --version
pip3 --version
```

### 2. Instalar dependencias

Desde la carpeta del proyecto (`TP 3 - IA`):

**Windows (PowerShell / CMD)**

```powershell
cd "ruta\al\proyecto\TP 3 - IA"
python -m pip install -r requirements.txt
```

Alternativa con el launcher de Windows:

```powershell
py -m pip install -r requirements.txt
```

**Linux (bash)**

```bash
cd "ruta/al/proyecto/TP 3 - IA"
python3 -m pip install -r requirements.txt
```

Opcional (recomendado): crear un entorno virtual antes de instalar.

| Sistema | Crear venv | Activar | Instalar |
|---------|------------|---------|----------|
| Windows | `python -m venv .venv` | `.venv\Scripts\Activate.ps1` | `pip install -r requirements.txt` |
| Linux   | `python3 -m venv .venv` | `source .venv/bin/activate` | `pip install -r requirements.txt` |

---

## ▶️ Cómo ejecutar

Desde la raíz del proyecto (donde está `hopfield_prototipo.py`):

**Windows**

```powershell
python hopfield_prototipo.py
```

Si no funciona:

```powershell
py hopfield_prototipo.py
```

**Linux**

```bash
python3 hopfield_prototipo.py
```

Esto genera:
- `hopfield_report.txt` con accuracy y Hamming por caso.
- `hopfield_demo_results.png` con la grilla de imágenes.
- `hopfield_energy_*.png` con la convergencia de energía.

---

## 🔍 Qué vas a ver

- **Recuperación perfecta** con 10%–30% de ruido en la mayoría de los casos (ambas reglas).
- **Degradación** con 50% de flips (más notoria con pseudoinversa para algunos patrones).
- **Convergencia** de la energía decreciendo hasta un mínimo (atractor).

---

## 🧪 Personalización

- **Patrones**: editá/añadí generadores de matrices 10×10 en el script (ej. otro símbolo geométrico).
- **Ruido**: modificá la lista `noise_levels = [0.10, 0.30, 0.50]`.
- **Dinámica**: por defecto es **asíncrona** (mejor convergencia). Podés alternar a **síncrona** dentro de la función `recall(...)`.
- **Reglas de entrenamiento**:
  - **Hebb** (`Hopfield.train_hebb([...])`): simple, requiere patrones relativamente distintos para máxima estabilidad.
  - **Pseudoinversa** (`Hopfield.train_pseudoinverse([...])`): mayor fidelidad para patrones no ortogonales, puede ser más sensible a ruido extremo.

---

## 🧩 Alcances y limitaciones

- Excelente para **demostración** de **memoria autoasociativa**, **ruido** y **convergencia**.
- **No escala** a imágenes grandes: el tamaño de la matriz de pesos crece como \(N^2 \times N^2\).
- Memoria **finita**: demasiados patrones → **atractores espurios**.
- Reconoce **sólo** patrones **entrenados** (no generaliza como una CNN moderna).
