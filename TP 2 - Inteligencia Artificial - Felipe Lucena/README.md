# Trabajo Práctico: Búsqueda en el Espacio de Estados

Este directorio contiene los archivos del **TP2 — Inteligencia Artificial**
(Felipe Lucena), donde se implementan y comparan búsquedas para localizar el
punto de montaje **A** a partir de la posición teórica **B** sobre el eje **H**
de la cara lateral del block (línea de ensamble de motores).

## Contenido
- **busquedaExhaustiva.py**: búsqueda no informada 1D (patrón alternado ±k·ΔH) con palpado simulado.
- **busquedaHeuristica.py**: búsqueda informada 1D guiada por el **relieve** (Best-First / hill climbing), con comparación automática frente a una baseline exhaustiva.

## Cómo usar

### 1. Búsqueda exhaustiva
```bash
python busquedaExhaustiva.py
```

**Parámetros:**
- `--start`: posición inicial B (default: 0.0)
- `--target`: posición real A (default: 0.35)
- `--tolerance`: tolerancia de palpado ± (default: 0.05)
- `--step`: paso ΔH (default: 0.1)
- `--max-range`: rango máximo desde B (default: 2.0)
- `--json`: salida JSON

### 2. Búsqueda heurística
```bash
python busquedaHeuristica.py
```

**Parámetros:**
- `--start`: posición inicial B (default: 0.0)
- `--target`: centro real del anillo A (default: 0.55)
- `--step`: paso ΔH (default: 0.05)
- `--max-range`: rango máximo desde B (default: 2.0)
- `--success-tol`: umbral de la heurística de relieve (default: 0.05)
- `--peak`: amplitud del anillo en el modelo de relieve (default: 1.0)
- `--json`: salida JSON (incluye comparación con exhaustiva)

## Enfoque respecto de la consigna

| Requisito | Enfoque en el código |
|-----------|----------------------|
| Búsqueda exhaustiva B→A sobre H | `exhaustive_line_search`: capas +Δ/−Δ sin orientación previa |
| Búsqueda heurística con relieve | `heuristic_line_search`: h(x)=‖z(x)−z_anillo‖ elige el sentido |
| Comparación de enfoques | CLI heurística reporta Δpalpados y Δrecorrido vs baseline ciega |

**Conclusión práctica:** la exhaustiva garantiza hallar A dentro del rango, pero
explora ambos sentidos. La heurística por relieve decide el sentido y reduce
palpados/recorrido cuando el modelo del anillo es fiable; si el relieve es
ruidoso, pueden aparecer retrocesos y conviene combinar ambos criterios.
