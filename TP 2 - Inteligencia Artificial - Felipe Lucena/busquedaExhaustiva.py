# *********************************************************
# Nombre: Felipe Lucena
# Legajo: VINF016997
# *********************************************************
"""
Búsqueda exhaustiva (no informada) sobre el eje horizontal H.

Contexto del TP2
----------------
En la línea de montaje, el block puede desplazarse ligeramente sobre la cinta.
El robot parte de la posición teórica B y debe encontrar la posición real A
del punto de montaje (perforación del anillo lateral), sin ninguna pista sobre
si A quedó a la izquierda o a la derecha de B.

Estrategia adoptada
-------------------
Búsqueda exhaustiva en 1D con patrón alternado (estilo “espiral” en la recta):
    B → +ΔH → −ΔH → +2ΔH → −2ΔH → …

Es equivalente a explorar en anchura por “capas” de distancia a B: primero
todas las celdas a distancia ΔH, luego 2ΔH, etc. Garantiza encontrar A si
está dentro de `max_range` y el paso ΔH es menor o igual a la tolerancia
de palpado.

Ventajas: completa y simple; no requiere modelo del relieve.
Limitaciones: no sabe en qué sentido avanzar; el recorrido puede ser largo
porque alterna ambos lados en cada capa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import argparse
import json


@dataclass
class ScanResult:
    """Resultado de una corrida de búsqueda exhaustiva sobre el eje H."""

    found: bool
    """True si el palpado detectó el objetivo A dentro del rango."""

    position: Optional[float]
    """Coordenada H donde se detectó A, o None si no se encontró."""

    probes: int
    """Cantidad de palpados (evaluaciones del sensor) realizados."""

    path_length: float
    """Distancia total recorrida por el extremo del brazo a lo largo de H."""

    visited: List[float]
    """Secuencia ordenada de posiciones H donde se palpó."""


def exhaustive_line_search(
    start: float,
    sense: Callable[[float], bool],
    step: float = 0.1,
    max_range: float = 2.0,
) -> ScanResult:
    """
    Explora de forma exhaustiva una línea 1D centrada en `start` (punto B).

    En cada capa k = 1, 2, 3, ... se prueban las posiciones:
        start + k*step  y  start - k*step
    hasta que k*step supera `max_range` o el sensor confirma el objetivo.

    Parameters
    ----------
    start:
        Posición teórica inicial B sobre el eje H.
    sense:
        Función de palpado: recibe una coordenada H y devuelve True si en
        esa posición se detecta el punto de montaje A (éxito).
    step:
        Incremento ΔH entre posiciones exploradas. Debe ser lo bastante
        pequeño respecto a la tolerancia de montaje.
    max_range:
        Distancia máxima a explorar desde B en cada sentido.

    Returns
    -------
    ScanResult
        Métricas de la búsqueda (éxito, posición, palpados, recorrido, ruta).
    """
    if step <= 0:
        raise ValueError("step (ΔH) must be > 0")
    if max_range < 0:
        raise ValueError("max_range must be >= 0")

    visited: List[float] = []
    probes = 0
    path = 0.0
    last_x = start

    def palp(x: float) -> bool:
        """Registra un palpado en x y consulta el sensor."""
        nonlocal probes
        probes += 1
        visited.append(round(x, 6))
        return sense(x)

    # Caso trivial: A ya coincide con B (dentro de tolerancia del sensor).
    if palp(start):
        return ScanResult(True, round(start, 6), probes, path, visited)

    k = 1
    while k * step <= max_range + 1e-12:
        # Ambos sentidos por capa: el robot no sabe si A esta a +H o -H.
        for direction in (+1, -1):
            x = round(start + direction * k * step, 6)
            path += abs(x - last_x)
            last_x = x
            if palp(x):
                return ScanResult(True, x, probes, round(path, 6), visited)
        k += 1

    return ScanResult(False, None, probes, round(path, 6), visited)


def make_sensor(target: float, tol: float) -> Callable[[float], bool]:
    """
    Construye un sensor idealizado de palpado.

    En la planta real, el “palpado” sería contacto mecánico o visión local.
    Aquí se simula como: éxito si |x − A| ≤ tolerancia.
    """
    if tol < 0:
        raise ValueError("tolerance must be >= 0")

    def sense(x: float) -> bool:
        return abs(x - target) <= tol

    return sense


def main() -> None:
    """CLI de demostración: localiza A partiendo de B con búsqueda exhaustiva."""
    parser = argparse.ArgumentParser(
        description=(
            "Búsqueda exhaustiva 1D para localizar el punto de montaje A "
            "sobre el eje H a partir de la posición teórica B (TP2)."
        )
    )
    parser.add_argument("--start", type=float, default=0.0, help="Posición inicial B (default: 0.0)")
    parser.add_argument("--target", type=float, default=0.35, help="Posición real A (para simular el sensor)")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Tolerancia de palpado ± (default: 0.05)")
    parser.add_argument("--step", type=float, default=0.1, help="Paso ΔH de exploración (default: 0.1)")
    parser.add_argument("--max-range", type=float, default=2.0, help="Rango máximo a explorar desde B (default: 2.0)")
    parser.add_argument("--json", action="store_true", help="Imprime la salida en JSON")
    args = parser.parse_args()

    sensor = make_sensor(args.target, args.tolerance)
    result = exhaustive_line_search(
        start=args.start,
        sense=sensor,
        step=args.step,
        max_range=args.max_range,
    )

    if args.json:
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
        return

    # Mensajes en ASCII-compatible para consolas Windows (cp1252).
    print("=== Resultado de la busqueda exhaustiva 1D ===")
    print(f"  B (start)        : {args.start:.6f}")
    print(f"  A (target sim)   : {args.target:.6f}")
    print(f"  Tolerancia       : +/-{args.tolerance:.6f}")
    print(f"  Paso dH          : {args.step:.6f}")
    print(f"  Rango max.       : {args.max_range:.6f}")
    print("---------------------------------------------")
    print(f"  Encontrado?      : {result.found}")
    print(
        f"  Posicion detect. : "
        f"{None if result.position is None else f'{result.position:.6f}'}"
    )
    print(f"  Palpados (N)     : {result.probes}")
    print(f"  Recorrido total  : {result.path_length:.6f} (unidades de H)")
    print(f"  Visitados        : {result.visited}")


if __name__ == "__main__":
    main()
