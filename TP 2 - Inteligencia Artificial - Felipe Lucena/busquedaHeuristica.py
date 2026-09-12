# *********************************************************
# Nombre: Felipe Lucena
# Legajo: VINF016997
# DNI: 46.601.576
# *********************************************************
"""
Busqueda heuristica (informada) sobre el eje horizontal H.

Contexto del TP2
----------------
A diferencia de la busqueda exhaustiva, aqui el robot dispone de un indicador
del "relieve" de la cara lateral del block respecto de un plano patron. En cada
punto de apoyo conoce la altura z(H) y, con una funcion heuristica, estima que
tan lejos esta del anillo de montaje (punto A) y en que sentido conviene avanzar.

Metodo adoptado
---------------
Hill climbing / Best-First en 1D guiado por el relieve:

1. Se modela el perfil lateral z(x) con un maximo caracteristico en A
   (anillo prominente de la figura 2/5 de la consigna).
2. La heuristica h(x) = z_anillo_esperado - z(x) estima el "faltante" de
   relieve respecto del patron del anillo. Valores cercanos a 0 -> cerca de A.
3. Desde B se palpan vecinos +/- dH y se elige el sentido que mas reduce h(x)
   (movimiento mas prometedor). Si ambos empeoran, se permite un retroceso
   controlado (backtracking acotado) para escapar de mesetas locales.
4. Exito solo si el relieve esta cerca del patron Y la posicion es un maximo
   local (cumbre del anillo), para no detenerse en la ladera.

Por que este metodo (y no un laberinto 2D generico)
---------------------------------------------------
La consigna restringe el problema a la horizontal H (el vertical ya es correcto)
y al uso del relieve como conocimiento del dominio. Hill climbing / Best-First
1D es el metodo heuristicamente mas apropiado: usa la informacion disponible,
decide el sentido de avance y reduce el recorrido frente a la exploracion ciega.

Ventajas: orienta la busqueda; suele hallar A con un camino corto.
Limitaciones: depende de la calidad del modelo de relieve; mesetas o ruidos
pueden exigir retrocesos; no garantiza optimalidad global del recorrido.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Tuple
import argparse
import json
import math


@dataclass
class HeuristicScanResult:
    """Resultado de una corrida de busqueda heuristica sobre el eje H."""

    found: bool
    """True si se alcanzo A (cumbre del anillo dentro de tolerancia)."""

    position: Optional[float]
    """Ultima posicion H aceptada como A, o None."""

    probes: int
    """Cantidad de mediciones de relieve (palpados) realizadas."""

    path_length: float
    """Distancia total recorrida a lo largo de H."""

    visited: List[float]
    """Secuencia de posiciones visitadas (incluye retrocesos)."""

    backtracks: int
    """Cantidad de veces que se eligio un movimiento que no mejoro h(x)."""


def ring_relief(x: float, center: float, amplitude: float = 1.0, width: float = 0.25) -> float:
    """
    Perfil sintetico del relieve lateral alrededor del anillo de montaje.

    Combina un lobulo gaussiano centrado en `center` (A) con un leve ondulado
    de fondo, para simular nervaduras de fundicion que pueden crear mesetas.
    """
    bump = amplitude * math.exp(-((x - center) ** 2) / (2.0 * width ** 2))
    # Ondulado leve: simula nervaduras sin desplazar la cumbre del anillo.
    background = 0.015 * math.sin(6.0 * x) + 0.005 * math.cos(13.0 * x)
    return bump + background


def make_height_sensor(
    target: float,
    amplitude: float = 1.0,
    width: float = 0.25,
) -> Callable[[float], float]:
    """Devuelve z(x): altura del relieve en la coordenada H = x."""

    def measure(x: float) -> float:
        return ring_relief(x, center=target, amplitude=amplitude, width=width)

    return measure


def relief_heuristic(height: float, expected_peak: float) -> float:
    """
    Funcion heuristica h(x) a partir del relieve medido.

    Interpreta (z_anillo - z) como estimacion de alejamiento del patron
    esperado en A. No es la distancia euclidea real (desconocida), sino una
    aproximacion usable para orientar la subida hacia el anillo.
    """
    return expected_peak - height


def is_ring_peak(
    x: float,
    measure: Callable[[float], float],
    expected_peak: float,
    step: float,
    success_tol: float,
) -> bool:
    """
    Criterio de exito: relieve cercano al patron y maximo local en +/- step.

    Evita declarar A encontrado a mitad de la ladera del anillo.
    """
    z = measure(x)
    if abs(z - expected_peak) > success_tol:
        return False
    z_left = measure(x - step)
    z_right = measure(x + step)
    return z >= z_left and z >= z_right


def heuristic_line_search(
    start: float,
    measure: Callable[[float], float],
    expected_peak: float,
    step: float = 0.1,
    max_range: float = 2.0,
    success_tol: float = 0.05,
    max_steps: int = 80,
) -> HeuristicScanResult:
    """
    Busqueda Best-First / hill climbing sobre H usando el relieve como guia.

    En cada iteracion se evaluan candidatos a +/- step. Se elige el movimiento
    con menor h(x). Si el mejor candidato no mejora, se cuenta como retroceso
    pero se acepta el menos malo para no quedar atrapado (backtracking acotado).

    Parameters
    ----------
    start:
        Posicion teorica B.
    measure:
        Sensor de relieve: x -> z(x).
    expected_peak:
        Altura patron del anillo (conocimiento previo del modelo CAD / plano).
    step:
        Paso dH entre palpados.
    max_range:
        No se exploran posiciones fuera de [start - max_range, start + max_range].
    success_tol:
        Tolerancia |z - z_anillo| para aceptar la cumbre como A.
    max_steps:
        Tope de iteraciones para evitar bucles infinitos.
    """
    if step <= 0:
        raise ValueError("step (dH) must be > 0")
    if max_range < 0:
        raise ValueError("max_range must be >= 0")

    visited: List[float] = []
    probes = 0
    path = 0.0
    backtracks = 0
    x = round(start, 6)
    last_x = x
    lo, hi = start - max_range, start + max_range

    def probe(pos: float) -> Tuple[float, float]:
        """Palpa en `pos` y devuelve (altura, valor heuristico)."""
        nonlocal probes
        probes += 1
        height = measure(pos)
        return height, relief_heuristic(height, expected_peak)

    # Palpado inicial en B (3 mediciones extras para chequear maximo local).
    _, h_cur = probe(x)
    probes += 2  # vecinos del chequeo de cumbre
    visited.append(x)
    if is_ring_peak(x, measure, expected_peak, step, success_tol):
        return HeuristicScanResult(True, x, probes, path, visited, backtracks)

    for _ in range(max_steps):
        candidates: List[Tuple[float, float, float]] = []  # (h, pos, signed_delta)

        for delta in (+step, -step):
            nxt = round(x + delta, 6)
            if lo - 1e-12 <= nxt <= hi + 1e-12:
                _, h_n = probe(nxt)
                candidates.append((h_n, nxt, delta))

        if not candidates:
            break

        # Best-First: el vecino con menor h es el mas prometedor.
        candidates.sort(key=lambda t: (t[0], abs(t[2])))
        best_h, best_x, _best_delta = candidates[0]

        # Retroceso si el mejor vecino no mejora (meseta / ruido local).
        if best_h >= h_cur - 1e-12:
            backtracks += 1

        path += abs(best_x - last_x)
        last_x = best_x
        x = best_x
        h_cur = best_h
        visited.append(x)

        probes += 2  # chequeo de cumbre en la nueva posicion
        if is_ring_peak(x, measure, expected_peak, step, success_tol):
            return HeuristicScanResult(
                True, x, probes, round(path, 6), visited, backtracks
            )

        # Estancamiento: oscilacion 2-ciclica -> fracaso controlado.
        if len(visited) >= 4 and visited[-1] == visited[-3] and visited[-2] == visited[-4]:
            break

    return HeuristicScanResult(False, None, probes, round(path, 6), visited, backtracks)


def exhaustive_baseline(
    start: float,
    measure: Callable[[float], float],
    expected_peak: float,
    step: float,
    max_range: float,
    success_tol: float,
) -> HeuristicScanResult:
    """
    Linea base ciega (mismo patron alternado que busquedaExhaustiva) pero
    usando el criterio de cumbre del anillo para declarar exito. Sirve para
    comparar recorrido frente al metodo heuristic en el mismo escenario.
    """
    visited: List[float] = []
    probes = 0
    path = 0.0
    last_x = round(start, 6)

    def ok(pos: float) -> bool:
        nonlocal probes
        # Un palpado central + dos vecinos para validar maximo local.
        probes += 3
        visited.append(round(pos, 6))
        return is_ring_peak(pos, measure, expected_peak, step, success_tol)

    if ok(last_x):
        return HeuristicScanResult(True, last_x, probes, path, visited, 0)

    k = 1
    while k * step <= max_range + 1e-12:
        for direction in (+1, -1):
            x = round(start + direction * k * step, 6)
            path += abs(x - last_x)
            last_x = x
            if ok(x):
                return HeuristicScanResult(True, x, probes, round(path, 6), visited, 0)
        k += 1

    return HeuristicScanResult(False, None, probes, round(path, 6), visited, 0)


def main() -> None:
    """CLI: busqueda heuristica por relieve y comparacion con exhaustiva."""
    parser = argparse.ArgumentParser(
        description=(
            "Busqueda heuristica 1D guiada por el relieve del block "
            "(punto B -> anillo A sobre el eje H). Compara contra exhaustiva."
        )
    )
    parser.add_argument("--start", type=float, default=0.0, help="Posicion inicial B")
    parser.add_argument("--target", type=float, default=0.55, help="Posicion real A (centro del anillo)")
    parser.add_argument("--step", type=float, default=0.05, help="Paso dH")
    parser.add_argument("--max-range", type=float, default=2.0, help="Rango maximo desde B")
    parser.add_argument(
        "--success-tol",
        type=float,
        default=0.05,
        help="Umbral |z - z_A| para declarar la cumbre como exito",
    )
    parser.add_argument(
        "--peak",
        type=float,
        default=1.0,
        help="Amplitud del anillo en el modelo de relieve",
    )
    parser.add_argument("--json", action="store_true", help="Salida JSON")
    args = parser.parse_args()

    measure = make_height_sensor(args.target, amplitude=args.peak)
    # z_A teorico del modelo (sin ruido adicional): pico del anillo en A.
    expected_peak = ring_relief(args.target, args.target, amplitude=args.peak)

    heur = heuristic_line_search(
        start=args.start,
        measure=measure,
        expected_peak=expected_peak,
        step=args.step,
        max_range=args.max_range,
        success_tol=args.success_tol,
    )
    exh = exhaustive_baseline(
        start=args.start,
        measure=measure,
        expected_peak=expected_peak,
        step=args.step,
        max_range=args.max_range,
        success_tol=args.success_tol,
    )

    payload = {
        "scenario": {
            "start_B": args.start,
            "target_A": args.target,
            "step": args.step,
            "max_range": args.max_range,
            "success_tol": args.success_tol,
            "expected_peak": expected_peak,
        },
        "heuristic": asdict(heur),
        "exhaustive_baseline": asdict(exh),
        "comparison": {
            "probe_saving": exh.probes - heur.probes,
            "path_saving": round(exh.path_length - heur.path_length, 6),
        },
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # Mensajes ASCII-compatible para consolas Windows (cp1252).
    print("=== Busqueda heuristica 1D (relieve -> Best-First / hill climbing) ===")
    print(f"  B (start)        : {args.start:.6f}")
    print(f"  A (target sim)   : {args.target:.6f}")
    print(f"  Paso dH          : {args.step:.6f}")
    print(f"  z patron anillo  : {expected_peak:.6f}")
    print(f"  Umbral exito     : {args.success_tol:.6f}")
    print("---------------------------------------------")
    print("  [Heuristica]")
    print(f"    Encontrado?    : {heur.found}")
    print(
        f"    Posicion       : "
        f"{None if heur.position is None else f'{heur.position:.6f}'}"
    )
    print(f"    Palpados       : {heur.probes}")
    print(f"    Recorrido      : {heur.path_length:.6f}")
    print(f"    Retrocesos     : {heur.backtracks}")
    print(f"    Visitados      : {heur.visited}")
    print("  [Exhaustiva - misma escena]")
    print(f"    Encontrado?    : {exh.found}")
    print(
        f"    Posicion       : "
        f"{None if exh.position is None else f'{exh.position:.6f}'}"
    )
    print(f"    Palpados       : {exh.probes}")
    print(f"    Recorrido      : {exh.path_length:.6f}")
    print("---------------------------------------------")
    print("  Comparacion (exhaustiva - heuristica):")
    print(f"    dPalpados      : {exh.probes - heur.probes}")
    print(f"    dRecorrido     : {exh.path_length - heur.path_length:.6f}")


if __name__ == "__main__":
    main()
