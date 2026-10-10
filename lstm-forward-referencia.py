#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Referencia numérica para el simulador interactivo del paso hacia adelante de una LSTM.
Maestría en Ciencia de Datos — Universidad Santo Tomás · Módulo 2 (secuencias).

Calcula el paso hacia adelante de una LSTM ESCALAR (una sola unidad) para un caso
canónico y para los dos escenarios pedagógicos, y exporta los valores «dorados» a JSON.
Esos valores se incrustan en la autoprueba del HTML (?selftest=1) para garantizar
paridad matemática entre el motor JS del navegador y esta referencia.
La paridad real observada es ~1e-16 (redondeo de doble precisión); la autoprueba
del HTML exige una tolerancia holgada, epsilon = 1e-9, que es el mismo valor que
se escribe en el JSON.

Notación (CP0): C_t es el estado de celda y C̃_t la memoria candidata. En el código
se llaman `c` y `cand`; las claves del JSON se conservan por compatibilidad.

Uso (desde la carpeta del script):
    python3 lstm-forward-referencia.py      # imprime y escribe lstm-forward-referencia.json
Solo requiere la biblioteca estándar de Python (módulo math). No requiere NumPy ni PyTorch.
"""
import json
import math

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def forward_pass(inputs, W):
    """LSTM escalar. W: dict con w_/u_/b_ para f, i, c (candidato) y o. Devuelve la lista de pasos."""
    c, h = 0.0, 0.0
    hist = []
    for t, x in enumerate(inputs):
        zf = W["wf"] * x + W["uf"] * h + W["bf"]; f = sigmoid(zf)
        zi = W["wi"] * x + W["ui"] * h + W["bi"]; i = sigmoid(zi)
        zc = W["wc"] * x + W["uc"] * h + W["bc"]; cand = math.tanh(zc)
        zo = W["wo"] * x + W["uo"] * h + W["bo"]; o = sigmoid(zo)
        c_new = f * c + i * cand
        h_new = o * math.tanh(c_new)
        hist.append({
            "t": t, "x": x, "h_prev": h, "c_prev": c,
            "zf": zf, "f": f, "zi": zi, "i": i,
            "zc": zc, "cand": cand, "zo": zo, "o": o,
            "c": c_new, "h": h_new,
        })
        c, h = c_new, h_new
    return hist

# --- Caso canónico de la autoprueba (valores arbitrarios pero fijos) ----------
CANONICAL_W = {
    "wf": 0.7, "uf": 0.10, "bf": 0.00,
    "wi": 0.9, "ui": 0.20, "bi": -0.10,
    "wc": 0.8, "uc": 0.15, "bc": 0.05,
    "wo": 0.6, "uo": 0.25, "bo": 0.00,
}
CANONICAL_X = [0.5, -1.0, 2.0]

# --- Escenario A: "Cambio de contexto" ---------------------------------------
# Entradas positivas sostenidas y luego una entrada negativa fuerte que cierra
# la compuerta de olvido (f -> 0) y «limpia» el estado de celda acumulado.
CTX_W = {
    "wf": 2.0, "uf": 0.0, "bf": 1.5,   # con x>0 => olvido saturado ~1; con x muy negativo => olvido ~0
    "wi": 1.5, "ui": 0.0, "bi": 0.0,
    "wc": 1.0, "uc": 0.0, "bc": 0.0,
    "wo": 1.0, "uo": 0.0, "bo": 0.5,
}
CTX_X = [1.0, 1.0, 1.0, -4.0, 1.0]

# --- Escenario B: "Puente de dependencia larga" ------------------------------
# Información importante en t=0, luego ruido ~0. La compuerta de entrada se abre
# en t=0 y el olvido queda cerca de 1: la señal viaja casi intacta por la cinta C_t
# (f = σ(4) ≈ 0,982 por paso; tras 4 pasos se conserva ≈ 93 %).
BRIDGE_W = {
    "wf": 0.0, "uf": 0.0, "bf": 4.0,   # olvido ~1 siempre (la cinta conserva)
    "wi": 5.0, "ui": 0.0, "bi": -1.0,  # entrada se abre con x grande, casi cerrada con x~0
    "wc": 2.0, "uc": 0.0, "bc": 0.0,
    "wo": 0.0, "uo": 0.0, "bo": 2.0,   # salida abierta para poder leer la memoria
}
BRIDGE_X = [2.0, 0.0, 0.0, 0.0, 0.0]

def build():
    return {
        "epsilon": 1e-9,
        "canonical": {"weights": CANONICAL_W, "inputs": CANONICAL_X,
                      "history": forward_pass(CANONICAL_X, CANONICAL_W)},
        "context_switch": {"weights": CTX_W, "inputs": CTX_X,
                           "history": forward_pass(CTX_X, CTX_W)},
        "long_bridge": {"weights": BRIDGE_W, "inputs": BRIDGE_X,
                        "history": forward_pass(BRIDGE_X, BRIDGE_W)},
    }

if __name__ == "__main__":
    ref = build()
    with open("lstm-forward-referencia.json", "w", encoding="utf-8") as fp:
        json.dump(ref, fp, indent=2)
    print("== Caso canónico (para la autoprueba del HTML) ==")
    for s in ref["canonical"]["history"]:
        print(f"t={s['t']}  x={s['x']:+.2f}  f={s['f']:.6f}  i={s['i']:.6f}  "
              f"cand={s['cand']:+.6f}  o={s['o']:.6f}  c={s['c']:+.6f}  h={s['h']:+.6f}")
    print("\n== Escenario A: Cambio de contexto (compuerta de olvido) ==")
    for s in ref["context_switch"]["history"]:
        print(f"t={s['t']}  x={s['x']:+.1f}  f={s['f']:.4f}  c={s['c']:+.4f}")
    print("\n== Escenario B: Puente de dependencia larga ==")
    for s in ref["long_bridge"]["history"]:
        print(f"t={s['t']}  x={s['x']:+.1f}  f={s['f']:.4f}  i={s['i']:.4f}  c={s['c']:+.4f}  h={s['h']:+.4f}")
    print("\nEscrito: lstm-forward-referencia.json")
