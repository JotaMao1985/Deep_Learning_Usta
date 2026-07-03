# =============================================================================
#  Laboratorio Interactivo: Regularización, Estabilización y Diagnóstico
#  Curso: Profundización I — Redes Neuronales / Deep Learning (USTA)
#  Módulo 1 · Acompaña al Capítulo 5 (Regularización, Evaluación y Ética)
# -----------------------------------------------------------------------------
#  INSTALACIÓN DE DEPENDENCIAS:
#     pip install streamlit torch numpy matplotlib pandas scikit-learn
#
#  EJECUCIÓN:
#     streamlit run app-laboratorio-regularizacion.py
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn

import streamlit as st
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Semilla global para que los experimentos sean reproducibles.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")  # Datos diminutos: la CPU sobra y simplifica el despliegue.

# Paleta sobria para fondo/puntos de las dos clases.
CMAP_FONDO = ListedColormap(["#FCE4D6", "#D6E4FC"])
CMAP_PUNTOS = ListedColormap(["#D7263D", "#1B4965"])


# =============================================================================
#  1. DATOS SINTÉTICOS
# =============================================================================
@st.cache_data(show_spinner=False)
def generar_datos(nombre, n_muestras, ruido):
    """Crea un dataset 2D, lo estandariza y lo separa en train/validación.

    Devolvemos los datos ya escalados porque la red entrena y se evalúa
    en ese espacio; dibujar la frontera ahí evita inconsistencias visuales.
    El escalador se ajusta SOLO con el train (se evita la fuga de datos).
    """
    if nombre == "make_moons":
        X, y = make_moons(n_samples=n_muestras, noise=ruido, random_state=SEED)
    else:
        X, y = make_circles(n_samples=n_muestras, noise=ruido,
                            factor=0.5, random_state=SEED)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    escalador = StandardScaler().fit(X_tr)
    X_tr = escalador.transform(X_tr)
    X_val = escalador.transform(X_val)
    return X_tr, X_val, y_tr, y_val


def a_tensores(X_tr, X_val, y_tr, y_val):
    return (
        torch.tensor(X_tr, dtype=torch.float32, device=DEVICE),
        torch.tensor(X_val, dtype=torch.float32, device=DEVICE),
        torch.tensor(y_tr, dtype=torch.long, device=DEVICE),
        torch.tensor(y_val, dtype=torch.long, device=DEVICE),
    )


# =============================================================================
#  2. MODELO: MLP CONFIGURABLE
# =============================================================================
class MLP(nn.Module):
    """Perceptrón multicapa con BatchNorm y Dropout opcionales.

    Estructura de cada bloque oculto:
        Linear  ->  [BatchNorm]  ->  Activación  ->  [Dropout]

    El forward puede 'capturar' las activaciones antes y después de BN para
    poder visualizar el efecto estabilizador de la normalización por lotes.
    """

    def __init__(self, dim_entrada, capas_ocultas, dim_salida,
                 usar_bn, p_dropout, activacion="relu"):
        super().__init__()
        self.usar_bn = usar_bn

        self.lineales = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [dim_entrada] + capas_ocultas
        for i in range(len(capas_ocultas)):
            self.lineales.append(nn.Linear(dims[i], dims[i + 1]))
            # Creamos una BN por capa aunque no se use, para mantener índices alineados.
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.salida = nn.Linear(dims[-1], dim_salida)
        self.dropout = nn.Dropout(p_dropout)
        self.activacion = nn.ReLU() if activacion == "relu" else nn.Tanh()

        # Buffers para inspección visual (no participan del entrenamiento).
        self.pre_bn = {}   # activaciones ANTES de BatchNorm, por capa
        self.post_bn = {}  # activaciones DESPUÉS de BatchNorm, por capa

    def forward(self, x, capturar=False):
        for i, lineal in enumerate(self.lineales):
            z = lineal(x)
            if capturar:
                self.pre_bn[i] = z.detach().cpu().numpy()

            if self.usar_bn:
                z = self.bns[i](z)
                if capturar:
                    self.post_bn[i] = z.detach().cpu().numpy()

            x = self.dropout(self.activacion(z))
        return self.salida(x)


# =============================================================================
#  3. ENTRENAMIENTO
# =============================================================================
def penalizacion_regularizacion(modelo, lambda_l1, lambda_l2):
    """Suma manual de los términos L1 y L2 sobre los PESOS (no los sesgos).

        L1 = λ1 · Σ |w|        -> empuja pesos a CERO (dispersión / sparsity)
        L2 = λ2 · Σ w²         -> ATENÚA pesos grandes (shrinkage)
    """
    l1 = torch.tensor(0.0, device=DEVICE)
    l2 = torch.tensor(0.0, device=DEVICE)
    for nombre, p in modelo.named_parameters():
        if "weight" in nombre:
            l1 = l1 + p.abs().sum()
            l2 = l2 + p.pow(2).sum()
    return lambda_l1 * l1 + lambda_l2 * l2


def precision(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def entrenar(modelo, datos, lr, epocas, lambda_l1, lambda_l2, batch_size,
             contenedor_progreso, contenedor_curvas):
    """Bucle de entrenamiento con registro de métricas y actualización en vivo.

    Si `batch_size` es None se entrena en FULL-BATCH (todo el set por época);
    en caso contrario se usan MINI-LOTES vía DataLoader. Los mini-lotes hacen
    que BatchNorm calcule estadísticas distintas en cada paso —exactamente el
    régimen donde la normalización por lotes muestra su efecto estabilizador—.
    """
    X_tr, X_val, y_tr, y_val = datos
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.CrossEntropyLoss()

    historia = {"perdida_tr": [], "perdida_val": [],
                "acc_tr": [], "acc_val": []}

    # --- Preparación de los mini-lotes (o full-batch) ---
    n = X_tr.shape[0]
    usar_minilotes = batch_size is not None and batch_size < n
    if usar_minilotes:
        generador = torch.Generator().manual_seed(SEED)  # barajado reproducible
        ds = torch.utils.data.TensorDataset(X_tr, y_tr)
        cargador = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            drop_last=True,          # evita un último lote de tamaño 1 (rompe BN)
            generator=generador,
        )

    barra = contenedor_progreso.progress(0, text="Entrenando…")

    for epoca in range(epocas):
        # ---- Paso(s) de entrenamiento ----
        modelo.train()
        if usar_minilotes:
            for xb, yb in cargador:
                optimizador.zero_grad()
                perdida = criterio(modelo(xb), yb) \
                    + penalizacion_regularizacion(modelo, lambda_l1, lambda_l2)
                perdida.backward()
                optimizador.step()
        else:
            optimizador.zero_grad()
            perdida = criterio(modelo(X_tr), y_tr) \
                + penalizacion_regularizacion(modelo, lambda_l1, lambda_l2)
            perdida.backward()
            optimizador.step()

        # ---- Evaluación sobre el set completo (sin Dropout; BN usa stats móviles) ----
        modelo.eval()
        with torch.no_grad():
            logits_tr = modelo(X_tr)
            logits_val = modelo(X_val)
            perdida_tr = criterio(logits_tr, y_tr) \
                + penalizacion_regularizacion(modelo, lambda_l1, lambda_l2)
            perdida_val = criterio(logits_val, y_val)

            historia["perdida_tr"].append(perdida_tr.item())
            historia["perdida_val"].append(perdida_val.item())
            historia["acc_tr"].append(precision(logits_tr, y_tr))
            historia["acc_val"].append(precision(logits_val, y_val))

        # ---- Refresco en vivo cada ~5% de las épocas ----
        if epoca % max(1, epocas // 20) == 0 or epoca == epocas - 1:
            barra.progress((epoca + 1) / epocas,
                           text=f"Época {epoca + 1}/{epocas}")
            contenedor_curvas.pyplot(figura_curvas(historia), clear_figure=True)

    barra.empty()
    return historia


# =============================================================================
#  4. VISUALIZACIONES
# =============================================================================
def figura_frontera(modelo, datos):
    """Frontera de decisión sobre una malla, con los puntos de validación."""
    X_tr, X_val, y_tr, y_val = datos
    todos_X = np.vstack([X_tr.cpu().numpy(), X_val.cpu().numpy()])

    x_min, x_max = todos_X[:, 0].min() - 0.5, todos_X[:, 0].max() + 0.5
    y_min, y_max = todos_X[:, 1].min() - 0.5, todos_X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))

    malla = torch.tensor(np.c_[xx.ravel(), yy.ravel()],
                         dtype=torch.float32, device=DEVICE)
    modelo.eval()
    with torch.no_grad():
        Z = modelo(malla).argmax(dim=1).cpu().numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.contourf(xx, yy, Z, alpha=0.65, cmap=CMAP_FONDO)
    # Puntos: train tenue, validación resaltada (lo que mide el sobreajuste).
    ax.scatter(X_tr.cpu()[:, 0], X_tr.cpu()[:, 1], c=y_tr.cpu(),
               cmap=CMAP_PUNTOS, s=12, alpha=0.35, edgecolors="none")
    ax.scatter(X_val.cpu()[:, 0], X_val.cpu()[:, 1], c=y_val.cpu(),
               cmap=CMAP_PUNTOS, s=28, edgecolors="white", linewidths=0.5)
    ax.set_title("Frontera de decisión (validación resaltada)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def figura_curvas(historia):
    """Curvas de pérdida y precisión: train vs. validación."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    epocas = range(1, len(historia["perdida_tr"]) + 1)

    ax1.plot(epocas, historia["perdida_tr"], label="Entrenamiento", color="#1B4965")
    ax1.plot(epocas, historia["perdida_val"], label="Validación",
             color="#D7263D", linestyle="--")
    ax1.set_title("Pérdida (Loss)"); ax1.set_xlabel("Época")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(epocas, historia["acc_tr"], label="Entrenamiento", color="#1B4965")
    ax2.plot(epocas, historia["acc_val"], label="Validación",
             color="#D7263D", linestyle="--")
    ax2.set_title("Precisión (Accuracy)"); ax2.set_xlabel("Época")
    ax2.set_ylim(0, 1.02); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def figura_activaciones(modelo, datos, idx_capa):
    """Histograma de activaciones ANTES vs DESPUÉS de BatchNorm."""
    X_val = datos[1]
    modelo.eval()
    with torch.no_grad():
        modelo(X_val, capturar=True)  # dispara el llenado de pre_bn / post_bn

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    pre = modelo.pre_bn.get(idx_capa)
    ax.hist(pre.ravel(), bins=40, alpha=0.6, color="#D7263D",
            label="Antes de BN", density=True)

    if modelo.usar_bn and idx_capa in modelo.post_bn:
        post = modelo.post_bn[idx_capa]
        ax.hist(post.ravel(), bins=40, alpha=0.6, color="#1B4965",
                label="Después de BN", density=True)
        ax.set_title("Activaciones: BatchNorm recentra y reescala")
    else:
        ax.set_title("Activaciones (BatchNorm desactivado)")

    ax.axvline(0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("Valor de activación"); ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def figura_pesos(modelo, idx_capa):
    """Mapa de calor de la matriz de pesos + histograma de magnitudes."""
    W = modelo.lineales[idx_capa].weight.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    vmax = np.abs(W).max()
    im = ax1.imshow(W, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_title("Matriz de pesos W")
    ax1.set_xlabel("Entradas"); ax1.set_ylabel("Neuronas")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ax2.hist(W.ravel(), bins=40, color="#1B4965", alpha=0.8)
    ax2.axvline(0, color="#D7263D", lw=1)
    ax2.set_title("Distribución de pesos\n(L1 → pico en 0 · L2 → angosta)")
    ax2.set_xlabel("Valor del peso"); ax2.set_ylabel("Frecuencia")

    fig.tight_layout()
    return fig


# =============================================================================
#  5. DIAGNÓSTICO AUTOMÁTICO DE LAS CURVAS
# =============================================================================
def diagnosticar(historia):
    """Heurística simple para etiquetar el régimen de entrenamiento."""
    acc_tr = np.mean(historia["acc_tr"][-5:])
    acc_val = np.mean(historia["acc_val"][-5:])
    brecha = acc_tr - acc_val

    if acc_tr < 0.80 and acc_val < 0.80:
        return ("warning", "🔶 SUBAJUSTE (underfitting)",
                "Ambas precisiones son bajas: el modelo es demasiado simple o "
                "le falta entrenamiento. Prueba más capas/neuronas o más épocas, "
                "y reduce la regularización.")
    if brecha > 0.12:
        return ("error", "🔴 SOBREAJUSTE (overfitting)",
                "El modelo memoriza el entrenamiento pero generaliza mal "
                f"(brecha de {brecha:.0%}). Sube L1/L2 o el Dropout, activa "
                "BatchNorm o reduce la complejidad de la red.")
    return ("success", "🟢 AJUSTE ADECUADO",
            f"Train y validación van parejas (brecha de {brecha:.0%}). "
            "Buen punto de equilibrio sesgo-varianza.")


# =============================================================================
#  6. INTERFAZ STREAMLIT
# =============================================================================
st.set_page_config(page_title="Lab: Regularización y Diagnóstico",
                   layout="wide")
st.title("🧪 Regularización, Estabilización y Diagnóstico del Entrenamiento")
st.caption("Mueve los controles, entrena la red y observa cómo cada técnica "
           "moldea los pesos, las activaciones y la capacidad de generalizar.")

# ---------------------- BARRA LATERAL: HIPERPARÁMETROS ----------------------
with st.sidebar:
    st.header("⚙️ Datos")
    nombre_ds = st.selectbox("Conjunto de datos", ["make_moons", "make_circles"])
    n_muestras = st.slider("N° de muestras", 200, 2000, 600, step=100)
    ruido = st.slider("Ruido", 0.0, 0.50, 0.20, step=0.05)

    st.header("🏗️ Arquitectura")
    n_capas = st.slider("Capas ocultas", 1, 4, 2)
    n_neuronas = st.slider("Neuronas por capa", 2, 64, 16, step=2)
    activacion = st.selectbox("Activación", ["relu", "tanh"])

    st.header("🛡️ Regularización")
    lambda_l1 = st.slider("Tasa L1 (dispersión)", 0.0, 0.05, 0.0, step=0.001,
                          format="%.3f")
    lambda_l2 = st.slider("Tasa L2 (atenuación)", 0.0, 0.05, 0.0, step=0.001,
                          format="%.3f")
    p_dropout = st.slider("Probabilidad de Dropout", 0.0, 0.7, 0.0, step=0.05)

    st.header("⚡ Estabilización")
    usar_bn = st.toggle("Batch Normalization", value=False)

    st.header("📈 Optimización")
    lr = st.select_slider("Tasa de aprendizaje",
                          options=[0.001, 0.003, 0.01, 0.03, 0.1], value=0.01)
    epocas = st.slider("Épocas", 50, 1000, 300, step=50)
    # Tamaño de lote: mini-lotes hacen vívido el efecto de BatchNorm.
    batch_opcion = st.select_slider(
        "Tamaño de lote (batch size)",
        options=["8", "16", "32", "64", "128", "Full-batch"], value="32")
    batch_size = None if batch_opcion == "Full-batch" else int(batch_opcion)
    if usar_bn and batch_size is None:
        st.caption("💡 BatchNorm luce mejor con mini-lotes (p. ej. 16–32) que "
                   "en Full-batch.")

    entrenar_click = st.button("🚀 Entrenar red", use_container_width=True,
                               type="primary")

# ----------------------------- DATOS Y MODELO ------------------------------
datos_np = generar_datos(nombre_ds, n_muestras, ruido)
datos = a_tensores(*datos_np)
capas_ocultas = [n_neuronas] * n_capas

# ----------------------------- ENTRENAMIENTO -------------------------------
if entrenar_click:
    torch.manual_seed(SEED)  # mismo punto de partida en cada experimento
    modelo = MLP(2, capas_ocultas, 2, usar_bn, p_dropout, activacion).to(DEVICE)

    st.subheader("⏱️ Entrenamiento en vivo")
    cont_progreso = st.empty()
    cont_curvas = st.empty()
    historia = entrenar(modelo, datos, lr, epocas, lambda_l1, lambda_l2,
                        batch_size, cont_progreso, cont_curvas)

    # Persistimos para poder inspeccionar sin re-entrenar.
    st.session_state["modelo"] = modelo
    st.session_state["historia"] = historia
    st.session_state["datos"] = datos
    st.session_state["n_capas"] = n_capas

# --------------------------- PANEL DE RESULTADOS ---------------------------
if "modelo" in st.session_state:
    modelo = st.session_state["modelo"]
    historia = st.session_state["historia"]
    datos = st.session_state["datos"]
    n_capas_ent = st.session_state["n_capas"]

    # --- Estación 1: Regularización ---
    st.markdown("## 1 · Regularización — Frontera de decisión y pesos")
    col_a, col_b = st.columns([1, 1.1])
    with col_a:
        st.pyplot(figura_frontera(modelo, datos), clear_figure=True)
    with col_b:
        capa_pesos = st.selectbox(
            "Capa a inspeccionar (pesos)",
            range(n_capas_ent),
            format_func=lambda i: f"Capa oculta {i + 1}",
            key="sel_pesos",
        )
        st.pyplot(figura_pesos(modelo, capa_pesos), clear_figure=True)

    # --- Estación 2: Batch Normalization ---
    st.markdown("## 2 · Estabilización — Distribución de activaciones")
    capa_act = st.selectbox(
        "Capa a inspeccionar (activaciones)",
        range(n_capas_ent),
        format_func=lambda i: f"Capa oculta {i + 1}",
        key="sel_act",
    )
    st.pyplot(figura_activaciones(modelo, datos, capa_act), clear_figure=True)

    # --- Estación 3: Diagnóstico ---
    st.markdown("## 3 · Diagnóstico — Curvas de aprendizaje")
    st.pyplot(figura_curvas(historia), clear_figure=True)

    nivel, titulo, mensaje = diagnosticar(historia)
    getattr(st, nivel)(f"**{titulo}** — {mensaje}")

    c1, c2 = st.columns(2)
    c1.metric("Precisión entrenamiento", f"{historia['acc_tr'][-1]:.1%}")
    c2.metric("Precisión validación", f"{historia['acc_val'][-1]:.1%}")
else:
    st.info("👈 Configura los hiperparámetros y pulsa **🚀 Entrenar red** "
            "para comenzar el experimento.")
