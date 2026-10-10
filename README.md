# Deep Learning USTA — Profundización I: Redes Neuronales

Materiales web de aprendizaje autónomo del curso **Profundización I: Redes Neuronales – Deep Learning** (EA-N-F-004).

Maestría en Ciencia de Datos · Universidad Santo Tomás.

🌐 **Sitio:** https://jotamao1985.github.io/Deep_Learning_Usta/

El portal `index.html` es un punto de entrada multi-módulo (selector Módulo 1 / Módulo 2 / Módulo 3) que da acceso a los materiales de cada módulo y al Proyecto Centinela (Fases 1–3).

## Módulo 1 — Fundamentos del Aprendizaje Profundo

| Archivo | Descripción |
|---|---|
| `01_Modulo_1_Fundamentos_Deep_Learning.html` | Material principal — 5 capítulos |
| `00_ruta-aprendizaje-modulo-1.html` | Ruta de aprendizaje (mapa navegable) |
| `011_material-consulta-modulo-1.html` | Material de consulta |
| `012_infografia-contenidos-modulo-1.html` | Infografía de contenidos |
| `013_microlearning-modulo-1.html` | Microlearning |

## Módulo 2 — Arquitecturas Especializadas de Deep Learning

| Archivo | Descripción |
|---|---|
| `02_Modulo_2_Arquitecturas_Deep_Learning.html` | Material principal — 3 capítulos (CNN/ViT, RNN/LSTM/GRU + atención, generativos AE/VAE/GAN/difusión/Transformers) |
| `020_ruta-aprendizaje-modulo-2.html` | Ruta de aprendizaje (mapa navegable) |
| `021_material-consulta-modulo-2.html` | Material de consulta |
| `022_infografia-contenidos-modulo-2.html` | Infografía de contenidos |
| `023_microlearning-modulo-2.html` | Microlearning |

## Módulo 3 — Implementación y Herramientas Profesionales

| Archivo | Descripción |
|---|---|
| `03_Modulo_3_Implementacion_Deep_Learning.html` | Material principal — 3 capítulos (frameworks TF/Keras & PyTorch + Keras 3, HPC/GPU + precisión mixta + Colab/Kaggle, pipelines tf.data/DataLoader + Data Augmentation + ONNX/despliegue) |
| `030_ruta-aprendizaje-modulo-3.html` | Ruta de aprendizaje (mapa navegable) |
| `031_material-consulta-modulo-3.html` | Material de consulta |
| `032_infografia-contenidos-modulo-3.html` | Infografía de contenidos |
| `033_microlearning-modulo-3.html` | Microlearning |

## Herramientas interactivas embebidas

Exploradores autónomos (HTML + JS/SVG o three.js empaquetado en `vendor/`, sin CDN) que se embeben vía `iframe` con auto-ajuste de altura dentro del material principal de cada módulo. También funcionan abriéndolos directamente.

| Archivo | Módulo / Capítulo | Descripción |
|---|---|---|
| `capa-oculta-3d.html` | M1 · Cap 1.2 | Animación 3D (three.js): una red 2-3-1 entrenada en el navegador deforma el plano de entrada hasta la capa oculta, donde la frontera curva queda sobre el plano de la neurona de salida. `?selftest=1` |
| `cnn-anatomia-interactiva.html` | M2 · Cap 2.1 | Animación paso a paso del pipeline de una CNN (entrada → convolución → ReLU → pooling → clasificación) |
| `vit-anatomia-interactiva.html` | M2 · Cap 2.1 | Explorador ViT vs. CNN (anatomía del ViT, campo receptivo local vs. global, bloque encoder) |
| `lstm-forward-interactivo.html` | M2 · Cap 2.2 | El paso hacia adelante de una LSTM escalar, paso a paso, con sustitución numérica exacta. `?selftest=1` (paridad con `lstm-forward-referencia.py`) |
| `rnn-lstm-gru-interactivo.html` | M2 · Cap 2.2 | Explorador RNN·LSTM·GRU: misma secuencia en las 3 celdas, compuertas en vivo, desvanecimiento del gradiente y cuadro comparativo |
| `atencion-transformer-interactiva.html` | M2 · Cap 2.3 §6 | Autoatención en vivo sobre la palabra «vaina»: pesos QKV, matriz de atención y generación autorregresiva con temperatura |
| `ae-vae-gan-transformer-interactivo.html` | M2 · Cap 2.3 | Comparador AE·VAE·GAN·Transformer: diagramas, fichas técnicas y laboratorios 2-D/secuencias con entrenamiento real (retropropagación + Adam) en el navegador |

## Proyecto Centinela (longitudinal)

| Archivo | Descripción |
|---|---|
| `11_Modulo_1_Actividad_Fase1_Centinela.html` | Actividad evaluativa · Fase 1 (MLP) |
| `21_Modulo_2_Actividad_Fase2_Centinela.html` | Actividad evaluativa · Fase 2 (CNN + RNN + fusión) |
| `31_Modulo_3_Actividad_Fase3_Centinela.html` | Actividad evaluativa · Fase 3 (pipeline + despliegue) |

## Notebooks (Google Colab)

| Archivo | Descripción |
|---|---|
| `notebooks/01-lab-fundamentos-modulo-1.ipynb` | Laboratorio M1 (neurona → MLP en PyTorch) |
| `notebooks/02-scaffold-centinela-fase1.ipynb` | Andamiaje del proyecto · Fase 1 |
| `notebooks/03-lab-arquitecturas-modulo-2.ipynb` | Laboratorio M2 (CNN/FashionMNIST, LSTM, autoencoder) |
| `notebooks/04-scaffold-centinela-fase2.ipynb` | Andamiaje del proyecto · Fase 2 (ramas A/B/C) |
| `notebooks/08-lab-lstm-forward-paso-a-paso.ipynb` | Laboratorio M2 · Cap 2.2: la LSTM paso a paso, a mano, frente al simulador y a `nn.LSTMCell` |
| `notebooks/06-lab-implementacion-modulo-3.ipynb` | Laboratorio M3 (frameworks TF/PyTorch, GPU + precisión mixta, pipelines, ONNX) |
| `notebooks/07-scaffold-centinela-fase3.ipynb` | Andamiaje del proyecto · Fase 3 (pipeline + GPU + despliegue offline) |

## Laboratorio interactivo (Streamlit)

| Archivo | Descripción |
|---|---|
| `notebooks/app-laboratorio-regularizacion.py` | Laboratorio de **Regularización, Estabilización y Diagnóstico** (M1 · Cap 5): L1/L2/Dropout, BatchNorm y curvas de aprendizaje en vivo. Se ejecuta localmente con `streamlit run notebooks/app-laboratorio-regularizacion.py`. |

Sitio estático (React + Tailwind vía CDN en los materiales principales); no requiere compilación. Servido con GitHub Pages. Las animaciones 3D usan `vendor/three-r160.iife.min.js` (three.js r160, MIT; ver `vendor/BUILD.md`).
