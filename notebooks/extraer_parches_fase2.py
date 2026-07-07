# -*- coding: utf-8 -*-
"""Extracción offline (paso del instructor) del dataset de imagen de la Fase 2 —
'Centinela del Tulipán'. Produce parches Sentinel-2 por parcela, etiquetados con
el cultivo del registro abierto neerlandés BRP, para una CNN multiclase.

Corre en geo_env. Salida: un .npz cacheado (~<20 MB) que el notebook carga sin
geolibrerías ni credenciales, igual que el CSV del KNMI de la Fase 1.

Diseño (ver IMPLEMENTATION_PLAN_EJEMPLO_FASE2_TULIPANES.md):
- Imágenes: Sentinel-2 L2A vía Element84 STAC -> COG en S3 público (anónimo).
- Etiquetas: BRP Gewaspercelen (PDOK WFS), campo `gewas`, año 2025.
- Coherencia temporal: imágenes 2025 + etiquetas BRP 2025 (los cultivos rotan).
- Parche 32x32 px (320 m) centrado en el centroide de la parcela; 4 bandas R,G,B,NIR.
- Varias fechas despejadas -> más muestras (misma parcela = mismo split, sin fuga).
"""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, geopandas as gpd, requests
from collections import Counter
from pystac_client import Client
import rasterio
from rasterio.windows import from_bounds

# ─────────────────────────────────────────────────────────────────────
# Configuración
POLDER_BBOX_LL = (5.50, 52.58, 6.00, 52.86)      # lon/lat: Noordoostpolder + bordes
ANIO = 2025
STAC_URL = "https://earth-search.aws.element84.com/v1"
WFS = "https://service.pdok.nl/rvo/brpgewaspercelen/wfs/v1_0"
PATCH = 32            # px (a 10 m/px = 320 m)
# Fechas despejadas (tile 31UFU) repartidas por la temporada 2025 -> diversidad
# fenológica (abril emergencia/floración · junio desarrollo · agosto madurez).
FECHAS_OBJETIVO = ["2025-04-04", "2025-05-11", "2025-06-12", "2025-08-14"]
TILE = "31UFU"
CRS_ESCENA = "EPSG:32631"

# Cultivos objetivo (excluye sloot/natuur/braak/grasland = no-cultivo)
OBJETIVO = {
    "Tulp, bloembollen en -knollen": "tulipan",
    "Aardappelen, poot NAK":         "papa",
    "Aardappelen, consumptie":       "papa",
    "Tarwe, winter-":                "trigo",
    "Uien, gele zaai-":              "cebolla",
    "Uien, rode zaai-":              "cebolla",
    "Bieten, suiker-":               "remolacha",
    "Mais, snij-":                   "maiz",
}
BANDAS = ["red", "green", "blue", "nir"]

# ─────────────────────────────────────────────────────────────────────
# 1) Escenas Sentinel-2 (tile 31UFU) en las fechas objetivo de la temporada
print("1) Localizando escenas Sentinel-2 (tile 31UFU) en las fechas objetivo…")
cat = Client.open(STAC_URL)
escenas = []
for fecha in FECHAS_OBJETIVO:
    cand = [it for it in cat.search(collections=["sentinel-2-l2a"], bbox=POLDER_BBOX_LL,
                                    datetime=f"{fecha}/{fecha}").items()
            if it.id.split("_")[1] == TILE]
    if not cand:
        print(f"   ⚠️  sin escena {TILE} el {fecha}, se omite"); continue
    it = min(cand, key=lambda it: it.properties["eo:cloud_cover"])
    escenas.append(it)
    print(f"   · {it.id}  {fecha}  nubes {it.properties['eo:cloud_cover']:.1f}%")
assert escenas, "No hay escenas"

# ─────────────────────────────────────────────────────────────────────
# 2) Parcelas BRP (paginado) sobre el polder
print("2) Descargando parcelas BRP (paginado)…")
def brp_pagina(start):
    p = {"service":"WFS","version":"2.0.0","request":"GetFeature",
         "typeNames":"brpgewaspercelen:BrpGewas","outputFormat":"application/json",
         "srsName":"EPSG:4326","count":"1000","startIndex":str(start),
         "bbox":f"{POLDER_BBOX_LL[1]},{POLDER_BBOX_LL[0]},{POLDER_BBOX_LL[3]},{POLDER_BBOX_LL[2]},urn:ogc:def:crs:EPSG::4326"}
    return requests.get(WFS, params=p, timeout=120).text
partes, start = [], 0
while True:
    g = gpd.read_file(brp_pagina(start))
    if len(g) == 0: break
    partes.append(g); start += len(g)
    if len(g) < 1000: break
gdf = gpd.GeoDataFrame(__import__("pandas").concat(partes, ignore_index=True), crs="EPSG:4326")
gdf = gdf[gdf["gewas"].isin(OBJETIVO)].copy()
gdf["clase"] = gdf["gewas"].map(OBJETIVO)
gdf = gdf.to_crs(CRS_ESCENA)
gdf["cx"] = gdf.geometry.centroid.x; gdf["cy"] = gdf.geometry.centroid.y
print(f"   parcelas-cultivo: {len(gdf)}  |  clases: {dict(Counter(gdf['clase']))}")

# ─────────────────────────────────────────────────────────────────────
# 3) Para cada fecha: leer las 4 bandas del bbox del polder una vez; clip local
print("3) Leyendo bandas por fecha y recortando parches…")
# bbox del polder en CRS de escena
poly_utm = gpd.GeoSeries.from_xy([POLDER_BBOX_LL[0],POLDER_BBOX_LL[2]],
                                 [POLDER_BBOX_LL[1],POLDER_BBOX_LL[3]], crs="EPSG:4326").to_crs(CRS_ESCENA)
xmin, xmax = float(poly_utm.x.min()), float(poly_utm.x.max())
ymin, ymax = float(poly_utm.y.min()), float(poly_utm.y.max())

X, y, pid, fechas, clases_lista = [], [], [], [], sorted(set(OBJETIVO.values()))
clase_idx = {c:i for i,c in enumerate(clases_lista)}
half = PATCH // 2

with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                  GDAL_HTTP_MULTIPLEX="YES"):
    for it in escenas:
        fecha = it.properties["datetime"][:10]
        # leer las 4 bandas del bbox del polder en memoria
        arrs, transform = {}, None
        for b in BANDAS:
            with rasterio.open(it.assets[b].href) as src:
                win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
                arrs[b] = src.read(1, window=win)
                if transform is None:
                    transform = src.window_transform(win)
                    H, W = arrs[b].shape
        # índice de cada parcela dentro del array leído
        inv = ~transform
        n_ok = 0
        for _, row in gdf.iterrows():
            col, r = inv * (row.cx, row.cy)
            col, r = int(round(col)), int(round(r))
            if col-half < 0 or r-half < 0 or col+half > W or r+half > H:
                continue
            parche = np.stack([arrs[b][r-half:r+half, col-half:col+half] for b in BANDAS])  # (4,32,32)
            if parche.shape != (4, PATCH, PATCH) or parche.max() == 0:
                continue
            X.append(parche.astype(np.uint16)); y.append(clase_idx[row.clase])
            pid.append(str(row["id"])); fechas.append(fecha); n_ok += 1
        print(f"   {fecha}: {n_ok} parches")

X = np.stack(X); y = np.array(y, np.int64)
pid = np.array(pid); fechas = np.array(fechas)
print(f"\nTotal parches: {len(X)} · forma {X.shape} · {X.nbytes/1e6:.1f} MB en memoria")
print("balance por clase:", {clases_lista[i]: int((y==i).sum()) for i in range(len(clases_lista))})

# ─────────────────────────────────────────────────────────────────────
# 4) Guardar .npz cacheado
SAL = "/Users/javiermauriciosierra/Documents/Deep learning course/Material html/notebooks/data/sentinel2_parches_polder.npz"
np.savez_compressed(SAL, X=X, y=y, parcela_id=pid, fecha=fechas,
                    clases=np.array(clases_lista), patch=PATCH, bandas=np.array(BANDAS))
import os
print(f"\n✔ Guardado: {SAL} ({os.path.getsize(SAL)/1e6:.2f} MB comprimido)")
