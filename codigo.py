from pathlib import Path
import pandas as pd

# Mostrar tablas completas en el terminal
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

# Carpeta data/ junto a este archivo (funciona desde cualquier directorio)
CARPETA = Path(__file__).resolve().parent

# 1) Leer todos los CSV (separador ';'; probamos dos codificaciones)
def leer_csv(ruta):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(ruta, sep=";", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {ruta}")

ficheros = sorted(CARPETA.glob("300351-*.csv"))
print("Ficheros encontrados:", len(ficheros))
raw = pd.concat([leer_csv(f) for f in ficheros], ignore_index=True)

# Nombres de columna en mayúsculas y sin espacios
raw.columns = raw.columns.str.strip().str.upper()
print("Forma:", raw.shape)
print("Primeras columnas:", raw.columns.tolist()[:10])
print(raw.head(3))

# 2) Formato ancho (D01..D31) -> largo (una fila por día)
cols_dia = [f"D{i:02d}" for i in range(1, 32)]
cols_val = [f"V{i:02d}" for i in range(1, 32)]
id_cols = ["ESTACION", "MAGNITUD", "ANO", "MES"]

valores = raw.melt(id_vars=id_cols, value_vars=cols_dia,
                   var_name="dia", value_name="valor")
validez = raw.melt(id_vars=id_cols, value_vars=cols_val,
                   var_name="dia", value_name="valido")
valores["dia"] = valores["dia"].str[1:].astype(int)
valores["valido"] = validez["valido"].values   # mismo orden de filas

# 3) Fecha real; los días inexistentes (31 de febrero...) quedan como NaT
valores["fecha"] = pd.to_datetime(
    dict(year=valores["ANO"], month=valores["MES"], day=valores["dia"]),
    errors="coerce")
validos = valores[(valores["valido"] == "V") & valores["fecha"].notna()]

# 4) Qué hay disponible
print("Rango de fechas:", validos["fecha"].min(), "->", validos["fecha"].max())
print("Valores de validez:", valores["valido"].value_counts().to_dict())
tabla = validos.pivot_table(index="ESTACION", columns="MAGNITUD",
                            values="valor", aggfunc="count")
print(tabla)

print("------------------------------------PARTE2----------------------------------------")
# ---- PASO 2: tabla diaria por estación (una columna por variable) ----
NOMBRES = {81: "viento_vel", 83: "temp", 86: "humedad", 87: "presion", 89: "precip"}
ESTACIONES = [24, 56, 59, 102, 103, 106, 107, 108]

sel = validos[validos["ESTACION"].isin(ESTACIONES)
              & validos["MAGNITUD"].isin(NOMBRES)]
diario = (sel.pivot_table(index=["ESTACION", "fecha"],
                          columns="MAGNITUD", values="valor")
             .rename(columns=NOMBRES)
             .reset_index())

print(diario.shape)
print(diario.groupby("ESTACION").size())    # días por estación
print(diario.isna().mean().round(3))         # fracción de huecos por variable
print(diario.describe().round(2))
p = diario["precip"].dropna()
print("Días con lluvia (precip > 0):", round((p > 0).mean(), 3))
print("Días con lluvia (precip >= 0.1):", round((p >= 0.1).mean(), 3))

diario.to_csv("diario.csv", index=False)     # guardamos el resultado

print("------------------------------------PARTE3----------------------------------------")

# ---- PASO 3: huecos y valores raros ----
calendario = pd.date_range(diario["fecha"].min(), diario["fecha"].max())

# Días sin temperatura por estación y racha más larga de días seguidos
filas = {}
for est, g in diario.groupby("ESTACION"):
    s = g.set_index("fecha")["temp"].reindex(calendario)
    falta = s.isna()
    racha = falta.groupby((~falta).cumsum()).sum().max()
    filas[est] = {"dias_sin_temp": int(falta.sum()), "racha_max": int(racha)}
print(pd.DataFrame(filas).T)

# Valores sospechosos
raros = diario[(diario["presion"] < 900) | (diario["humedad"] < 5)
               | (diario["precip"] > 50) | (diario["temp"] < -5)]
print(len(raros))
print(raros.head(30))

import numpy as np

print("------------------------------------PARTE4----------------------------------------")

# ---- PASO 4: limpieza y localizar los huecos largos ----
limpio = diario.copy()
limpio.loc[limpio["presion"] < 900, "presion"] = np.nan   # sensor roto (678, 800...)
limpio.loc[limpio["humedad"] < 5, "humedad"] = np.nan     # 3-4 % inverosímil (est. 108)

# Reindexar cada estación al calendario completo (los días que faltan pasan a NaN)
partes = []
for est, g in limpio.groupby("ESTACION"):
    g = g.set_index("fecha").reindex(calendario)
    g["ESTACION"] = est
    g.index.name = "fecha"
    partes.append(g.reset_index())
completo = pd.concat(partes, ignore_index=True)

# Dos huecos más largos de temperatura por estación, con fechas
for est, g in completo.groupby("ESTACION"):
    falta = g.set_index("fecha")["temp"].isna()
    fechas = falta[falta].index.to_series()
    nueva = (fechas.diff() != pd.Timedelta(days=1)).cumsum()
    rachas = fechas.groupby(nueva.values).agg(["min", "max", "size"])
    print("Estación", est)
    print(rachas.sort_values("size", ascending=False).head(2).to_string())

completo.to_csv("diario_limpio.csv", index=False)

from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

print("------------------------------------PARTE5----------------------------------------")

# ---- PASO 5: variables, objetivos, partición y modelos de referencia ----
VARS = ["temp", "humedad", "presion", "viento_vel", "precip"]
H = [1, 2, 3]          # horizontes de predicción (días)
LAGS = [0, 1, 2]       # t, t-1, t-2

def construir(g):
    # g es una estación con el calendario completo: shift() no cruza huecos
    g = g.sort_values("fecha").set_index("fecha")
    X = pd.DataFrame(index=g.index)
    # Entradas: valores de hoy y de los dos días anteriores
    for v in VARS:
        for l in LAGS:
            X[f"{v}_t-{l}" if l else f"{v}_t"] = g[v].shift(l)
    # Época del año como seno y coseno (para que el 31-dic quede cerca del 1-ene)
    doy = g.index.dayofyear
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    # Objetivos: temperatura a 1, 2 y 3 días; lluvia mañana (>= 0,1 mm)
    for h in H:
        X[f"y_temp_t+{h}"] = g["temp"].shift(-h)
    sig = g["precip"].shift(-1)
    X["y_lluvia_t+1"] = np.where(sig.isna(), np.nan, (sig >= 0.1).astype(float))
    X["ESTACION"] = g["ESTACION"].iloc[0]
    return X.reset_index()

datos = pd.concat([construir(g) for _, g in completo.groupby("ESTACION")],
                  ignore_index=True)
cols_x = [c for c in datos.columns
          if c not in ("fecha", "ESTACION") and not c.startswith("y_")]

def particion(d):
    f = d["fecha"]
    return (d[f < "2025-01-01"],
            d[(f >= "2025-01-01") & (f < "2026-01-01")],
            d[f >= "2026-01-01"])

# Regresión: nos quedamos con las filas sin ningún dato que falte
reg = datos.dropna(subset=cols_x + [f"y_temp_t+{h}" for h in H])
tr, va, te = particion(reg)
print("Regresión - filas:", len(reg), "| train/val/test:", len(tr), len(va), len(te))
for h in H:
    for nombre, d in (("val", va), ("test", te)):
        mae = mean_absolute_error(d[f"y_temp_t+{h}"], d["temp_t"])
        print(f"Persistencia t+{h} | {nombre}: MAE = {mae:.2f} °C")

# Clasificación (lluvia)
cl = datos.dropna(subset=cols_x + ["y_lluvia_t+1"])
tr_c, va_c, te_c = particion(cl)
print("Clasificación - filas:", len(cl), "| train/val/test:", len(tr_c), len(va_c), len(te_c))
print("Proporción de lluvia en train:", round(tr_c["y_lluvia_t+1"].mean(), 3))
for nombre, d in (("val", va_c), ("test", te_c)):
    y = d["y_lluvia_t+1"]
    nunca = np.zeros(len(d))
    hoy = (d["precip_t"] >= 0.1).astype(float)
    print(nombre, "| nunca llueve: acc=%.3f F1=%.3f" % (accuracy_score(y, nunca), f1_score(y, nunca, zero_division=0)))
    print(nombre, "| como hoy:     acc=%.3f F1=%.3f" % (accuracy_score(y, hoy), f1_score(y, hoy)))

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score

print("------------------------------------PARTE6a----------------------------------------")

# ---- PASO 6a: perceptrón multicapa (temperatura a t+1, t+2 y t+3) ----
cols_y = [f"y_temp_t+{h}" for h in H]
mlp = make_pipeline(
    StandardScaler(),                       # estandariza usando SOLO el train
    MLPRegressor(hidden_layer_sizes=(32, 16),  # 2 capas ocultas: 32 y 16 neuronas
                 alpha=1e-3,                   # regularización L2
                 max_iter=500, random_state=42))
mlp.fit(tr[cols_x], tr[cols_y])             # una red, 3 salidas a la vez

print("\n=== Perceptrón multicapa ===")
for nombre, d in (("val", va), ("test", te)):
    pred = mlp.predict(d[cols_x])
    for i, h in enumerate(H):
        mae_mlp = mean_absolute_error(d[f"y_temp_t+{h}"], pred[:, i])
        mae_per = mean_absolute_error(d[f"y_temp_t+{h}"], d["temp_t"])
        mejora = 100 * (mae_per - mae_mlp) / mae_per
        print(f"MLP t+{h} | {nombre}: MAE = {mae_mlp:.2f} °C "
              f"(persistencia {mae_per:.2f}; mejora {mejora:.1f} %)")

print("------------------------------------PARTE6b----------------------------------------")

# ---- PASO 6b: regresión logística (¿lloverá mañana?) ----
print("\n=== Regresión logística ===")
for etiqueta, cw in (("sin pesos", None), ("balanceada", "balanced")):
    logi = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, class_weight=cw))
    logi.fit(tr_c[cols_x], tr_c["y_lluvia_t+1"])
    for nombre, d in (("val", va_c), ("test", te_c)):
        y = d["y_lluvia_t+1"]
        p = logi.predict(d[cols_x])
        print(f"Logística {etiqueta} | {nombre}: acc={accuracy_score(y, p):.3f} "
              f"prec={precision_score(y, p, zero_division=0):.3f} "
              f"recall={recall_score(y, p):.3f} F1={f1_score(y, p):.3f}")

import matplotlib.pyplot as plt
import seaborn as sns

print("----------------------------PARTE7-----------------------------------")

# ---- PASO 7: figuras del EDA (se guardan en figuras/) ----
FIG = Path(__file__).resolve().parent / "figuras"
FIG.mkdir(exist_ok=True)

# Figura 1: serie temporal de la temperatura (media de las estaciones disponibles cada día)
media_dia = completo.groupby("fecha")["temp"].mean()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(media_dia.index, media_dia.values, lw=0.8)
ax.axvspan(pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-31"),
           color="red", alpha=0.25, label="Sin datos (dic-2024)")
ax.set_title("Temperatura media diaria (media de las 8 estaciones)")
ax.set_ylabel("°C"); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "01_serie_temperatura.png", dpi=150); plt.close(fig)

# Figura 2: cobertura de datos por estación y mes (muestra los huecos)
cob = (completo.assign(mes=completo["fecha"].dt.to_period("M").astype(str),
                       ok=completo["temp"].notna())
       .pivot_table(index="ESTACION", columns="mes", values="ok", aggfunc="mean"))
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(cob, cmap="viridis", vmin=0, vmax=1, ax=ax, xticklabels=6,
            cbar_kws={"label": "fracción de días con temperatura"})
ax.set_title("Cobertura de datos por estación y mes")
ax.set_xlabel("Mes"); ax.set_ylabel("Estación")
fig.tight_layout(); fig.savefig(FIG / "02_cobertura.png", dpi=150); plt.close(fig)

# Figura 3: proporción de días de lluvia por mes
ll = completo.dropna(subset=["precip"]).copy()
ll["llueve"] = ll["precip"] >= 0.1
ll["mes"] = ll["fecha"].dt.month
prop = ll.groupby("mes")["llueve"].mean() * 100
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(prop.index, prop.values)
ax.set_xticks(range(1, 13)); ax.set_xlabel("Mes")
ax.set_ylabel("% de días con lluvia (≥ 0,1 mm)")
ax.set_title("Proporción de días de lluvia por mes"); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "03_lluvia_por_mes.png", dpi=150); plt.close(fig)
print("Lluvia por mes (%):", prop.round(1).to_dict())

# Figura 4: correlaciones entre variables
corr = completo[VARS].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlación entre variables")
fig.tight_layout(); fig.savefig(FIG / "04_correlaciones.png", dpi=150); plt.close(fig)
print(corr.round(2))