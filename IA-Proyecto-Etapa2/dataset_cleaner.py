"""
Limpieza del dataset de emociones (dataset exclusivamente de texto).

NOTA: El limpiador no realiza particiones del dataset, pero sí prepara el para 
      que pueda ser particionado eficientemente.

Pasos:
  1) normalizar solo para comparar (el texto original no se modifica)
  2) eliminar textos con etiquetas contradictorias (todas sus copias)
  3) eliminar duplicados exactos
  4) detectar cuasi-duplicados (TF-IDF + coseno, candidatos por términos raros)

Salidas (carpeta --out):
  cleaned_full_data.csv       id, text, label, group
  pares_cuasi_duplicados.csv  los pares detectados, para revisar a mano
  reporte_limpieza.json       cifras de cada paso

Uso:
  python dataset_cleaner.py --csv Emotions.csv --out salida --modo eliminar
  --modo eliminar : conserva 1 texto por grupo de cuasi-duplicados
  --modo agrupar  : conserva todo y deja la columna `group`; al particionar,
                    cada grupo debe caer completo en una sola particion
"""

import argparse
import gc
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize


def parsear_argumentos():
    """Configura y lee los parámetros de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Limpieza eficiente de dataset de texto.")
    parser.add_argument("--csv", default="./data/raw/text.csv", help="Ruta al archivo CSV de entrada.")
    parser.add_argument("--out", default="salida", help="Carpeta de salida para reportes y datos limpios.")
    parser.add_argument("--umbral", type=float, default=0.90, help="Coseno mínimo para cuasi-duplicado.")
    parser.add_argument("--raro", type=float, default=0.002, help="Un término es raro si aparece en <= 0.2%% de textos.")
    parser.add_argument("--modo", choices=["eliminar", "agrupar"], default="eliminar", help="Estrategia ante cuasi-duplicados.")
    parser.add_argument("--lote_texto", type=int, default=50_000, help="Tamaño de lote.")
    parser.add_argument("--limite", type=int, default=0, help="Límite de filas para pruebas rápidas (0 = todo).")
    return parser.parse_args()


# Funciones de procesamiento y TF-IDF (Lógica intacta de cuasi-duplicados)

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

def extraer_hashes_caracteristicas(texto):
    """Extrae hashes de 64 bits para unigramas y bigramas de un texto."""
    palabras = _TOKEN_RE.findall(texto)
    unigramas = [hash(w) for w in palabras]
    bigramas = [hash(palabras[i] + " " + palabras[i + 1]) for i in range(len(palabras) - 1)]
    return np.array(unigramas + bigramas, dtype=np.int64)


def construir_matriz_tfidf_eficiente(textos, tamano_lote, n_total):
    """
    Construye la matriz TF-IDF (unigramas + bigramas) en 2 pasadas por lotes sin guardar
    el vocabulario en memoria, usando hashes de 64 bits.
    """
    # Pasada 1: Frecuencia documental
    gk, gcnt = np.empty(0, np.int64), np.empty(0, np.int32)
    for i in range(0, n_total, tamano_lote):
        f = np.concatenate([np.unique(extraer_hashes_caracteristicas(t)) for t in textos[i:i + tamano_lote]])
        k, c = np.unique(f, return_counts=True)
        allk = np.concatenate([gk, k])
        allc = np.concatenate([gcnt, c.astype(np.int32)])
        
        orden = np.argsort(allk, kind="stable")
        allk, allc = allk[orden], allc[orden]
        
        indices_unicos = np.flatnonzero(np.r_[True, allk[1:] != allk[:-1]])
        gk = allk[indices_unicos]
        gcnt = np.add.reduceat(allc, indices_unicos).astype(np.int32)

    # Filtrar min_df=2 y max_df=2%
    filtro_df = (gcnt >= 2) & (gcnt <= 0.02 * n_total)
    llaves_validas = gk[filtro_df]
    df_terminos = gcnt[filtro_df].astype(np.int64)
    del gk, gcnt, allk, allc, filtro_df
    gc.collect()

    # Pasada 2: Matriz TF en formato ralo (CSR)
    partes_csr = []
    for i in range(0, n_total, tamano_lote):
        indices_cols, datos, punteros = [], [], [0]
        for t in textos[i:i + tamano_lote]:
            k, c = np.unique(extraer_hashes_caracteristicas(t), return_counts=True)
            posiciones = np.minimum(np.searchsorted(llaves_validas, k), len(llaves_validas) - 1)
            coinciden = llaves_validas[posiciones] == k
            
            indices_cols.append(posiciones[coinciden])
            datos.append(c[coinciden])
            punteros.append(punteros[-1] + int(coinciden.sum()))
            
        matriz_lote = csr_matrix(
            (np.concatenate(datos).astype(np.float32), np.concatenate(indices_cols), np.array(punteros)),
            shape=(len(punteros) - 1, len(llaves_validas))
        )
        partes_csr.append(matriz_lote)

    X = vstack(partes_csr).tocsr()
    del partes_csr, llaves_validas
    
    # Aplicar TF sublineal y IDF suave
    X.data = 1.0 + np.log(X.data)
    idf_suave = (np.log((1 + n_total) / (1 + df_terminos)) + 1.0).astype(np.float32)
    X.data *= idf_suave[X.indices]
    
    # Normalización L2
    X = normalize(X, copy=False)
    
    return X, df_terminos


# Pipeline principal

def ejecutar_pipeline():
    args = parsear_argumentos()
    os.makedirs(args.out, exist_ok=True)
    reporte = {"parametros": vars(args)}

    # Carga de los datos, y normalización de los registros
    df = pd.read_csv(args.csv, usecols=["text", "label"], dtype={"label": "int8"})
    reporte["sha256_csv_entrada"] = hashlib.sha256(open(args.csv, "rb").read()).hexdigest()
    reporte["filas_originales"] = len(df)
    
    df["norm"] = df["text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["norm"].str.len() > 0]

    # Eliminar textos con etiquetas contradictorias
    etiquetas_por_texto = df.groupby("norm")["label"].transform("nunique")
    reporte["filas_eliminadas_etiquetas_contradictorias"] = int((etiquetas_por_texto > 1).sum())
    reporte["textos_distintos_contradictorios"] = int(df.loc[etiquetas_por_texto > 1, "norm"].nunique())
    df = df[etiquetas_por_texto == 1]

    # Eliminación de copias exactas
    reporte["copias_exactas_eliminadas"] = int(df["norm"].duplicated().sum())
    df = df.drop_duplicates(subset="norm").reset_index(drop=True)
    df["id"] = df["norm"].map(lambda s: hashlib.md5(s.encode()).hexdigest())
    
    assert df["id"].is_unique
    if args.limite:
        df = df.iloc[: args.limite].reset_index(drop=True)
        
    n_registros = len(df)
    reporte["textos_tras_dedup_exacta"] = n_registros
    print(f"Tras etiquetas contradictorias y duplicados exactos: {n_registros}")

    # Detección de cuasi-duplicados (TF-IDF + Coseno - Lógica Intacta)
    t_inicio = time.time()
    textos_array = df["norm"].values
    
    X, df_terminos = construir_matriz_tfidf_eficiente(textos_array, args.lote_texto, n_registros)
    df.drop(columns="norm", inplace=True)
    gc.collect()

    # Selección de términos raros como candidatos
    es_raro = df_terminos <= max(2, args.raro * n_registros)
    X_raros = (X[:, es_raro] > 0).astype(np.float32).tocsr()
    X_raros_T = X_raros.T.tocsr()

    filas_pares, cols_pares = [], []
    num_candidatos = 0
    tamano_bloque = 100

    for idx_inicio in range(0, n_registros, tamano_bloque):
        sim_candidatos = (X_raros[idx_inicio:idx_inicio + tamano_bloque] @ X_raros_T).tocoo()
        masca_triangular = sim_candidatos.row + idx_inicio < sim_candidatos.col
        
        r_cand = sim_candidatos.row[masca_triangular] + idx_inicio
        c_cand = sim_candidatos.col[masca_triangular]
        num_candidatos += len(r_cand)
        
        # Verificación de similitud en lotes
        for sub_i in range(0, len(r_cand), 50_000):
            r_sub = r_cand[sub_i:sub_i + 50_000]
            c_sub = c_cand[sub_i:sub_i + 50_000]
            
            cosenos = np.asarray(X[r_sub].multiply(X[c_sub]).sum(1)).ravel()
            pasa_umbral = cosenos >= args.umbral
            
            filas_pares.append(r_sub[pasa_umbral])
            cols_pares.append(c_sub[pasa_umbral])
            
        if (idx_inicio // tamano_bloque) % 500 == 0:
            print(f"  cuasi-duplicados: {idx_inicio}/{n_registros}  ({time.time() - t_inicio:.0f}s)", flush=True)

    filas_pares = np.concatenate(filas_pares) if filas_pares else np.array([], int)
    cols_pares = np.concatenate(cols_pares) if cols_pares else np.array([], int)

    # Componentes conexas para agrupar cuasi-duplicados
    matriz_grafos = coo_matrix((np.ones(len(filas_pares)), (filas_pares, cols_pares)), shape=(n_registros, n_registros))
    _, etiquetas_grupos = connected_components(matriz_grafos, directed=False)
    
    df["group"] = etiquetas_grupos
    conteo_grupos = df["group"].map(df["group"].value_counts())

    reporte.update(
        pares_candidatos=int(num_candidatos),
        pares_cuasi_dup=int(len(filas_pares)),
        grupos_multiples=int((df["group"].value_counts() > 1).sum()),
        textos_en_grupos_multiples=int((conteo_grupos > 1).sum()),
        grupos_con_etiquetas_mixtas=int((df[conteo_grupos > 1].groupby("group")["label"].nunique() > 1).sum())
    )

    # Exportar pares detectados para revisión manual
    pd.DataFrame({
        "a": df["text"].values[filas_pares],
        "b": df["text"].values[cols_pares],
        "label_a": df["label"].values[filas_pares],
        "label_b": df["label"].values[cols_pares]
    }).to_csv(f"{args.out}/pares_cuasi_duplicados.csv", index=False)

    # Aplicar modo de salida
    if args.modo == "eliminar":
        total_previo = len(df)
        df = df.drop_duplicates("group", keep="first").reset_index(drop=True)
        reporte["cuasi_duplicados_eliminados"] = total_previo - len(df)

    print(f"Registros limpios finales: {len(df)}")
    reporte["registros_finales"] = len(df)

    # SALIDAS Y REPORTES
    df[["id", "text", "label", "group"]].to_csv(f"{args.out}/cleaned_full_data.csv", index=False)
    
    reporte["distribucion_clases_pct"] = (df["label"].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()

    with open(f"{args.out}/reporte_limpieza.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=1, default=str)

    print(json.dumps(reporte, indent=1, default=str))


if __name__ == "__main__":
    ejecutar_pipeline()