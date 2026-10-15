# Contrato experimental: Etapa 2 (Track B)

Benchmark de reconocimiento de emociones en texto (CARER, 6 clases).
Versión: 0.1 (borrador, no congelado).

Este documento fija las reglas que valen para los tres integrantes y para todos los modelos. Una regla solo cambia por commit, con la razón escrita en el "Historial de cambios" (sección 16). Nadie ajusta una regla de forma local en su máquina.

Marcas usadas:

- **[MEDIR]**: valor que se fija con datos reales durante la Fase 0 (sección 13).
- **[POR DEFINIR]**: espera acuerdo explícito del equipo.

---

## 1. Qué se compara y qué no se puede contaminar

Ejes de estudio: familia arquitectural, escala de parámetros y objetivo de entrenamiento.

| Par controlado | Eje que aísla |
|---|---|
| DistilBERT vs BERT-base | Escala (encoders) |
| Qwen2.5-0.5B vs Qwen2.5-1.5B | Escala (decoders) |
| Gemma 4 E2B vs Gemma 4 E4B | Escala (decoders, otra familia) |
| Qwen2.5-1.5B vs Gemma 4 E2B | Familia (decoders) |
| Qwen2.5-1.5B vs DeepSeek-R1-Distill-Qwen-1.5B | Objetivo de entrenamiento |

**Regla de pares controlados.** Dentro de un par, todo es idéntico salvo la variable bajo estudio: misma partición, mismo conjunto de entrenamiento, mismo `max_len`, misma loss, mismo optimizador y planificación, mismos módulos y rango de LoRA, mismo régimen de cuantización y mismo dtype de cómputo. Si el hardware de una máquina impide cumplirlo, el modelo se corre en otra máquina antes de romper el par.

**Verificación pendiente.** Confirmar en la model card sobre qué base fue destilado DeepSeek-R1-Distill-Qwen-1.5B. Si no es Qwen2.5-1.5B, el par no aísla solo el objetivo de entrenamiento y el artículo debe decirlo.

---

## 2. Datos

- Fuente: dataset de Kaggle "Emotions" (derivado de CARER, Saravia et al., 2018), ~390K filas. Es el único dataset de toda la Etapa 2.
- Se guarda el SHA-256 del archivo original descargado y la fecha de descarga: `data/DATASET_INFO.md`.
- El CSV no se versiona en git. Se versiona un script de descarga y el hash.
- Mapeo de etiquetas: el del dataset, verificado y fijado en `labels.json`. Ese orden de clases se usa en todos los archivos de predicción.
- Se reportan conteos por clase antes y después de la limpieza (sección 3).
- **Chequeo obligatorio:** revisar si el texto ya viene en minúsculas y sin puntuación. Si es así, la afirmación del artículo de la Etapa 1 sobre conservar puntuación y mayúsculas no aplica y se corrige.

---

## 3. Limpieza y deduplicación (anti-fuga)

El objetivo no es embellecer el texto. Es que ningún texto, ni uno casi igual, aparezca en dos particiones.

Orden fijo:

1. **Normalización solo para comparar:** Unicode NFKC, minúsculas, espacios colapsados. El texto original se conserva y es el que se usa para entrenar.
2. **Duplicados exactos** (texto normalizado, misma etiqueta): se conserva una copia.
3. **Conflicto de etiqueta** (mismo texto normalizado, etiquetas distintas): se eliminan todas las copias y se registran en `data/conflicts.csv`.
4. **Cuasi-duplicados:** detección con MinHash/LSH sobre n-gramas de palabras, o vecinos cercanos sobre TF-IDF. Umbral inicial de similitud 0.90 **[MEDIR]**: se revisan a mano 30 pares cerca del umbral y se ajusta. Los cuasi-duplicados forman clusters, y cada cluster completo va a una sola partición (sección 4). No se eliminan.
5. No se aplica stemming, eliminación de stopwords ni limpieza de puntuación o emoticones (decisión de la Etapa 1).

Salida: `data/clean.parquet` y un reporte `data/cleaning_report.md` con conteos de cada paso.

---

## 4. Partición

- 80 / 10 / 10 (train / val / test), estratificada por etiqueta y **agrupada por cluster** de cuasi-duplicados.
- Semilla de partición: 42. La partición se genera una sola vez y no depende de la semilla de entrenamiento.
- Se guardan `splits/train_ids.csv`, `splits/val_ids.csv`, `splits/test_ids.csv` y el hash de cada uno.
- `scripts/check_splits.py` verifica: intersecciones vacías, ningún texto normalizado repetido entre particiones, ningún cluster repartido, proporciones por clase dentro de ±0.5 puntos porcentuales de las globales. Debe pasar antes de cualquier entrenamiento.
- **Subconjunto de entrenamiento principal S:** subconjunto estratificado del train, de tamaño fijo, igual para todos los modelos y guardado en `splits/train_S_ids.csv`. Se calcula como:
  `S_max = throughput (muestras/s) x horas de presupuesto x 3600 / (épocas x semillas)`
  usando el modelo más caro (Gemma 4 E4B en la 3080). El valor final de S y las horas de presupuesto son **[MEDIR]** y **[POR DEFINIR]**.
- Validación y prueba se usan completas, con una excepción: durante el entrenamiento, el early stopping evalúa sobre una **val de selección** (10,000 muestras estratificadas, IDs en `splits/val_select_ids.csv`) para ahorrar tiempo.
- Opcional: curva de aprendizaje de los modelos baratos (TF-IDF, BiLSTM, DistilBERT) con 10 %, 25 %, 50 % y 100 % del train, para mostrar cuánto se pierde con S.

---

## 5. Entrada de los modelos

- Cada familia usa su tokenizador nativo sobre el texto original (WordPiece en BERT/DistilBERT, BPE en Qwen, Gemma y DeepSeek).
- `max_len` común para todos los modelos: el mayor de los percentiles 99 de longitud en tokens de train entre los tokenizadores, redondeado a múltiplo de 8. Valor de partida 64 **[MEDIR]**. Truncado a la derecha, padding dinámico por lote.
- TF-IDF: `TfidfVectorizer` ajustado solo con train. Configuración inicial: n-gramas (1, 2), `min_df` y `max_features` definidos en la búsqueda (sección 9).
- BiLSTM: vocabulario construido solo con train. Embeddings estáticos pre-entrenados (por defecto GloVe 300d), congelados. Nota: el artículo de la Etapa 1 menciona "embeddings estáticos" y también "embeddings contextualizados" para esta línea base. Se unifica a estáticos y se corrige el artículo.

---

## 6. Formulación de la tarea y cabezas de clasificación

- **Encoders:** cabeza lineal sobre la representación agrupada ([CLS] / pooled), fine-tuning completo.
- **Decoders:** clasificación con cabeza de secuencia (`AutoModelForSequenceClassification`, `task_type=SEQ_CLS` en PEFT). La cabeza se entrena completa y los adaptadores LoRA van en las proyecciones elegidas. Si Gemma 4 no soporta esa clase en `transformers`, se implementa una cabeza manual sobre el estado oculto del último token no-padding, con el mismo diseño.
- No se usa generación de texto ni plantilla de prompt: la entrada es el texto crudo. Esto da probabilidades sobre las 6 clases y permite usar la loss ponderada directamente. Es una desviación de la formulación generativa descrita en la Sección II.B del artículo de la Etapa 1, y el artículo debe reflejarla.
- LoRA, igual para **todos** los decoders: r = 16 (valor de partida, ver sección 9), alpha = 2r, dropout 0.05, módulos objetivo `q_proj, k_proj, v_proj, o_proj`. Se corrige la inconsistencia del artículo, que decía configuraciones idénticas para Qwen y DeepSeek pero la Tabla VI daba más módulos a DeepSeek.
- Cuantización: **QLoRA NF4 para todos los decoders**, para que la escala y la familia no se confundan con el régimen de memoria. Dtype de cómputo: bf16 en la RTX 3080 y en Mac donde aplique. Para el par de escala Qwen 0.5B vs 1.5B, ambos se entrenan con el mismo dtype de cómputo; si la 2060 (sin bf16 nativo) lo impide, las corridas finales de Qwen2.5-0.5B se hacen en la 3080.

---

## 7. Desbalance y función de pérdida

- Entropía cruzada con pesos por clase `w_c = N / (K x n_c)`, con `N`, `K = 6` y `n_c` calculados solo sobre el conjunto de entrenamiento que usa el modelo (S).
- Misma loss ponderada para BiLSTM, encoders y decoders.
- TF-IDF + LogReg: `class_weight` equivalente.
- Las métricas de evaluación nunca se ponderan.
- `DummyClassifier`: no usa pesos. Se reportan dos estrategias, `most_frequent` y `stratified` (esta última con las tres semillas).

---

## 8. Entrenamiento

| Parámetro | Valor |
|---|---|
| Optimizador | AdamW, weight decay 0.01 |
| Planificación | Warmup lineal 6 % de los pasos, decaimiento lineal a 0 |
| Clip de gradiente | Norma 1.0 |
| Tamaño de lote efectivo | 32 para todos (acumulación de gradiente si no cabe) |
| Épocas máximas | 3 sobre S para modelos de transformers [MEDIR según presupuesto]; hasta 10 para BiLSTM |
| Evaluación en val de selección | Cada 1/4 de época |
| Parada temprana | Macro-F1 de la val de selección, paciencia de 3 evaluaciones |
| Checkpoint a reportar | El de mayor Macro-F1 en val de selección |
| Semillas (finales) | 42, 1337, 2026, para **todos** los modelos neuronales |
| Semilla de la búsqueda | 42 |
| Precisión | Declarada por modelo: fp16 en Turing (1650, 2060), bf16 en Ampere (3080) |

- Las semillas controlan inicialización de la cabeza, orden de los datos y dropout. No tocan la partición.
- Se fijan `random`, `numpy` y `torch` (CPU y CUDA), y se activan los flags deterministas de cuDNN donde sea viable. Se declara que las corridas con cuantización y GPU no son reproducibles bit a bit.
- TF-IDF + LogReg es determinista con solver y semilla fijos.

---

## 9. Búsqueda de hiperparámetros

Acotada, sistemática y documentada. Cada intento se guarda en `results/hpo/<familia>/<trial>.json` con su configuración y resultado. La selección usa solo la val de selección, nunca el test.

| Familia | Qué se explora | Presupuesto |
|---|---|---|
| TF-IDF + LogReg | `C` en {0.1, 1, 10}, n-gramas (1,1) y (1,2) | Rejilla completa, 6 intentos |
| BiLSTM | lr en {1e-3, 3e-3}, hidden en {128, 256}, dropout en {0.3, 0.5} | Rejilla completa, 8 intentos, subset reducido |
| Encoders (en BERT-base) | lr en {2e-5, 3e-5, 5e-5} | 3 intentos; DistilBERT hereda el lr elegido |
| Decoders (en Qwen2.5-1.5B) | lr en {1e-4, 2e-4, 3e-4} y r en {8, 16, 32} (con alpha = 2r), uno a la vez alrededor del punto central | 5 intentos |

- Los intentos de la búsqueda de decoders se corren en un 20 % de S, con la semilla 42, y 1 época **[MEDIR]**.
- La configuración elegida en Qwen2.5-1.5B se reutiliza en el resto de los decoders para mantener los pares controlados. Si un modelo diverge (loss que explota o no baja), se ajusta solo el lr, se documenta y se declara como posible confusor.
- Se declara el criterio de parada de la búsqueda (presupuesto agotado) y por qué se paró ahí. Variar r además responde a la limitación que el propio artículo reconoce: que r = 16 fijo a priori podría ser una restricción de capacidad.

---

## 10. Evaluación y estadística

- Un único script, `scripts/evaluate.py`, califica todas las corridas. Los modelos solo producen predicciones.
- Cada corrida guarda `predictions.csv` con las columnas: `id, y_true, y_pred, p_<clase_1>, ..., p_<clase_6>` (en el orden de `labels.json`).
- `evaluate.py` genera `metrics.json` con: Macro-F1 (principal), Weighted-F1, F1 por clase, precisión y recall por clase, matriz de confusión 6x6, y los metadatos de la sección 11.
- El conjunto de prueba se evalúa una sola vez por configuración final. No se usa para elegir hiperparámetros, umbrales, calibración ni early stopping.
- Resultados: media ± desviación estándar de Macro-F1 sobre las tres semillas.
- Comparaciones por pares (eje controlado): bootstrap pareado sobre el test (1,000 remuestreos, mismos IDs para ambos modelos), con intervalo de confianza del 95 % de la diferencia de Macro-F1. Una diferencia se considera informativa solo si el intervalo excluye el cero. No se usa una prueba sobre las semillas porque n = 3 no alcanza.
- Se reportan siempre tres referencias: baseline trivial (Dummy), baseline clásico (TF-IDF + LogReg) y baseline de literatura. Los números de literatura se incluyen solo cuando su origen fue verificado; si no, se recalcula o se marca como propio.

---

## 11. Costo y hardware

Las tres máquinas tienen GPUs distintas, así que tiempos y latencias medidos en cada una no son comparables entre sí.

- **Se reporta por corrida y por máquina** (dato descriptivo): GPU, versión de driver, torch, tiempo total de entrenamiento, muestras/s, VRAM pico de entrenamiento (con `max_memory_allocated`, reiniciando el contador antes de entrenar), parámetros totales y entrenables, tamaño de lote, `max_len` y dtype.
- **Costo para la frontera de Pareto:** parámetros entrenables, VRAM pico de inferencia y **latencia unificada**, todos medidos en la misma máquina de referencia (RTX 3080) con un benchmark común. Cada integrante comparte los adaptadores o checkpoints finales (son pequeños) y se corre el benchmark una sola vez para todos.
- **Protocolo de latencia:** 500 muestras fijas del test (`splits/latency_ids.csv`), lote 1 y lote 32, 50 pasos de calentamiento, `torch.cuda.synchronize()`, se reportan mediana y p95 en ms. La tokenización se mide aparte. Se mide el modelo tal como fue evaluado (base cuantizada más adaptador). Los modelos de CPU (Dummy, TF-IDF) se miden en la CPU de la máquina de referencia.
- **Tiempo de entrenamiento:** se declara por máquina y no se compara entre GPUs distintas.
- **Mac M5 (24 GB):** solo para pruebas auxiliares, plan B de Gemma 4 E4B y la Etapa 3. Sus números se etiquetan aparte y no entran a la Pareto principal. Si E4B termina en la Mac, ese punto se marca como régimen distinto (MLX, cuantización diferente a NF4).

---

## 12. Reproducibilidad y trazabilidad

```
/
  README.md                  instrucciones de ejecución de punta a punta
  EXPERIMENT_CONTRACT.md     este documento
  requirements.txt           versiones fijas (o requirements/<modelo>.txt si hay conflictos)
  labels.json
  configs/                   un YAML por corrida
  data/                      sin el CSV; solo DATASET_INFO.md, reportes y scripts
  splits/                    IDs de cada partición (versionados)
  src/
  scripts/                   prepare_data.py, check_splits.py, evaluate.py
  results/<modelo>/seed_<n>/ predictions.csv, metrics.json, config.yaml, run.log
  results/hpo/
  docs/ai_usage_log.md
```

- Cada corrida guarda el hash del commit, su configuración completa y las versiones de las librerías.
- Los tres integrantes hacen commits de su propio trabajo desde el inicio. La autoría individual se determina por commits.
- No se versionan datasets, checkpoints pesados ni dependencias instaladas (`.venv`). Los adaptadores se comparten por un medio externo.
- `docs/ai_usage_log.md`: nota corta por sesión de trabajo con IA (qué se pidió, qué se verificó a mano). Alimenta la declaración de uso de IA obligatoria.

---

## 13. Fase 0: orden de ejecución

Nadie entrena un modelo final hasta completar los pasos 1 a 6.

1. Repo con estructura, `requirements.txt` base y README inicial.
2. Descarga del dataset, hash y `DATASET_INFO.md`.
3. Limpieza y deduplicación (sección 3), con `cleaning_report.md`.
4. Partición, subconjuntos y `check_splits.py` en verde (sección 4).
5. `evaluate.py` y formato de predicciones, probados con `DummyClassifier`.
6. Prueba de humo del entorno en cada máquina (un paso de entrenamiento de uno de sus modelos). En la 3080 incluye la prueba de viabilidad de Gemma 4 E4B en 10 GB.
7. Medición de throughput (500 pasos por decoder), cálculo de S y de épocas, y medición de percentiles de longitud para `max_len`.
8. Verificaciones de literatura: origen de los números de la Tabla II y base de DeepSeek-R1-Distill.
9. Congelar el contrato como v1.0 (todas las marcas **[MEDIR]** resueltas).

---

## 14. Reglas anti-fuga (resumen)

- Toda estadística aprendida de los datos (vocabularios, pesos de clase, TF-IDF, normalización) se calcula solo con el conjunto de entrenamiento que usa el modelo.
- El test no se usa para early stopping, búsqueda de hiperparámetros, calibración ni umbrales.
- Una sola partición para todos los modelos, nunca regenerada por modelo.
- Los cuasi-duplicados nunca se reparten entre particiones.
- Si se detecta una fuga, se corrige y se reentrenan los modelos afectados antes de reportar.

---

## 15. Decisiones clave y alternativas descartadas

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Partición agrupada por cluster | Partición aleatoria simple | Un split aleatorio reparte cuasi-duplicados entre train y test e infla las métricas |
| Eliminar copias con etiqueta conflictiva | Votación por mayoría | Si el mismo texto tiene dos etiquetas, ninguna es confiable |
| Subconjunto S común para todos | Train completo para modelos baratos y subconjunto para decoders | Usar datos distintos rompe la comparación controlada |
| Cabeza de clasificación en decoders | Generación de la etiqueta como texto | Da probabilidades por clase (calibración, baja confianza en Etapa 3) y permite la loss ponderada sin depender de prompt ni de parseo |
| QLoRA NF4 en todos los decoders | Mezclar LoRA 16 bits y QLoRA según la GPU | El régimen de memoria se confundiría con la escala y la familia |
| Misma configuración de LoRA en todos los decoders | Módulos distintos para DeepSeek | Aislar el efecto del objetivo de entrenamiento exige configuración idéntica |
| Benchmark de latencia en una sola GPU | Latencias medidas en cada máquina | GPUs distintas hacen incomparables los tiempos |
| Bootstrap pareado sobre el test | Prueba t sobre 3 semillas | Con n = 3 la prueba no tiene poder; el bootstrap usa las miles de muestras de test |
| Tres semillas para todos los neuronales | Solo para decoders | Los encoders y el BiLSTM son baratos y así el ± es comparable entre familias |
| Val de selección de 10K | Val completa en cada evaluación | Evaluar 39K muestras cada cuarto de época es caro y no cambia la decisión de parada |
| `max_len` común | `max_len` por modelo | Mantiene la entrada idéntica entre modelos; es un parámetro más del control |

---

## 16. Historial de cambios

| Versión | Fecha | Cambio | Autor |
|---|---|---|---|
| 0.1 | | Borrador inicial | |
