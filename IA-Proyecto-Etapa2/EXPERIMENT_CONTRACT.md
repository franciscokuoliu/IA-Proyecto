# Contrato experimental: Etapa 2 (Track B)

Benchmark de reconocimiento de emociones en texto (CARER, 6 clases).
Integrantes: Francisco Kuo Liu, Jimmy Feng Liu y Danielo Wu Zhong.
Versión: 0.4 (2026-11-06). Fase 0 cerrada y corridas finales terminadas (ocho modelos con tres semillas; TF-IDF y Dummy most-frequent son deterministas y tienen una sola corrida). Siguen abiertas las verificaciones de literatura, la revisión de los 30 pares de la limpieza y el XAI sobre BERT-base. Pasa a 1.0 cuando se resuelvan las marcas **[POR DEFINIR]**.

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
| Qwen2.5-1.5B vs Gemma 4 E2B | Familia (decoders) |
| Qwen2.5-1.5B vs DeepSeek-R1-Distill-Qwen-1.5B | Objetivo de entrenamiento |

**Regla de pares controlados.** Dentro de un par, todo es idéntico salvo la variable bajo estudio: misma partición, mismo conjunto de entrenamiento (S), mismo `max_len`, misma loss, mismo optimizador y planificación, mismos módulos y rango de LoRA, mismo régimen de cuantización y mismo dtype de cómputo. Si el hardware de una máquina impide cumplirlo, el modelo se corre en otra máquina antes de romper el par.

**Decisión (Danielo, 2026-11-06): Gemma 4 E4B queda fuera del plan.** E4B no cabe en la RTX 3080 (10 GB) ni en 4 bits: solo sus tablas de embeddings ocupan unos 6.5 GB y bitsandbytes no las cuantiza. El estimado total en 4 bits es de 8.77 GB (medido con `scripts/inspect_model.py`), sin margen para activaciones ni optimizador. En la Mac M5 (24 GB de memoria unificada) sí corrió un paso de prueba, en bf16 sin cuantizar y con unos 19.6 GB, pero esa corrida es de otro régimen (bf16 sin cuantizar contra NF4 en E2B), así que compararlas mezclaría escala con cuantización. Por eso el par E2B vs E4B no forma parte del estudio y E4B se reporta solo como hallazgo de viabilidad. Además, la Mac lo procesa tan lento que sería un cuello de botella para la fecha de entrega. Las corridas finales de los otros ocho modelos terminaron sin E4B. Si el equipo decidiera retomarlo, vuelve el régimen B de la versión 0.2 y hay que reabrir las secciones 1 y 11.

**Excepción declarada: par de objetivo de entrenamiento.** DeepSeek-R1-Distill-Qwen-1.5B no se destiló sobre Qwen2.5-1.5B, sino sobre **Qwen2.5-Math-1.5B** (fuente primaria: DeepSeek-AI, 2025, arXiv:2501.12948, sección 2.4; verificada el 2026-11-06). Por eso el par Qwen2.5-1.5B contra DeepSeek no aísla solo el objetivo de entrenamiento: también cambia el modelo base, que está especializado en matemáticas. Hay que declararlo en el artículo como confusor.

**Excepción declarada: par de escala de Qwen.** Qwen2.5-0.5B se entrena en la RTX 2060 de Jimmy con fp16, y Qwen2.5-1.5B en la RTX 3080 con bf16. Eso rompe la regla de dtype idéntico dentro del par; se declara como posible confusor (decisión de Danielo, 2026-11-04).
---

## 2. Datos

- Fuente: dataset de Kaggle "Emotions" (derivado de CARER, Saravia et al., 2018). Es el único dataset de toda la Etapa 2.
- El archivo descargado tiene 416,809 filas. Su SHA-256 (`5f2254945d620f7013d8b516c1bde76791af54cc6c94984404c136886285b430`) está fijado en `download_data.py` y se verifica en cada descarga. Después de la limpieza (sección 3) quedan **371,527 filas**.
- El CSV no se versiona en git. Se versionan el script de descarga y las huellas (`data/expected_hashes.json`). Información de la fuente: `data/DATASET_INFO.md`.
- Mapeo de etiquetas: el del dataset, fijado en `labels.json` (orden: sadness, joy, love, anger, fear, surprise). Ese orden se usa en todos los archivos de predicción.
- Distribución después de la limpieza: sadness 31.23 %, joy 34.82 %, love 6.56 %, anger 14.08 %, fear 10.65 %, surprise 2.65 %. La limpieza quitó proporcionalmente más a las clases minoritarias (surprise cerca de 34 %, love cerca de 29 %, fear cerca de 17 % respecto al original), y el artículo debe reportarlo.
- **Chequeo de minúsculas y puntuación (resuelto el 2026-11-04):** en `cleaned_full_data.csv` ningún texto tiene mayúsculas (0.000 %) ni signos de puntuación o apóstrofos (0.000 %). El texto viene en minúsculas y sin puntuación, así que la afirmación del artículo de la Etapa 1 sobre conservar puntuación y mayúsculas no aplica y debe corregirse.
---

## 3. Limpieza y deduplicación (anti-fuga)

El objetivo no es embellecer el texto. Es que ningún texto, ni uno casi igual, aparezca en dos particiones. La implementa `scripts/dataset_cleaner.py` (modo `agrupar`, con `PYTHONHASHSEED=0` fijado por `prepare_data.py` para que el resultado sea idéntico en todas las máquinas).

Orden real:

1. **Normalización solo para comparar.** El texto original se conserva y es el que se usa para entrenar.
2. **Conflicto de etiqueta** (mismo texto normalizado, etiquetas distintas): se eliminan todas las copias. Fueron 22,295 textos distintos, 44,650 filas.
3. **Duplicados exactos** (misma etiqueta): se conserva una copia. Se eliminaron 632 copias.
4. **Cuasi-duplicados:** TF-IDF con similitud coseno y umbral 0.90. Los candidatos salen de términos raros (frecuencia relativa menor a 0.002), lo que redujo el cálculo a 351,161,827 pares candidatos. Se detectaron 5,536 pares, que forman 3,220 grupos de 2 o más textos (7,509 textos en total), 16 de ellos con etiquetas mixtas. Los grupos **no se eliminan**: cada grupo completo va a una sola partición (sección 4).
5. No se aplica stemming, eliminación de stopwords ni limpieza de puntuación o emoticones (decisión de la Etapa 1).

Conteo: 416,809 filas originales, menos 44,650 por etiquetas contradictorias, menos 632 copias exactas, igual a **371,527** filas finales.

Salidas (no versionadas, se regeneran): `salida/cleaned_full_data.csv` (`id, text, label, group`), `salida/pares_cuasi_duplicados.csv` y `salida/reporte_limpieza.json`.

`dataset_cleaner.py` lo escribió Jimmy Feng Liu. **Pendiente:** la revisión manual de 30 pares cerca del umbral 0.90 no está registrada; confirmar con Jimmy si se hizo. **[POR DEFINIR]**
---

## 4. Partición

- Aproximadamente 80 / 10 / 10 (train / val / test), estratificada por etiqueta y **agrupada por grupo** de cuasi-duplicados: `StratifiedGroupKFold` de 10 pliegues, semilla 42. El pliegue 0 es test, el pliegue 1 es validación y los otros 8 son train.
- Tamaños: train 297,220, val 37,154, test 37,153. La partición se genera una sola vez y no depende de la semilla de entrenamiento.
- Los archivos de ids (`splits/*_ids.csv`) **no se versionan**: cada persona los regenera con `prepare_data.py`. Se versionan sus huellas SHA-256 en `data/expected_hashes.json`, y `scripts/check_splits.py` compara contra ellas.
- `check_splits.py` verifica: intersecciones vacías, ningún texto normalizado repetido entre particiones, ningún grupo repartido, proporciones por clase dentro de ±0.5 puntos porcentuales de las globales, y que S sea un subconjunto válido de train. Debe pasar antes de cualquier entrenamiento.
- **Val de selección:** durante el entrenamiento, el early stopping evalúa sobre 10,000 muestras de val, estratificadas, semilla 42 (`splits/val_select_ids.csv`), para ahorrar tiempo.

### Subconjunto de entrenamiento principal S

- **S = 32,000 muestras**, estratificado por emoción, semilla 42, tomado solo de train y guardado en `splits/train_S_ids.csv` (lo genera `make_splits.py`). Equivale a cerca del 11 % de train. Huella SHA-256: `2d518490c80f10c8...` (en `data/expected_hashes.json`).
- Es el mismo para todos los modelos. Al ser un subconjunto de train, hereda la ausencia de grupos cruzados con val y test.
- Fórmula: `S = muestras/s x horas de presupuesto x 3600 / (épocas x semillas)`, con el modelo más lento. Ese modelo es Gemma 4 E2B en la RTX 3080, con la cifra conservadora de 10.17 muestras/s (una prueba de humo posterior dio 9.7) y unas 8 h de presupuesto de entrenamiento: 10.17 x 8 x 3600 / (3 x 3) = 32,544, redondeado a 32,000.
- La búsqueda de hiperparámetros usa el 20 % de S (6,400 muestras).

Mediciones de Fase 0 (muestras/s de entrenamiento, segundos por evaluación sobre la val de selección) y estimado de horas para S = 32,000, 3 semillas y 3 épocas (10 para BiLSTM), evaluación cada 1/4 de época:

| Modelo | Máquina | Muestras/s | s por evaluación | Horas estimadas |
|---|---|---|---|---|
| BiLSTM | Fran | 286.6 | 7.6 | 1.2 |
| DistilBERT | Jimmy (RTX 2060) | 409.0 | 5.7 | 0.3 |
| BERT-base | Jimmy (RTX 2060) | 246.7 | 9.4 | 0.4 |
| Qwen2.5-0.5B | Jimmy (RTX 2060) | 27.7 | 131.6 | 4.2 |
| Qwen2.5-1.5B | Dani (RTX 3080) | 34.8 | 131.3 | 3.6 |
| DeepSeek-R1-Distill-Qwen-1.5B | Dani (RTX 3080) | 33.6 | 141.0 | 3.8 |
| Gemma 4 E2B | Dani (RTX 3080) | 10.2 | 199.5 | 9.9 |

Son estimados: las corridas reales pueden variar alrededor de 15 %. La evaluación periódica pesa cerca de un tercio del tiempo en los decoders.

- Opcional: curva de aprendizaje de los modelos baratos (TF-IDF, BiLSTM, DistilBERT) con 10 %, 25 %, 50 % y 100 % del train, para mostrar cuánto se pierde con S.
---

## 5. Entrada de los modelos

- Cada familia usa su tokenizador nativo sobre el texto original (WordPiece en BERT/DistilBERT, BPE en Qwen, Gemma y DeepSeek).
- `max_len` común para todos los modelos: el mayor de los percentiles 99 de longitud en tokens entre los tokenizadores, redondeado a múltiplo de 8. Medido con `scripts/token_lengths.py` sobre 20,000 textos de S: el p99 máximo es 57 tokens (BERT), y con 64 el peor tokenizador recorta el 0.15 % de los textos (Qwen, Gemma y DeepSeek recortan 0.07 %; el BiLSTM, que cuenta palabras, 0.02 %). **Valor fijado: 64.** Truncado a la derecha, padding dinámico por lote.
- TF-IDF: `TfidfVectorizer` ajustado solo con train. Configuración inicial: n-gramas (1, 2), `min_df` y `max_features` definidos en la búsqueda (sección 9).
- BiLSTM: vocabulario construido solo con train. Embeddings estáticos pre-entrenados (GloVe de 300 dimensiones, que se pasa con `--glove`; sin esa opción el arnés usa vectores aleatorios congelados), congelados. Nota: el artículo de la Etapa 1 menciona "embeddings estáticos" y también "embeddings contextualizados" para esta línea base. Se unifica a estáticos y se corrige el artículo.

---

## 6. Formulación de la tarea y cabezas de clasificación

- **Encoders:** cabeza lineal sobre la representación agrupada ([CLS] / pooled), fine-tuning completo.
- **Decoders:** clasificación con cabeza de secuencia (`AutoModelForSequenceClassification`, `task_type=SEQ_CLS` en PEFT). La cabeza se entrena completa y los adaptadores LoRA van en las proyecciones elegidas. Gemma 4 no tiene esa clase en `transformers`, así que usa una cabeza manual (lineal) sobre el estado oculto del último token no-padding, con el mismo diseño. Gemma rellena por la izquierda, y el índice del último token se calcula con la máscara (no con la suma de la máscara); una versión anterior con ese error dio Macro-F1 0.354 contra 0.975 de Qwen y se corrigió (commit `a4e0cef`). En Gemma, LoRA se aplica solo a las capas de lenguaje, no a la torre de visión.
- No se usa generación de texto ni plantilla de prompt: la entrada es el texto crudo. Esto da probabilidades sobre las 6 clases y permite usar la loss ponderada directamente. Es una desviación de la formulación generativa descrita en la Sección II.B del artículo de la Etapa 1, y el artículo debe reflejarla.
- LoRA, igual para **todos** los decoders: r = 16 (valor de partida, ver sección 9), alpha = 2r, dropout 0.05, módulos objetivo `q_proj, k_proj, v_proj, o_proj`. Se corrige la inconsistencia del artículo, que decía configuraciones idénticas para Qwen y DeepSeek pero la Tabla VI daba más módulos a DeepSeek.
- Cuantización: **QLoRA NF4 para todos los decoders**, para que la escala y la familia no se confundan con el régimen de memoria. Dtype de cómputo: bf16 en la RTX 3080, y fp16 con escalador de gradiente en las GPUs Turing (1650, 2060), con los parámetros entrenables en fp32. Para el par de escala Qwen 0.5B vs 1.5B ambos deben usar el mismo dtype de cómputo. Qwen2.5-0.5B se entrena en la 2060 (fp16) y Qwen2.5-1.5B en la 3080 (bf16), lo que rompe la regla de pares. **Decisión (Danielo, 2026-11-04):** las corridas finales de Qwen2.5-0.5B se hacen en la 2060 con fp16 y el dtype distinto se declara como posible confusor (sección 1).

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
| Épocas máximas | 3 sobre S para modelos de transformers; hasta 10 para BiLSTM |
| Evaluación en val de selección | Cada 1/4 de época |
| Parada temprana | Macro-F1 de la val de selección, paciencia de 3 evaluaciones |
| Checkpoint a reportar | El de mayor Macro-F1 en val de selección |
| Semillas (finales) | 42, 1337, 2026, para **todos** los modelos neuronales |
| Semilla de la búsqueda | 42 |
| Precisión | Declarada por modelo: fp16 en Turing (1650, 2060), bf16 en Ampere (3080). La Mac M5 (bf16 en MPS) no entra al experimento principal |

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

- Los intentos de la búsqueda de decoders se corren en un 20 % de S (6,400 muestras), con la semilla 42 y 1 época (200 pasos con lote efectivo 32).

**Resultado de decoders (Qwen2.5-1.5B, RTX 3080, 2026-11-04), Macro-F1 en la val de selección:**

| Intento | Macro-F1 | Observación |
|---|---|---|
| lr 2e-4, r 16 | **0.874** | Elegido |
| lr 3e-4, r 16 | 0.798 | Colapsó después del paso 100 (predice una sola clase); el arnés conservó el mejor checkpoint |
| lr 2e-4, r 8 | 0.715 | Aprende más lento |
| lr 1e-4, r 16 | 0.453 | Aprende más lento |
| lr 2e-4, r 32 | 0.245 | Colapsó desde el paso 100 |

Configuración elegida: **lr 2e-4, r 16** (alpha 32), que coincide con los valores por defecto del catálogo. Dos advertencias para el artículo: (1) el punto elegido está cerca del borde de estabilidad, porque lr 3e-4 y r 32 colapsaron; en las corridas finales (unos 3,000 pasos, calentamiento de unos 180) el arranque es más suave, y los logs de las doce corridas de decoders no muestran colapso (la mayor caída del Macro-F1 de val de selección respecto de su mejor valor fue de 0.0024); (2) con una sola época se favorecen los lr altos. Resultados en `results/hpo/decoder/`.

**Las otras familias (Macro-F1 en la val de selección, una semilla).** Encoders (BERT-base, S completo, 3 épocas): lr 5e-5 dio 0.9921 (elegido), 3e-5 dio 0.9905 y 2e-5 dio 0.9901; DistilBERT hereda lr 5e-5. TF-IDF (S completo): C 10 con n-gramas (1,2) dio 0.9480 (elegido) y el peor intento, C 0.1 con (1,2), dio 0.8525. BiLSTM (20 % de S, 10 épocas): lr 3e-3, hidden 128 y dropout 0.3 dio 0.9475 (elegido) y el peor intento dio 0.9059. Detalle en `results/hpo/` y en `results/report/results_table.md`.

**Advertencia de borde de rejilla.** Los ganadores de encoders, TF-IDF y BiLSTM quedaron en el extremo de su rejilla (lr 5e-5; C 10 y n-gramas (1,2); lr 3e-3, hidden 128 y dropout 0.3), así que no se puede afirmar que sean el óptimo real. Las rejillas se fijaron en la Fase 0 y no se ampliaron por falta de tiempo antes de la entrega; se declara como limitación.
- La configuración elegida en Qwen2.5-1.5B se reutiliza en el resto de los decoders para mantener los pares controlados. Si un modelo diverge (loss que explota o no baja), se ajusta solo el lr, se documenta y se declara como posible confusor. No hizo falta: los cuatro decoders corrieron con lr 2e-4 y r 16.
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

### Análisis posteriores a las corridas

Estos pasos leen `results/` y no cambian ningún resultado.

- **Informe agregado.** `scripts/make_report.py` genera `results/report/` (tablas y figuras). Es la fuente de las cifras del artículo.
- **Pares.** `scripts/compare_pairs.py` aplica el bootstrap pareado de arriba a cada semilla del par (semilla k de A contra la k de B, 1,000 remuestreos, semilla 42) y guarda `results/pairs/<a>__vs__<b>.json`. Una diferencia se considera sistemática solo si el IC excluye el cero con el mismo signo en las tres semillas.
- **Errores.** `scripts/error_analysis.py` mide errores por clase, confusiones, confianza, longitud y errores compartidos o exclusivos entre modelos. Decir que un error es "ruido de etiqueta" es una hipótesis hasta revisar a mano una muestra (mínimo 30 textos que fallan todos los modelos).
- **XAI (`scripts/xai.py`).** Se explica BERT-base (semilla 42) y, si alcanza el tiempo, DistilBERT. No se explican los decoders (decisión de Danielo, 2026-11-06): son cuantizados y con atención causal, y explicarlos costaría mucho más; se declara como limitación. El método principal es Integrated Gradients (Sundararajan et al., 2017) sobre los embeddings de entrada, con baseline [PAD] y 64 puntos de Gauss-Legendre, contrastado con la atención de [CLS] y el attention rollout (Abnar y Zuidema, 2020), que no se toman como explicación por sí solas (Jain y Wallace, 2019). Antes de interpretar se comprueba que IG cumple la completitud (error relativo medio menor a 5 %) y que las atribuciones cambian con una cabeza o un modelo aleatorio (Adebayo et al., 2018). La fidelidad se mide quitando, o dejando solo, las k palabras más atribuidas frente a k palabras al azar (DeYoung et al., 2020). `scripts/test_xai_toy.py` valida el código con un modelo diminuto de pistas conocidas; no valida a BERT-base.

---

## 11. Costo y hardware

Las tres máquinas tienen GPUs distintas, así que tiempos y latencias medidos en cada una no son comparables entre sí.

- **Se reporta por corrida y por máquina** (dato descriptivo): GPU, versión de driver, torch, tiempo total de entrenamiento, muestras/s, VRAM pico de entrenamiento (con `max_memory_allocated`, reiniciando el contador antes de entrenar), parámetros totales y entrenables, tamaño de lote, `max_len` y dtype. Todo queda en `run_metadata.json`.
- **Costo (decisión de Danielo, 2026-11-06, en lugar del plan de la versión 0.3):** por modelo se reportan los parámetros totales y entrenables, la VRAM pico de entrenamiento y el tiempo de entrenamiento por corrida, cada uno con la GPU donde se midió. No se comparan tiempos ni VRAM entre GPUs distintas. Los parámetros de los decoders cuantizados salen del conteo publicado del modelo base, porque el arnés cuenta cada par de pesos de 4 bits como un elemento (`results/report/summary.json`, campo `n_params_source`).
- **Recortado:** el benchmark de latencia unificada (500 muestras del test en la RTX 3080) y la VRAM de inferencia no se hicieron, por falta de tiempo antes de la entrega. El artículo debe decir que no hay medición de latencia.
- **Tiempo de entrenamiento:** se declara por máquina y no se compara entre GPUs distintas.
- **Un solo régimen (confirmado):** GPUs NVIDIA, decoders en QLoRA NF4, con bf16 en Ampere y fp16 en Turing. TF-IDF y BiLSTM corrieron en CPU.
- **Mac M5 (fuera del experimento principal):** el arnés soporta MPS en bf16 sin cuantizar. Se usó para la prueba de viabilidad de Gemma 4 E4B (sección 1). Las corridas en MPS se etiquetan `device: mps`, no entran a la Pareto, y su memoria (`memory_kind: mps_driver_allocated`) es un orden de magnitud, no comparable con `max_memory_allocated`.
- **Costo de la evaluación periódica:** evaluar la val de selección cada 1/4 de época tarda de 130 a 215 s en los decoders y pesa cerca de un tercio de cada corrida. Es parte del tiempo declarado.
---

## 12. Reproducibilidad y trazabilidad

```
/
  README.md                  instrucciones de ejecución de punta a punta
  setup_env.ps1
  IA-Proyecto-Etapa2/
    EXPERIMENT_CONTRACT.md   este documento
    requirements.txt
    labels.json
    data/                    DATASET_INFO.md y expected_hashes.json (huellas); sin el CSV
    splits/                  ids de cada partición y S (no versionados, se regeneran)
    src/harness/             arnés de entrenamiento
    scripts/                 prepare_data.py, train.py, evaluate.py, check_splits.py, token_lengths.py,
                             make_report.py, compare_pairs.py, error_analysis.py, share_predictions.py,
                             xai.py, test_xai_toy.py (y de apoyo: download_data.py, dataset_cleaner.py,
                             make_splits.py, splits_common.py, run_dummy.py, inspect_model.py)
    results/<modelo>/seed_<n>/   metrics.json, config.yaml, run.log, run_metadata.json,
                                 predictions.csv (no versionado) y model/ (adaptadores, con --save-model;
                                 solo model_info.json se versiona)
    results/hpo/<familia>/   un resumen y una carpeta por intento
    results/report/          tablas y figuras agregadas (make_report.py)
    results/pairs/           bootstrap pareado de los pares (compare_pairs.py)
    results/errors/          análisis de errores (error_analysis.py)
    results/xai/<modelo>/seed_<n>/   explicabilidad (xai.py)
    docs/ai_usage_log.md
```

- Cada corrida guarda el hash del commit, su configuración completa y las versiones de las librerías. Excepción: las corridas finales de Jimmy (DistilBERT, BERT-base y Qwen2.5-0.5B) quedaron con `commit: null` en `run_metadata.json`. El código de entrenamiento (`src/` y `scripts/train.py`) no cambió en el repositorio desde `87beccc`, el commit que registran las corridas de Danielo; falta confirmar con Jimmy qué versión tenía en su copia.
- Los tres integrantes hacen commits de su propio trabajo desde el inicio. La autoría individual se determina por commits.
- No se versionan datasets, checkpoints pesados ni dependencias instaladas (`.venv`). Los adaptadores se comparten por un medio externo.
- `docs/ai_usage_log.md`: nota corta por sesión de trabajo con IA (qué se pidió, qué se verificó a mano). Alimenta la declaración de uso de IA obligatoria.

---

## 13. Fase 0: orden de ejecución

Nadie entrena un modelo final hasta completar los pasos 1 a 6.

1. Hecho. Repo con estructura, `requirements.txt` y README.
2. Hecho. Descarga del dataset con SHA-256 verificado.
3. Hecho. Limpieza y deduplicación (sección 3).
4. Hecho. Partición, S y `check_splits.py` en verde (sección 4).
5. Hecho. `evaluate.py` y formato de predicciones, probados con `DummyClassifier`.
6. Hecho. Prueba de humo del entorno en las tres máquinas. En la 3080, la prueba de viabilidad de Gemma 4 E4B dio que no cabe (sección 1).
7. Hecho. Throughput (sección 4), S = 32,000 y `max_len` = 64. Queda por aclarar la velocidad de Gemma 4 E2B, medida de dos formas (10.2 y 15.1 muestras/s); se usó la conservadora.
8. En curso (Francisco). Verificaciones de literatura: la base de DeepSeek-R1-Distill resultó ser Qwen2.5-Math-1.5B (fuente primaria verificada el 2026-11-06: DeepSeek-AI, 2025, sección 2.4). Falta verificar el origen de las cifras de la tabla de baselines publicados sobre CARER del artículo de la Etapa 1: cita a Saravia 2018 y Zhang 2024, pero otra tabla del mismo artículo da 0.647 de Macro-F1 para el BiLSTM de Saravia (la primera dice cerca de 84 %), y Zhang 2024 se resume como "Standard Sentiment Corpora", no CARER. No usar esas cifras hasta verificarlas.
9. Pendiente. Congelar el contrato como v1.0: requiere cerrar el paso 8 y la revisión de los 30 pares de la limpieza (Jimmy).

Correcciones pendientes al artículo de la Etapa 1: el dataset (20,000 instancias en el texto, frente a 371,527 filas limpias); "Gemma-2" en el texto, cuando el modelo es Gemma 4; Gemma 4 E4B, que sale del plan; el texto en minúsculas y sin puntuación; los módulos de LoRA de DeepSeek (el artículo los extiende a las capas feed-forward, el contrato usa los mismos que los demás); y la base de DeepSeek.

Estado de la ejecución al 2026-11-06: la búsqueda de hiperparámetros de las cuatro familias y las corridas finales de los ocho modelos están terminadas (secciones 9 y 10; resultados en `results/`). Pendientes: repetir el Dummy con S (Francisco), el XAI sobre BERT-base (Jimmy), los pares que necesitan predicciones de dos máquinas, la revisión manual de los errores compartidos, las verificaciones de literatura y la revisión de los 30 pares de la limpieza.
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
| Sin latencia unificada: el costo es parámetros, VRAM y tiempo de entrenamiento, cada uno con su GPU | Benchmark de latencia en una sola GPU, o latencias medidas en cada máquina | No cupo antes de la entrega, y medir en cada máquina daría cifras incomparables |
| Bootstrap pareado sobre el test | Prueba t sobre 3 semillas | Con n = 3 la prueba no tiene poder; el bootstrap usa las miles de muestras de test |
| Tres semillas para todos los neuronales | Solo para decoders | Los encoders y el BiLSTM son baratos y así el ± es comparable entre familias |
| Val de selección de 10K | Val completa en cada evaluación | Evaluar 39K muestras cada cuarto de época es caro y no cambia la decisión de parada |
| `max_len` común | `max_len` por modelo | Mantiene la entrada idéntica entre modelos; es un parámetro más del control |
| Gemma 4 E4B fuera del plan (decidido por Danielo, 2026-11-06) | E4B en la Mac en bf16 (régimen B) con E2B de puente | Mezcla escala con cuantización, usa otra máquina y no cabe en la RTX 3080 |
| S = 32,000 | Más muestras | Es lo que cabe en unas 8 h de entrenamiento del modelo más lento (E2B) con 3 semillas y 3 épocas, para cerrar el domingo 2026-11-08 |
| `max_len` = 64 | 128 | Recorta 0.15 % en el peor tokenizador, y menos longitud baja el costo |
| HPO de decoders con 20 % de S y 1 época | HPO completo | Cabe en el tiempo; sesgo hacia lr altos declarado en la sección 9 |
| XAI sobre BERT-base (y DistilBERT si alcanza el tiempo) | XAI sobre un decoder cuantizado | BERT-base es el mejor en Macro-F1 y se puede explicar en una RTX 2060; los decoders tienen atención causal y pesos NF4, y explicarlos costaría mucho más. Se declara como limitación |
| Integrated Gradients con verificaciones (completitud, cordura, fidelidad) | Mostrar solo la atención | La atención por sí sola no es una explicación (Jain y Wallace, 2019); IG tiene propiedades que se pueden comprobar |
| Baseline trivial (Dummy) entrenado con S | Dummy entrenado con el train completo | Debe ver la misma información que el resto. Los primeros Dummy usaron el train completo y se repiten con S |

---

## 16. Historial de cambios

| Versión | Fecha | Cambio | Autor |
|---|---|---|---|
| 0.1 | | Borrador inicial | |
| 0.2 | 2026-11-02 | Gemma 4 E4B no cabe en 10 GB: se agrega el régimen B (Mac M5, bf16 sin cuantizar) con E2B como puente, se aclara que la Pareto es solo del régimen A y se registra el costo de memoria en MPS | Dani |
| 0.3 | 2026-11-04 | Se propone retirar el régimen B y el par E2B vs E4B (E4B quedaría como hallazgo de viabilidad, pendiente de confirmación del equipo); se corrigen filas (371,527) y método real de limpieza; se fijan S = 32,000, `max_len` = 64 y las mediciones de Fase 0; se registra el resultado del HPO de decoders (lr 2e-4, r 16); se marca el estado de la Fase 0 y los pendientes | Dani |
| 0.4 | 2026-11-06 | Se cierra Gemma 4 E4B (fuera); se registran el HPO de encoders, TF-IDF y BiLSTM con la advertencia de borde de rejilla y la ausencia de colapsos en las corridas finales; se recorta la latencia unificada; se agregan los análisis posteriores (informe, pares, errores, XAI); se corrige la nota de GloVe del BiLSTM; se verifica en la fuente primaria que DeepSeek-R1-Distill-Qwen-1.5B parte de Qwen2.5-Math-1.5B; se actualizan la estructura del repositorio y el estado de la ejecución | Dani |
