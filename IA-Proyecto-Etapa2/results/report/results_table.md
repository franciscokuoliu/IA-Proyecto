# Resultados agregados de la Etapa 2

> Archivo generado por `scripts/make_report.py`. No editar a mano: se regenera con `python scripts/make_report.py`.

## 1. Tabla principal (test, 37,153 textos)

| Modelo | Familia | Macro-F1 test | Exactitud test | Params totales | Params entrenables | VRAM pico (GB) | Tiempo/corrida (min) | Hardware (dtype) | Corridas |
|---|---|---|---|---|---|---|---|---|---|
| BERT-base | Encoder (full fine-tuning) | 0.9895 ± 0.0005 | 0.9924 ± 0.0002 | 109.5M | 109.5M | 2.2 | 16.6 | RTX 2060 (fp16) | 3 |
| DistilBERT | Encoder (full fine-tuning) | 0.9891 ± 0.0003 | 0.9919 ± 0.0002 | 67.0M | 67.0M | 1.4 | 9.6 | RTX 2060 (fp16) | 3 |
| Qwen2.5-1.5B | Decoder (QLoRA, 4-bit) | 0.9887 ± 0.0012 | 0.9916 ± 0.0010 | 1.54B † | 4.4M | 2.7 | 72.7 (59.0–81.1) | RTX 3080 (bf16) | 3 |
| Qwen2.5-0.5B | Decoder (QLoRA, 4-bit) | 0.9877 ± 0.0010 | 0.9905 ± 0.0009 | 0.49B † | 2.2M | 1.2 | 85.5 | RTX 2060 (fp16) | 3 |
| DeepSeek-R1-Distill-1.5B | Decoder (QLoRA, 4-bit) | 0.9874 ± 0.0008 | 0.9901 ± 0.0008 | 1.54B † | 4.4M | 2.7 | 79.0 | RTX 3080 (bf16) | 3 |
| BiLSTM + GloVe | BiLSTM + GloVe | 0.9786 ± 0.0028 | 0.9814 ± 0.0033 | 7.0M | 442K | - | 9.6 (9.1–10.2) | CPU (fp32) | 3 |
| Gemma 4 E2B | Decoder (QLoRA, 4-bit) | 0.9544 ± 0.0032 | 0.9642 ± 0.0010 | 5.1B † | 5.4M | 9.2 | 149.4 | RTX 3080 (bf16) | 3 |
| TF-IDF + LogReg | TF-IDF + LogReg | 0.9493 (1 corrida) | 0.9564 (1 corrida) | 344K | 344K | - | 0.2 | CPU | 1 |
| Dummy (stratified) | Dummy | 0.1662 ± 0.0026 | 0.2544 ± 0.0024 | - | - | - | <0.1 | CPU | 3 |
| Dummy (most frequent) | Dummy | 0.0861 (1 corrida) | 0.3482 (1 corrida) | - | - | - | <0.1 | CPU | 1 |

- Macro-F1 y exactitud: media ± desviación estándar muestral (ddof=1) sobre las semillas [42, 1337, 2026]. La exactitud se calcula desde la matriz de confusión (no es una métrica oficial del arnés).
- TF-IDF y Dummy most-frequent son deterministas: una sola corrida, sin desviación estándar.
- † Los decoders se entrenan con pesos de 4 bits (QLoRA/NF4) y el arnés cuenta cada par de pesos como un solo elemento, así que su `n_params` subestima (Qwen2.5-1.5B sale 0.89B y tiene 1.54B). La tabla usa el conteo publicado del modelo base; ver `summary.json` (`n_params_source`) y marcar las fuentes por verificar antes de citarlas.
  - Qwen2.5-0.5B: arnés 317,292,928; reporte 0.49B: Qwen2.5 (qwenlm.github.io/blog/qwen2.5-llm): 0.49B
  - Qwen2.5-1.5B: arnés 892,993,024; reporte 1.54B: Qwen2.5 (qwenlm.github.io/blog/qwen2.5-llm): 1.54B
  - DeepSeek-R1-Distill-1.5B: arnés 892,993,024; reporte 1.54B: inferido: mismo n_params del arnes que Qwen2.5-1.5B (la ficha publica 1.78B contando un lm_head sin uso) [verificar]
  - Gemma 4 E2B: arnés 3,942,566,438; reporte 5.1B: ficha de Gemma 4 citada por apxml.com: 2.3B efectivos, 5.1B con Per-Layer Embeddings [verificar fuente primaria]
- Tiempos y VRAM: no son comparables entre GPU distintas (contrato, sección 11). VRAM pico es de entrenamiento (`cuda_max_allocated`), no de inferencia. Los modelos en CPU no tienen VRAM.
- Entre paréntesis, rango de tiempos cuando una corrida difiere más de 10% de la media (una corrida que paró antes por paciencia dura menos).

- Dummy (stratified) y Dummy (most frequent) se entrenaron con n_train=297,220 (train completo), no con el subconjunto S (32,000); repetirlos con S para que la comparación sea justa.

## 2. Resultados por semilla

| Modelo | Semilla | Macro-F1 test | Mejor Macro-F1 val_select | Pasos | Paró por paciencia | Tiempo (min) | Commit |
|---|---|---|---|---|---|---|---|
| BERT-base | 42 | 0.9898 | 0.9921 | 3000 | no | 16.7 | no registrado |
| BERT-base | 1337 | 0.9897 | 0.9911 | 3000 | no | 16.5 | no registrado |
| BERT-base | 2026 | 0.9889 | 0.9910 | 3000 | no | 16.6 | no registrado |
| DistilBERT | 42 | 0.9889 | 0.9907 | 3000 | no | 9.6 | no registrado |
| DistilBERT | 1337 | 0.9895 | 0.9900 | 3000 | no | 9.5 | no registrado |
| DistilBERT | 2026 | 0.9889 | 0.9899 | 3000 | no | 9.6 | no registrado |
| Qwen2.5-1.5B | 42 | 0.9884 | 0.9897 | 3000 | no | 81.1 | 87beccc |
| Qwen2.5-1.5B | 1337 | 0.9900 | 0.9928 | 3000 | no | 77.8 | 87beccc |
| Qwen2.5-1.5B | 2026 | 0.9877 | 0.9904 | 2250 | sí | 59.0 | 87beccc |
| Qwen2.5-0.5B | 42 | 0.9886 | 0.9907 | 3000 | no | 84.8 | no registrado |
| Qwen2.5-0.5B | 1337 | 0.9867 | 0.9903 | 3000 | no | 86.1 | no registrado |
| Qwen2.5-0.5B | 2026 | 0.9878 | 0.9897 | 3000 | no | 85.7 | no registrado |
| DeepSeek-R1-Distill-1.5B | 42 | 0.9877 | 0.9897 | 3000 | no | 79.8 | 87beccc |
| DeepSeek-R1-Distill-1.5B | 1337 | 0.9865 | 0.9884 | 3000 | no | 78.7 | 87beccc |
| DeepSeek-R1-Distill-1.5B | 2026 | 0.9880 | 0.9895 | 3000 | no | 78.4 | 87beccc |
| BiLSTM + GloVe | 42 | 0.9812 | 0.9826 | 4000 | sí | 9.5 | c400f84 |
| BiLSTM + GloVe | 1337 | 0.9792 | 0.9795 | 4250 | sí | 10.2 | c400f84 |
| BiLSTM + GloVe | 2026 | 0.9756 | 0.9793 | 4000 | sí | 9.1 | c400f84 |
| Gemma 4 E2B | 42 | 0.9507 | 0.9530 | 3000 | no | 152.0 | 87beccc |
| Gemma 4 E2B | 1337 | 0.9567 | 0.9588 | 3000 | no | 153.7 | 87beccc |
| Gemma 4 E2B | 2026 | 0.9558 | 0.9557 | 3000 | no | 142.5 | 87beccc |
| TF-IDF + LogReg | 42 | 0.9493 | - | - | - | 0.2 | c400f84 |
| Dummy (stratified) | 42 | 0.1635 | - | - | - | 0.0 | c400f84 |
| Dummy (stratified) | 1337 | 0.1663 | - | - | - | 0.0 | c400f84 |
| Dummy (stratified) | 2026 | 0.1687 | - | - | - | 0.0 | c400f84 |
| Dummy (most frequent) | 42 | 0.0861 | - | - | - | 0.0 | c400f84 |

- `Paró por paciencia`: el entrenamiento terminó antes del máximo de pasos porque val_select no mejoró en 3 evaluaciones seguidas; se restaura el mejor checkpoint.
- `Commit: no registrado`: la corrida no guardó el hash del commit en sus metadatos (declararlo en la sección de reproducibilidad).

## 3. F1 por clase (media de las semillas, test)

| Modelo | sadness | joy | love | anger | fear | surprise |
|---|---|---|---|---|---|---|
| BERT-base | 0.9942 | 0.9956 | 0.9954 | 0.9871 | 0.9850 | 0.9795 |
| DistilBERT | 0.9941 | 0.9949 | 0.9948 | 0.9863 | 0.9844 | 0.9800 |
| Qwen2.5-1.5B | 0.9946 | 0.9944 | 0.9962 | 0.9850 | 0.9821 | 0.9798 |
| Qwen2.5-0.5B | 0.9932 | 0.9933 | 0.9958 | 0.9828 | 0.9829 | 0.9780 |
| DeepSeek-R1-Distill-1.5B | 0.9931 | 0.9932 | 0.9956 | 0.9835 | 0.9797 | 0.9791 |
| BiLSTM + GloVe | 0.9834 | 0.9852 | 0.9864 | 0.9728 | 0.9751 | 0.9691 |
| Gemma 4 E2B | 0.9753 | 0.9706 | 0.9687 | 0.9466 | 0.9430 | 0.9222 |
| TF-IDF + LogReg | 0.9607 | 0.9641 | 0.9449 | 0.9460 | 0.9454 | 0.9346 |
| Dummy (stratified) | 0.3123 | 0.3469 | 0.0673 | 0.1394 | 0.1063 | 0.0248 |
| Dummy (most frequent) | 0.0000 | 0.5166 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- Soporte en test: sadness 11,604, joy 12,937, love 2,438, anger 5,233, fear 3,958, surprise 983.

## 4. Validación vs test

| Modelo | Mejor Macro-F1 val_select (media) | Macro-F1 test (media) | Diferencia |
|---|---|---|---|
| BERT-base | 0.9914 | 0.9895 | +0.0019 |
| DistilBERT | 0.9902 | 0.9891 | +0.0011 |
| Qwen2.5-1.5B | 0.9910 | 0.9887 | +0.0022 |
| Qwen2.5-0.5B | 0.9903 | 0.9877 | +0.0026 |
| DeepSeek-R1-Distill-1.5B | 0.9892 | 0.9874 | +0.0018 |
| BiLSTM + GloVe | 0.9804 | 0.9786 | +0.0018 |
| Gemma 4 E2B | 0.9559 | 0.9544 | +0.0015 |

- val_select es el 10% de train que decide el mejor checkpoint; por eso es algo optimista frente a test (sesgo de selección). La diferencia debe ser pequeña y de signo positivo; una diferencia grande sería señal de problema.

## 5. Búsqueda de hiperparámetros (val_select, una semilla)

### bilstm

Configuración: modelo `bilstm`, n_train=6,399, 10 épocas, fracción 0.2 de S.

| Prueba | Macro-F1 val_select | Nota |
|---|---|---|
| **lr0.003_h128_d0.3** | 0.9475 | ganador |
| lr0.003_h256_d0.5 | 0.9461 |  |
| lr0.003_h256_d0.3 | 0.9408 |  |
| lr0.001_h256_d0.3 | 0.9203 |  |
| lr0.001_h256_d0.5 | 0.9140 |  |
| lr0.001_h128_d0.3 | 0.9119 |  |
| lr0.003_h128_d0.5 | 0.9073 |  |
| lr0.001_h128_d0.5 | 0.9059 |  |

- El ganador está en el borde de la rejilla: dropout=0.3 (mínimo explorado), hidden=128 (mínimo explorado), learning_rate=0.003 (máximo explorado). No se puede afirmar que sea el óptimo real; declarar como limitación.

### decoder

Configuración: modelo `qwen2.5-1.5b`, n_train=6,399, 1 época, fracción 0.2 de S.

| Prueba | Macro-F1 val_select | Nota |
|---|---|---|
| **lr2e-4_r16** | 0.8739 | ganador |
| lr3e-4_r16 | 0.7983 |  |
| lr2e-4_r8 | 0.7151 |  |
| lr1e-4_r16 | 0.4534 | no convergió |
| lr2e-4_r32 | 0.2451 | no convergió |

### encoder

Configuración: modelo `bert-base`, n_train=32,000, 3 épocas.

| Prueba | Macro-F1 val_select | Nota |
|---|---|---|
| **lr5e-05** | 0.9921 | ganador |
| lr3e-05 | 0.9905 |  |
| lr2e-05 | 0.9901 |  |

- El ganador está en el borde de la rejilla: learning_rate=5e-05 (máximo explorado). No se puede afirmar que sea el óptimo real; declarar como limitación.

### tfidf

Configuración: modelo `tfidf-logreg`, n_train=32,000.

| Prueba | Macro-F1 val_select | Nota |
|---|---|---|
| **C10.0_ngram12** | 0.9480 | ganador |
| C10.0_ngram11 | 0.9420 |  |
| C1.0_ngram12 | 0.9346 |  |
| C1.0_ngram11 | 0.9320 |  |
| C0.1_ngram11 | 0.8938 |  |
| C0.1_ngram12 | 0.8525 |  |

- El ganador está en el borde de la rejilla: C=10.0 (máximo explorado), ngram=[1, 2] (máximo explorado). No se puede afirmar que sea el óptimo real; declarar como limitación.

## 6. Comparaciones pareadas (bootstrap sobre el test)

| Par (A − B) | Semilla | Macro-F1 A | Macro-F1 B | Δ (A − B) | IC 95% | ¿Informativa? |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B − Gemma 4 E2B | 42 | 0.9884 | 0.9507 | +0.0377 | [+0.0346, +0.0413] | sí |
| Qwen2.5-1.5B − Gemma 4 E2B | 1337 | 0.9900 | 0.9567 | +0.0333 | [+0.0305, +0.0360] | sí |
| Qwen2.5-1.5B − Gemma 4 E2B | 2026 | 0.9877 | 0.9558 | +0.0319 | [+0.0293, +0.0349] | sí |
| Qwen2.5-1.5B − DeepSeek-R1-Distill-1.5B | 42 | 0.9884 | 0.9877 | +0.0008 | [-0.0004, +0.0019] | no (incluye 0) |
| Qwen2.5-1.5B − DeepSeek-R1-Distill-1.5B | 1337 | 0.9900 | 0.9865 | +0.0035 | [+0.0024, +0.0047] | sí |
| Qwen2.5-1.5B − DeepSeek-R1-Distill-1.5B | 2026 | 0.9877 | 0.9880 | -0.0003 | [-0.0020, +0.0014] | no (incluye 0) |

Resumen por par:

| Par (A − B) | Δ medio | Semillas con IC que excluye 0 | Mismo signo en todas |
|---|---|---|---|
| Qwen2.5-1.5B − Gemma 4 E2B | +0.0343 | 3/3 | sí |
| Qwen2.5-1.5B − DeepSeek-R1-Distill-1.5B | +0.0013 | 1/3 | no |

- Bootstrap pareado por id, 1000 remuestreos, IC percentil 95% de la diferencia de Macro-F1 (contrato, sección 10). Una diferencia es informativa si el IC no incluye 0. Se compara la semilla k de A con la semilla k de B.
- Si el IC excluye 0 con el mismo signo en las tres semillas, hay evidencia de una diferencia sistemática; si el signo cambia entre semillas o el IC incluye 0, no se puede afirmar que un modelo sea mejor.

- Pares pendientes (faltan sus JSON en `results/pairs/`; se calculan con `scripts/compare_pairs.py`): DistilBERT − BERT-base; Qwen2.5-0.5B − Qwen2.5-1.5B.

## 7. Figuras

- `fig_macro_f1`: Macro-F1 de test por modelo (puntos = semillas, marca = media, línea = ±1 sd). Los Dummy quedan fuera de escala (ver tabla 1).
- `fig_val_curves`: Macro-F1 en val_select durante el entrenamiento, una línea por semilla; el punto marca el mejor checkpoint.
- `fig_confusion`: matriz de confusión normalizada por clase real (semillas sumadas) de cinco modelos; `fig_confusion_all`, de todos.
- `fig_cost_perf`: Macro-F1 frente a VRAM pico de entrenamiento y frente a parámetros totales.
