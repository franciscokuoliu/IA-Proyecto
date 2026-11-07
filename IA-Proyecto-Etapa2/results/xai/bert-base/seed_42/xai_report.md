# Explicabilidad (XAI): bert-base, semilla 42

Generado por `scripts/xai.py`. No se edita a mano: si algo cambia, se vuelve a correr el script. La interpretación va en el artículo, apoyada en estos números y en `examples.html`.

## 0. Qué se hizo

- Modelo: `bert-base-uncased` ajustado (checkpoint `results/bert-base/seed_42/model/trainable.pt`).
- Muestra: 600 textos del test, estratificada por clase (hasta 100 por clase, semilla 42), más 40 errores del test para el análisis cualitativo (el modelo se equivocó en 287 textos del test).
- Integrated Gradients: 64 puntos de Gauss-Legendre, baseline = [PAD] en cada palabra ([CLS] y [SEP] intactos), objetivo = logit de la clase predicha. La atribución de una palabra es la suma de la de sus sub-tokens.
- Atención: fila de [CLS] en la última capa (media de cabezas) y attention rollout (con conexiones residuales).
- Fidelidad: quitar (o dejar solo) las k palabras más importantes, con k en [1, 2, 3, 5]. "Quitar" pone en cero la máscara de atención de sus sub-tokens. Línea base: 5 sorteos de k palabras al azar por texto.
- Cálculo en fp32 en `cuda` (torch 2.9.1+cu128, transformers 5.22.1).

## 1. ¿Se puede confiar en el método? (verificaciones previas)

| Verificación | Resultado | Umbral orientativo |
|---|---|---|
| Predicciones recalculadas iguales a `predictions.csv` | 100.0% (0 de 640 difieren) | ≥ 99% (la corrida usó fp16, aquí se usa fp32) |
| Completitud de IG: \|suma de atribuciones − (F(x) − F(base))\| / \|F(x) − F(base)\| | media 1.6%, mediana 0.2%, p95 6.4% | media < 5% |
| Texto base ([CLS] [PAD] … [SEP]) clasificado como la clase que se explica | 15.8% de los textos | bajo; si es alto, las atribuciones de esa clase son pequeñas |
| Cordura: cabeza de clasificación aleatoria (Spearman con las atribuciones del modelo entrenado, 100 textos) | rho medio -0.120; misma palabra top-1 en 20.0% | rho cerca de 0 |
| Cordura: modelo entero aleatorio (Spearman con las atribuciones del modelo entrenado, 100 textos) | rho medio -0.053; misma palabra top-1 en 4.0% | rho cerca de 0 |

Si la completitud o la cordura fallan, las atribuciones no se interpretan: se sube `--ig-steps` o se revisa el checkpoint.

## 2. Fidelidad: ¿las palabras marcadas importan de verdad?

Caída de la probabilidad de la clase predicha. **Quitar** las k palabras más importantes: más caída = más fiel. **Dejar solo** esas k palabras: menos caída = más fiel. Entre paréntesis, el IC 95% aproximado (1.96 errores estándar). n es la cantidad de textos con más de k palabras.

| k | Método | n | Caída al quitar | Cambia la predicción al quitar | Caída al dejar solo | La decisión se mantiene al dejar solo |
|---|---|---|---|---|---|---|
| 1 | Integrated Gradients | 600 | 0.748 (±0.033) | 75.0% | 0.014 (±0.009) | 98.5% |
| 1 | Attention rollout | 600 | 0.169 (±0.030) | 16.7% | 0.638 (±0.034) | 35.5% |
| 1 | [CLS] attention, last layer | 600 | 0.448 (±0.039) | 45.0% | 0.419 (±0.038) | 56.5% |
| 1 | Random words | 600 | 0.072 (±0.010) | 7.3% | 0.754 (±0.019) | 24.7% |
| 2 | Integrated Gradients | 599 | 0.838 (±0.026) | 84.0% | 0.001 (±0.003) | 99.7% |
| 2 | Attention rollout | 599 | 0.308 (±0.036) | 30.9% | 0.493 (±0.037) | 50.8% |
| 2 | [CLS] attention, last layer | 599 | 0.621 (±0.037) | 62.4% | 0.160 (±0.029) | 83.6% |
| 2 | Random words | 599 | 0.137 (±0.015) | 13.9% | 0.682 (±0.021) | 31.5% |
| 3 | Integrated Gradients | 597 | 0.863 (±0.024) | 86.6% | -0.000 (±0.002) | 99.8% |
| 3 | Attention rollout | 597 | 0.410 (±0.038) | 41.0% | 0.397 (±0.037) | 60.6% |
| 3 | [CLS] attention, last layer | 597 | 0.701 (±0.034) | 69.7% | 0.082 (±0.021) | 91.6% |
| 3 | Random words | 597 | 0.184 (±0.018) | 18.6% | 0.630 (±0.022) | 36.7% |
| 5 | Integrated Gradients | 569 | 0.890 (±0.021) | 89.6% | -0.001 (±0.002) | 99.8% |
| 5 | Attention rollout | 569 | 0.566 (±0.038) | 56.1% | 0.237 (±0.033) | 76.3% |
| 5 | [CLS] attention, last layer | 569 | 0.759 (±0.031) | 75.7% | 0.025 (±0.013) | 97.4% |
| 5 | Random words | 569 | 0.268 (±0.021) | 27.0% | 0.537 (±0.024) | 46.0% |

AOPC (promedio de la caída al quitar sobre los k): Integrated Gradients 0.835, Attention rollout 0.363, [CLS] attention, last layer 0.632, Random words 0.165.

## 3. ¿Qué palabras usa el modelo?

| Clase predicha | Palabras con mas atribucion (aparece en N textos; * = palabra marcadora del entrenamiento) |
|---|---|
| sadness | feel (73), feeling (30), i (154), old (5), so (21), my (27), like (21), and (59) |
| joy | feel (64), feeling (32), i (150), and (52), for (22), the* (63), to (56), will (11) |
| love | feel (68), tender* (9), feeling (31), horny* (7), liked* (7), gentle* (6), lovely* (5), supportive* (6) |
| anger | rebellious* (6), rushed* (6), irritated* (5), feel (65), insulted* (5), bitter* (6), cold* (5), i (166) |
| fear | intimidated* (8), uncomfortable* (9), shaken* (5), paranoid* (5), afraid* (5), confused* (5), scared* (5), frightened* (5) |
| surprise | amazed* (19), shocked* (18), funny* (19), feel (68), surprised* (10), dazed (9), impressed* (10), curious* (10) |

Sobre 600 textos de la muestra (con al menos 2 palabras):

- La palabra con más atribución (top-1) es una palabra marcadora de la clase predicha en 77.5% de los textos con IG, 19.8% con rollout y 46.2% con la atención de [CLS]. Si se eligiera una palabra al azar del texto, acertaría en 11.6%. Marcadoras = las 50 palabras con mayor razón de log-odds de cada clase en el subconjunto S de entrenamiento.
- El top-1 de IG es una stopword en 0.0% de los textos (al azar sería 53.4%).
- Posición del top-1 de IG respecto de la primera palabra de la familia feel/feeling/felt: antes de la palabra feel 0.7%, la propia palabra feel 2.5%, texto sin palabra feel 1.7%, 1 palabra después 33.3%, 2 palabras después 20.3%, 3 o más palabras después 41.5%.
- Acuerdo de IG con rollout: misma palabra top-1 en 21.5%, Spearman medio 0.174.
- Acuerdo de IG con la atención de [CLS]: misma palabra top-1 en 51.8%, Spearman medio 0.272.
- Masa de atención que cae en [CLS]/[SEP]: 20.2% (última capa) y 59.7% (rollout). Una masa alta en tokens especiales es una de las razones por las que la atención cruda se interpreta con cuidado.

## 4. Ejemplos

Ver `fig_xai_examples.png` (figura para el artículo) y `examples.html` (más ejemplos, con IG, rollout y atención). Errores analizados: palabras con mayor atribución hacia la clase predicha y hacia la verdadera.

- verdadera **joy**, predicha **sadness** (p=1.00): "when my boyfriend once again told me that he loved me after a period in which he felt very depressive" → hacia la predicha: depressive (+3.91), loved (+1.12), felt (+0.58); hacia la verdadera: loved (+0.42), boyfriend (+0.31), after (+0.04)
- verdadera **anger**, predicha **sadness** (p=0.79): "when i got home i found that the electrical supply had been disconnected despite my having paid zesco the full bill a few days earlier" → hacia la predicha: disconnected (+2.00), found (+0.36), supply (+0.29); hacia la verdadera: when (+0.18), been (+0.16), had (+0.16)
- verdadera **fear**, predicha **joy** (p=0.73): "introducing myself at a first tutorial this year at university" → hacia la predicha: tutorial (+0.93), this (+0.36), year (+0.26); hacia la verdadera: introducing (+1.37), myself (+0.81), first (+0.48)
- verdadera **fear**, predicha **surprise** (p=0.60): "i was feeling really overwhelmed trying to remember all the names of the members but i think i learned at least names or so so i think within a couple of weeks i will know the members pretty well" → hacia la predicha: overwhelmed (+2.93), feeling (+2.72), really (+0.39); hacia la verdadera: overwhelmed (+2.37), feeling (+0.59), know (+0.41)
- verdadera **anger**, predicha **sadness** (p=0.55): "i felt unfairly treated at an airport" → hacia la predicha: unfairly (+0.67), i (+0.55), airport (+0.51); hacia la verdadera: unfairly (+0.75), an (+0.67), at (+0.47)
- verdadera **anger**, predicha **fear** (p=0.50): "i trusted a friend of mine" → hacia la predicha: trusted (+2.97), i (+0.36), friend (+0.13); hacia la verdadera: a (+0.72), of (+0.39), i (+0.13)
- verdadera **joy**, predicha **sadness** (p=0.65): "i feel so robbed of such a precious life gift" → hacia la predicha: robbed (+8.23), i (+0.75), a (-0.22); hacia la verdadera: precious (+7.10), feel (+2.78), so (+0.92)
- verdadera **anger**, predicha **joy** (p=0.75): "my grandmother is a highly authoritative person" → hacia la predicha: authoritative (+4.00), highly (+0.00), my (-0.16); hacia la verdadera: a (+0.75), person (+0.55), grandmother (+0.48)
- verdadera **love**, predicha **anger** (p=0.99): "i wasnt feeling that horny just annoyed about work and i guess he wasnt either" → hacia la predicha: annoyed (+4.48), feeling (+0.96), about (+0.55); hacia la verdadera: horny (+3.82), that (+0.99), guess (+0.45)
- verdadera **surprise**, predicha **fear** (p=0.74): "i feel overwhelmed by my circumstance in all of my mere human ness i will remember that god has landed here" → hacia la predicha: overwhelmed (+1.25), by (+0.94), feel (+0.67); hacia la verdadera: overwhelmed (+3.29), feel (+1.69), by (+0.27)
- verdadera **sadness**, predicha **joy** (p=0.59): "im over it and i want fall and im feeling angsty and gloomy and bleh" → hacia la predicha: feeling (+3.37), im (+0.47), and (+0.17); hacia la verdadera: gloomy (+2.91), and (+0.27), it (-0.02)
- verdadera **sadness**, predicha **joy** (p=0.97): "i cant help feeling this way" → hacia la predicha: feeling (+3.54), help (+0.16), i (-0.17); hacia la verdadera: way (-0.02), i (-0.07), help (-0.16)

## 5. Lectura y límites (para el artículo y la defensa)

- La atribución explica **al modelo**, no a los datos: que una palabra reciba atribución dice que el modelo la usó, no que sea la causa real de la emoción.
- IG explica la diferencia entre el texto y el baseline. Si el baseline ya favorece la clase explicada, las atribuciones de esa clase son pequeñas y menos fiables. Otro baseline (cero, [MASK]) puede dar otro reparto; no se probó.
- Quitar una palabra con la máscara de atención no equivale a borrarla: el modelo ve un texto que nunca vio en el entrenamiento. Por eso se compara siempre contra quitar palabras al azar.
- La atención no es una explicación por sí sola (Jain y Wallace, 2019). Se reporta para contrastarla con IG, no como evidencia independiente.
- Las palabras marcadoras salen de la misma distribución del entrenamiento. Si el modelo las usa, aprendió pistas léxicas parecidas a las de un modelo lineal; es coherente con TF-IDF + regresión logística, pero no prueba que entienda la emoción.
- La muestra es de 600 textos y una sola semilla (42); los IC son aproximados y no cubren la variación entre semillas.
