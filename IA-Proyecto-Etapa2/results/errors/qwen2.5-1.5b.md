# Análisis de errores: Qwen2.5-1.5B

> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, 37,153 textos, semillas [42, 1337, 2026]). Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.

## 1. Resumen

| Semilla | Errores | Tasa de error | Macro-F1 recalculado | vs metrics.json |
|---|---|---|---|---|
| 42 | 327 | 0.88% | 0.9884 | coincide |
| 1337 | 269 | 0.72% | 0.9900 | coincide |
| 2026 | 342 | 0.92% | 0.9877 | coincide |

- Textos mal clasificados en **todas** las semillas (persistentes): 130 (0.35% del test).
- Textos mal clasificados en **una sola** semilla: 268. Un error que aparece solo en una semilla suele ser ruido del entrenamiento; uno persistente apunta al dato (ambigüedad o etiqueta dudosa) o a una limitación del modelo.
- Errores totales (suma de semillas): 938; de ellos, en textos persistentes: 41.6%.

## 2. Por clase (semillas sumadas)

| Clase | Soporte (test) | Recall | Precisión | Predichas / reales |
|---|---|---|---|---|
| sadness | 11,604 | 99.5% | 99.4% | 1.00 |
| joy | 12,937 | 99.3% | 99.5% | 1.00 |
| love | 2,438 | 99.7% | 99.5% | 1.00 |
| anger | 5,233 | 98.6% | 98.4% | 1.00 |
| fear | 3,958 | 98.1% | 98.3% | 1.00 |
| surprise | 983 | 98.7% | 97.3% | 1.01 |

- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.

## 3. Pares de confusión más frecuentes (top 5)

| Real → Predicho | Errores | % de la clase real | % de todos los errores |
|---|---|---|---|
| joy → anger | 96 | 0.25% | 10.2% |
| joy → sadness | 87 | 0.22% | 9.3% |
| anger → fear | 87 | 0.55% | 9.3% |
| fear → anger | 87 | 0.73% | 9.3% |
| fear → surprise | 72 | 0.61% | 7.7% |

## 4. Confianza de la predicción

| Probabilidad de la clase predicha | Predicciones | Tasa de error |
|---|---|---|
| < 0.50 | 165 | 67.27% |
| 0.50 - 0.70 | 453 | 50.99% |
| 0.70 - 0.90 | 651 | 39.17% |
| 0.90 - 0.99 | 950 | 18.63% |
| >= 0.99 | 109,240 | 0.15% |

- Errores con confianza >= 0.90: 36.4% de los errores. Confianza media: aciertos 0.998, errores 0.770.

## 5. Longitud del texto (palabras)

| Palabras | Textos (test) | Tasa de error |
|---|---|---|
| <= 11 | 10,536 | 1.10% |
| 12 - 17 | 8,680 | 0.76% |
| 18 - 25 | 8,825 | 0.69% |
| > 25 | 9,112 | 0.76% |

- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el truncamiento a `max_len` o la dilución de la señal importan.

## 6. Tipo de texto

| Subconjunto | Textos (test) | % del test | Tasa de error | % de los errores típicos del modelo |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 98.49% | 0.39% | 38.9% |
| No contiene feel | 562 | 1.51% | 30.37% | 61.1% |
| Empieza con "when" | 168 | 0.45% | 35.52% | 22.6% |

- Casi todo el conjunto son tweets de estilo CARER ("i feel ..."). Los textos sin "feel" y los que empiezan con "when" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.

## 7. Ejemplos

Elegidos al azar (semilla 42) entre los textos que el modelo falla en la mayoría de las semillas, preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.

### joy → anger

- "when my elder brother"  
  real = **joy**, predicho = **anger** (confianza media 0.84; falla en 3/3 semillas)
- "when i was with someone i love"  
  real = **joy**, predicho = **anger** (confianza media 0.57; falla en 3/3 semillas)
- "when i made up with a girlfriend of mine after having been on unfriendly terms with her my fault i cried with joy that evening"  
  real = **joy**, predicho = **anger** (confianza media 0.92; falla en 3/3 semillas)
- "when you are satisfied with something"  
  real = **joy**, predicho = **anger** (confianza media 0.74; falla en 3/3 semillas)

### joy → sadness

- "i feel so displaced my thoughts so astray that i know nothing else can calm me nothing can help me put things in perspective than a session of baking therapy"  
  real = **joy**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "when my boyfriend once again told me that he loved me after a period in which he felt very depressive"  
  real = **joy**, predicho = **sadness** (confianza media 0.81; falla en 3/3 semillas)
- "an encounter with a man whom i love"  
  real = **joy**, predicho = **sadness** (confianza media 0.60; falla en 3/3 semillas)
- "a very good friend of mine had been away for three months when we met again"  
  real = **joy**, predicho = **sadness** (confianza media 0.61; falla en 3/3 semillas)

### anger → fear

- "when i was working as a nurse in a mental hospital"  
  real = **anger**, predicho = **fear** (confianza media 0.61; falla en 3/3 semillas)
- "my first day to work in hospital"  
  real = **anger**, predicho = **fear** (confianza media 0.64; falla en 3/3 semillas)
- "i was sitting in school"  
  real = **anger**, predicho = **fear** (confianza media 0.59; falla en 3/3 semillas)
- "when i heard about the disaster in bophal india"  
  real = **anger**, predicho = **fear** (confianza media 0.60; falla en 3/3 semillas)

### fear → anger

- "when i have to leave the baby in the carriage and go shopping"  
  real = **fear**, predicho = **anger** (confianza media 0.77; falla en 3/3 semillas)
- "not applicable to myself"  
  real = **fear**, predicho = **anger** (confianza media 0.58; falla en 3/3 semillas)
- "when i heard that my father had found a letter from my boyfriend and he was calling me to talk about it"  
  real = **fear**, predicho = **anger** (confianza media 0.34; falla en 3/3 semillas)
- "i was hunting"  
  real = **fear**, predicho = **anger** (confianza media 0.68; falla en 3/3 semillas)

### fear → surprise

- "ive already been feeling overwhelmed"  
  real = **fear**, predicho = **surprise** (confianza media 0.82; falla en 3/3 semillas)
- "i am feeling very overwhelmed and i just dont feel like i can give things the attention they deserve"  
  real = **fear**, predicho = **surprise** (confianza media 0.88; falla en 3/3 semillas)
- "i was feeling really overwhelmed trying to remember all the names of the members but i think i learned at least names or so so i think within a couple of weeks i will know the members pretty well"  
  real = **fear**, predicho = **surprise** (confianza media 0.87; falla en 3/3 semillas)
- "i feel overwhelmed i want to walk out and just run"  
  real = **fear**, predicho = **surprise** (confianza media 0.71; falla en 3/3 semillas)

### Errores persistentes de mayor confianza

Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.

- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i feel fear quite often but this feeling was very strong last year when my husband was quite unexpectedly sent to the hospital"  
  real = **fear**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i feel like getting real snarky here but the other part of me just feels awful for lavolo"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i still have a fever and i m still feeling pretty pissy"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "when i was a child i was playing on the upper deck of the bed with my brother my parents slept on the lower deck my father felt disgusted and beat me up i felt very sad as my father had never beaten me or scolded [...]"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i have been stuffed in a cafe all day feeling some what costaphobic and lost"  
  real = **sadness**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)
- "i had to leave to work feeling supermegapissedoff and lethargic but i found solace in the awesome traffic while everyone else is still away from the city"  
  real = **sadness**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i feel so displaced my thoughts so astray that i know nothing else can calm me nothing can help me put things in perspective than a session of baking therapy"  
  real = **joy**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)

