# Análisis de errores: BERT-base

> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, 37,153 textos, semillas [42, 1337, 2026]). Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.

## 1. Resumen

| Semilla | Errores | Tasa de error | Macro-F1 recalculado | vs metrics.json |
|---|---|---|---|---|
| 42 | 287 | 0.77% | 0.9898 | coincide |
| 1337 | 273 | 0.73% | 0.9897 | coincide |
| 2026 | 290 | 0.78% | 0.9889 | coincide |

- Textos mal clasificados en **todas** las semillas (persistentes): 144 (0.39% del test).
- Textos mal clasificados en **una sola** semilla: 190. Un error que aparece solo en una semilla suele ser ruido del entrenamiento; uno persistente apunta al dato (ambigüedad o etiqueta dudosa) o a una limitación del modelo.
- Errores totales (suma de semillas): 850; de ellos, en textos persistentes: 50.8%.

## 2. Por clase (semillas sumadas)

| Clase | Soporte (test) | Recall | Precisión | Predichas / reales |
|---|---|---|---|---|
| sadness | 11,604 | 99.6% | 99.3% | 1.00 |
| joy | 12,937 | 99.4% | 99.7% | 1.00 |
| love | 2,438 | 99.7% | 99.4% | 1.00 |
| anger | 5,233 | 98.8% | 98.6% | 1.00 |
| fear | 3,958 | 98.3% | 98.7% | 1.00 |
| surprise | 983 | 98.2% | 97.7% | 1.00 |

- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.

## 3. Pares de confusión más frecuentes (top 5)

| Real → Predicho | Errores | % de la clase real | % de todos los errores |
|---|---|---|---|
| anger → sadness | 107 | 0.68% | 12.6% |
| joy → sadness | 87 | 0.22% | 10.2% |
| sadness → anger | 71 | 0.20% | 8.4% |
| fear → anger | 70 | 0.59% | 8.2% |
| joy → anger | 68 | 0.18% | 8.0% |

## 4. Confianza de la predicción

| Probabilidad de la clase predicha | Predicciones | Tasa de error |
|---|---|---|
| < 0.50 | 76 | 65.79% |
| 0.50 - 0.70 | 294 | 50.00% |
| 0.70 - 0.90 | 341 | 46.04% |
| 0.90 - 0.99 | 590 | 35.93% |
| >= 0.99 | 110,158 | 0.26% |

- Errores con confianza >= 0.90: 58.4% de los errores. Confianza media: aciertos 0.999, errores 0.853.

## 5. Longitud del texto (palabras)

| Palabras | Textos (test) | Tasa de error |
|---|---|---|
| <= 11 | 10,536 | 0.98% |
| 12 - 17 | 8,680 | 0.66% |
| 18 - 25 | 8,825 | 0.59% |
| > 25 | 9,112 | 0.78% |

- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el truncamiento a `max_len` o la dilución de la señal importan.

## 6. Tipo de texto

| Subconjunto | Textos (test) | % del test | Tasa de error | % de los errores típicos del modelo |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 98.49% | 0.36% | 40.7% |
| No contiene feel | 562 | 1.51% | 27.28% | 59.3% |
| Empieza con "when" | 168 | 0.45% | 30.95% | 20.9% |

- Casi todo el conjunto son tweets de estilo CARER ("i feel ..."). Los textos sin "feel" y los que empiezan con "when" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.

## 7. Ejemplos

Elegidos al azar (semilla 42) entre los textos que el modelo falla en la mayoría de las semillas, preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.

### anger → sadness

- "disagreements with mother when i moved from kankaanpss"  
  real = **anger**, predicho = **sadness** (confianza media 0.76; falla en 3/3 semillas)
- "i feel that this ungratefulness has lead me to be selfish self absorbed and proud and that s not a good feeling for me at all"  
  real = **anger**, predicho = **sadness** (confianza media 0.99; falla en 3/3 semillas)
- "when i heard the bad news about the death of a close person"  
  real = **anger**, predicho = **sadness** (confianza media 0.91; falla en 3/3 semillas)
- "in autumn i had to move to another lodging my landlords"  
  real = **anger**, predicho = **sadness** (confianza media 0.96; falla en 3/3 semillas)

### joy → sadness

- "talking to a close friend"  
  real = **joy**, predicho = **sadness** (confianza media 0.75; falla en 3/3 semillas)
- "when i passed the whole weekend with my boyfriend"  
  real = **joy**, predicho = **sadness** (confianza media 0.93; falla en 3/3 semillas)
- "when after months of national service i was downgraded and did not have to do the remaining months"  
  real = **joy**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "spending the nitht with close friends"  
  real = **joy**, predicho = **sadness** (confianza media 0.94; falla en 3/3 semillas)

### sadness → anger

- "i was together with my family one night"  
  real = **sadness**, predicho = **anger** (confianza media 0.53; falla en 3/3 semillas)
- "i spouting off in the most laughable way about how shocked he is damsgaard lambasting the afld for not testing enough the teams vs uci conflict seemingly threatening to kill off the whole bio passport plan and [...]"  
  real = **sadness**, predicho = **anger** (confianza media 0.47; falla en 3/3 semillas)
- "when my best friend promised to come at my house and i had to prepare so many expensive things for her but only to learn at last that she would not turn up"  
  real = **sadness**, predicho = **anger** (confianza media 0.76; falla en 3/3 semillas)
- "im over it and i want fall and im feeling angsty and gloomy and bleh"  
  real = **sadness**, predicho = **anger** (confianza media 0.80; falla en 3/3 semillas)

### fear → anger

- "at the railway station"  
  real = **fear**, predicho = **anger** (confianza media 0.58; falla en 3/3 semillas)
- "when my father was very drunk and raging at home"  
  real = **fear**, predicho = **anger** (confianza media 0.99; falla en 3/3 semillas)
- "i was sexually assaulted one night by a stranger"  
  real = **fear**, predicho = **anger** (confianza media 0.88; falla en 3/3 semillas)
- "not applicable to myself"  
  real = **fear**, predicho = **anger** (confianza media 0.90; falla en 3/3 semillas)

### joy → anger

- "when i was with someone i love"  
  real = **joy**, predicho = **anger** (confianza media 0.71; falla en 3/3 semillas)
- "i feel more and more repulsed by my own religion to the point where there are moments that go by when i am briefly convinced that i dont believe in any god at all"  
  real = **joy**, predicho = **anger** (confianza media 0.82; falla en 3/3 semillas)
- "one of the greatest joiy was when i succeded in acquiring a motorcycle through my own efforts"  
  real = **joy**, predicho = **anger** (confianza media 0.69; falla en 3/3 semillas)
- "a fellow player"  
  real = **joy**, predicho = **anger** (confianza media 0.72; falla en 3/3 semillas)

### Errores persistentes de mayor confianza

Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.

- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i wasnt really feeling like being snarky or funny"  
  real = **surprise**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i feel like i am a knowledgeble person when it comes to fruits and vegetables so i was surprised that i had never tried one before"  
  real = **surprise**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i close my eyes i see her so vividly that makes me feel that time is only ticking away timidly where do i begin my cute li"  
  real = **joy**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)
- "i feel fear quite often but this feeling was very strong last year when my husband was quite unexpectedly sent to the hospital"  
  real = **fear**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i still have a fever and i m still feeling pretty pissy"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i have had to choose to give those feelings and terrible moments of indecisiveness over to god"  
  real = **sadness**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)
- "as the time to get the results drew near i became more and more anxious my friend called me and said"  
  real = **joy**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)

