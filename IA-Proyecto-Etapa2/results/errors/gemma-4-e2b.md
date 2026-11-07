# Análisis de errores: Gemma 4 E2B

> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, 37,153 textos, semillas [42, 1337, 2026]). Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.

## 1. Resumen

| Semilla | Errores | Tasa de error | Macro-F1 recalculado | vs metrics.json |
|---|---|---|---|---|
| 42 | 1,365 | 3.67% | 0.9507 | coincide |
| 1337 | 1,333 | 3.59% | 0.9567 | coincide |
| 2026 | 1,294 | 3.48% | 0.9558 | coincide |

- Textos mal clasificados en **todas** las semillas (persistentes): 870 (2.34% del test).
- Textos mal clasificados en **una sola** semilla: 372. Un error que aparece solo en una semilla suele ser ruido del entrenamiento; uno persistente apunta al dato (ambigüedad o etiqueta dudosa) o a una limitación del modelo.
- Errores totales (suma de semillas): 3,992; de ellos, en textos persistentes: 65.4%.

## 2. Por clase (semillas sumadas)

| Clase | Soporte (test) | Recall | Precisión | Predichas / reales |
|---|---|---|---|---|
| sadness | 11,604 | 96.3% | 98.8% | 0.98 |
| joy | 12,937 | 97.1% | 97.0% | 1.00 |
| love | 2,438 | 96.3% | 97.4% | 0.99 |
| anger | 5,233 | 96.4% | 93.0% | 1.04 |
| fear | 3,958 | 95.3% | 93.3% | 1.02 |
| surprise | 983 | 93.2% | 91.3% | 1.02 |

- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.

## 3. Pares de confusión más frecuentes (top 5)

| Real → Predicho | Errores | % de la clase real | % de todos los errores |
|---|---|---|---|
| sadness → joy | 530 | 1.52% | 13.3% |
| joy → anger | 466 | 1.20% | 11.7% |
| sadness → anger | 378 | 1.09% | 9.5% |
| joy → fear | 290 | 0.75% | 7.3% |
| anger → joy | 256 | 1.63% | 6.4% |

## 4. Confianza de la predicción

| Probabilidad de la clase predicha | Predicciones | Tasa de error |
|---|---|---|
| < 0.50 | 930 | 77.85% |
| 0.50 - 0.70 | 1,755 | 72.71% |
| 0.70 - 0.90 | 1,606 | 65.82% |
| 0.90 - 0.99 | 1,467 | 42.54% |
| >= 0.99 | 105,701 | 0.29% |

- Errores con confianza >= 0.90: 23.4% de los errores. Confianza media: aciertos 0.996, errores 0.705.

## 5. Longitud del texto (palabras)

| Palabras | Textos (test) | Tasa de error |
|---|---|---|
| <= 11 | 10,536 | 3.61% |
| 12 - 17 | 8,680 | 3.50% |
| 18 - 25 | 8,825 | 3.56% |
| > 25 | 9,112 | 3.64% |

- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el truncamiento a `max_len` o la dilución de la señal importan.

## 6. Tipo de texto

| Subconjunto | Textos (test) | % del test | Tasa de error | % de los errores típicos del modelo |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 98.49% | 3.23% | 89.2% |
| No contiene feel | 562 | 1.51% | 26.75% | 10.8% |
| Empieza con "when" | 168 | 0.45% | 29.17% | 3.7% |

- Casi todo el conjunto son tweets de estilo CARER ("i feel ..."). Los textos sin "feel" y los que empiezan con "when" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.

## 7. Ejemplos

Elegidos al azar (semilla 42) entre los textos que el modelo falla en la mayoría de las semillas, preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.

### sadness → joy

- "i can feel the staples but they hurt less"  
  real = **sadness**, predicho = **joy** (confianza media 0.50; falla en 3/3 semillas)
- "i am just feeling too lethargic to write something good"  
  real = **sadness**, predicho = **joy** (confianza media 0.66; falla en 3/3 semillas)
- "i feel that i wasnt totally that terrible"  
  real = **sadness**, predicho = **joy** (confianza media 0.59; falla en 3/3 semillas)
- "i am tired of struggling financially and im feeling guilty because i dont work"  
  real = **sadness**, predicho = **joy** (confianza media 0.87; falla en 3/3 semillas)

### joy → anger

- "i feel like it gives that cool image im looking for"  
  real = **joy**, predicho = **anger** (confianza media 0.51; falla en 3/3 semillas)
- "i feel at my most useful when listening to others hurts and dilemmas and even better when i can help them parse out a strategy or solution to make their lives better"  
  real = **joy**, predicho = **anger** (confianza media 0.59; falla en 3/3 semillas)
- "i feel casual about"  
  real = **joy**, predicho = **anger** (confianza media 0.26; falla en 3/3 semillas)
- "i am feeling more peaceful from day to day and more committed to keeping most of this up when the days completes"  
  real = **joy**, predicho = **anger** (confianza media 0.31; falla en 3/3 semillas)

### sadness → anger

- "i feel i have disappointed my parents even though my environmentalist mother sings my praises for escaping the commuting life and my father has never shown anything but pride in me"  
  real = **sadness**, predicho = **anger** (confianza media 0.54; falla en 3/3 semillas)
- "i always feel guilty when i do that because it kind of puts me on a different schedule than my husband"  
  real = **sadness**, predicho = **anger** (confianza media 0.27; falla en 3/3 semillas)
- "i feel that he shamed a franchise but simply because hes bad at what he does"  
  real = **sadness**, predicho = **anger** (confianza media 0.28; falla en 3/3 semillas)
- "i couldnt have fathomed welcoming people to feel my stomach move and i hope i havent dismayed some of my students"  
  real = **sadness**, predicho = **anger** (confianza media 0.16; falla en 3/3 semillas)

### joy → fear

- "i returned home and ever since i simply forgot about it and thank god i feel fine"  
  real = **joy**, predicho = **fear** (confianza media 0.62; falla en 3/3 semillas)
- "first public speaking"  
  real = **joy**, predicho = **fear** (confianza media 0.93; falla en 3/3 semillas)
- "i feel i need to start being more truthful"  
  real = **joy**, predicho = **fear** (confianza media 0.47; falla en 3/3 semillas)
- "when i received my report card in cou i was afraid of obtaining low marks"  
  real = **joy**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)

### anger → joy

- "im back in gloucester albeit with flu bad sinuses and feeling generally crap and irritable"  
  real = **anger**, predicho = **joy** (confianza media 0.51; falla en 3/3 semillas)
- "i step out of the house it feels hostile"  
  real = **anger**, predicho = **joy** (confianza media 0.67; falla en 3/3 semillas)
- "i personally think its infection im coughing alot still and it feel all a bit irritable"  
  real = **anger**, predicho = **joy** (confianza media 0.51; falla en 3/3 semillas)
- "ive been feeling annoyed at myself that i remember barely more than a few details of the later books"  
  real = **anger**, predicho = **joy** (confianza media 0.50; falla en 3/3 semillas)

### Errores persistentes de mayor confianza

Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.

- "i close my eyes i see her so vividly that makes me feel that time is only ticking away timidly where do i begin my cute li"  
  real = **joy**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)
- "i am mostly feeling contentedly terrified about it all"  
  real = **fear**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i could feel the coldness easing my aching shins within those five minutes and it lasted for over an hour"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i feel fear quite often but this feeling was very strong last year when my husband was quite unexpectedly sent to the hospital"  
  real = **fear**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i am feeling the coldness and numbness of death in my lively heart"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i usually get this feeling of discontentment or i get really restless and feel like i always need to do something"  
  real = **fear**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "when my mother scolded me for something"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)

