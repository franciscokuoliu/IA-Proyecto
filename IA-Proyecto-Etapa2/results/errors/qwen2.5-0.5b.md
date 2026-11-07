# Análisis de errores: Qwen2.5-0.5B

> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, 37,153 textos, semillas [42, 1337, 2026]). Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.

## 1. Resumen

| Semilla | Errores | Tasa de error | Macro-F1 recalculado | vs metrics.json |
|---|---|---|---|---|
| 42 | 326 | 0.88% | 0.9886 | coincide |
| 1337 | 393 | 1.06% | 0.9867 | coincide |
| 2026 | 345 | 0.93% | 0.9878 | coincide |

- Textos mal clasificados en **todas** las semillas (persistentes): 168 (0.45% del test).
- Textos mal clasificados en **una sola** semilla: 246. Un error que aparece solo en una semilla suele ser ruido del entrenamiento; uno persistente apunta al dato (ambigüedad o etiqueta dudosa) o a una limitación del modelo.
- Errores totales (suma de semillas): 1,064; de ellos, en textos persistentes: 47.4%.

## 2. Por clase (semillas sumadas)

| Clase | Soporte (test) | Recall | Precisión | Predichas / reales |
|---|---|---|---|---|
| sadness | 11,604 | 99.4% | 99.2% | 1.00 |
| joy | 12,937 | 99.4% | 99.3% | 1.00 |
| love | 2,438 | 99.6% | 99.5% | 1.00 |
| anger | 5,233 | 98.4% | 98.2% | 1.00 |
| fear | 3,958 | 97.9% | 98.7% | 0.99 |
| surprise | 983 | 97.3% | 98.4% | 0.99 |

- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.

## 3. Pares de confusión más frecuentes (top 5)

| Real → Predicho | Errores | % de la clase real | % de todos los errores |
|---|---|---|---|
| anger → joy | 111 | 0.71% | 10.4% |
| joy → sadness | 109 | 0.28% | 10.2% |
| fear → anger | 100 | 0.84% | 9.4% |
| sadness → anger | 93 | 0.27% | 8.7% |
| joy → anger | 92 | 0.24% | 8.6% |

## 4. Confianza de la predicción

| Probabilidad de la clase predicha | Predicciones | Tasa de error |
|---|---|---|
| < 0.50 | 117 | 67.52% |
| 0.50 - 0.70 | 424 | 57.55% |
| 0.70 - 0.90 | 537 | 47.86% |
| 0.90 - 0.99 | 868 | 27.07% |
| >= 0.99 | 109,513 | 0.23% |

- Errores con confianza >= 0.90: 45.5% de los errores. Confianza media: aciertos 0.998, errores 0.807.

## 5. Longitud del texto (palabras)

| Palabras | Textos (test) | Tasa de error |
|---|---|---|
| <= 11 | 10,536 | 1.13% |
| 12 - 17 | 8,680 | 0.94% |
| 18 - 25 | 8,825 | 0.81% |
| > 25 | 9,112 | 0.90% |

- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el truncamiento a `max_len` o la dilución de la señal importan.

## 6. Tipo de texto

| Subconjunto | Textos (test) | % del test | Tasa de error | % de los errores típicos del modelo |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 98.49% | 0.43% | 40.6% |
| No contiene feel | 562 | 1.51% | 35.05% | 59.4% |
| Empieza con "when" | 168 | 0.45% | 40.67% | 20.3% |

- Casi todo el conjunto son tweets de estilo CARER ("i feel ..."). Los textos sin "feel" y los que empiezan con "when" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.

## 7. Ejemplos

Elegidos al azar (semilla 42) entre los textos que el modelo falla en la mayoría de las semillas, preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.

### anger → joy

- "i was double crossing mercy and maria they one day came to know each other i was found by mercy redhaded when i was with maria"  
  real = **anger**, predicho = **joy** (confianza media 0.85; falla en 3/3 semillas)
- "at my summer job"  
  real = **anger**, predicho = **joy** (confianza media 0.60; falla en 3/3 semillas)
- "i was a kid growing up in the s drought i remember when i first left the water on when brushing my teeth because it made me feel sort of badass and rebellious i was a very good kid okay"  
  real = **anger**, predicho = **joy** (confianza media 0.87; falla en 3/3 semillas)
- "when i saw the film memorie of prison i saw a scene of masturbation of a man in the hold of a ship apart form fact that the film wasnt pleasant i had a leg of coconut candies in my hands"  
  real = **anger**, predicho = **joy** (confianza media 0.83; falla en 3/3 semillas)

### joy → sadness

- "i don t know about you but i m feeling cents underpaid on the dollar c mon that is creative"  
  real = **joy**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "i feel that sleeve variations are one of the more underused ways to make a garment special"  
  real = **joy**, predicho = **sadness** (confianza media 0.94; falla en 3/3 semillas)
- "when after months of national service i was downgraded and did not have to do the remaining months"  
  real = **joy**, predicho = **sadness** (confianza media 0.76; falla en 3/3 semillas)
- "a very good friend of mine had been away for three months when we met again"  
  real = **joy**, predicho = **sadness** (confianza media 0.51; falla en 3/3 semillas)

### fear → anger

- "while swimming"  
  real = **fear**, predicho = **anger** (confianza media 0.56; falla en 3/3 semillas)
- "when i heard that my father had found a letter from my boyfriend and he was calling me to talk about it"  
  real = **fear**, predicho = **anger** (confianza media 0.53; falla en 3/3 semillas)
- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i was sexually assaulted one night by a stranger"  
  real = **fear**, predicho = **anger** (confianza media 0.57; falla en 3/3 semillas)

### sadness → anger

- "when the train taking me to my military service left the station"  
  real = **sadness**, predicho = **anger** (confianza media 0.68; falla en 3/3 semillas)
- "when i was a child i was playing on the upper deck of the bed with my brother my parents slept on the lower deck my father felt disgusted and beat me up i felt very sad as my father had never beaten me or scolded [...]"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i seized up and started feeling twitchy and unhappy after the first page"  
  real = **sadness**, predicho = **anger** (confianza media 0.84; falla en 3/3 semillas)
- "i could feel the coldness easing my aching shins within those five minutes and it lasted for over an hour"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)

### joy → anger

- "i just did something that i never expected i would ever feel the need to do and that s delete a post actually two posts because the content offended somebody"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i am feeling the coldness and numbness of death in my lively heart"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "when i knew that there were"  
  real = **joy**, predicho = **anger** (confianza media 0.63; falla en 3/3 semillas)
- "a fellow player"  
  real = **joy**, predicho = **anger** (confianza media 0.97; falla en 3/3 semillas)

### Errores persistentes de mayor confianza

Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.

- "i still have a fever and i m still feeling pretty pissy"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i close my eyes i see her so vividly that makes me feel that time is only ticking away timidly where do i begin my cute li"  
  real = **joy**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)
- "i usually get this feeling of discontentment or i get really restless and feel like i always need to do something"  
  real = **fear**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "im trying to compare something to this feeling of coolness but im drawin a blank"  
  real = **sadness**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i wrote this i could smell and feel the coolness of her garage on a hot summers day"  
  real = **love**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "when i was a child i was playing on the upper deck of the bed with my brother my parents slept on the lower deck my father felt disgusted and beat me up i felt very sad as my father had never beaten me or scolded [...]"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i loved that feeling of perfecting a combination and was amazed by the freedom for movement that the ballet attire allowed for"  
  real = **surprise**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)

