# Análisis de errores: DeepSeek-R1-Distill-1.5B

> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, 37,153 textos, semillas [42, 1337, 2026]). Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.

## 1. Resumen

| Semilla | Errores | Tasa de error | Macro-F1 recalculado | vs metrics.json |
|---|---|---|---|---|
| 42 | 350 | 0.94% | 0.9877 | coincide |
| 1337 | 399 | 1.07% | 0.9865 | coincide |
| 2026 | 349 | 0.94% | 0.9880 | coincide |

- Textos mal clasificados en **todas** las semillas (persistentes): 177 (0.48% del test).
- Textos mal clasificados en **una sola** semilla: 251. Un error que aparece solo en una semilla suele ser ruido del entrenamiento; uno persistente apunta al dato (ambigüedad o etiqueta dudosa) o a una limitación del modelo.
- Errores totales (suma de semillas): 1,098; de ellos, en textos persistentes: 48.4%.

## 2. Por clase (semillas sumadas)

| Clase | Soporte (test) | Recall | Precisión | Predichas / reales |
|---|---|---|---|---|
| sadness | 11,604 | 99.4% | 99.2% | 1.00 |
| joy | 12,937 | 99.2% | 99.5% | 1.00 |
| love | 2,438 | 99.6% | 99.5% | 1.00 |
| anger | 5,233 | 98.5% | 98.2% | 1.00 |
| fear | 3,958 | 97.7% | 98.3% | 0.99 |
| surprise | 983 | 99.4% | 96.5% | 1.03 |

- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.

## 3. Pares de confusión más frecuentes (top 5)

| Real → Predicho | Errores | % de la clase real | % de todos los errores |
|---|---|---|---|
| joy → sadness | 137 | 0.35% | 12.5% |
| joy → anger | 101 | 0.26% | 9.2% |
| anger → fear | 96 | 0.61% | 8.7% |
| sadness → anger | 92 | 0.26% | 8.4% |
| fear → surprise | 91 | 0.77% | 8.3% |

## 4. Confianza de la predicción

| Probabilidad de la clase predicha | Predicciones | Tasa de error |
|---|---|---|
| < 0.50 | 197 | 73.10% |
| 0.50 - 0.70 | 520 | 52.50% |
| 0.70 - 0.90 | 726 | 39.94% |
| 0.90 - 0.99 | 1,032 | 22.29% |
| >= 0.99 | 108,984 | 0.15% |

- Errores con confianza >= 0.90: 35.6% de los errores. Confianza media: aciertos 0.998, errores 0.765.

## 5. Longitud del texto (palabras)

| Palabras | Textos (test) | Tasa de error |
|---|---|---|
| <= 11 | 10,536 | 1.20% |
| 12 - 17 | 8,680 | 0.89% |
| 18 - 25 | 8,825 | 0.82% |
| > 25 | 9,112 | 0.99% |

- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el truncamiento a `max_len` o la dilución de la señal importan.

## 6. Tipo de texto

| Subconjunto | Textos (test) | % del test | Tasa de error | % de los errores típicos del modelo |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 98.49% | 0.47% | 41.2% |
| No contiene feel | 562 | 1.51% | 34.58% | 58.8% |
| Empieza con "when" | 168 | 0.45% | 39.48% | 20.6% |

- Casi todo el conjunto son tweets de estilo CARER ("i feel ..."). Los textos sin "feel" y los que empiezan con "when" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.

## 7. Ejemplos

Elegidos al azar (semilla 42) entre los textos que el modelo falla en la mayoría de las semillas, preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.

### joy → sadness

- "seeing relations from overseas after a few years"  
  real = **joy**, predicho = **sadness** (confianza media 0.48; falla en 3/3 semillas)
- "im happy for her and want the best but it just got me thinking about how i feel my ra has denied me all those wonderful normal things you just take for granted like a proposal"  
  real = **joy**, predicho = **sadness** (confianza media 0.75; falla en 3/3 semillas)
- "i don t say day after i feel suckish and mellow"  
  real = **joy**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "i feel sympathy for jonas the one i atone many ask for release but well be left helpless because now we see it doesnt bring peace only sadness"  
  real = **joy**, predicho = **sadness** (confianza media 0.64; falla en 3/3 semillas)

### joy → anger

- "when i made up with a girlfriend of mine after having been on unfriendly terms with her my fault i cried with joy that evening"  
  real = **joy**, predicho = **anger** (confianza media 0.76; falla en 3/3 semillas)
- "i still have a fever and i m still feeling pretty pissy"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "went to a movie with a date"  
  real = **joy**, predicho = **anger** (confianza media 0.68; falla en 3/3 semillas)
- "i had a quarrel with my fianct when he admitted his fault"  
  real = **joy**, predicho = **anger** (confianza media 0.92; falla en 3/3 semillas)

### anger → fear

- "one night i could not practise my violin as my family was watching tv with the lights off"  
  real = **anger**, predicho = **fear** (confianza media 0.73; falla en 3/3 semillas)
- "when a friend didnt invite her own parents to her graduation ceremony"  
  real = **anger**, predicho = **fear** (confianza media 0.30; falla en 3/3 semillas)
- "when i was working as a nurse in a mental hospital"  
  real = **anger**, predicho = **fear** (confianza media 0.80; falla en 3/3 semillas)
- "i would never wish harm to come to anyone that i like if i don t like then they re in trouble just kidding i don t really want anyone to get lost or do i mu hahahahahahahahahahahahahahaha i guess you ll never know [...]"  
  real = **anger**, predicho = **fear** (confianza media 0.37; falla en 3/3 semillas)

### sadness → anger

- "i had a quarrell with my boyfriend and decided to finish with him and and my easter holidays"  
  real = **sadness**, predicho = **anger** (confianza media 0.45; falla en 3/3 semillas)
- "when the neighbour of a friend of mine was stabbed to death by thieves in kabwe they stole his van"  
  real = **sadness**, predicho = **anger** (confianza media 0.79; falla en 3/3 semillas)
- "when the train taking me to my military service left the station"  
  real = **sadness**, predicho = **anger** (confianza media 0.78; falla en 3/3 semillas)
- "when someone i love told me that she did not love me anymore and that she wanted all the snaps that she had given me in exchange for the ones i had given her"  
  real = **sadness**, predicho = **anger** (confianza media 0.80; falla en 3/3 semillas)

### fear → surprise

- "i have a habit of examining the ends of my hair with harsh scrutiny whenever i feel mentally and or emotionally and or creatively overwhelmed"  
  real = **fear**, predicho = **surprise** (confianza media 0.78; falla en 3/3 semillas)
- "i often tell my first timers that it is going to feel overwhelmed the first time and probably a little lost and to at least give me two classes before they decided on whether or not this is for them"  
  real = **fear**, predicho = **surprise** (confianza media 0.91; falla en 3/3 semillas)
- "i feel slightly overwhelmed but i know that i must tell"  
  real = **fear**, predicho = **surprise** (confianza media 0.92; falla en 3/3 semillas)
- "i am feeling overwhelmed you can catch me opening the pantry and admiring my own organization skills"  
  real = **fear**, predicho = **surprise** (confianza media 0.97; falla en 3/3 semillas)

### Errores persistentes de mayor confianza

Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.

- "i had to leave to work feeling supermegapissedoff and lethargic but i found solace in the awesome traffic while everyone else is still away from the city"  
  real = **sadness**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i have been stuffed in a cafe all day feeling some what costaphobic and lost"  
  real = **sadness**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)
- "when i was a child i was playing on the upper deck of the bed with my brother my parents slept on the lower deck my father felt disgusted and beat me up i felt very sad as my father had never beaten me or scolded [...]"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "when i received my report card in cou i was afraid of obtaining low marks"  
  real = **joy**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)
- "i still have a fever and i m still feeling pretty pissy"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i feel like getting real snarky here but the other part of me just feels awful for lavolo"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "as the time to get the results drew near i became more and more anxious my friend called me and said"  
  real = **joy**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)

