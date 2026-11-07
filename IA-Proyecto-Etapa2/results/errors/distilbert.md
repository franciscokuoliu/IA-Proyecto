# Análisis de errores: DistilBERT

> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, 37,153 textos, semillas [42, 1337, 2026]). Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.

## 1. Resumen

| Semilla | Errores | Tasa de error | Macro-F1 recalculado | vs metrics.json |
|---|---|---|---|---|
| 42 | 297 | 0.80% | 0.9889 | coincide |
| 1337 | 294 | 0.79% | 0.9895 | coincide |
| 2026 | 311 | 0.84% | 0.9889 | coincide |

- Textos mal clasificados en **todas** las semillas (persistentes): 152 (0.41% del test).
- Textos mal clasificados en **una sola** semilla: 190. Un error que aparece solo en una semilla suele ser ruido del entrenamiento; uno persistente apunta al dato (ambigüedad o etiqueta dudosa) o a una limitación del modelo.
- Errores totales (suma de semillas): 902; de ellos, en textos persistentes: 50.6%.

## 2. Por clase (semillas sumadas)

| Clase | Soporte (test) | Recall | Precisión | Predichas / reales |
|---|---|---|---|---|
| sadness | 11,604 | 99.5% | 99.3% | 1.00 |
| joy | 12,937 | 99.4% | 99.6% | 1.00 |
| love | 2,438 | 99.6% | 99.4% | 1.00 |
| anger | 5,233 | 98.8% | 98.4% | 1.00 |
| fear | 3,958 | 98.0% | 98.9% | 0.99 |
| surprise | 983 | 98.3% | 97.7% | 1.01 |

- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.

## 3. Pares de confusión más frecuentes (top 5)

| Real → Predicho | Errores | % de la clase real | % de todos los errores |
|---|---|---|---|
| joy → sadness | 97 | 0.25% | 10.8% |
| fear → anger | 88 | 0.74% | 9.8% |
| anger → sadness | 84 | 0.54% | 9.3% |
| sadness → anger | 78 | 0.22% | 8.6% |
| joy → anger | 70 | 0.18% | 7.8% |

## 4. Confianza de la predicción

| Probabilidad de la clase predicha | Predicciones | Tasa de error |
|---|---|---|
| < 0.50 | 90 | 73.33% |
| 0.50 - 0.70 | 322 | 54.35% |
| 0.70 - 0.90 | 438 | 46.58% |
| 0.90 - 0.99 | 670 | 31.34% |
| >= 0.99 | 109,939 | 0.22% |

- Errores con confianza >= 0.90: 50.7% de los errores. Confianza media: aciertos 0.999, errores 0.824.

## 5. Longitud del texto (palabras)

| Palabras | Textos (test) | Tasa de error |
|---|---|---|
| <= 11 | 10,536 | 0.96% |
| 12 - 17 | 8,680 | 0.66% |
| 18 - 25 | 8,825 | 0.68% |
| > 25 | 9,112 | 0.90% |

- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el truncamiento a `max_len` o la dilución de la señal importan.

## 6. Tipo de texto

| Subconjunto | Textos (test) | % del test | Tasa de error | % de los errores típicos del modelo |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 98.49% | 0.37% | 40.7% |
| No contiene feel | 562 | 1.51% | 29.18% | 59.3% |
| Empieza con "when" | 168 | 0.45% | 30.75% | 19.6% |

- Casi todo el conjunto son tweets de estilo CARER ("i feel ..."). Los textos sin "feel" y los que empiezan con "when" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.

## 7. Ejemplos

Elegidos al azar (semilla 42) entre los textos que el modelo falla en la mayoría de las semillas, preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.

### joy → sadness

- "i met the girls at the chalet but i didnt take pictures w them played games like truth or dare and got to know a lot of secrets hehe it was a little awkward for me at first because i havent seen them for quite some [...]"  
  real = **joy**, predicho = **sadness** (confianza media 0.86; falla en 3/3 semillas)
- "an encounter with a man whom i love"  
  real = **joy**, predicho = **sadness** (confianza media 0.55; falla en 3/3 semillas)
- "i feel in some way that this regressed step has appeared as a means to underline a lesson i have yet to gain something with intelligent intent"  
  real = **joy**, predicho = **sadness** (confianza media 0.89; falla en 3/3 semillas)
- "relating to my closest family"  
  real = **joy**, predicho = **sadness** (confianza media 0.71; falla en 3/3 semillas)

### fear → anger

- "at the dentists"  
  real = **fear**, predicho = **anger** (confianza media 0.92; falla en 3/3 semillas)
- "in an aeroplane"  
  real = **fear**, predicho = **anger** (confianza media 0.58; falla en 3/3 semillas)
- "earth crake"  
  real = **fear**, predicho = **anger** (confianza media 0.82; falla en 3/3 semillas)
- "when i have to leave the baby in the carriage and go shopping"  
  real = **fear**, predicho = **anger** (confianza media 0.75; falla en 3/3 semillas)

### anger → sadness

- "when i heard the bad news about the death of a close person"  
  real = **anger**, predicho = **sadness** (confianza media 0.91; falla en 3/3 semillas)
- "a friend of mine came to visit and stayed with my roommate and me my roommate and she became good friends after she left"  
  real = **anger**, predicho = **sadness** (confianza media 0.57; falla en 3/3 semillas)
- "when i left after the examination to enter the university"  
  real = **anger**, predicho = **sadness** (confianza media 0.65; falla en 3/3 semillas)
- "not being alowed on a school camp with the class id been in for years i had stayed down due to illness"  
  real = **anger**, predicho = **sadness** (confianza media 0.84; falla en 3/3 semillas)

### sadness → anger

- "when my boyfriend said he did not want to see me anymore which turned out to be untrue"  
  real = **sadness**, predicho = **anger** (confianza media 0.99; falla en 3/3 semillas)
- "when my best friend promised to come at my house and i had to prepare so many expensive things for her but only to learn at last that she would not turn up"  
  real = **sadness**, predicho = **anger** (confianza media 0.95; falla en 3/3 semillas)
- "the news of the natural calamity"  
  real = **sadness**, predicho = **anger** (confianza media 0.28; falla en 3/3 semillas)
- "when i was a child i was playing on the upper deck of the bed with my brother my parents slept on the lower deck my father felt disgusted and beat me up i felt very sad as my father had never beaten me or scolded [...]"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)

### joy → anger

- "relatives visiting me"  
  real = **joy**, predicho = **anger** (confianza media 0.66; falla en 3/3 semillas)
- "during lectures"  
  real = **joy**, predicho = **anger** (confianza media 0.91; falla en 3/3 semillas)
- "a fellow player"  
  real = **joy**, predicho = **anger** (confianza media 0.79; falla en 3/3 semillas)
- "when my elder brother"  
  real = **joy**, predicho = **anger** (confianza media 0.79; falla en 3/3 semillas)

### Errores persistentes de mayor confianza

Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.

- "i feel fear quite often but this feeling was very strong last year when my husband was quite unexpectedly sent to the hospital"  
  real = **fear**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i still have a fever and i m still feeling pretty pissy"  
  real = **joy**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "when i was a child i was playing on the upper deck of the bed with my brother my parents slept on the lower deck my father felt disgusted and beat me up i felt very sad as my father had never beaten me or scolded [...]"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i wasnt really feeling like being snarky or funny"  
  real = **surprise**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "when i received my report card in cou i was afraid of obtaining low marks"  
  real = **joy**, predicho = **fear** (confianza media 1.00; falla en 3/3 semillas)
- "i am mostly feeling contentedly terrified about it all"  
  real = **fear**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i think the blood had rushed to her head and she was thinking weeeeeeeeeeeeeeeeeeeeeeeeeeeeeee this is faaaaaaaaaaaaaaaaaaaaaaaaaast and im feeling dizzzzzzzzzzzzzzzzzzzzzzzzzzzy but she still had lovely flourishes"  
  real = **love**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)

