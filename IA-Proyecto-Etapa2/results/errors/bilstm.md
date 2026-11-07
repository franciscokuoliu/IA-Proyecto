# Análisis de errores: BiLSTM + GloVe

> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, 37,153 textos, semillas [42, 1337, 2026]). Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.

## 1. Resumen

| Semilla | Errores | Tasa de error | Macro-F1 recalculado | vs metrics.json |
|---|---|---|---|---|
| 42 | 616 | 1.66% | 0.9812 | coincide |
| 1337 | 621 | 1.67% | 0.9792 | coincide |
| 2026 | 834 | 2.24% | 0.9756 | coincide |

- Textos mal clasificados en **todas** las semillas (persistentes): 249 (0.67% del test).
- Textos mal clasificados en **una sola** semilla: 796. Un error que aparece solo en una semilla suele ser ruido del entrenamiento; uno persistente apunta al dato (ambigüedad o etiqueta dudosa) o a una limitación del modelo.
- Errores totales (suma de semillas): 2,071; de ellos, en textos persistentes: 36.1%.

## 2. Por clase (semillas sumadas)

| Clase | Soporte (test) | Recall | Precisión | Predichas / reales |
|---|---|---|---|---|
| sadness | 11,604 | 98.5% | 98.1% | 1.00 |
| joy | 12,937 | 98.0% | 99.1% | 0.99 |
| love | 2,438 | 98.9% | 98.4% | 1.01 |
| anger | 5,233 | 98.0% | 96.5% | 1.02 |
| fear | 3,958 | 97.1% | 97.9% | 0.99 |
| surprise | 983 | 98.4% | 95.5% | 1.03 |

- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.

## 3. Pares de confusión más frecuentes (top 5)

| Real → Predicho | Errores | % de la clase real | % de todos los errores |
|---|---|---|---|
| joy → sadness | 378 | 0.97% | 18.3% |
| joy → anger | 213 | 0.55% | 10.3% |
| sadness → anger | 210 | 0.60% | 10.1% |
| sadness → joy | 184 | 0.53% | 8.9% |
| anger → sadness | 143 | 0.91% | 6.9% |

## 4. Confianza de la predicción

| Probabilidad de la clase predicha | Predicciones | Tasa de error |
|---|---|---|
| < 0.50 | 874 | 59.38% |
| 0.50 - 0.70 | 1,235 | 44.05% |
| 0.70 - 0.90 | 1,701 | 25.22% |
| 0.90 - 0.99 | 4,769 | 8.09% |
| >= 0.99 | 102,880 | 0.19% |

- Errores con confianza >= 0.90: 28.0% de los errores. Confianza media: aciertos 0.992, errores 0.692.

## 5. Longitud del texto (palabras)

| Palabras | Textos (test) | Tasa de error |
|---|---|---|
| <= 11 | 10,536 | 1.66% |
| 12 - 17 | 8,680 | 1.52% |
| 18 - 25 | 8,825 | 1.86% |
| > 25 | 9,112 | 2.40% |

- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el truncamiento a `max_len` o la dilución de la señal importan.

## 6. Tipo de texto

| Subconjunto | Textos (test) | % del test | Tasa de error | % de los errores típicos del modelo |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 98.49% | 1.18% | 50.3% |
| No contiene feel | 562 | 1.51% | 46.32% | 49.7% |
| Empieza con "when" | 168 | 0.45% | 54.37% | 17.7% |

- Casi todo el conjunto son tweets de estilo CARER ("i feel ..."). Los textos sin "feel" y los que empiezan con "when" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.

## 7. Ejemplos

Elegidos al azar (semilla 42) entre los textos que el modelo falla en la mayoría de las semillas, preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.

### joy → sadness

- "i received a letter from a distant friend"  
  real = **joy**, predicho = **sadness** (confianza media 0.27; falla en 3/3 semillas)
- "i feel like julius casear in bed with the gorgeous cleopatra right now watching the sunrise over the nile and pyramids naked in bed and watching you eat your grapes so seductively"  
  real = **joy**, predicho = **sadness** (confianza media 0.51; falla en 3/3 semillas)
- "i feel like david against goliath accused bishop says title share on delicious rel nofollow target blank img src http www"  
  real = **joy**, predicho = **sadness** (confianza media 0.99; falla en 3/3 semillas)
- "i feel sympathy for jonas the one i atone many ask for release but well be left helpless because now we see it doesnt bring peace only sadness"  
  real = **joy**, predicho = **sadness** (confianza media 0.98; falla en 3/3 semillas)

### joy → anger

- "talking to a close friend"  
  real = **joy**, predicho = **anger** (confianza media 0.39; falla en 3/3 semillas)
- "when my nephew was born the birth was hard"  
  real = **joy**, predicho = **anger** (confianza media 0.43; falla en 3/3 semillas)
- "when my boyfriend once again told me that he loved me after a period in which he felt very depressive"  
  real = **joy**, predicho = **anger** (confianza media 0.32; falla en 3/3 semillas)
- "i was in a restaurant with my boyfriend when i saw my best female friend after not seeing her for two weeks we rejoyced"  
  real = **joy**, predicho = **anger** (confianza media 0.45; falla en 3/3 semillas)

### sadness → anger

- "when out of work"  
  real = **sadness**, predicho = **anger** (confianza media 0.38; falla en 3/3 semillas)
- "when my mother scolded me for something"  
  real = **sadness**, predicho = **anger** (confianza media 0.72; falla en 3/3 semillas)
- "there was a conflict of interest between me and a close friend of mine he only thought about his own interest and not about our friendship this behaviour hurt his friends another incident is when a friend isolated [...]"  
  real = **sadness**, predicho = **anger** (confianza media 0.68; falla en 3/3 semillas)
- "it was when i failed standard exams while all my friends passed and went to secondary school in class i was always getting more marks than they did"  
  real = **sadness**, predicho = **anger** (confianza media 0.30; falla en 3/3 semillas)

### sadness → joy

- "i know how important it is to listen to your body but tomorrow is a cardio recovery whatever that means so i figured if i feel like death tomorrow it hopefully wont be as bad of workout"  
  real = **sadness**, predicho = **joy** (confianza media 0.95; falla en 3/3 semillas)
- "i dont feel anything anymore"  
  real = **sadness**, predicho = **joy** (confianza media 0.68; falla en 3/3 semillas)
- "i thought i would never find keep or feel praise for the other survivors who insure i no longer walk this path alone praise for two sisters who chose me as i chose them and love me in the manner of any family the [...]"  
  real = **sadness**, predicho = **joy** (confianza media 0.97; falla en 3/3 semillas)
- "i feel in my legs are worth it besides of course the perverse satisfaction i feel on a personal level"  
  real = **sadness**, predicho = **joy** (confianza media 0.62; falla en 3/3 semillas)

### anger → sadness

- "i was exhausted and i had that headache and scratchy throat feeling that usually portends the onset of a cold"  
  real = **anger**, predicho = **sadness** (confianza media 0.43; falla en 2/3 semillas)
- "doctors ignored relative who complained about pain"  
  real = **anger**, predicho = **sadness** (confianza media 0.61; falla en 2/3 semillas)
- "receiving personal criticisms early in the morning"  
  real = **anger**, predicho = **sadness** (confianza media 0.34; falla en 2/3 semillas)
- "i had a discussion with my mother concerning my sisters divorce"  
  real = **anger**, predicho = **sadness** (confianza media 0.36; falla en 2/3 semillas)

### Errores persistentes de mayor confianza

Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.

- "i do at least feel reasonably rejuvenated after a rather lovely week followed by an equally lovely weekend so im cautiously confident that ill be able to hit my targets this week"  
  real = **love**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "i really the psychopath i feel like sometimes or the well spoken compassionate man i try to be or at least to look like"  
  real = **joy**, predicho = **love** (confianza media 1.00; falla en 3/3 semillas)
- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, predicho = **anger** (confianza media 1.00; falla en 3/3 semillas)
- "i am feeling much better quite drained and a bit of a headache but so glad to be home"  
  real = **joy**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "i feel like my free lonely spaces are being filled up"  
  real = **joy**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "i was feeling chris was underwhelming in this role but as the movie went on he shined in the vulnerable moments"  
  real = **fear**, predicho = **sadness** (confianza media 1.00; falla en 3/3 semillas)
- "i have a feeling that all i really did was guarantee that this kid gets the crap beaten out of him this afternoon"  
  real = **sadness**, predicho = **joy** (confianza media 1.00; falla en 3/3 semillas)
- "im not just a person who has been through a lot lately but i feel that i am truthful and extremely loyal to those i call my friends"  
  real = **joy**, predicho = **love** (confianza media 1.00; falla en 3/3 semillas)

