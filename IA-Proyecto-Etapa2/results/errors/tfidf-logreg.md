# Análisis de errores: TF-IDF + LogReg

> Generado por `scripts/error_analysis.py` a partir de `predictions.csv` (test, 37,153 textos, semillas [42]). Los datos son hechos medidos; las lecturas marcadas como hipótesis requieren revisión manual.

## 1. Resumen

| Semilla | Errores | Tasa de error | Macro-F1 recalculado | vs metrics.json |
|---|---|---|---|---|
| 42 | 1,620 | 4.36% | 0.9493 | coincide |

- Textos mal clasificados en **todas** las semillas (persistentes): 1,620 (4.36% del test).

## 2. Por clase (semillas sumadas)

| Clase | Soporte (test) | Recall | Precisión | Predichas / reales |
|---|---|---|---|---|
| sadness | 11,604 | 95.6% | 96.6% | 0.99 |
| joy | 12,937 | 96.3% | 96.6% | 1.00 |
| love | 2,438 | 96.7% | 92.4% | 1.05 |
| anger | 5,233 | 94.3% | 94.9% | 0.99 |
| fear | 3,958 | 94.3% | 94.8% | 1.00 |
| surprise | 983 | 98.1% | 89.3% | 1.10 |

- `Predichas / reales` > 1 significa que el modelo usa esa clase de más (sus errores desembocan ahí); < 1, de menos.

## 3. Pares de confusión más frecuentes (top 5)

| Real → Predicho | Errores | % de la clase real | % de todos los errores |
|---|---|---|---|
| sadness → joy | 201 | 1.73% | 12.4% |
| joy → sadness | 184 | 1.42% | 11.4% |
| sadness → anger | 130 | 1.12% | 8.0% |
| anger → sadness | 123 | 2.35% | 7.6% |
| anger → joy | 107 | 2.04% | 6.6% |

## 4. Confianza de la predicción

| Probabilidad de la clase predicha | Predicciones | Tasa de error |
|---|---|---|
| < 0.50 | 1,785 | 35.18% |
| 0.50 - 0.70 | 4,767 | 13.13% |
| 0.70 - 0.90 | 11,404 | 2.53% |
| 0.90 - 0.99 | 13,991 | 0.49% |
| >= 0.99 | 5,206 | 0.17% |

- Errores con confianza >= 0.90: 4.8% de los errores. Confianza media: aciertos 0.861, errores 0.573.

## 5. Longitud del texto (palabras)

| Palabras | Textos (test) | Tasa de error |
|---|---|---|
| <= 11 | 10,536 | 2.23% |
| 12 - 17 | 8,680 | 3.06% |
| 18 - 25 | 8,825 | 4.00% |
| > 25 | 9,112 | 8.41% |

- Los rangos son los cuartiles de longitud del test. Una tasa de error que sube con la longitud sugeriría que el truncamiento a `max_len` o la dilución de la señal importan.

## 6. Tipo de texto

| Subconjunto | Textos (test) | % del test | Tasa de error | % de los errores típicos del modelo |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 98.49% | 3.74% | 84.4% |
| No contiene feel | 562 | 1.51% | 45.02% | 15.6% |
| Empieza con "when" | 168 | 0.45% | 47.62% | 4.9% |

- Casi todo el conjunto son tweets de estilo CARER ("i feel ..."). Los textos sin "feel" y los que empiezan con "when" (frases de otro estilo) son una minoría; si concentran una parte desproporcionada de los errores, el límite del modelo está en esa minoría. Es una observación sobre el conjunto de datos, no una prueba de su origen.

## 7. Ejemplos

Elegidos al azar (semilla 42) entre los textos que el modelo falla en la mayoría de las semillas, preferiendo los persistentes. Los textos están en minúsculas y sin puntuación, como en el conjunto limpio.

### sadness → joy

- "im a good person and my life will be all right are broken the inability to feel anything but despair doesnt actually prove anything about you except that youre feeling despairing"  
  real = **sadness**, predicho = **joy** (confianza media 0.41; falla en 1/1 semillas)
- "i end the day feeling exhausted and productive and while i might not be jazzed at the idea of waking up at am its hella more peaceful than how i used to feel when laying my head down in istanbul"  
  real = **sadness**, predicho = **joy** (confianza media 0.66; falla en 1/1 semillas)
- "i feel like she is doing something just to please me i have to remember that it is what she wants and that i can find my submissive joy in that"  
  real = **sadness**, predicho = **joy** (confianza media 0.39; falla en 1/1 semillas)
- "i wrote down ways i think i could be more happy and feel more joy and then broke it down into three steps"  
  real = **sadness**, predicho = **joy** (confianza media 0.56; falla en 1/1 semillas)

### joy → sadness

- "im sorry that i feel like im being judged when someone makes an innocent comment im sorry that i feel like i cant show my face to tuckers parents again because they probably think little of me too"  
  real = **joy**, predicho = **sadness** (confianza media 0.36; falla en 1/1 semillas)
- "i am feeling relieved and at the same time feeling bad of being lost thinking a lot of what other will start thinking and gossiping"  
  real = **joy**, predicho = **sadness** (confianza media 0.55; falla en 1/1 semillas)
- "finding out my grades at the end of my freshman year"  
  real = **joy**, predicho = **sadness** (confianza media 0.33; falla en 1/1 semillas)
- "one of the greatest joiy was when i succeded in acquiring a motorcycle through my own efforts"  
  real = **joy**, predicho = **sadness** (confianza media 0.39; falla en 1/1 semillas)

### sadness → anger

- "i am feeling very gloomy ultra moody and very much irritated"  
  real = **sadness**, predicho = **anger** (confianza media 0.84; falla en 1/1 semillas)
- "i royally pissed off because im an asshole and have terrible communication skills i was clear up front about not having time to check every day but feelings were hurt and im sorry about that obviously its not a match"  
  real = **sadness**, predicho = **anger** (confianza media 0.53; falla en 1/1 semillas)
- "i begin to feel guilty and guiltier of spending free money on selfish things"  
  real = **sadness**, predicho = **anger** (confianza media 0.75; falla en 1/1 semillas)
- "i feel shamed and insulted exactly as if they had called me a gorilla"  
  real = **sadness**, predicho = **anger** (confianza media 0.68; falla en 1/1 semillas)

### anger → sadness

- "i meltdown today just feel so irritable feel so alone and bored always bored"  
  real = **anger**, predicho = **sadness** (confianza media 0.62; falla en 1/1 semillas)
- "i get to feeling this way i get very irritable sometimes i lash out at people for stupid things stupid reasons"  
  real = **anger**, predicho = **sadness** (confianza media 0.66; falla en 1/1 semillas)
- "when one of my parents verbally"  
  real = **anger**, predicho = **sadness** (confianza media 0.65; falla en 1/1 semillas)
- "i want him to feel like i do disgusted with himself ashamed of how he looks and heartbroken at the thought that maybe his not good enough"  
  real = **anger**, predicho = **sadness** (confianza media 0.69; falla en 1/1 semillas)

### anger → joy

- "the day the results of the chinese university was announced i should have been very happy however my grandmother"  
  real = **anger**, predicho = **joy** (confianza media 0.63; falla en 1/1 semillas)
- "i might have enjoyed it more but i was feeling very grumpy on account of not feeling well"  
  real = **anger**, predicho = **joy** (confianza media 0.67; falla en 1/1 semillas)
- "i know how to control my emotions whenever i feel extremely furious or happy"  
  real = **anger**, predicho = **joy** (confianza media 0.46; falla en 1/1 semillas)
- "i almost always seven to twelve that row every time i see beside me in the penalty mentor called reciprocal achievements of the students i feel very disgusted they just did not test well try in other respects are [...]"  
  real = **anger**, predicho = **joy** (confianza media 0.51; falla en 1/1 semillas)

### Errores persistentes de mayor confianza

Todas las semillas dan la misma clase equivocada con confianza alta. **Hipótesis**: la etiqueta del conjunto es dudosa o el texto es ambiguo. Hay que leerlos y decidir a mano; no se asume ruido de etiqueta sin revisión.

- "i turned fifty eight years ago suddenly i felt hot not a normal hot its that sweaty hot sticky hot horrible hot that makes one feel disgusted about life"  
  real = **anger**, predicho = **love** (confianza media 1.00; falla en 1/1 semillas)
- "i feel super hurt disappointment emotional sad unhappy feel like crying"  
  real = **joy**, predicho = **sadness** (confianza media 1.00; falla en 1/1 semillas)
- "i have been feeling a little needy a little fearful and little anxious"  
  real = **sadness**, predicho = **fear** (confianza media 1.00; falla en 1/1 semillas)
- "im feeling totally gloomy agitated anxious uneasy restless and stuff"  
  real = **sadness**, predicho = **fear** (confianza media 1.00; falla en 1/1 semillas)
- "i know that when i walk into the magic kingdom i feel what i felt as a child excited surprised curious wonder imagination but mostly"  
  real = **joy**, predicho = **surprise** (confianza media 1.00; falla en 1/1 semillas)
- "im basically sat in the cold crying headphones in feeling so lost helpless angry frustrated that i was late and couldnt seem to time manage today annoyed that im working my ass off to be a good mum but one night [...]"  
  real = **sadness**, predicho = **anger** (confianza media 1.00; falla en 1/1 semillas)
- "i also feel like a prisoner i am mad suspicious and maybe even paranoid"  
  real = **anger**, predicho = **fear** (confianza media 0.99; falla en 1/1 semillas)
- "im still feeling a little bit lonely over there thank you to all loyal readers who liked fine fine books already"  
  real = **sadness**, predicho = **love** (confianza media 0.99; falla en 1/1 semillas)

