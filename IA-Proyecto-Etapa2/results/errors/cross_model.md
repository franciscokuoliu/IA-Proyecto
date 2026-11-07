# Errores compartidos entre modelos

> Generado por `scripts/error_analysis.py`. Modelos: BERT-base, BiLSTM + GloVe, DeepSeek-R1-Distill-1.5B, DistilBERT, Gemma 4 E2B, Qwen2.5-0.5B, Qwen2.5-1.5B, TF-IDF + LogReg. Textos en común: 37,153. Un texto cuenta como error de un modelo si lo falla en la mayoría de sus semillas.

## 1. Errores típicos por modelo

| Modelo | Errores típicos | Tasa | Solo de este modelo | % de sus errores que nadie más comete |
|---|---|---|---|---|
| BERT-base | 258 | 0.69% | 14 | 5.4% |
| BiLSTM + GloVe | 513 | 1.38% | 159 | 31.0% |
| DeepSeek-R1-Distill-1.5B | 335 | 0.90% | 25 | 7.5% |
| DistilBERT | 280 | 0.75% | 12 | 4.3% |
| Gemma 4 E2B | 1,375 | 3.70% | 1,101 | 80.1% |
| Qwen2.5-0.5B | 325 | 0.87% | 16 | 4.9% |
| Qwen2.5-1.5B | 270 | 0.73% | 6 | 2.2% |
| TF-IDF + LogReg | 1,620 | 4.36% | 1,246 | 76.9% |

- Errores que cometen **todos** los modelos: 64 (0.17% del test).
- Un error que solo comete un modelo no es culpa del dato: es una limitación de ese modelo o de su entrenamiento. Un error que cometen todos apunta al dato.

## 2. Solapamiento de errores (Jaccard)

|  | BERT-base | BiLSTM + GloVe | DeepSeek-R1-Distill-1.5B | DistilBERT | Gemma 4 E2B | Qwen2.5-0.5B | Qwen2.5-1.5B | TF-IDF + LogReg |
|---|---|---|---|---|---|---|---|---|
| BERT-base | 1.00 | 0.25 | 0.42 | 0.62 | 0.09 | 0.40 | 0.46 | 0.09 |
| BiLSTM + GloVe | 0.25 | 1.00 | 0.30 | 0.28 | 0.08 | 0.26 | 0.28 | 0.15 |
| DeepSeek-R1-Distill-1.5B | 0.42 | 0.30 | 1.00 | 0.44 | 0.10 | 0.44 | 0.54 | 0.10 |
| DistilBERT | 0.62 | 0.28 | 0.44 | 1.00 | 0.10 | 0.43 | 0.47 | 0.10 |
| Gemma 4 E2B | 0.09 | 0.08 | 0.10 | 0.10 | 1.00 | 0.13 | 0.11 | 0.06 |
| Qwen2.5-0.5B | 0.40 | 0.26 | 0.44 | 0.43 | 0.13 | 1.00 | 0.48 | 0.09 |
| Qwen2.5-1.5B | 0.46 | 0.28 | 0.54 | 0.47 | 0.11 | 0.48 | 1.00 | 0.09 |
| TF-IDF + LogReg | 0.09 | 0.15 | 0.10 | 0.10 | 0.06 | 0.09 | 0.09 | 1.00 |

## 3. Todos los modelos coinciden en la misma clase equivocada

Textos: 22. **Hipótesis**: etiqueta dudosa o texto ambiguo; revisar a mano.

- "i feel fear quite often but this feeling was very strong last year when my husband was quite unexpectedly sent to the hospital"  
  real = **fear**, todos predicen **joy** (confianza media 0.94)
- "when i received my report card in cou i was afraid of obtaining low marks"  
  real = **joy**, todos predicen **fear** (confianza media 0.93)
- "as the time to get the results drew near i became more and more anxious my friend called me and said"  
  real = **joy**, todos predicen **fear** (confianza media 0.91)
- "when my mother scolded me for something"  
  real = **sadness**, todos predicen **anger** (confianza media 0.91)
- "when i made up with a girlfriend of mine after having been on unfriendly terms with her my fault i cried with joy that evening"  
  real = **joy**, todos predicen **anger** (confianza media 0.87)
- "someone told me i was chosen for the english lectures because i am good looking"  
  real = **anger**, todos predicen **joy** (confianza media 0.87)
- "the day the results of the chinese university was announced i should have been very happy however my grandmother"  
  real = **anger**, todos predicen **joy** (confianza media 0.86)
- "when the neighbour of a friend of mine was stabbed to death by thieves in kabwe they stole his van"  
  real = **sadness**, todos predicen **anger** (confianza media 0.84)

## 4. Dónde caen los errores de consenso

| Subconjunto | Textos (test) | Errores de consenso | % de los errores de consenso | Tasa de error de consenso en el subconjunto |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 9 | 14.1% | 0.02% |
| No contiene feel | 562 | 55 | 85.9% | 9.79% |
| Empieza con "when" | 168 | 22 | 34.4% | 13.10% |

- Si una minoría pequeña del test concentra casi todos los errores de consenso, el techo de los modelos se explica por esa minoría (hipótesis sobre el conjunto de datos) y no por la arquitectura.

## 5. Muestra para auditoría manual

30 textos al azar (semilla 42) entre los 64 errores de consenso (los que fallan todos los modelos, sección 1). Para cada uno, anotar a mano: **A** = etiqueta dudosa o ambigua, **B** = el modelo se equivoca claramente, **C** = no se puede decidir. El conteo A/B/C es la evidencia para hablar (o no) de ruido de etiqueta en el artículo.

- [ ] `06da8db4fdc1404f0ac89351a521986b` real = **joy**, predicciones erradas = **anger / sadness**: "an encounter with a man whom i love"
- [ ] `0a7b8bfaa80773cfd30cfb2d7337015b` real = **anger**, predicciones erradas = **fear / joy**: "in the evening"
- [ ] `122f85e2a9d5e93d3d5fe77f5752f4bc` real = **joy**, predicciones erradas = **anger / fear / sadness**: "when after months of national service i was downgraded and did not have to do the remaining months"
- [ ] `1d52e36665546eee5a4b849361501650` real = **anger**, predicciones erradas = **joy / sadness**: "cant remember having had this feeling"
- [ ] `21a38b749030e267efdee4b360270a63` real = **sadness**, predicciones erradas = **anger / fear / joy**: "the news of the natural calamity"
- [ ] `3e85b764c770113d141b409e6c23ba65` real = **fear**, predicciones erradas = **joy / surprise**: "i was feeling really overwhelmed trying to remember all the names of the members but i think i learned at least names or so so i think within a couple of weeks i will know the members pretty well"
- [ ] `5b9984cbb197ef3256f237243d87a735` real = **fear**, predicciones erradas = **anger / joy / sadness**: "after getting a fail notice for a math class"
- [ ] `65ff10c079271d347423f24ed1d4ec2a` real = **fear**, predicciones erradas = **anger / joy / sadness**: "when i got to know that the friend i am closest to got engaged"
- [ ] `6753c8d4b23aa8e81b0fdefa52b90ef5` real = **fear**, predicciones erradas = **anger / joy**: "when i heard that my father had found a letter from my boyfriend and he was calling me to talk about it"
- [ ] `725cc978e337bf0229b93d98dcb8d506` real = **sadness**, predicciones erradas = **anger / joy**: "when i received the syjc results"
- [ ] `7548ca0d1154001173cecc9670e2be80` real = **fear**, predicciones erradas = **anger**: "somebody told her friend that she would hit me wherever we would meet"
- [ ] `7b692e990a5d4db2bb2f31bea7a21480` real = **fear**, predicciones erradas = **anger / joy / sadness**: "when i have to leave the baby in the carriage and go shopping"
- [ ] `7e0ef442d066df921028555d11901271` real = **joy**, predicciones erradas = **anger / sadness**: "relatives visiting me"
- [ ] `9d4a539de0664e4791458847efa77b5a` real = **fear**, predicciones erradas = **anger**: "i got a very nasty electrical shock when i was tampering with some electrical applainces"
- [ ] `a36c6001a49d2be364452ddbdc15f340` real = **joy**, predicciones erradas = **anger / sadness**: "when my elder brother"
- [ ] `a457fd15b47fa53d3470b4c41c34b5df` real = **sadness**, predicciones erradas = **anger**: "when my mother scolded me for something"
- [ ] `ab7311f17a8db99ac2074c6198762224` real = **joy**, predicciones erradas = **anger**: "when i made up with a girlfriend of mine after having been on unfriendly terms with her my fault i cried with joy that evening"
- [ ] `ac3d0fe1bf7437fcb2d77e1e9db4fe5f` real = **fear**, predicciones erradas = **surprise**: "i am feeling overwhelmed that the ms is causing such significant and crippling symptoms"
- [ ] `ad91722091668788942b0366730689bd` real = **sadness**, predicciones erradas = **anger / fear**: "i was together with my family one night"
- [ ] `bc9ddc7a3aa91ec3ff9fcf40039be398` real = **sadness**, predicciones erradas = **anger / joy**: "the second time i went to the entrance exam to vsp and didnt get approved"
- [ ] `c4592a3c91044bb84c36e0abf93b96b7` real = **sadness**, predicciones erradas = **anger**: "when the neighbour of a friend of mine was stabbed to death by thieves in kabwe they stole his van"
- [ ] `dc00e704db53bb6044e62ea6c17b94b8` real = **anger**, predicciones erradas = **fear / sadness**: "when i left after the examination to enter the university"
- [ ] `e3395da9f532b82dfa48f70af20501e5` real = **fear**, predicciones erradas = **anger**: "when i did not get the compound which is known"
- [ ] `e6c64b66bed95f5f3e207ad698615617` real = **fear**, predicciones erradas = **anger / joy / sadness**: "i have not felt this emotion yet"
- [ ] `e934f358ca312779f83e14ec7092f3b4` real = **anger**, predicciones erradas = **fear**: "i was sitting in school"
- [ ] `f3441abb1df14ad6923ff7565336c5e3` real = **joy**, predicciones erradas = **fear**: "when i received my report card in cou i was afraid of obtaining low marks"
- [ ] `f4c63fd3392b4a9f5246faffeb362a2b` real = **sadness**, predicciones erradas = **anger / joy**: "not winning a medal in the swimming contest"
- [ ] `f71cbe2c750d4fdc4616093df558c23a` real = **sadness**, predicciones erradas = **anger / joy**: "i met again a very good friend after a year and realized that there had grown a certain distance between us"
- [ ] `f7649ebc8be3754f10fce3d893347c87` real = **sadness**, predicciones erradas = **anger / joy**: "when i heard that my girlfriend was getting married to another person"
- [ ] `fba8fc249d4dde103015152a90ebb2a3` real = **anger**, predicciones erradas = **fear / sadness**: "my mother carried on using mental pressure concerning something which was serious and a real burden to me"

