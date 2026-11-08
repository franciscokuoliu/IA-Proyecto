# Auditoría manual de errores de consenso

Copia fija de las 30 calificaciones. `cross_model.md` se regenera con `scripts/error_analysis.py` y pisaría las marcas hechas a mano; este archivo no se regenera.

**Muestra.** 30 textos al azar (semilla 42) entre los 64 errores de consenso (los que fallan los ocho modelos).

**Criterio.** A = etiqueta dudosa o ambigua dado el texto. B = etiqueta plausible y los modelos se equivocan claramente. C = no se puede decidir.

**Conteo.** A = 11 (37%), B = 15 (50%), C = 4 (13%).

**Cómo se hizo.** Claude hizo una primera pasada con una razón por texto. Danielo calificó los 30 después de verla y sus calificaciones coinciden con las de Claude en los 30. Es una sola persona calificando, sin acuerdo entre anotadores, y el artículo lo declara como límite.

| # | id | real | predicciones erradas | voto | texto |
|---|---|---|---|---|---|
| 1 | `06da8db4fdc1404f0ac89351a521986b` | joy | anger / sadness | B | an encounter with a man whom i love |
| 2 | `0a7b8bfaa80773cfd30cfb2d7337015b` | anger | fear / joy | A | in the evening |
| 3 | `122f85e2a9d5e93d3d5fe77f5752f4bc` | joy | anger / fear / sadness | B | when after months of national service i was downgraded and did not have to do the remaining months |
| 4 | `1d52e36665546eee5a4b849361501650` | anger | joy / sadness | A | cant remember having had this feeling |
| 5 | `21a38b749030e267efdee4b360270a63` | sadness | anger / fear / joy | B | the news of the natural calamity |
| 6 | `3e85b764c770113d141b409e6c23ba65` | fear | joy / surprise | B | i was feeling really overwhelmed trying to remember all the names of the members but i think i learned at least names or so so i think within a couple of weeks i will know the members pretty well |
| 7 | `5b9984cbb197ef3256f237243d87a735` | fear | anger / joy / sadness | A | after getting a fail notice for a math class |
| 8 | `65ff10c079271d347423f24ed1d4ec2a` | fear | anger / joy / sadness | A | when i got to know that the friend i am closest to got engaged |
| 9 | `6753c8d4b23aa8e81b0fdefa52b90ef5` | fear | anger / joy | B | when i heard that my father had found a letter from my boyfriend and he was calling me to talk about it |
| 10 | `725cc978e337bf0229b93d98dcb8d506` | sadness | anger / joy | C | when i received the syjc results |
| 11 | `7548ca0d1154001173cecc9670e2be80` | fear | anger | B | somebody told her friend that she would hit me wherever we would meet |
| 12 | `7b692e990a5d4db2bb2f31bea7a21480` | fear | anger / joy / sadness | B | when i have to leave the baby in the carriage and go shopping |
| 13 | `7e0ef442d066df921028555d11901271` | joy | anger / sadness | C | relatives visiting me |
| 14 | `9d4a539de0664e4791458847efa77b5a` | fear | anger | B | i got a very nasty electrical shock when i was tampering with some electrical applainces |
| 15 | `a36c6001a49d2be364452ddbdc15f340` | joy | anger / sadness | C | when my elder brother |
| 16 | `a457fd15b47fa53d3470b4c41c34b5df` | sadness | anger | A | when my mother scolded me for something |
| 17 | `ab7311f17a8db99ac2074c6198762224` | joy | anger | B | when i made up with a girlfriend of mine after having been on unfriendly terms with her my fault i cried with joy that evening |
| 18 | `ac3d0fe1bf7437fcb2d77e1e9db4fe5f` | fear | surprise | B | i am feeling overwhelmed that the ms is causing such significant and crippling symptoms |
| 19 | `ad91722091668788942b0366730689bd` | sadness | anger / fear | A | i was together with my family one night |
| 20 | `bc9ddc7a3aa91ec3ff9fcf40039be398` | sadness | anger / joy | B | the second time i went to the entrance exam to vsp and didnt get approved |
| 21 | `c4592a3c91044bb84c36e0abf93b96b7` | sadness | anger | B | when the neighbour of a friend of mine was stabbed to death by thieves in kabwe they stole his van |
| 22 | `dc00e704db53bb6044e62ea6c17b94b8` | anger | fear / sadness | A | when i left after the examination to enter the university |
| 23 | `e3395da9f532b82dfa48f70af20501e5` | fear | anger | C | when i did not get the compound which is known |
| 24 | `e6c64b66bed95f5f3e207ad698615617` | fear | anger / joy / sadness | A | i have not felt this emotion yet |
| 25 | `e934f358ca312779f83e14ec7092f3b4` | anger | fear | A | i was sitting in school |
| 26 | `f3441abb1df14ad6923ff7565336c5e3` | joy | fear | A | when i received my report card in cou i was afraid of obtaining low marks |
| 27 | `f4c63fd3392b4a9f5246faffeb362a2b` | sadness | anger / joy | B | not winning a medal in the swimming contest |
| 28 | `f71cbe2c750d4fdc4616093df558c23a` | sadness | anger / joy | B | i met again a very good friend after a year and realized that there had grown a certain distance between us |
| 29 | `f7649ebc8be3754f10fce3d893347c87` | sadness | anger / joy | B | when i heard that my girlfriend was getting married to another person |
| 30 | `fba8fc249d4dde103015152a90ebb2a3` | anger | fear / sadness | A | my mother carried on using mental pressure concerning something which was serious and a real burden to me |
