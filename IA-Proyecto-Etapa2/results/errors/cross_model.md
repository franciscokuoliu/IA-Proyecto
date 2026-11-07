# Errores compartidos entre modelos

> Generado por `scripts/error_analysis.py`. Modelos: Qwen2.5-1.5B, DeepSeek-R1-Distill-1.5B, Gemma 4 E2B. Textos en común: 37,153. Un texto cuenta como error de un modelo si lo falla en la mayoría de sus semillas.

## 1. Errores típicos por modelo

| Modelo | Errores típicos | Tasa | Solo de este modelo | % de sus errores que nadie más comete |
|---|---|---|---|---|
| Qwen2.5-1.5B | 270 | 0.73% | 32 | 11.9% |
| DeepSeek-R1-Distill-1.5B | 335 | 0.90% | 100 | 29.9% |
| Gemma 4 E2B | 1,375 | 3.70% | 1,192 | 86.7% |

- Errores que cometen **todos** los modelos: 132 (0.36% del test).
- Un error que solo comete un modelo no es culpa del dato: es una limitación de ese modelo o de su entrenamiento. Un error que cometen todos apunta al dato.

## 2. Solapamiento de errores (Jaccard)

|  | Qwen2.5-1.5B | DeepSeek-R1-Distill-1.5B | Gemma 4 E2B |
|---|---|---|---|
| Qwen2.5-1.5B | 1.00 | 0.54 | 0.11 |
| DeepSeek-R1-Distill-1.5B | 0.54 | 1.00 | 0.10 |
| Gemma 4 E2B | 0.11 | 0.10 | 1.00 |

## 3. Todos los modelos coinciden en la misma clase equivocada

Textos: 91. **Hipótesis**: etiqueta dudosa o texto ambiguo; revisar a mano.

- "i had climbed on a cherry tree alone and there was a thick caterpillar beside my fingers i feel disgusted by caterpillars and snakes i was terribly afraid of the caterpillar crawling on my fingers out of the fear i [...]"  
  real = **fear**, todos predicen **anger** (confianza media 1.00)
- "i have been stuffed in a cafe all day feeling some what costaphobic and lost"  
  real = **sadness**, todos predicen **fear** (confianza media 1.00)
- "when i was a child i was playing on the upper deck of the bed with my brother my parents slept on the lower deck my father felt disgusted and beat me up i felt very sad as my father had never beaten me or scolded [...]"  
  real = **sadness**, todos predicen **anger** (confianza media 1.00)
- "i still have a fever and i m still feeling pretty pissy"  
  real = **joy**, todos predicen **anger** (confianza media 1.00)
- "i feel fear quite often but this feeling was very strong last year when my husband was quite unexpectedly sent to the hospital"  
  real = **fear**, todos predicen **joy** (confianza media 1.00)
- "i had to leave to work feeling supermegapissedoff and lethargic but i found solace in the awesome traffic while everyone else is still away from the city"  
  real = **sadness**, todos predicen **joy** (confianza media 1.00)
- "when i received my report card in cou i was afraid of obtaining low marks"  
  real = **joy**, todos predicen **fear** (confianza media 0.99)
- "when my mother scolded me for something"  
  real = **sadness**, todos predicen **anger** (confianza media 0.99)

## 4. Dónde caen los errores de consenso

| Subconjunto | Textos (test) | Errores de consenso | % de los errores de consenso | Tasa de error de consenso en el subconjunto |
|---|---|---|---|---|
| Contiene feel / feeling / felt | 36,591 | 39 | 29.5% | 0.11% |
| No contiene feel | 562 | 93 | 70.5% | 16.55% |
| Empieza con "when" | 168 | 36 | 27.3% | 21.43% |

- Si una minoría pequeña del test concentra casi todos los errores de consenso, el techo de los modelos se explica por esa minoría (hipótesis sobre el conjunto de datos) y no por la arquitectura.

## 5. Muestra para auditoría manual

30 textos al azar (semilla 42) de la sección 3. Para cada uno, anotar a mano: **A** = etiqueta dudosa o ambigua, **B** = el modelo se equivoca claramente, **C** = no se puede decidir. El conteo A/B/C es la evidencia para hablar (o no) de ruido de etiqueta en el artículo.

- [ ] `0a438bdd58296e75389c8b6a00993c1f` real = **joy**, modelos = **fear**: "in southern tyrol"
- [ ] `0a7b8bfaa80773cfd30cfb2d7337015b` real = **anger**, modelos = **fear**: "in the evening"
- [ ] `1d52e36665546eee5a4b849361501650` real = **anger**, modelos = **joy**: "cant remember having had this feeling"
- [ ] `241b2525555718e9243afd811411abfb` real = **fear**, modelos = **joy**: "introducing myself at a first tutorial this year at university"
- [ ] `24c0fd9bd81ccaba59a3c067effe7c55` real = **fear**, modelos = **joy**: "i was on this amusement park ride at marriots great america it went way up in the air and then around and around heights are not my favorite spots in life"
- [ ] `505f234ae9e0d8d7fd7ee925029a60dc` real = **sadness**, modelos = **anger**: "the quarrel between my parents"
- [ ] `509400f1d2804afe1e179b3e5dd7a9d6` real = **sadness**, modelos = **fear**: "alone i am coming back home in the night after the serial explanation"
- [ ] `65591596a28b3cd4a8abf866ca27c9ba` real = **fear**, modelos = **surprise**: "i walk in a room let me rant when im feeling overwhelmed make my ice cream cone first never belittle or berate push me to do new things and laugh and dance with me always"
- [ ] `664b2beccec006e1150514eb04dc85da` real = **fear**, modelos = **surprise**: "i was feeling at my most overwhelmed and burdened meditation is what i turned to help draw me out of that"
- [ ] `675858088e30f829eedf330aa931e0f4` real = **fear**, modelos = **anger**: "a film on zombies"
- [ ] `725cc978e337bf0229b93d98dcb8d506` real = **sadness**, modelos = **joy**: "when i received the syjc results"
- [ ] `7548ca0d1154001173cecc9670e2be80` real = **fear**, modelos = **anger**: "somebody told her friend that she would hit me wherever we would meet"
- [ ] `8e818f18c403677b76bafa0012c1c056` real = **sadness**, modelos = **anger**: "i feel like getting real snarky here but the other part of me just feels awful for lavolo"
- [ ] `91a40185fa6c2debda39e6d19cfdd8c7` real = **anger**, modelos = **sadness**: "i had a discussion with my mother concerning my sisters divorce"
- [ ] `9d4a539de0664e4791458847efa77b5a` real = **fear**, modelos = **anger**: "i got a very nasty electrical shock when i was tampering with some electrical applainces"
- [ ] `a457fd15b47fa53d3470b4c41c34b5df` real = **sadness**, modelos = **anger**: "when my mother scolded me for something"
- [ ] `a48ef59767792961902f8fa6b162a50a` real = **surprise**, modelos = **anger**: "i wasnt really feeling like being snarky or funny"
- [ ] `a9f88de94f717334f5f8c2964dafaeee` real = **anger**, modelos = **sadness**: "when my daughter began staying away from classes at the age of nine only"
- [ ] `b29f0eb80599be92943b234dee87ae68` real = **fear**, modelos = **surprise**: "i am feeling overwhelmed with thankfulness for the support of my friends and family"
- [ ] `b5eb45f375755f07bb0429a86d4045bb` real = **fear**, modelos = **anger**: "when my father was very drunk and raging at home"
- [ ] `b66117a09350338a9813c4b730d53337` real = **anger**, modelos = **fear**: "one night i could not practise my violin as my family was watching tv with the lights off"
- [ ] `bea75a2886ffe905c6daa32a2476840b` real = **sadness**, modelos = **anger**: "when my best friend promised to come at my house and i had to prepare so many expensive things for her but only to learn at last that she would not turn up"
- [ ] `c383a638a3336dd1971b577964948b61` real = **joy**, modelos = **anger**: "during lectures"
- [ ] `c4592a3c91044bb84c36e0abf93b96b7` real = **sadness**, modelos = **anger**: "when the neighbour of a friend of mine was stabbed to death by thieves in kabwe they stole his van"
- [ ] `c86853141ffa7cff5cb0ea426f190178` real = **sadness**, modelos = **anger**: "when out of work"
- [ ] `d16d31ce6db3f68a474a97f5253e1ecb` real = **fear**, modelos = **anger**: "when we forced a friend to chew a poisonous herb in primary school his tongue was swollen and he told his mother about it the mother reported it to the headmaster who threatened to expel us"
- [ ] `d567b52f5fff19012a700d06efa94ca7` real = **anger**, modelos = **joy**: "someone told me i was chosen for the english lectures because i am good looking"
- [ ] `ddb93dadd2956077a4eff19c753c21a4` real = **surprise**, modelos = **fear**: "i skim through this forty pound book at least once a year so i can feel wholly and truly overwhelmed on this homeschooling journey"
- [ ] `ef2f46f788fbaa3e81199b94b109d7e3` real = **sadness**, modelos = **joy**: "i had to leave to work feeling supermegapissedoff and lethargic but i found solace in the awesome traffic while everyone else is still away from the city"
- [ ] `ff3b9eaae41cf7535fa949b5d9cc2858` real = **fear**, modelos = **surprise**: "i am feeling overwhelmed with a few things outside of the farm these days so am really looking forward to december which is thus far promising to be a slow low key time of family fun and traditions"

