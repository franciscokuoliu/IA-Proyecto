# Uso de IA y criterio técnico

Equipo: Francisco Kuo Liu, Jimmy Feng Liu y Danielo Wu Zhong. Este documento es el de **Danielo Wu Zhong**; Francisco y Jimmy agregan el suyo al final.

## Cómo trabajé con la IA

Usé Claude como herramienta de aceleración: para estructurar ideas, escribir y depurar código, y redactar documentación. El criterio es otra cosa, y es lo que me corresponde poner. Por eso este documento no se limita a contar qué hizo la herramienta: se detiene en las decisiones que tomé, en por qué las tomé, en qué alternativas descarté y en los errores que encontramos, porque ahí se ve si se entiende lo que se está haciendo.

Mi regla fue sencilla: la IA propone, yo decido, y ninguna cifra entra al trabajo si no sale de una corrida guardada o de un script que se pueda repetir.

**Herramientas.** Claude en dos modos: chat en claude.ai durante la Etapa 1 y el planteamiento del proyecto, y agente con acceso al repositorio y a mis máquinas durante la Etapa 2. Casi todo mi uso de IA fue Claude. Si Francisco y Jimmy usaron otras herramientas, lo declaran ellos.

## Quién hizo qué

| Parte del trabajo | Qué hizo la IA | Qué hice yo |
|---|---|---|
| Planteamiento y Etapa 1 | Propuso la estructura de la matriz de 8 modelos, sugirió referencias y redactó borradores de la revisión de literatura | Definí el alcance con la retroalimentación del profesor, busqué y seleccioné mis 7 referencias en Web of Science (descartando las que no aplicaban) y ajusté el texto. Calculo que cerca de un tercio del texto final es mío, concentrado en las decisiones críticas; el resto del enriquecimiento y el formato fue de Claude |
| Pipeline de datos | Escribió `prepare_data.py`, el reinicio, el README y los ajustes a la partición y su verificación | Pedí que los resultados no se versionaran, un comando de reinicio y un README que mis compañeros pudieran seguir; corrí el flujo completo y comprobé que las huellas coinciden con la referencia del equipo |
| Arnés de entrenamiento | Corrigió los fallos que aparecieron al medir | Probé cada corrección en mis máquinas y verifiqué que no alteraba mis corridas |
| Subconjunto S y tiempos | Calculó S con las mediciones de los tres | Llevé la propuesta al equipo y la defendí con sus límites |
| Búsqueda de hiperparámetros | Interpretó los resultados | Lancé y vigilé la búsqueda en mi RTX 3080 y leí el log completo |
| Documentación | Redactó el contrato (v0.3) y el README | Revisé lo escrito, exigí que reflejara lo que realmente ocurrió y lo corregí |
| Cierre de la Etapa 2 | Escribió los scripts del informe agregado, del análisis de errores, del bootstrap pareado y de la explicabilidad (esta con una prueba sobre un modelo de juguete), y actualizó README y contrato (v0.4) | Decidí el alcance (XAI en BERT-base y no en los decoders, sin medición de latencia, sin E4B) y repartí el trabajo entre los tres |

El script de limpieza (`dataset_cleaner.py`) lo escribió Jimmy Feng Liu, y la base del arnés, Francisco Kuo Liu (commit `b702fd9`). Lo que usaron de IA lo declaran ellos.

## Decisiones críticas

### 1. Qué hacer con Gemma 4 E4B

**Situación.** E4B no cupo en la RTX 3080: aun en 4 bits se estima en 8.77 GB, porque sus tablas de embeddings (unos 6.5 GB) no se cuantizan, y eso no deja margen para entrenar en 10 GB. En mi MacBook M5 sí corrió un paso de prueba, con unos 19.6 GB de memoria, pero mucho más lento que el resto.

**Alternativas.** Mantenerlo en el Mac como régimen aparte (bf16 sin cuantizar) usando E2B de puente; dejarlo opcional; o sacarlo del plan.

**Decisión y razón.** Primero seguí el plan B y lo llevé al Mac, para aprovechar la máquina. Cuando vi lo lento que procesaba, propuse sacarlo del plan principal: esperarlo sería un cuello de botella enorme para una entrega que cierra el domingo 8 de noviembre, y ponía en riesgo todos los demás modelos. La IA agregó un argumento metodológico que comparto: compararlo en bf16 contra E2B en NF4 mezclaría el efecto de la escala con el de la cuantización.

**Estado.** Cerrado el 2026-11-06: E4B sale del plan y los otros ocho modelos terminaron sin él. El contrato (v0.4) lo registra como decisión. Hay que corregir también el artículo de la Etapa 1, que lo menciona.

### 2. Cuánto entrena cada modelo (el subconjunto S)

**Situación.** Tres máquinas muy distintas y un plazo fijo. Para que la comparación sea justa, todos los modelos deben ver exactamente los mismos datos, pero el modelo más lento fija cuánto cabe.

**Decisión y razón.** S = 32,000 textos, calculado con el modelo más lento (Gemma 4 E2B en mi 3080) usando su velocidad más conservadora y unas 8 horas de presupuesto. Lo llevé al equipo, que lo aceptó con la reserva de que son solo cerca del 11 % del train. Esa reserva es real: el contrato prevé, como extra, una curva de aprendizaje con porcentajes mayores de datos para mostrar cuánto se pierde.

**Alternativa descartada.** 40,000 textos: subía Gemma E2B de 10 a casi 12 horas y no cabía con holgura antes de la entrega.

### 3. El dtype de Qwen2.5-0.5B

**Situación.** Qwen2.5-0.5B corre en la RTX 2060 de Jimmy, que solo tiene fp16 de forma nativa, y Qwen2.5-1.5B en mi 3080 con bf16. En un par de escala controlado, todo debería ser idéntico salvo el tamaño.

**Alternativas.** Correr las finales de Qwen2.5-0.5B en mi 3080 para igualar el dtype, o mantenerlas en la 2060.

**Decisión y razón.** Se quedan en la 2060 con fp16 y se declara la diferencia como posible confusor del par. Respeta el reparto de trabajo del equipo y no acumula más horas sobre mi máquina, a cambio de un par que ya no es perfectamente controlado, y el artículo debe decirlo.

### 4. Aceptar un cambio solo después de comprobarlo

Para que el entrenamiento de Jimmy arrancara, se pasaron a fp32 los parámetros entrenables cuando se usa fp16. Antes de aceptarlo pregunté si afectaba mis corridas y lo comprobé yo mismo: en mi 3080 el pico de memoria en una corrida de un paso salió idéntico (2,092,308,480 bytes). El cambio no se activa en mi hardware.

### 5. Reproducibilidad antes que comodidad

Pedí que el dataset no se versione y que cualquiera lo regenere con un comando, con huellas SHA-256 que confirman que los tres tenemos exactamente lo mismo; que los resultados pesados no entren a Git; un comando de reinicio para probar el flujo desde cero; commits convencionales; y que el uso de IA se declare aparte, para no llenar el historial de ruido. Con eso un tercero puede reproducir el trabajo sin preguntarnos nada.

### 6. Qué explicamos con XAI

**Situación.** La explicabilidad pide recalcular gradientes con los pesos de cada modelo, y esos pesos están solo en la máquina de quien lo entrenó.

**Alternativas.** Explicar un decoder (Qwen2.5-1.5B) o explicar BERT-base.

**Decisión y razón.** BERT-base (semilla 42), corrido por Jimmy, que tiene los pesos. DistilBERT no alcanzó el tiempo. BERT-base es el mejor modelo en Macro-F1 y cabe en una RTX 2060. Los decoders son cuantizados y de atención causal, y explicarlos costaba más tiempo del que queda. El costo es que el artículo no puede decir nada sobre qué usan los decoders, y lo declara como limitación.

### 7. Recortar la medición de latencia

**Situación.** El plan pedía medir la latencia de todos los modelos en una sola GPU, lo que exige juntar los checkpoints de las tres máquinas.

**Decisión y razón.** Se recorta. El costo se reporta con parámetros, VRAM de entrenamiento y tiempo de entrenamiento, cada uno con su GPU y sin compararlos entre GPUs distintas. No cabía antes del domingo, y el artículo lo declara como limitación.

### 8. Quién califica la auditoría manual de errores

**Situación.** El contrato pide revisar a mano 30 textos que fallan todos los modelos, para ver si hay ruido en las etiquetas.

**Alternativas.** Que la IA califique los 30; que los califique yo solo; o que la IA haga una primera pasada y yo califique después.

**Decisión y razón.** Claude hizo una primera pasada con una razón por texto y yo califiqué los 30 después de verla. Mis calificaciones coinciden con las suyas en los 30 (11 A, 15 B, 4 C). Lo hicimos así por el tiempo que quedaba. El costo es que no hay una segunda opinión independiente, así que no se puede reportar acuerdo entre anotadores, y el artículo lo declara como límite. Los votos quedan en `results/errors/auditoria_manual.md`.

## Errores encontrados y qué aprendimos

| Error | Consecuencia | Cómo lo detectamos | Lección |
|---|---|---|---|
| La IA dio por buena una cifra de filas del dataset (unas 390K, luego 393,822) sin verificarla | Habría entrado al artículo y al cálculo de S | Al ejecutar el pipeline: 416,809 filas, 371,527 tras la limpieza | Las cifras salen del reporte de limpieza, no de la memoria de una IA |
| Gemma 4 tomaba mal el último token con relleno a la izquierda (origen: el arnés base, commit `b702fd9`) | Macro-F1 de 0.354 contra 0.975 de Qwen; las corridas finales habrían sido inválidas | Comparar con Qwen en la misma prueba y revisar cómo rellena Gemma | Un resultado muy por debajo de un modelo hermano es una alarma, no un dato. Corregido (commit `a4e0cef`): 0.937 en la prueba corta |
| LoRA no encontraba las capas de atención de Gemma 4 | La corrida fallaba al iniciar | Error de PEFT al cargar | Aplicar LoRA solo a las capas de lenguaje, no a la torre de visión |
| El escalador de gradiente fallaba con bf16 en la RTX 2060 | Las corridas de Qwen2.5-0.5B no arrancaban | Error reportado por Jimmy | Declarar el dtype por modelo y comprobar el efecto en las otras máquinas |
| La velocidad de Gemma E2B salió distinta en dos mediciones (2.11 contra 3.15 segundos por paso) | Cambia entre 7 y 10 horas el tiempo estimado | Comparar mis mediciones | Sin resolver; se usó la cifra conservadora |
| `prepare_data.py` no incluía `train_S_ids.csv` en la lista de particiones que decide si un paso ya está hecho | Con particiones viejas, sin S, el paso 3 se omitía y el error aparecía después, en la verificación | Revisión de los scripts antes de la entrega | Un archivo nuevo del flujo también va en las listas que deciden si un paso se omite. Corregido en esta entrega |
| El README no pasaba `--glove` en los comandos del BiLSTM | Sin esa opción el arnés usa vectores aleatorios congelados, que no es lo que describe el contrato | Comparar el README con los `config.yaml` de las corridas de Francisco, que sí usaron GloVe | Los comandos documentados se comprueban contra lo que las corridas guardaron. Corregido en esta entrega |
| El baseline Dummy se corrió con el train completo y no con S | No veía los mismos datos que el resto de los modelos | El informe agregado lo avisa | Se repitió con S (Francisco) y ya está en el repo |

Dos hallazgos más de la revisión de hoy, que corrigen el artículo de la Etapa 1: el dataset viene sin mayúsculas ni puntuación (ningún texto las tiene), y DeepSeek-R1-Distill-Qwen-1.5B se destiló sobre Qwen2.5-Math-1.5B y no sobre Qwen2.5-1.5B, de modo que su par con Qwen no aísla solo el objetivo de entrenamiento.

## Lo que puedo explicar hoy y lo que estoy repasando

**Puedo explicar con mis palabras** las decisiones de arriba y su razón: por qué sale E4B, cómo se fija S, qué se pierde con el dtype distinto de Qwen 0.5B, y cómo comprobamos que ningún cambio alterara las demás corridas.

**Estoy repasando** antes de la defensa, con ayuda de la IA y hasta poder explicarlo sin apoyo: el flujo de datos en detalle (cómo el reparto por grupos evita fugas entre train y test); por qué los intentos con lr 3e-4 y con r 32 colapsaron en la búsqueda de hiperparámetros y por qué elegimos lr 2e-4 con r 16; y la fuente primaria sobre la base de DeepSeek.

## Versión corta para los entregables

> Usamos Claude (Anthropic) como asistente durante todo el proyecto: en la Etapa 1 para estructurar el alcance y redactar borradores de la revisión de literatura, y en la Etapa 2 como agente de programación para el pipeline de datos, la corrección de fallos del arnés, el cálculo del subconjunto de entrenamiento, los scripts de análisis (informe agregado, errores, comparaciones pareadas y explicabilidad) y la redacción de la documentación. Las decisiones críticas (qué modelos entran, el tamaño del subconjunto, el tratamiento de las diferencias de hardware) las tomamos nosotros y se justifican en el contrato experimental. Verificamos por nuestra cuenta cada cifra reportada con las corridas guardadas en el repositorio, comprobamos cada cambio de código en nuestras máquinas y registramos los errores de la IA que detectamos. El detalle de cada integrante está en `docs/ai_usage_log.md`.

## Cómo trabajé con la IA — Francisco Kuo Liu

Usé la IA para escribir código, armar comandos y contrastar cifras con los PDF. El criterio es mío: qué modelos corro, con qué archivo de entrenamiento, qué hiperparámetros se quedan y qué número puede entrar a un texto. Ninguna cifra entra si no sale de una corrida guardada en `results/` o de una fuente que yo pueda abrir.

**Herramientas.** Cursor, como agente dentro del repositorio, en casi toda la Etapa 2: el script de evaluación, el baseline Dummy, la descarga del dataset, el arnés (commit `b702fd9`), la separación del tiempo de evaluación y los comandos de búsqueda y de corrida final. Claude, en chat, para revisar el aviso al equipo antes de que cada quien lanzara su búsqueda de hiperparámetros. Esta sección la redacté con Cursor a partir de esos commits y de las corridas.

Mi regla fue la misma que la del equipo: la IA propone, yo decido, y si un comando no coincide con el contrato lo paro y lo repito.

## Quién hizo qué

| Parte del trabajo | Qué hizo la IA | Qué hice yo |
|---|---|---|
| Evaluación y formato común | Escribió `evaluate.py` a partir del contrato: Macro-F1 como métrica principal, `predictions.csv` con id, etiqueta real, predicción y probabilidades, y el JSON por corrida | Pedí el punto 5 por partes, primero el script y después las pruebas con Dummy. Comprobé el humo: `most_frequent` sobre la distribución de 20,000 textos queda cerca de 0.08, que es el F1 de la clase mayoritaria dividido entre 6 |
| Dataset | Escribió `download_data.py`, fijó la versión 1 de `nelgiriyewithana/emotions` y calculó el SHA-256 | Decidí no subir el CSV. Pregunté si el hash es del archivo y corrí `prepare_data.py` hasta que la verificación quedó en verde contra la referencia del equipo. El archivo real tiene 416,809 filas, no las 20,000 de la tabla de la Etapa 1 |
| Arnés | Escribió `scripts/train.py` y `src/harness/` (catálogo, datos, TF-IDF, bucle de entrenamiento e informe) cuando se lo pedí | Exigí un solo camino para todos los modelos, con las reglas del contrato ya metidas: AdamW, lote efectivo 32, paro por paciencia, test solo al final. Corrí el humo de TF-IDF antes de dar el arnés por usable |
| Velocidad para fijar S | Separó segundos por paso, muestras por segundo y segundos por evaluación, que antes iban mezclados | Pedí esas tres cifras el 2026-11-02, las leí en el log y se las pasé a Danielo. En la corrida final del BiLSTM, semilla 42, quedaron 0.113 s/paso, 284.3 muestras/s y 7.3 s por evaluación |
| Búsqueda y corridas finales | Armó los comandos de la rejilla y los de las finales con el ganador de cada familia | Lancé y vigilé las 6 pruebas de TF-IDF y las 8 del BiLSTM en mi máquina. Me quedé con lo que ganó en `val_select` y corrí las finales. El reparto me dejó BiLSTM, TF-IDF y el Dummy; DistilBERT y BERT-base los corrió Jimmy en la RTX 2060 |
| Dummy con S | Cambió el comando que apuntaba a `train_ids.csv` | El 2026-11-07 pedí repetir el Dummy con S, porque es el piso de los demás modelos. `n_train` pasó de 297,220 a 32,000. El 2026-11-08 versioné sus predicciones para el análisis de errores |
| Literatura de la Etapa 1 | Leyó los PDF de Saravia 2018 y Zhang 2024 y reescribió las cifras que no estaban en esos artículos | Pedí la verificación el 2026-11-07 y acepté el cambio en el commit `9b9ea7b`. El artículo de la Etapa 2 no usa esas cifras. El contrato las deja como pendiente de la Etapa 3, y yo tengo que poder explicar cada una sin el PDF delante |

El limpiador es de Jimmy y el cierre del contrato, el cálculo de S y los scripts de análisis son de Danielo. El fallo del último token de Gemma salió de este arnés; lo detectó y lo corrigió Danielo (commit `a4e0cef`).

## Decisiones críticas

### 1. Un solo arnés, y el test se toca una vez

**Situación.** Sin un script común, cada modelo habría escrito métricas distintas y el Dummy no habría servido de prueba del formato.

**Alternativas.** Un script por familia, o calificar a mano con `sklearn` al final de cada corrida.

**Decisión y razón.** Un solo `train.py` y un solo `evaluate.py`. La búsqueda y el paro temprano miran `val_select`. El test se evalúa al terminar, con el mejor checkpoint. Lo pedí así porque el contrato ya lo exigía y porque el humo del Dummy tenía que ejercer el mismo lector que iban a usar BERT y los decoders. El costo es que un error en el arnés toca a los ocho modelos: el del relleno de Gemma es el ejemplo.

### 2. Parar el BiLSTM y repetirlo con GloVe

**Situación.** El 2026-11-04 el BiLSTM estaba corriendo sin `--glove`. Sin esa opción el arnés congela vectores aleatorios de 300 dimensiones y lo anota como `random-frozen`. Eso no es la línea base del contrato.

**Alternativas.** Dejar terminar esa corrida y declarar los vectores aleatorios, o pararla y repetirla con el archivo.

**Decisión y razón.** La paré y la repetí con `glove.6B.300d.txt`. Los `config.yaml` de la búsqueda y de las tres semillas dicen `embeddings: glove-300d-frozen`. Los vectores quedan congelados: de 7.0 M de parámetros solo se entrenan 442 K (el LSTM y la capa de salida). El vocabulario se arma solo con train. El archivo que usé es una copia de Kaggle del mismo `glove.6B.300d.txt`, no el zip oficial de Stanford, y hay que decirlo.

### 3. Con qué configuración me quedo, y por qué TF-IDF lleva una sola corrida

**Situación.** La rejilla ya estaba fijada en la Fase 0. TF-IDF: `C` en {0.1, 1, 10} y n-gramas (1, 1) o (1, 2), sobre los 32,000 de S. BiLSTM: lr en {1e-3, 3e-3}, hidden en {128, 256}, dropout en {0.3, 0.5}, 8 pruebas sobre el 20 % de S (6,399 textos, 10 épocas), porque el contrato pide un subconjunto reducido y en CPU una rejilla completa con S no cabía holgada.

**Decisión y razón.** Gana quien tenga el mejor Macro-F1 en `val_select`, semilla 42. TF-IDF: `C` 10 y n-gramas (1, 2), 0.9480; el peor intento, `C` 0.1 con (1, 2), dio 0.8525. BiLSTM: lr 3e-3, hidden 128 y dropout 0.3, 0.9475; el peor dio 0.9059. Esos dos ganadores están en el borde de la rejilla, así que no son un óptimo demostrado. No amplié la rejilla: se había congelado y no quedaba tiempo. Las finales del BiLSTM, ya con S completo, dan 0.9786 ± 0.0028 en test; el 0.9475 es de la búsqueda, con menos datos, y no se reporta como resultado final.

TF-IDF y Dummy `most_frequent` llevan una sola corrida porque con solver y semilla fijos no se mueven. El BiLSTM sí lleva 42, 1337 y 2026. Las tres pararon por paciencia y restauraron el mejor checkpoint. En la semilla 42 el mejor `val_select` es 0.9826 en el paso 3250 y el test queda en 0.9812.

### 4. La velocidad no incluye la evaluación

**Situación.** Danielo necesitaba muestras por segundo para calcular S, y el arnés metía el tiempo de `val_select` dentro de esa cifra. Así el modelo parece más lento de lo que entrena.

**Alternativas.** Pasar el tiempo de reloj completo, o separar entrenamiento y evaluación.

**Decisión y razón.** Pedí las tres cifras por separado y acepté el cambio solo después de verlas en el log (commit `b945174`). Si hay GPU, el reloj espera a `cuda.synchronize()` para no contar un paso que todavía no terminó. En mi máquina el BiLSTM corre en CPU, así que esa espera no cambia nada; igual dejé el arreglo porque el mismo bucle lo usan los decoders. S no lo fijé yo: lo limita Gemma 4 E2B en la RTX 3080. Mis números solo decían que el BiLSTM no era el cuello de botella.

### 5. El Dummy ve S, igual que los demás

**Situación.** La primera corrida oficial usó `train_ids.csv` (297,220 textos). Un piso entrenado con todo el train no es comparable con modelos que solo vieron S.

**Alternativas.** Dejar esa corrida y declararla, o repetirla con `train_S_ids.csv`.

**Decisión y razón.** La repetí el 2026-11-07. `most_frequent` predice siempre `joy` y saca Macro-F1 0.0861 (el F1 de joy, 0.5166, partido entre seis clases). `stratified`, con las tres semillas y sin pesos de clase, saca 0.1696 ± 0.0014. No compiten: sirven para ver cuánto hay de marco. Las predicciones quedaron en el repositorio para el análisis de errores.

### 6. BiLSTM y TF-IDF en CPU

**Situación.** Esta máquina tiene una GTX 1650, pero el PyTorch instalado es `2.15.0+cpu`. Los metadatos guardan el nombre de la GPU y `device: cpu`.

**Alternativas.** Instalar un PyTorch con CUDA y pasar el BiLSTM a la 1650, o dejarlo en CPU en fp32.

**Decisión y razón.** Se quedan en CPU. TF-IDF tarda unos 0.2 minutos y el BiLSTM unos 9.5 (entre 9.1 y 10.2). Moverlos de dispositivo a mitad del estudio habría mezclado hardware con modelo. Esos tiempos no se comparan con los de la RTX 2060 ni con los de la 3080, y el artículo los reporta por máquina.

### 7. No dejar en la Etapa 1 una cifra que el PDF no trae

**Situación.** La tabla de baselines de la Etapa 1 citaba a Saravia 2018 y a Zhang 2024 con números que además se contradecían entre sí (un BiLSTM cerca de 84 % y, en otra tabla, Macro-F1 0.647).

**Decisión y razón.** Pedí contrastar los PDF con el paper. Saravia no publica ese BiLSTM ni ese 0.647: su modelo con Bi-LSTM es DeepMoji (F1 promedio 0.59 en ocho clases de Plutchik) y su CARER con CNN llega a F1 promedio 0.79. Zhang 2024 entrena RNN, GRU y LSTM sobre una muestra balanceada del dataset de Kaggle; no entrena BERT ni DistilBERT, y su mejor Bi-LSTM de dos capas imprime accuracy 0.9321 y F1 0.93, que no sirve de techo para nuestro corpus desbalanceado. Acepté reemplazar la tabla de la Etapa 1 con lo que esos PDF sí imprimen. El artículo de la Etapa 2 no cita esas cifras. Sigo repasándolas, porque el contrato no da esta verificación por cerrada.

## Errores encontrados y qué aprendimos

| Error | Consecuencia | Cómo lo detectamos | Lección |
|---|---|---|---|
| El arnés base (commit `b702fd9`) tomaba mal el último token cuando el relleno va a la izquierda | En una prueba corta Gemma quedó en Macro-F1 0.354 contra 0.975 de Qwen. Las finales habrían sido inválidas | Lo vio Danielo al comparar los dos decoders. El arreglo está en `a4e0cef`: la posición es el máximo de `mascara * índice`, no `suma(mascara) - 1` | Un resultado muy por debajo de un modelo hermano es una alarma. El que pide el arnés tiene que poder explicar la cabeza, aunque no entrene ese modelo |
| La primera medición de muestras/s incluía la evaluación | S habría salido más chico de lo que el entrenamiento realmente permite | Al pedir por separado tiempo de paso, muestras/s y tiempo de evaluación | El reloj de entrenar y el reloj de evaluar no se suman. Corregido en `b945174` |
| El Dummy oficial se entrenó con el train completo | El piso no veía los mismos 32,000 textos que el resto | `n_train: 297220` en el `config.yaml`, y el informe agregado lo avisó | El baseline trivial usa `train_S_ids.csv`. Repetido el 2026-11-07 |
| Una corrida de BiLSTM iba sin `--glove` | Habría quedado como línea base un modelo de vectores aleatorios congelados | Al revisar el comando contra el contrato, antes de guardar la búsqueda | Si el `config.yaml` no dice `glove-300d-frozen`, esa corrida no entra |
| Las cifras de Saravia y de Zhang en la Etapa 1 no están en esos PDF | Habrían quedado como techo publicado de un experimento que no es el nuestro | Lectura de los dos PDF el 2026-11-07 | Una cita entra si el PDF la imprime. El 0.69 sin ponderar de Saravia es un cálculo nuestro, no un número suyo, y así hay que decirlo |

## Lo que puedo explicar hoy y lo que estoy repasando

**Puedo explicar con mis palabras** por qué hay un solo arnés y por qué el test no entra a la búsqueda; qué pasa si el BiLSTM no recibe `--glove`; por qué TF-IDF y el Dummy `most_frequent` tienen una corrida y el BiLSTM tres; por qué el ganador de la rejilla no es un óptimo demostrado; por qué el Dummy se repitió con S y por qué su Macro-F1 de 0.0861 es el F1 de `joy` dividido entre seis; y por qué mis tiempos van en CPU y no se comparan con las otras dos GPUs.

**Estoy repasando** antes de la defensa, hasta poder decirlo sin abrir el código ni el PDF: cómo elige el arnés el último token real cuando el relleno es a la izquierda; por qué el 0.9475 de la búsqueda del BiLSTM y el 0.9786 de la final no se contradicen; y qué número de Saravia 2018 y de Zhang 2024 sí está impreso y cuál salió de la tabla de la Etapa 1.

## Cómo trabajé con la IA - Jimmy Feng Liu
La IA escribió gran parte del script de limpieza: etiquetas contradictorias, detector de cuasi-duplicados (TF-IDF con hashes + coseno, con candidatos por términos raros), modos de salida y reporte. Mi parte fue la carga, la normalización y la eliminación de duplicados exactos. El modelo utilizado fue Claude, para depurar código, plantear la lógica de limpieza, redactar documentación y estructurar ideas.

Mi regla: todo material generado por IA fue revisado a mano antes de pasar a la siguiente etapa de desarrollo, y ninguna cifra del dataset entra al trabajo si no sale de `reporte_limpieza.json`

## Quién hizo qué
| Parte | Qué hizo la IA | Qué hice yo |
|---|---|---|
| Carga, normalización y duplicados exactos | Depuró el código y verificó condiciones que faltaban | Escribí parte del código y revisé los cambios que hizo la IA en la normalización y en la eliminación de duplicados exactos |
| Detección de cuasi-duplicados (TF-IDF + coseno) | Escribió el detector: hashes de 64 bits en dos pasadas, filtro por términos raros y componentes conexas | Decidí el umbral de 0.90: busco eliminar los registros más emparentados sin comprometer los que parecen similares pero significan algo distinto. Aprobé TF-IDF + coseno por ser un método ligero, que no exige carga extra al equipo físico |
| Etiquetas contradictorias | Escribió el código | Comprobé los resultados. Este paso elimina todas las copias de un texto que aparece con dos o más etiquetas distintas, y corre antes de los duplicados exactos para que la etiqueta que sobreviva no dependa del orden del CSV |
| Reporte JSON y huella SHA-256 | Escribió el código | Verifiqué a mano los reportes generados en las limpiezas |

## Decisiones críticas
### 1. Normalizar solo para comparar

Uso una columna `norm` (minúsculas, espacios colapsados) para detectar duplicados, pero el texto que sale es el original. Así no alteramos los datos que verán los modelos, y la limpieza solo decide qué filas entran.

**Alternativa descartada:** guardar el texto normalizado, que es más simple pero cambia la entrada de los modelos.

### 2. Textos con etiquetas contradictorias: eliminar todas sus copias

Si el mismo texto aparece con dos etiquetas, no hay forma de saber cuál es la correcta, así que lo quito completo en lugar de elegir una. El costo es perder algunos ejemplos legítimos; el beneficio es no meter ruido de etiqueta en train ni en test. El reporte registra cuántas filas y cuántos textos distintos se perdieron por esto.

Este paso va **antes** de los duplicados exactos. Si fuera después, `drop_duplicates` conservaría la primera copia y su etiqueta dependería del orden del CSV.

### 3. TF-IDF + coseno para los cuasi-duplicados

Los cuasi-duplicados son la principal fuente de fuga entre train y test, por eso no basta con eliminar duplicados exactos. Elegí TF-IDF + coseno por su ligereza: no requiere cargar modelos pesados ni sobrecargar el equipo. Para que cupiera en memoria con unos 370K textos, no se guarda el vocabulario: unigramas y bigramas se representan con hashes de 64 bits, en dos pasadas por lotes. Solo se comparan pares que comparten al menos un término raro (aparece en ≤ 0.2 % de los textos), en lugar de las ~7×10¹⁰ comparaciones de todos contra todos.

### 4. Umbral de coseno ≥ 0.90

Decidí 0.90 porque busco eliminar los registros con mayor parentesco sin comprometer los que parecen similares pero tienen significados distintos. Un umbral más bajo arriesga agrupar textos que solo hablan de lo mismo; uno más alto deja pasar copias casi idénticas. 0.90 es el valor por defecto y fue revisado en una muestra pequeña.

### 5. Dos modos: `eliminar` y `agrupar`

`eliminar` deja un texto por grupo; `agrupar` conserva todo y deja la columna group para que la partición mantenga cada grupo completo en un solo lado. Esa columna es lo que permite repartir por grupos sin fugas. En el flujo del equipo, prepare_data.py corre el limpiador con `--modo agrupar` y `PYTHONHASHSEED=0`, y `check_splits.py` comprueba después que ningún grupo de cuasi-duplicados quede partido entre train, val y test, y que ningún texto normalizado aparezca en dos particiones. `make_splits.py` reparte con StratifiedGroupKFold (10 pliegues, semilla 42) usando la columna group: el pliegue 0 es test, el 1 es validación y los otros 8 son train, estratificado por emoción y sin partir ningún grupo. Reemplaza el número de grupo por el menor id del grupo, porque ese número es arbitrario, y así el reparto no depende de cómo lo numeró el limpiador. Esa verificación debe pasar en verde antes de entrenar, y además compara las huellas de los grupos con la referencia del equipo.

Modo usado: agrupar (según reporte_limpieza.json), así que no se eliminó ningún cuasi-duplicado: los 371,527 textos se conservan y la partición reparte los grupos completos. [COMPLETAR: por qué elegiste agrupar en vez de eliminar]

### 6. Reproducibilidad

El `id` es el MD5 del texto normalizado, así que no depende del orden de las filas. El reporte guarda el SHA-256 del CSV de entrada, los parámetros y las cifras de cada paso. Con eso los tres podemos comprobar que partimos de lo mismo.

## Limitaciones y errores a declarar

| Punto | Consecuencia | Qué hacer |
|---|---|---|
| `hash()` de Python se aleatoriza en cada proceso (`PYTHONHASHSEED`) | Cambia el orden de columnas y puede alterar los últimos decimales del coseno, así que un par justo en 0.90 podría entrar o no entre corridas | Fijar `PYTHONHASHSEED=0` en el comando, o usar un hash estable (por ejemplo `hashlib.blake2b` de 8 bytes). Verificar que las huellas coincidan entre las tres máquinas |
| Solo se comparan pares con algún término raro (y se descartan términos con df > 2 %) | Cuasi-duplicados formados solo por palabras comunes no se detectan: el recall es menor al 100 % | Declararlo como limitación |
| Componentes conexas | Una cadena A~B~C agrupa textos que no son similares entre sí, y en modo `eliminar` se borrarían de más | El reporte da `grupos_multiples`; revisar el tamaño máximo de grupo |
| El paso de etiquetas contradictorias solo detecta textos idénticos tras normalizar | Dos textos casi iguales con etiquetas distintas pasan este filtro | Los atrapa el detector de cuasi-duplicados; el reporte los cuenta en `grupos_con_etiquetas_mixtas` |

**Errores de la IA que detecté:** 
Principalmente con los algortitmos de limpieza de cuasi-duplicados, donde algunas de las sugerencias iniciales generaba diferentes reportes de limpieza no reproducibles para el experimento.

## Cifras (de `reporte_limpieza.json`)

| Paso | Cifra | Campo del reporte |
|---|---|---|
| Filas originales | 416,809 | `filas_originales` |
| Eliminadas por etiquetas contradictorias | 44,650 | `filas_eliminadas_etiquetas_contradictorias` |
| Copias exactas eliminadas | 632 | `copias_exactas_eliminadas` |
| Pares de cuasi-duplicados detectados | 5,536 | `pares_cuasi_dup` |
| Cuasi-duplicados eliminados | 0 (fueron agrupados) | `cuasi_duplicados_eliminados` |
| Registros finales | 371,527 | `registros_finales` |

## Lo que puedo explicar hoy / lo que estoy repasando

**Puedo explicar con mis palabras:** la normalización y la eliminación de duplicados exactos (mi parte del código), por qué elegí el umbral 0.90, por qué TF-IDF + coseno y no un método más pesado, y qué hace el paso de etiquetas contradictorias y por qué va antes de los duplicados exactos.

**Estoy repasando** antes de la defensa, hasta poder explicarlo sin apoyo: el funcionamiento del filtro por términos raros, porqué la columna `group` evita fugas durante la partición y el funcionamiento general del algoritmo de cuasi-duplicados con TF-IDF + Coseno.




