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

## Sesiones de Francisco Kuo Liu

[completar por Francisco]

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




