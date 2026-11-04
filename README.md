# IA-Proyecto

Proyecto semestral del curso de IA (Track B: Deep Learning, NLP). Integrantes: Francisco Kuo Liu, Jimmy Feng Liu y Danielo Wu Zhong. Benchmark controlado de 9 modelos para clasificación de emociones en 6 clases (dataset Kaggle "Emotions", derivado de CARER).

- `IA-Proyecto-Etapa1/`: documento y entregables de la Etapa 1.
- `IA-Proyecto-Etapa2/`: modelado, XAI y evaluación. Las reglas del experimento están en `EXPERIMENT_CONTRACT.md`.

## 1. Preparar el entorno

Requiere Python 3.10 o superior.

**Windows (PowerShell), desde la raíz del repo:**

```powershell
.\setup_env.ps1            # crea .venv e instala IA-Proyecto-Etapa2/requirements.txt
.\.venv\Scripts\Activate.ps1
```

Si algo se corrompe: `.\setup_env.ps1 -Recreate`.

**macOS / Linux:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r IA-Proyecto-Etapa2/requirements.txt
```

## 2. Preparar los datos (un solo comando)

El dataset **no está en el repo**. Cada persona lo regenera y el script comprueba que sea idéntico al del equipo.

```bash
python IA-Proyecto-Etapa2/scripts/prepare_data.py
```

Hace, en orden:

| Paso | Script | Salida (no versionada) |
|---|---|---|
| 1 | `download_data.py` | `data/raw/text.csv` (verifica SHA-256) |
| 2 | `dataset_cleaner.py --modo agrupar` (con `PYTHONHASHSEED=0` fijado) | `salida/cleaned_full_data.csv` |
| 3 | `make_splits.py` | `splits/{train,val,test,val_select}_ids.csv`, `splits/train_S_ids.csv` (subconjunto S), `data/labels.csv` |
| 4 | `check_splits.py --require-reference` | verificación |

Los pasos ya hechos se omiten; `--force` rehace todo. Si el paso 4 termina en verde (incluye `subconjunto S: n y huella`), tus particiones y tu S son idénticos a los de todos. Si falla en las huellas (fingerprints), revisa las versiones de pandas/scikit-learn contra `requirements.txt` y avisa al equipo antes de entrenar.

### Referencia rápida de comandos

Todos se corren desde la raíz del repo con el entorno activado. Abreviamos `prepare_data.py` como `PD` (ruta completa: `IA-Proyecto-Etapa2/scripts/prepare_data.py`).

| Quiero... | Comando |
|---|---|
| Preparar todo por primera vez | `python PD` |
| Solo verificar que mis datos coinciden con el equipo | `python PD` (omite lo ya hecho y siempre verifica) |
| Rehacer todo sin borrar antes | `python PD --force` |
| Borrar lo generado (pide confirmación) | `python PD --reset` |
| Borrar lo generado y también `results/` | `python PD --reset --reset-results` |
| Borrar sin preguntar | `python PD --reset --yes` |
| Rehacer solo particiones y verificación | borrar `splits/` y `data/labels.csv` y correr `python PD` |
| Ayuda | `python PD --help` |

`--reset` borra `data/raw/`, `salida/`, `splits/` y `data/labels.csv`. Nunca borra `data/expected_hashes.json`, el código ni `labels.json`. Después de un reset, `python PD` vuelve a descargar y limpiar (es lo más lento).

Ejemplo de reinicio completo:

```bash
python IA-Proyecto-Etapa2/scripts/prepare_data.py --reset
python IA-Proyecto-Etapa2/scripts/prepare_data.py
```

Debe terminar con `Listo: tus datos son identicos a la referencia del equipo`.

### Si algo falla

- **Falla la descarga o el SHA-256 no coincide:** Kaggle entregó otro archivo. No sigas; avisa al equipo.
- **Falla en las huellas (paso 4) pero lo demás está bien:** casi siempre es una versión distinta de pandas o scikit-learn. Compara `pip freeze` con el de un compañero.
- **Error de importación (por ejemplo `sklearn.utils.validation`):** el venv quedó corrupto. En Windows: `.\setup_env.ps1 -Recreate`. Si persiste, créalo en una ruta corta fuera de Documents.
- **Falta `data/expected_hashes.json`:** haz `git pull`. Solo quien define la referencia usa `--write-reference`.

### Qué se versiona y qué no

- Sí: scripts, `labels.json`, `data/DATASET_INFO.md` y `data/expected_hashes.json` (huellas SHA-256 de ids, etiquetas y grupos).
- No: `data/raw/`, `salida/`, `splits/`, `data/labels.csv` (están en `.gitignore`).

### Partición (determinista)

`StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` sobre grupos de casi-duplicados: fold 0 = test, fold 1 = val, resto = train (~80/10/10). `val_select` son 10 000 muestras estratificadas de val (semilla 42). **S** (`splits/train_S_ids.csv`) son 32 000 muestras de train, estratificadas por emoción (semilla 42): es el conjunto con el que entrenan todos los modelos. Los grupos evitan fugas de casi-duplicados entre train y test.

### Cambiar la referencia (solo con acuerdo del equipo)

```bash
python IA-Proyecto-Etapa2/scripts/prepare_data.py --write-reference --force-reference
```

Luego hacer commit de `data/expected_hashes.json` y avisar a todos.

## 3. Entrenar

Requisito: `prepare_data.py` en verde (huella de S `2d518490c80f`). Todo desde la raíz del repo, con el entorno activado. Las reglas del experimento están en `IA-Proyecto-Etapa2/EXPERIMENT_CONTRACT.md`.

### Una corrida

```powershell
python IA-Proyecto-Etapa2/scripts/train.py train --model qwen2.5-1.5b --seed 42
```

Modelos: `tfidf-logreg`, `bilstm`, `distilbert`, `bert-base`, `qwen2.5-0.5b`, `qwen2.5-1.5b`, `gemma-4-e2b`, `deepseek-r1-distill-qwen-1.5b` (`gemma-4-e4b` existe en el arnés, pero no se entrena mientras el equipo no confirme si entra al plan).

| Opción | Para qué |
|---|---|
| `--seed` | 42, 1337 y 2026 en las corridas finales |
| `--max-steps N` | Prueba rápida (2 pasos) para ver que todo funciona |
| `--predict-split test` | Calificar en test. **Solo en las corridas finales**, una vez por configuración |
| `--save-model` | Guarda los adaptadores (o el modelo, en encoders y BiLSTM) en `<corrida>/model/`. Se necesita para el XAI. Úsalo solo en las corridas finales |
| `--batch-size` | 8 por defecto; el lote efectivo siempre es 32 (se acumula el gradiente). Si hay error de memoria, baja a 4 |
| `--output-root` | Carpeta de resultados. Para pruebas usa `IA-Proyecto-Etapa2/results/smoke` (Git la ignora). Ojo: una ruta relativa se cuenta desde donde corras el comando |

No cambies `--max-len` (64 para todos). Cada corrida escribe en `IA-Proyecto-Etapa2/results/<modelo>/seed_<n>/`: `metrics.json`, `config.yaml`, `run.log`, `run_metadata.json` y, sin versionar, `predictions.csv` y `model/`.

### Búsqueda de hiperparámetros (HPO)

```powershell
python IA-Proyecto-Etapa2/scripts/train.py hpo --family decoder    # Dani, hecho: lr 2e-4, r 16
python IA-Proyecto-Etapa2/scripts/train.py hpo --family encoder    # Jimmy (corre en BERT-base)
python IA-Proyecto-Etapa2/scripts/train.py hpo --family tfidf      # Fran
python IA-Proyecto-Etapa2/scripts/train.py hpo --family bilstm     # Fran
```

Califica solo en `val_select` y guarda cada intento en `results/hpo/<familia>/`. Haz commit de esa carpeta al terminar.

### Corridas finales

Tres semillas por modelo, una semilla completa a la vez, y se salta lo que ya terminó (si se corta, relanza el mismo bloque). Cambia `$modelos` por los tuyos:

```powershell
$modelos = "qwen2.5-1.5b","deepseek-r1-distill-qwen-1.5b","gemma-4-e2b"   # los de Dani
foreach ($seed in 42,1337,2026) {
  foreach ($m in $modelos) {
    if (Test-Path "IA-Proyecto-Etapa2\results\$m\seed_$seed\metrics.json") { continue }
    python IA-Proyecto-Etapa2/scripts/train.py train --model $m --seed $seed --predict-split test --save-model 2>&1 | Tee-Object -Append -FilePath "final.log"
  }
}
```

| Quién | Modelos |
|---|---|
| Dani (RTX 3080) | `qwen2.5-1.5b`, `deepseek-r1-distill-qwen-1.5b`, `gemma-4-e2b` |
| Jimmy (RTX 2060) | `distilbert`, `bert-base`, `qwen2.5-0.5b` |
| Fran | `bilstm` (3 semillas, 10 épocas), `tfidf-logreg`, y el baseline trivial (sección 4) |

Mientras corre la cola **no cambies de rama ni hagas `pull` con código nuevo**: cada corrida lee el código del disco al arrancar, y las semillas quedarían con versiones distintas. Cambiar a una rama que solo toca `.md` es seguro.

### Medir la longitud de los textos

```powershell
python IA-Proyecto-Etapa2/scripts/token_lengths.py
```

Ya está medido: `max_len` = 64 recorta como máximo el 0.15 % de los textos.

### Si algo sale raro

- **Texto en rojo al principio** (`Redirects are currently not supported`, `triton not found`, `NativeCommandError`): son avisos de PyTorch y de PowerShell, no errores. `score.weight MISSING` también es normal: es la capa de clasificación nueva.
- **Caracteres raros en la consola** (`Ôûê`): solo se ve mal en pantalla. El log queda bien.
- **Error de memoria (OutOfMemory):** agrega `--batch-size 4`.
- **`Torch not compiled with CUDA enabled`:** `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- **El proceso se detuvo de noche:** desactiva la suspensión de Windows y pausa las actualizaciones.
- **Una semilla colapsó** (el Macro-F1 cae a unos 0.08 y se queda ahí): avisa al equipo antes de seguir; el arnés conserva el mejor checkpoint, pero hay que declararlo.

## 4. Baseline trivial y evaluación

```bash
cd IA-Proyecto-Etapa2
python scripts/run_dummy.py --help
python scripts/evaluate.py --help     # subcomandos: score, aggregate, compare
```

Resultados en `IA-Proyecto-Etapa2/results/`.

## Uso de IA

Parte de los scripts (entorno, datos, partición y verificación, arnés de entrenamiento, subconjunto S) y de la documentación se hicieron con asistencia de Claude. Cada integrante debe poder explicar lo que presenta. El registro está en `IA-Proyecto-Etapa2/docs/ai_usage_log.md`.
