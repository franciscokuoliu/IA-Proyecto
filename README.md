# IA-Proyecto

Proyecto semestral del curso de IA (Track B: Deep Learning, NLP). Benchmark controlado de 9 modelos para clasificación de emociones en 6 clases (dataset Kaggle "Emotions", derivado de CARER).

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
| 3 | `make_splits.py` | `splits/{train,val,test,val_select}_ids.csv`, `data/labels.csv` |
| 4 | `check_splits.py --require-reference` | verificación |

Los pasos ya hechos se omiten; `--force` rehace todo. Si el paso 4 imprime **OK**, tus particiones son idénticas a las de todos. Si falla en las huellas (fingerprints), revisa las versiones de pandas/scikit-learn contra `requirements.txt` y avisa al equipo antes de entrenar.

### Empezar desde cero

```bash
python IA-Proyecto-Etapa2/scripts/prepare_data.py --reset                    # borra data/raw, salida, splits y labels.csv (pide confirmación)
python IA-Proyecto-Etapa2/scripts/prepare_data.py --reset --reset-results    # además borra results/
python IA-Proyecto-Etapa2/scripts/prepare_data.py                            # regenera todo
```

Nunca se borra `data/expected_hashes.json` ni el código. `--yes` salta la confirmación.

### Qué se versiona y qué no

- Sí: scripts, `labels.json`, `data/DATASET_INFO.md` y `data/expected_hashes.json` (huellas SHA-256 de ids, etiquetas y grupos).
- No: `data/raw/`, `salida/`, `splits/`, `data/labels.csv` (están en `.gitignore`).

### Partición (determinista)

`StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` sobre grupos de casi-duplicados: fold 0 = test, fold 1 = val, resto = train (~80/10/10). `val_select` son 10 000 muestras estratificadas de val (semilla 42). Los grupos evitan fugas de casi-duplicados entre train y test.

### Cambiar la referencia (solo con acuerdo del equipo)

```bash
python IA-Proyecto-Etapa2/scripts/prepare_data.py --write-reference --force-reference
```

Luego hacer commit de `data/expected_hashes.json` y avisar a todos.

## 3. Baseline trivial y evaluación

```bash
cd IA-Proyecto-Etapa2
python scripts/run_dummy.py --help
python scripts/evaluate.py --help     # subcomandos: score, aggregate, compare
```

Resultados en `IA-Proyecto-Etapa2/results/`.

## Uso de IA

Parte de los scripts de entorno y de partición/verificación fueron generados con asistencia de Claude. Cada integrante debe poder explicarlos. Ver el registro de uso de IA cuando se cree (`docs/ai_usage_log.md`).
