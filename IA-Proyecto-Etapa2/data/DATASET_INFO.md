# Dataset

Fuente única de la Etapa 2.

| Campo | Valor |
|---|---|
| Nombre | Emotions |
| Autor | Nidula Elgiriyewithana |
| URL | https://www.kaggle.com/datasets/nelgiriyewithana/emotions |
| Versión de Kaggle | 1 |
| Handle | `nelgiriyewithana/emotions/versions/1` |
| Fecha de descarga | 2026-10-17 |
| Archivo | `data/raw/text.csv` |
| Tamaño | 44,916,111 bytes |
| SHA-256 | `5f2254945d620f7013d8b516c1bde76791af54cc6c94984404c136886285b430` |
| Filas | 416,809 |
| Columnas | índice sin nombre, `text`, `label` |

El CSV no se versiona. Se vuelve a obtener con `python scripts/download_data.py`. El script rechaza el archivo si el SHA-256 no coincide con el de esta ficha.

## Etiquetas

El orden es el de la columna `label`, tomado de la descripción del propio dataset. Quedó fijado en `labels.json`.

| Código | Clase | Filas |
|---|---|---|
| 0 | sadness | 121,187 |
| 1 | joy | 141,067 |
| 2 | love | 34,554 |
| 3 | anger | 57,317 |
| 4 | fear | 47,712 |
| 5 | surprise | 14,972 |

## Texto

Las 416,809 filas usan solo letras ASCII, dígitos y espacios. Ninguna tiene mayúsculas, puntuación ni caracteres fuera de ASCII. Los apóstrofos ya vienen quitados (`dont`, `ive`, `won t`).

La afirmación de la Etapa 1 sobre conservar mayúsculas y puntuación no aplica a este archivo. El artículo tiene que corregirla.
