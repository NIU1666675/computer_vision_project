# Vision xG from SoccerNet shot frames

Este proyecto deja preparado el pipeline descrito en el PDF para estimar xG como una clasificacion binaria gol / no gol desde frames de video.

El dataset actual ya contiene los frames filtrados:

```text
Dataset_final/Dataset_final/Frames_Bons_Definitius/<shot_id>/
  frame_shot.jpg
  frame_prev_0.jpg
  frame_prev_1.jpg
  frame_prev_2.jpg
  frame_prev_3.jpg
  frame_prev_4.jpg
  label.txt
```

La ruta se resuelve automaticamente aunque se pase `Dataset_final` o `Frames_Bons_Definitius`.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

En Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Para la rama opcional de homografia:

```bash
pip install -e ".[homography]"
```

## Auditoria del dataset

```bash
python -m xg_vision.inspect_dataset --dataset-root Dataset_final --write-splits
```

En este equipo el dataset contiene 1418 disparos: 1218 no goles y 200 goles. Todos los disparos tienen los 6 frames esperados y las imagenes son `398x224`.

## Entrenamiento visual

CNN baseline sobre `frame_shot.jpg`:

```bash
python -m xg_vision.train --model cnn --dataset-root Dataset_final --output-dir runs/cnn
```

CNN + LSTM sobre la secuencia `prev_4 ... prev_0, shot`:

```bash
python -m xg_vision.train --model lstm --dataset-root Dataset_final --output-dir runs/lstm
```

CNN + self-attention sobre la misma secuencia:

```bash
python -m xg_vision.train --model attention --dataset-root Dataset_final --output-dir runs/attention
```

Los entrenamientos crean automaticamente:

```text
runs/<modelo>/
  best.pt
  last.pt
  config.yaml
  splits.csv
  split_summary.csv
  history.csv
  metrics.json
  val_predictions.csv
  test_predictions.csv
```

Por defecto el split es por partido (`group`) para reducir fuga entre train, validation y test. La clase positiva esta desbalanceada, asi que el loss usa `pos_weight = negativos / positivos` calculado solo en train.

## Entrenar los tres modelos

```bash
python scripts/train_all_models.py --dataset-root Dataset_final --output-root runs/server
```

Se pueden pasar overrides comunes:

```bash
python scripts/train_all_models.py --dataset-root Dataset_final --output-root runs/server --epochs 80 --batch-size 32 --num-workers 8
```

## Evaluacion e inferencia

Evaluar un checkpoint:

```bash
python -m xg_vision.evaluate --checkpoint runs/cnn/best.pt --split test
```

Predecir un disparo:

```bash
python -m xg_vision.predict --checkpoint runs/cnn/best.pt --shot-dir "Dataset_final/Dataset_final/Frames_Bons_Definitius/<shot_id>"
```

Crear una visualizacion con el xG superpuesto:

```bash
python -m xg_vision.visualize --checkpoint runs/cnn/best.pt --shot-dir "Dataset_final/Dataset_final/Frames_Bons_Definitius/<shot_id>" --output-dir outputs/visualizations
```

## Rama de homografia

La homografia queda preparada como modulo final opcional en `xg_vision.homography`.

Incluye:

- calculo de homografia imagen -> campo con puntos correspondientes;
- proyeccion de puntos y centros inferiores de bounding boxes;
- features tabulares de xG: distancia a porteria, angulo, defensores en el cono de tiro, distancia al defensor mas cercano y features del portero;
- entrenamiento tabular con regresion logistica y, opcionalmente, XGBoost.

Cuando tengais detecciones/correspondencias:

```python
from xg_vision.homography import compute_homography, project_points, features_from_projected_positions
```

Entrenar baseline tabular desde un CSV de features:

```bash
python -m xg_vision.train_tabular --features-csv outputs/homography/features.csv --model both --output-dir runs/tabular
```

## Smoke test

Antes de lanzar en servidor:

```bash
python scripts/smoke_test.py --dataset-root Dataset_final
```

Comprueba resolucion de dataset, splits, datasets y forward pass de CNN, CNN+LSTM y CNN+attention.
