# Trabajo Final CEIA - Segmentación de Imágenes Satelitales

Este proyecto corresponde al trabajo final de la Especialización en Inteligencia Artificial (CEIA) y aborda la segmentación semántica de imágenes satelitales Sentinel-2 sobre territorio argentino.

Se utiliza una arquitectura flexible que permite entrenar distintos modelos de segmentación, tanto definidos manualmente (como DuckNet), como también provenientes de librerías externas como **segmentation-models-pytorch (SMP)** y **MONAI**. El pipeline está basado en PyTorch Lightning.

---

## 📁 Estructura del Proyecto

```
ceia_final_project/
├── constants/              # Constantes globales (mean, std)
├── datasets/               # Dataset personalizado con imágenes Sentinel-2
├── models/                 # Modelos definidos manualmente (ej. DuckNet)
├── modules/                # LightningModule con lógica de entrenamiento
├── transforms/             # Aumentos y normalización para segmentación
├── inference/              # Clase Segmenter para predicción desde imagen
```

---

## ⚙️ Instalación

Se recomienda trabajar en un entorno virtual:

```bash
pip install -e .
# O instalar manualmente desde pyproject.toml
```

Requiere Python 3.9 o superior.

---

## 🏋️‍♂️ Entrenamiento

El entrenamiento se realiza mediante PyTorch Lightning utilizando el módulo `LightningSegmentation`, que encapsula la lógica de entrenamiento, validación y checkpointing.

Es posible entrenar distintos modelos:
- Manualmente definidos en `models/`
- Externos desde MONAI o SMP, especificando su nombre en la configuración

Los datos son cargados mediante un `Dataset` personalizado y procesados con transformaciones específicas para segmentación.

---

## 🚀 Inferencia

El módulo `Segmenter` permite aplicar un modelo entrenado sobre nuevas imágenes en formato `.png` o `.jpg`.

```python
from ceia_final_project.inference import Segmenter

segmenter = Segmenter(
    checkpoint_path="path/to/model.ckpt",
    threshold=0.5
)

input_tensor, mask = segmenter.segment(
    input_path="path/to/image.png",
    output_path="output/mask.png",  # opcional
    normalize_input=True
)
```

- `input_tensor` es la imagen de entrada como tensor.
- `mask` es la máscara binaria predicha.
- Si se proporciona `output_path`, se guarda la máscara como imagen.

---

## 🧱 Requisitos

Las dependencias principales incluyen:

- `torch==2.4.0`
- `torchvision==0.19.0`
- `lightning==2.5.1`
- `segmentation-models-pytorch==0.4.0`
- `monai==1.4.0`
- `fastai==2.7.19`
- `numpy==1.24.3`

Todas las dependencias están definidas en `pyproject.toml`.

---

## 👨‍💻 Autor

Trabajo realizado por **Kevin Cajachuán** como parte de la Especialización en Inteligencia Artificial (UBA - CEIA).
