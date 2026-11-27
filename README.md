# TP2 - Sistema de Detección y Clasificación de Razas de Perros

Plantilla base para desarrollar un pipeline completo de Computer Vision:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md) (versión PDF: [Trabajo Práctico N°2.pdf](<Trabajo Práctico N°2.pdf>)).

La cátedra provee la infraestructura completa (Docker, aplicación Gradio, base de datos vectorial, API, scripts y herramientas de evaluación y visualización). El grupo hace un **fork** de este repositorio, trabaja en una **feature branch** y completa **únicamente** las funciones indicadas en cada etapa. Cualquier modificación fuera de esas funciones debe justificarse en el Pull Request.

## Objetivo del backend

API asincrónica en Python que permite:

- Búsqueda por similitud (`/search`): imagen consultada, Top-K imágenes similares y raza predicha.
- Clasificación supervisada (`/classify`): raza y score con el modelo entrenado seleccionado.
- Detección + clasificación (`/detect`): bounding boxes, raza y scores de confianza.
- Consulta del estado del procesamiento asincrónico (`/status/{job_id}`).
- Selección dinámica del modelo de embeddings (`/models` + campo `model` en `/search`).

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-dog-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Etapa 2: entrenamiento, evaluación (incl. NDCG@10 de la Etapa 1) y comparación
├── ENUNCIADO.md                 # Enunciado completo del TP
├── Trabajo Práctico N°2.pdf     # Enunciado en PDF
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra (una pestaña por etapa)
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /search, /classify, /detect, /status, /models, /files
│       ├── bootstrap.py         # Construcción de los servicios (compartida por la API y los scripts)
│       ├── config.py            # Configuración vía .env
│       ├── files.py
│       ├── schemas.py
│       ├── preprocesamiento.py      # Preprocesamiento y augmentation compartidos (notebooks y app)
│       ├── models/
│       │   └── clasificadores.py    # Arquitecturas del Modelo A (ResNetRazas) y del Modelo B (CNNPropia)
│       ├── evaluation/
│       │   └── metrics.py       # NDCG@k, precision/recall/F1, specificity
│       ├── visualization/
│       │   └── draw.py          # Dibujo de bounding boxes
│       ├── services/
│       │   ├── similarity_service.py   # Etapa 1: búsqueda por similitud
│       │   ├── classifier_service.py   # Etapa 2: entrenamiento, evaluación y embeddings de los modelos
│       │   ├── detection_service.py    # Etapa 3: detección con YOLO y clasificación de recortes
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py      # Base vectorial en JSON
│           └── pgvector_store.py       # Base vectorial en PostgreSQL + pgvector
├── scripts/
│   ├── download_dataset.py      # Descarga el dataset de Kaggle
│   ├── build_index.py           # Indexa el dataset en la base vectorial (usa data/splits/splits.csv)
│   └── train_classifier.py      # Entrena/evalúa el clasificador (alternativa local a 2-train.ipynb)
├── data/
│   ├── dataset/                 # 70 Dog Breeds Image Dataset, no se versiona
│   ├── splits/splits.csv        # Splits train/valid/test definidos en 1-pre-processing.ipynb
│   └── embeddings.json          # Base vectorial JSON (si USE_PGVECTOR=false)
├── models/                      # Checkpoints entrenados (.pth / .onnx), no se versionan
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente

## Requisitos

- Python 3.12
- Docker (Docker Desktop incluye Docker Compose)
- Git
- Cuenta de Kaggle (para descargar el dataset)

## Dataset

Dataset principal: [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set).

```bash
python scripts/download_dataset.py
```

Requiere [credenciales de Kaggle](https://www.kaggle.com/docs/api) configuradas. Si no las tienen, descarguen el dataset manualmente desde Kaggle y descompriman `train/`, `valid/` y `test/` dentro de `data/dataset`.

## Configurá tus modelos

Entrená tus modelos (Etapa 2) con `2-train.ipynb` (en Google Colab, con GPU) y guardá los checkpoints dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `SIMILARITY_THRESHOLD`) para que apunte a tus modelos.

> El backend arranca aunque falten los checkpoints: `/health` informa cuáles encuentra en `models/`. Las pestañas de las Etapas 2 y 3 fallan hasta que existan.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

Es la forma en que la cátedra evalúa el trabajo.

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

Servicios disponibles: frontend en `http://localhost:8080`, backend en `http://localhost:8000` y PostgreSQL en `localhost:5432`.

### 2. Indexá la galería

Con la Etapa 1 implementada y el dataset en `data/dataset`, indexá el split de entrenamiento desde el contenedor del backend:

```bash
docker compose exec backend python /scripts/build_index.py --split train
```

## Opción 2 - Configurando el ambiente local

### 1. Cloná tu fork

```bash
git clone <url-de-tu-fork>
cd tuia-dog-recognition-app
git checkout -b feature/tp2-razas-perros
```

### 2. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/). También pueden usar `python3.12 -m venv .venv` y `pip`.

### 3. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 4. Activá el ambiente virtual

```bash
# Linux / Mac
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 5. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 6. Configurá las variables de entorno

Ajustá el archivo `.env.local.example` a tus necesidades y copialo a `src/.env`. El backend local y los scripts de `scripts/` leen la configuración desde ese archivo.

```bash
cp .env.local.example src/.env
```

### 7. Descargá el dataset

```bash
python scripts/download_dataset.py
```

### 8. Iniciá la base de datos

```bash
docker compose up postgres -d
```

Solo es necesaria con `USE_PGVECTOR=true` (por defecto). Con `USE_PGVECTOR=false` se usa la base vectorial JSON (`data/embeddings.json`) y no hace falta Docker para trabajar en local.

### 9. Iniciá el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciá el frontend (en otra terminal, con el ambiente virtual activado)

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificá que todo esté corriendo

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000/docs` (Swagger) y `http://localhost:8000/health` (muestra el modelo seleccionado y los checkpoints encontrados)

## Scripts provistos

Se ejecutan desde la raíz del repositorio con el ambiente local activo (y PostgreSQL corriendo si `USE_PGVECTOR=true`). Leen la configuración de `src/.env` si existe:

```bash
# Descargar el dataset de Kaggle a data/dataset
python scripts/download_dataset.py

# Indexar la galería en la base vectorial (requiere la Etapa 1 implementada)
python scripts/build_index.py --split train
# ... o lo mismo desde el contenedor del backend (con docker compose up -d corriendo)
docker compose exec backend python /scripts/build_index.py --split train

# Entrenar y evaluar el clasificador localmente (alternativa a 2-train.ipynb)
python scripts/train_classifier.py --model resnet18_finetuned
```

Con `USE_PGVECTOR=false`, la indexación del split completo es lenta: el JSON se reescribe en cada inserción. Para pruebas rápidas, usen `--limit` (máximo de imágenes por raza).

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten el modelo de embeddings (`EMBEDDING_MODEL`, `EMBEDDING_DIM`), la búsqueda (`SIMILARITY_METRIC`, `SIMILARITY_THRESHOLD`, `TOP_K`), los checkpoints (`RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME`), YOLO (`YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID`) y los paths.
3. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

| Endpoint | Descripción |
|----------|-------------|
| `POST /upload` | Sube una imagen al servidor |
| `POST /search` | Etapa 1: búsqueda por similitud (acepta `model` y `top_k`) |
| `POST /classify` | Etapa 2: clasificación con el modelo entrenado (acepta `model`) |
| `POST /detect` | Etapa 3: detección + clasificación |
| `GET /status/{job_id}` | Estado del procesamiento asincrónico |
| `GET /models` | Modelos de embeddings disponibles |
| `GET /health` | Estado del backend y checkpoints encontrados |

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`) y manejo de jobs.
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3: detección → recorte → clasificación → JSON).
- Selección dinámica del modelo de embeddings: `baseline`, `resnet18_finetuned` o `cnn_custom`.
- Carga de checkpoints (`.pth` / `.onnx`) desde `models/` y selección del modelo activo.
- Similitud configurable (`cosine` o `l2`) y manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Funciones auxiliares de evaluación (`lib/evaluation/metrics.py`) y de visualización (`lib/visualization/draw.py`).
- Frontend Gradio con una pestaña por etapa: Etapa 1 (imagen consultada, Top-K similares, raza predicha y selección de modelo), Etapa 2 (raza y score con el modelo entrenado) y Etapa 3 (bounding boxes, razas y scores).

### Implementado por el grupo

Funciones completadas (ninguna lanza `NotImplementedError`):

| Etapa | Función | Archivo |
|-------|---------|---------|
| 1 | `extract_embedding(image)` | `src/lib/services/similarity_service.py` |
| 1 | `search_similar_images(embedding, top_k)` | `src/lib/services/similarity_service.py` |
| 1 | `predict_breed_from_neighbors(results)` | `src/lib/services/similarity_service.py` |
| 2 | `train_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `evaluate_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `extract_custom_embedding(image)` | `src/lib/services/classifier_service.py` |
| 3 | `detect_dogs(image)` | `src/lib/services/detection_service.py` |
| 3 | `classify_detected_dog(crop)` | `src/lib/services/detection_service.py` |

En las Etapas 1 y 3 el trabajo se limita a completar las funciones. En la Etapa 2, además, el grupo entrena sus propios modelos (Modelo A: fine-tuning de ResNet18/ResNet50; Modelo B: CNN propia, **ambos obligatorios**) y hace un estudio de resultados.

Ver el contrato de cada función, recomendaciones y errores frecuentes en [ENUNCIADO.md](ENUNCIADO.md#24-funciones-a-implementar).


### Pipeline implementado

Todas las etapas usan el mismo preprocesamiento (`src/lib/preprocesamiento.py`): la imagen llega en
BGR (OpenCV), se convierte a RGB, se rellena hasta hacerla cuadrada (sin deformarla ni recortarla),
se lleva a 224×224 y se normaliza con la media y el desvío de ImageNet. Las notebooks y los
servicios importan este módulo, por lo que entrenamiento e inferencia procesan las imágenes igual.

1. **Etapa 1 (búsqueda por similitud).** `extract_embedding` obtiene un vector de 512 dimensiones
   con una ResNet18 preentrenada en ImageNet sin la capa de clasificación, normalizado L2. La
   galería (`train`, 6.001 imágenes) se indexa en PostgreSQL + pgvector. Para cada consulta se
   recuperan los 10 vecinos más similares (coseno) y la raza se decide por voto ponderado por
   similitud. El score es la similitud media de los vecinos de la raza ganadora; si es menor que
   `SIMILARITY_THRESHOLD` (0,78) la respuesta es `unknown`.
2. **Etapa 2 (clasificación).** El Modelo A (ResNet18 ajustada a las 70 razas) clasifica la imagen
   completa y devuelve la raza con su probabilidad (softmax). Los modelos guardan la lista de razas
   dentro del checkpoint, así que la app no depende del dataset para traducir la predicción.
3. **Etapa 3 (detección).** YOLOv8n preentrenado en COCO detecta los perros (clase 16, confianza
   mínima 0,25). Cada caja se recorta y se clasifica con el Modelo A usando el mismo
   preprocesamiento.

### Orden de trabajo sugerido

1. `1-pre-processing.ipynb`: EDA, filtrado, conjunto independiente, preprocesamiento, augmentation y splits.
2. Etapa 1 en `similarity_service.py`. Indexá la galería con `python scripts/build_index.py --split train` y probá la pestaña de la Etapa 1.
3. Etapa 2 en `classifier_service.py`, entrenando y comparando los modelos con `2-train.ipynb`. Descargá los checkpoints a `models/`.
4. Etapa 3 en `detection_service.py` y validación end-to-end con `docker compose up`.

## Notebooks

| Notebook | Contenido |
|----------|-----------|
| `1-pre-processing.ipynb` | Dataset, EDA, control de calidad, conjunto independiente, preprocesamiento, data augmentation y splits |
| `2-train.ipynb` | Evaluación de la búsqueda (NDCG@10), entrenamiento del Modelo A y del Modelo B, métricas, análisis de errores, comparación y exportación de checkpoints |

Se pueden entregar commiteadas en el repositorio o en Google Colab, con el link a la notebook de Colab en una celda Markdown al inicio de la notebook del repositorio. En ambos casos deben estar **ejecutadas y con las salidas visibles**: una notebook sin salidas se considera **no entregada** (ver [ENUNCIADO.md](ENUNCIADO.md#82-notebooks-repositorio-o-google-colab)).

## Dataset

Detalle completo, con gráficos y ejemplos, en `1-pre-processing.ipynb`.

- **Dataset:** 70 Dog Breeds Image Dataset (Kaggle, licencia CC0), 9.346 imágenes JPEG de 224×224.
  Se corrige el nombre de la carpeta `American  Spaniel` (doble espacio) en `valid`.
- **Control de calidad** (aplicado a todos los splits):

  | Regla | Descartadas |
  |---|---|
  | Archivo corrupto | 0 |
  | Resolución menor a 112 px | 0 |
  | Borrosa (varianza del Laplaciano < 30) | 5 |
  | Misma foto en carpetas de razas distintas | 44 |
  | Duplicado o casi duplicado (coseno > 0,95 entre embeddings) | 723 |
  | **Total** | **772 (8,3%)** |

  Los splits originales tenían fuga de datos: el 17% de `valid` y de `test` tenía una copia en
  `train`.
- **Splits:** las 8.574 imágenes restantes se dividen 70/15/15, estratificado por raza y con
  semilla 42: train 6.001, valid 1.286 y test 1.287 (entre 11 y 31 imágenes por raza en valid y
  test). Quedan en `data/splits/splits.csv`, que usan las notebooks, `train_classifier` y
  `scripts/build_index.py`.
- **Conjunto independiente:** 162 imágenes de 56 razas (Oxford-IIIT Pet, CC BY-SA 4.0, y Stanford
  Dogs, uso académico), 10 escenas con varios perros, personas u objetos, y 19 imágenes sin perros
  (incluye 15 gatos). Se descartaron las imágenes con un casi duplicado en el dataset. Fuentes y
  licencias en `independiente.csv` y en la sección 13 de `1-pre-processing.ipynb`.
  Link de solo lectura: https://drive.google.com/drive/folders/1qWz0Yv6uGes9m9oyMNONr8NX6CvlzQD0
- **Data augmentation** (solo `train`): recorte aleatorio (70–100% del área), flip horizontal,
  rotación de ±15°, brillo y contraste de ±20%, desenfoque y ruido leves.

## Modelos y entrenamiento

Detalle en `2-train.ipynb` (ejecutada en Kaggle con GPU T4).

- **Embeddings de la Etapa 1 (baseline):** ResNet18 de ImageNet sin la capa final (512 dimensiones,
  igual a `EMBEDDING_DIM`). Se eligió por su buen rendimiento sin entrenamiento y porque es liviana
  para CPU.
- **Modelo A (`resnet18_finetuned`):** ResNet18 preentrenada con una capa final de 70 salidas. Se
  compararon *feature extraction* (87,1% en valid), *fine-tuning* completo con learning rates
  diferenciados (1e-4 en la red preentrenada y 1e-3 en la capa nueva, 91,0%) y fine-tuning con
  *class weights* (91,1%, elegido).
- **Modelo B (`cnn_custom`):** CNN propia de 5 bloques convolucionales (dos convoluciones 3×3 con
  BatchNorm y ReLU, y MaxPool), global average pooling, capa de embedding de 512 y dropout 0,3
  (5,0 M de parámetros). Se compararon 3 experimentos (4 bloques sin augmentation, 4 bloques con
  augmentation y 5 bloques con augmentation).
- **Hiperparámetros:** AdamW (weight decay 1e-4), scheduler coseno, batch size 64, label smoothing
  0,1, early stopping por accuracy en valid (se guarda la mejor época). Modelo A: hasta 10 épocas;
  Modelo B: 30 épocas.
- **Detector:** YOLOv8n preentrenado en COCO, sin entrenar. `YOLO_CONF_THRESHOLD = 0.25`: con 0,15
  no detecta más perros y con 0,40 pierde perros lejanos sin eliminar falsos positivos.
- **Umbral de similitud:** `SIMILARITY_THRESHOLD = 0.78`, percentil 1 del score de las
  predicciones correctas en valid. Rechaza 17 de 19 imágenes sin perro y el 1% de las consultas
  de test.
- **Configuración final:** `EMBEDDING_MODEL=baseline`, `EMBEDDING_DIM=512`, `TOP_K=10`; la app
  clasifica con el Modelo A.
- **Checkpoints** (link público de solo lectura): https://drive.google.com/drive/folders/1oszseGN2kq3BSq7K7fjzWvzhb_ryYmoc
  (`resnet18_finetuned.pth` y `cnn_custom.pth`, a copiar en `models/`).

## Resultados

**Etapa 1: búsqueda por similitud** (galería `train`, TOP_K = 10)

| Embeddings | NDCG@10 test | NDCG@10 independiente | Accuracy test | Accuracy independiente |
|---|---|---|---|---|
| Baseline (ImageNet) | 0,819 | 0,698 | 88,7% | 77,8% |
| Modelo A | 0,880 | 0,630 | 90,8% | 65,4% |
| Modelo B | 0,560 | 0,161 | 63,8% | 21,0% |

Cosine y l2 dan los mismos resultados porque los embeddings tienen norma 1.

**Etapa 2: clasificación** (métricas macro)

| Métrica | Modelo A (ResNet18 fine-tuned) | Modelo B (CNN propia) |
|---|---|---|
| Accuracy (test) | 89,8% | 65,4% |
| Precision / Recall / F1 (test) | 90,4% / 89,9% / 89,8% | 67,3% / 65,2% / 65,3% |
| Specificity (test) | 99,85% | 99,5% |
| Accuracy (independiente) | 59,9% | 17,3% |
| Parámetros / tamaño | 11,2 M / 42,8 MB | 5,0 M / 19,1 MB |
| Inferencia en CPU | 29,9 ms/imagen | 46,2 ms/imagen |

**Conclusiones**

- La app clasifica con el **Modelo A** y busca con los **embeddings del baseline**: los del Modelo A
  son mejores en test, pero el baseline es más robusto con fotos de otras fuentes, que es lo que
  recibe la app.
- La CNN propia rinde ≈24 puntos menos: no tiene preentrenamiento, la tarea es de grano fino y hay
  ≈86 imágenes por raza.
- Los errores se concentran en razas muy parecidas (Lhasa/Shih-Tzu, Labradoodle/Cockapoo, American
  Hairless/Mex Hairless) y en etiquetas erróneas del dataset (la carpeta "Bulldog" contiene Boston
  Terriers y Bulldogs Franceses).
- **Detección:** 8 de 10 escenas con el conteo de perros exacto (incluye perros superpuestos y
  escenas con personas u objetos). Sin falsos positivos con gatos; sí con un zorro y un peluche con
  forma de perro. Fallos con cachorros amontonados y con perros lejanos.

## Problemas encontrados y soluciones

- **Fuga de datos en el dataset original:** el 17% de valid y test tenía copias en train. Se
  deduplicó con embeddings y se redefinieron los splits.
- **`build_index.py` indexaba las carpetas originales:** con los splits nuevos, eso habría
  indexado imágenes de nuestro test. Se modificó para leer `data/splits/splits.csv`.
- **Caída fuera del dominio:** el Modelo A baja de 89,8% en test a 59,9% en el conjunto
  independiente. Se diagnosticó que el modelo nunca vio bandas de relleno, porque todas las
  imágenes del dataset son cuadradas: con recorte central sube a 72,2%. Se mantuvo el relleno
  (necesario para los recortes de YOLO) y se propone como mejora entrenar con relleno y encuadres
  aleatorios.
- **Sin GPU en Colab:** el entrenamiento se hizo en Kaggle. La Etapa 1 dio los mismos resultados en
  ambos entornos.

## Cómo reproducir la entrega

1. Descargar el dataset de Kaggle y descomprimir `train/`, `valid/` y `test/` en `data/dataset/`.
2. Descargar los checkpoints del link de la sección *Modelos y entrenamiento* y copiarlos en
   `models/`.
3. `docker compose build` y `docker compose up -d`.
4. Indexar la galería: `docker compose exec backend python /scripts/build_index.py --split train`.
5. Abrir el frontend en `http://localhost:8080`.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- `EMBEDDING_DIM` debe coincidir con la dimensión del embedding del modelo elegido. Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- La base vectorial guarda los embeddings de un modelo a la vez: al cambiar de modelo de embeddings hay que reindexar (`scripts/build_index.py`).
- Las rutas de la galería se guardan relativas a `DATA_PATH` (por ejemplo, `dataset/train/<raza>/<imagen>.jpg`): una base indexada desde el entorno local también sirve para el backend en Docker, porque comparten PostgreSQL y la carpeta `data/`.
- Los modelos (`*.pth`, `*.pt`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen los checkpoints en `models/` y súbanlos a un link **público de solo lectura** para que los docentes puedan descargarlos.
- Los pesos preentrenados (torchvision/timm y YOLO) se descargan en el primer uso, por lo que se necesita conexión a internet.

## Entrega

- Pull Request desde su feature branch hacia `main` **de su propio repositorio**, **sin mergear**, con el link subido al campus. No abran el PR contra el repositorio de la cátedra.
- Si el repositorio es privado, los docentes deben tener acceso.
- Notebooks `1-pre-processing.ipynb` y `2-train.ipynb` ejecutadas, con sus salidas.
- Para aprobar, el sistema debe andar sin errores con:

```bash
docker compose build
docker compose up
```

**Fecha de entrega:** 27/11/2026, 23:59. Solo se consideran los commits hasta esa fecha y hora.

El detalle del contenido del PR, la plantilla de descripción y el checklist están en [ENUNCIADO.md](ENUNCIADO.md#8-entrega-pull-request).
