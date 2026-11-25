from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import onnxruntime

from lib.evaluation.metrics import precision_recall_f1, specificity
from lib.preprocesamiento import aumentacion_entrenamiento, leer_imagen, preprocesar

logger = logging.getLogger(__name__)

# Hiperparámetros por defecto de cada modelo; train_classifier(**cambios) permite modificarlos
# desde la notebook sin tocar el código (y quedan registrados en el historial del entrenamiento).
HIPERPARAMETROS: dict[str, dict[str, Any]] = {
    "resnet18_finetuned": {
        "estrategia": "fine_tuning",  # fine_tuning | feature_extraction
        "epocas": 10,
        "batch_size": 64,
        "lr": 1e-3,  # cabeza (capa final)
        "lr_backbone": 1e-4,  # resto de la red (solo en fine_tuning)
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "paciencia": 3,  # early stopping: épocas sin mejorar en valid
        "aumentacion": True,
        "class_weights": False,
        "semilla": 42,
    },
    "cnn_custom": {
        "canales": (32, 64, 128, 256, 512),
        "dropout": 0.3,
        "epocas": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "paciencia": 6,
        "aumentacion": True,
        "class_weights": False,
        "semilla": 42,
    },
}


class _ImagenesRaza(torch.utils.data.Dataset):
    """Lee las imágenes de una lista de rutas y aplica el preprocesamiento compartido."""

    def __init__(self, rutas: list[str], etiquetas: list[int], image_size: int, aumentacion=None) -> None:
        self.rutas, self.etiquetas = rutas, etiquetas
        self.image_size, self.aumentacion = image_size, aumentacion

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, i: int):
        x = preprocesar(leer_imagen(self.rutas[i]), self.image_size, self.aumentacion)
        return x, self.etiquetas[i]


def _fijar_semillas(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}
        # Resultado de la última evaluación (matriz de confusión, predicciones) para la notebook
        self.ultima_evaluacion: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Auxiliares de la Etapa 2
    # ------------------------------------------------------------------

    def _leer_split(self, split: str) -> tuple[list[str], list[str]]:
        """Rutas y razas de un split de data/splits/splits.csv (definido en 1-pre-processing.ipynb)."""
        archivo = self.dataset_path.parent / "splits" / "splits.csv"
        if not archivo.is_file():
            raise ValueError(f"No existe {archivo}: generar los splits en 1-pre-processing.ipynb")
        with open(archivo, encoding="utf-8") as f:
            filas = [fila for fila in csv.DictReader(f) if fila["split"] == split]
        return [str(self.dataset_path / fila["ruta"]) for fila in filas], [fila["raza"] for fila in filas]

    @staticmethod
    def _dispositivo() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _crear_modelo(self, razas: list[str], hp: dict[str, Any]) -> torch.nn.Module:
        from lib.models.clasificadores import CNNPropia, ResNetRazas

        if self.active_model_name == "resnet18_finetuned":
            modelo = ResNetRazas(razas)
            if hp["estrategia"] == "feature_extraction":
                for p in modelo.backbone.parameters():
                    p.requires_grad = False  # solo se entrena la capa final
            return modelo
        if self.active_model_name == "cnn_custom":
            return CNNPropia(razas, canales=tuple(hp["canales"]), dropout=hp["dropout"])
        raise ValueError(f"Modelo desconocido: {self.active_model_name}")

    def _grupos_parametros(self, modelo: torch.nn.Module, hp: dict[str, Any]) -> list[dict[str, Any]]:
        """Learning rates diferenciados: más bajo para la ResNet preentrenada, más alto para la cabeza."""
        if self.active_model_name == "resnet18_finetuned" and hp["estrategia"] == "fine_tuning":
            return [
                {"params": modelo.backbone.parameters(), "lr": hp["lr_backbone"]},
                {"params": modelo.clasificador.parameters(), "lr": hp["lr"]},
            ]
        return [{"params": [p for p in modelo.parameters() if p.requires_grad], "lr": hp["lr"]}]

    def predecir_imagenes(self, rutas: list[str], batch_size: int = 64) -> np.ndarray:
        """Probabilidades (softmax) del modelo activo para una lista de imágenes: (n, n_razas)."""
        modelo = self.load_model()
        dispositivo = self._dispositivo()
        modelo.to(dispositivo).eval()
        datos = _ImagenesRaza(list(rutas), [0] * len(rutas), self.image_size)
        cargador = torch.utils.data.DataLoader(datos, batch_size=batch_size, num_workers=2)
        salidas = []
        with torch.no_grad():
            for x, _ in cargador:
                salidas.append(torch.softmax(modelo(x.to(dispositivo)), dim=1).cpu().numpy())
        modelo.to("cpu")
        return np.concatenate(salidas)

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self, ruta_checkpoint: Optional[Path] = None, **cambios: Any) -> None:
        """
        Entrena el modelo activo (resnet18_finetuned | cnn_custom) con los splits de
        data/splits/splits.csv, el preprocesamiento compartido y la augmentation solo en train.

        - Hiperparámetros: HIPERPARAMETROS[modelo], modificables con `cambios`
          (ej: train_classifier(estrategia="feature_extraction", epocas=8)).
        - Guarda el **mejor modelo según accuracy en valid** (early stopping) como modelo completo
          (torch.save(model)) en self.active_checkpoint o en `ruta_checkpoint`.
        - Guarda el historial por época en output_path/historial_<checkpoint>.json.
        """
        hp = {**HIPERPARAMETROS[self.active_model_name], **cambios}
        _fijar_semillas(hp["semilla"])
        destino = Path(ruta_checkpoint) if ruta_checkpoint else self.active_checkpoint
        dispositivo = self._dispositivo()

        rutas_tr, razas_tr = self._leer_split("train")
        rutas_va, razas_va = self._leer_split("valid")
        razas = sorted(set(razas_tr))
        indice = {r: i for i, r in enumerate(razas)}
        y_tr = [indice[r] for r in razas_tr]
        y_va = [indice[r] for r in razas_va]

        aumentacion = aumentacion_entrenamiento(self.image_size) if hp["aumentacion"] else None
        generador = torch.Generator().manual_seed(hp["semilla"])
        cargador_tr = torch.utils.data.DataLoader(
            _ImagenesRaza(rutas_tr, y_tr, self.image_size, aumentacion),
            batch_size=hp["batch_size"], shuffle=True, num_workers=2, generator=generador,
            pin_memory=dispositivo.type == "cuda",
        )
        cargador_va = torch.utils.data.DataLoader(
            _ImagenesRaza(rutas_va, y_va, self.image_size),
            batch_size=hp["batch_size"], num_workers=2, pin_memory=dispositivo.type == "cuda",
        )

        modelo = self._crear_modelo(razas, hp).to(dispositivo)
        pesos = None
        if hp["class_weights"]:
            # Peso inversamente proporcional a la cantidad de imágenes de cada raza (media = 1)
            conteo = np.bincount(y_tr, minlength=len(razas)).astype(np.float32)
            pesos = torch.tensor(conteo.sum() / (len(razas) * conteo), device=dispositivo)
        criterio = torch.nn.CrossEntropyLoss(weight=pesos, label_smoothing=hp["label_smoothing"])
        optimizador = torch.optim.AdamW(self._grupos_parametros(modelo, hp), weight_decay=hp["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=hp["epocas"])
        escalador = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")  # precisión mixta en GPU

        historial: list[dict[str, float]] = []
        mejor_acc, sin_mejorar = -1.0, 0
        destino.parent.mkdir(parents=True, exist_ok=True)
        for epoca in range(1, hp["epocas"] + 1):
            inicio = time.time()
            modelo.train()
            perdida_tr, aciertos_tr, n_tr = 0.0, 0, 0
            for x, y in cargador_tr:
                x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
                optimizador.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dispositivo.type, enabled=dispositivo.type == "cuda"):
                    logits = modelo(x)
                    perdida = criterio(logits, y)
                escalador.scale(perdida).backward()
                escalador.step(optimizador)
                escalador.update()
                perdida_tr += perdida.item() * len(y)
                aciertos_tr += (logits.argmax(1) == y).sum().item()
                n_tr += len(y)
            scheduler.step()

            modelo.eval()
            perdida_va, aciertos_va, n_va = 0.0, 0, 0
            with torch.no_grad():
                for x, y in cargador_va:
                    x, y = x.to(dispositivo), y.to(dispositivo)
                    logits = modelo(x)
                    perdida_va += criterio(logits, y).item() * len(y)
                    aciertos_va += (logits.argmax(1) == y).sum().item()
                    n_va += len(y)

            fila = {
                "epoca": epoca,
                "loss_train": perdida_tr / n_tr, "acc_train": aciertos_tr / n_tr,
                "loss_valid": perdida_va / n_va, "acc_valid": aciertos_va / n_va,
                "segundos": round(time.time() - inicio, 1),
            }
            historial.append(fila)
            logger.info("Época %d: %s", epoca, {k: round(v, 4) for k, v in fila.items()})

            if fila["acc_valid"] > mejor_acc:
                mejor_acc, sin_mejorar = fila["acc_valid"], 0
                modelo.to("cpu")
                modelo.image_size = self.image_size
                torch.save(modelo, destino)  # modelo completo, con la lista de razas
                modelo.to(dispositivo)
            else:
                sin_mejorar += 1
                if sin_mejorar >= hp["paciencia"]:
                    logger.info("Early stopping en la época %d", epoca)
                    break

        self.output_path.mkdir(parents=True, exist_ok=True)
        registro = {"modelo": self.active_model_name, "checkpoint": str(destino),
                    "hiperparametros": {k: list(v) if isinstance(v, tuple) else v for k, v in hp.items()},
                    "mejor_acc_valid": mejor_acc, "historial": historial}
        (self.output_path / f"historial_{destino.stem}.json").write_text(
            json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
        # La próxima carga lee el checkpoint recién guardado
        self._loaded.pop(self.active_model_name, None)
        logger.info("Mejor accuracy en valid: %.4f → %s", mejor_acc, destino)

    def evaluate_classifier(
        self, rutas: Optional[list[str]] = None, razas_reales: Optional[list[str]] = None
    ) -> dict[str, float]:
        """
        Evalúa el modelo activo sobre `test` (o sobre las imágenes indicadas, ej. el conjunto
        independiente). Métricas macro: precision, recall y F1 se calculan por raza y se
        promedian; la especificidad de cada raza es one-vs-rest TN / (TN + FP).
        La matriz de confusión y las predicciones quedan en self.ultima_evaluacion.
        """
        if rutas is None:
            rutas, razas_reales = self._leer_split("test")
        razas = list(self.load_model().razas)
        indice = {r: i for i, r in enumerate(razas)}
        y_real = np.array([indice[r] for r in razas_reales])
        probas = self.predecir_imagenes(rutas)
        y_pred = probas.argmax(1)

        n = len(razas)
        matriz = np.zeros((n, n), dtype=int)
        np.add.at(matriz, (y_real, y_pred), 1)
        # Promedio macro sobre las razas presentes en el conjunto evaluado
        presentes = np.unique(y_real)
        prec, rec, f1s, esp = [], [], [], []
        for c in presentes:
            tp = matriz[c, c]
            fp = matriz[:, c].sum() - tp
            fn = matriz[c, :].sum() - tp
            tn = matriz.sum() - tp - fp - fn
            p, r, f = precision_recall_f1(tp, fp, fn)
            prec.append(p); rec.append(r); f1s.append(f); esp.append(specificity(tn, fp))

        self.ultima_evaluacion = {"razas": razas, "matriz": matriz, "y_real": y_real,
                                  "y_pred": y_pred, "probas": probas, "rutas": list(rutas)}
        return {
            "accuracy": float((y_real == y_pred).mean()),
            "precision": float(np.mean(prec)),
            "recall": float(np.mean(rec)),
            "specificity": float(np.mean(esp)),
            "f1": float(np.mean(f1s)),
        }

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Embedding de la capa previa al clasificador del modelo activo (ResNet fine-tuned o CNN
        propia), con el mismo preprocesamiento del entrenamiento y normalizado L2.
        Recibe BGR uint8 y devuelve EMBEDDING_DIM floats.
        """
        modelo = self.load_model()
        if not hasattr(modelo, "embedding"):
            raise ValueError("El checkpoint activo no expone embedding() (se requiere un .pth de lib.models)")
        modelo.eval()
        x = preprocesar(image, self.image_size).unsqueeze(0)
        with torch.no_grad():
            emb = modelo.embedding(x)[0].numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb.astype(float).tolist()
