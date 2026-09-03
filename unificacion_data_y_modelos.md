# Instrucciones de Unificación de Rutas y Corrección de Formato de Modelo para Antigravity

**DIRIGIDO A:** Antigravity (Agente de Programación y Optimización Autónoma)  
**ROL:** Arquitecto de Software Principal & Ingeniero de ML  
**PROYECTO:** Ajuste de Rutas y Formato de Redes en el Traductor de Lengua de Señas Peruana (LSP)

---

### 🚨 DIAGNÓSTICO DEL ERROR DE CARGA
El error `"no encuentra el modelo .h5 números"` y el conflicto de codificación UTF-8 ocurren por dos razones críticas en el codebase actual:
1.  **Dispersión de Carpetas (Path Mismatch):** El módulo de entrenamiento (`model_trainer.py` o `trainer.py`) está exportando el modelo en una carpeta (ej. `modelos_exportados/`), pero el servicio de pruebas en vivo (`tester_service.py` o `app.py`) lo busca en una ruta distinta o bajo una carpeta antigua del repositorio `TraductorLSP`.
2.  **Conflicto de Formato Legacy (.h5 vs .keras):** Keras 3 / TensorFlow 2.16+ marcan el formato `.h5` como heredado (legacy) y generan advertencias de consola. Debemos estandarizar **todo el sistema** para guardar y cargar únicamente el formato oficial nativo: **`model.keras`**.

---

### 📂 ARQUITECTURA DE DIRECTORIOS PROPUESTA (EL ESTÁNDAR)

Para mantener la máxima limpieza, modularidad y rigor académico en tu tesis, **todas las muestras físicas y los modelos entrenados deben vivir dentro de una única carpeta raíz llamada `data/`**. Esto evita contaminar la carpeta de código fuente `src/` con datos binarios y facilita la configuración del archivo `.gitignore`.

Debes reestructurar el sistema para que use exactamente este árbol de directorios:

```text
traductor-lsp-admin/
├── src/                          # Código fuente puro (Síncrono y Asíncrono)
│   ├── app.py                    # Punto de entrada
│   ├── ui_components.py          # Interfaz Flet (Celeste y Blanco)
│   ├── vision_service.py         # OpenCV + MediaPipe
│   ├── voice_service.py          # Vosk (Comando "recopila")
│   ├── normalizer.py             # 37 puntos faciales y anclaje en el cuello
│   ├── model_trainer.py          # Entrenamiento de CNN 1D
│   └── tester_service.py         # Inferencia en tiempo real (Sliding Window)
│
└── data/                         # Capa de Persistencia Única (Ignorada en Git si pesa mucho)
    ├── muestras/                 # Reemplaza a 'data_historica' o carpetas sueltas
    │   └── numeros/              # Categoría
    │       ├── metadata.json     # Metadatos del dataset
    │       ├── uno/              # Palabra
    │       │   ├── seq_0.csv     # Coordenadas normalizadas (30, 159)
    │       │   └── ...
    │       └── dos/
    │
    └── modelos/                  # Reemplaza a 'modelos_exportados'
        └── numeros/              # Modelos locales entrenados
            ├── labels.json       # Mapeo numérico de palabras {"0": "uno", "1": "dos"}
            ├── model.keras       # Modelo binario oficial y único
            └── tfjs_model/       # Modelo compilado para la web de los alumnos
                ├── model.json
                └── ...
```

---

### 🛠️ IMPLEMENTACIÓN DE CAMBIOS EN CÓDIGO (ÓRDENES IMPERATIVAS)

#### 1. Unificación en `app.py` (Punto de Entrada)
Al inicializar las clases de servicio en `src/app.py`, debes inyectar las rutas de la carpeta centralizada `data/` para garantizar la consistencia absoluta:

```python
# Inicialización limpia en src/app.py
from src.data_manager import LSPDatasetManager
from src.model_trainer import LSPTrainer
from src.tester_service import LSPTesterService

# 1. El gestor de datos apunta a las muestras
db_manager = LSPDatasetManager(base_dir="data/muestras")

# 2. El entrenador apunta al gestor y guarda en la carpeta de modelos de data/
trainer = LSPTrainer(db_manager, export_base_dir="data/modelos")

# 3. El probador en vivo busca los modelos en la misma carpeta unificada
tester_service = LSPTesterService(model_base_dir="data/modelos")
```

#### 2. Guardado Exclusivo en Formato `.keras` (`src/model_trainer.py`)
En el archivo de entrenamiento de la red CNN 1D, asegúrate de guardar el modelo con la extensión `.keras` y no `.h5`:

```python
# Corrección en la exportación del entrenamiento
category_export_path = os.path.join(self.export_base_dir, category_name.lower().strip())
keras_model_path = os.path.join(category_export_path, "model.keras")

# Guardar de forma nativa en Keras
model.save(keras_model_path)
print(f"[ENTRENAMIENTO] Modelo guardado en formato moderno: {keras_model_path}")
```

#### 3. Carga Segura y Desacoplada (`src/tester_service.py`)
Reescribe por completo el método de carga de tu servicio de pruebas en vivo. Debe cargar de forma independiente el archivo de texto plano (`labels.json`) y los pesos binarios (`model.keras`), anulando por completo el error de codificación UTF-8:

```python
# Corrección de importación en src/tester_service.py
import os
import json
import tensorflow as tf

class LSPTesterService:
    def __init__(self, model_base_dir="data/modelos"):
        self.model_base_dir = model_base_dir
        self.model = None
        self.labels = {}

    def load_trained_model(self, category_name):
        """Carga el modelo binario y su mapeo de etiquetas de texto."""
        category_dir = os.path.join(self.model_base_dir, category_name.lower().strip())
        model_path = os.path.join(category_dir, "model.keras")
        labels_path = os.path.join(category_dir, "labels.json")

        # 1. Validaciones físicas previas (Evita colisiones de lectura)
        if not os.path.exists(model_path):
            # Fallback por si existe un modelo antiguo .h5 en la transición
            alt_path = os.path.join(category_dir, "model.h5")
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"No se encontró el modelo 'model.keras' en {category_dir}")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"No se encontró el archivo de etiquetas 'labels.json' en {category_dir}")

        # 2. Carga del Modelo Binario (EXCLUSIVAMENTE con la API de Keras)
        try:
            # Forzar la carga limpia del archivo binario comprimido
            self.model = tf.keras.models.load_model(model_path)
            print(f"[TESTER] Modelo binario importado de forma segura desde: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error al abrir los pesos del modelo con Keras: {str(e)}")

        # 3. Carga del archivo de Etiquetas de Texto (Utilizando UTF-8 limpio)
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            print(f"[TESTER] Archivo de mapeo de clases decodificado: {self.labels}")
        except Exception as e:
            raise RuntimeError(f"Error al decodificar etiquetas JSON: {str(e)}")

        return True
```

---

### 🤖 ORDEN DE EJECUCIÓN AUTÓNOMA
1.  **Refactoriza las rutas** en todos los módulos de servicios para unificar la persistencia bajo la carpeta `/data`.
2.  **Corrige el cargador** en `tester_service.py` con el código provisto arriba para erradicar el crash de codificación UTF-8.
3.  **Ejecuta el sistema** con `flet run src/app.py` para probar de extremo a extremo que el entrenamiento guarde en la nueva ruta y la pestaña de pruebas cargue el modelo de forma inmediata y sin trabas.
