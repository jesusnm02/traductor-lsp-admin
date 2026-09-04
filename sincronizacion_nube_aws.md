# Directrices de Integración de Nube (AWS S3) y Conversión a TensorFlow.js (v1)

**DIRIGIDO A:** Antigravity (Agente de Desarrollo y Sincronización)  
**PROYECTO:** Traductor de Lengua de Señas Peruana (LSP) - Módulo Administrador  
**ESTADO DE INFRAESTRUCTURA:** El archivo `.env` ya está creado y configurado en la raíz con credenciales seguras de IAM y acceso al Bucket S3.

---

## 🚨 OBJETIVO TÉCNICO DE ESTA ITERACIÓN
Debes implementar de forma autónoma el módulo de **Sincronización en la Nube (AWS S3)** en la aplicación de escritorio de Flet (`traductor-lsp-admin`). Esto permitirá al docente:
1.  Ver en tiempo real qué categorías de señas están sincronizadas, cuáles solo existen en la PC local y cuáles están desactualizadas.
2.  Convertir automáticamente el modelo entrenado (`model.keras`) al formato optimizado para web de **TensorFlow.js (TF.js)** (`model.json` y fragmentos binarios `shard*`).
3.  Subir y desplegar el compilado de TF.js y el diccionario `labels.json` a AWS S3 con un solo clic.
4.  Eliminar o reemplazar modelos existentes en S3 directamente desde la interfaz gráfica.

---

## 🛠️ REQUISITOS TÉCNICOS E IMPERATIVOS

### 1. Gestión de Dependencias Locales (Entorno de Windows)
Asegúrate de que el entorno virtual del usuario (`venv`) cuente con las librerías necesarias. Ejecuta o verifica de forma silenciosa antes de levantar la app:
```bash
pip install boto3 python-dotenv tensorflowjs
```

### 2. Estructura de Rutas de Sincronización en S3
Toda la Sincronización con S3 debe replicar la arquitectura de persistencia local en `data/modelos/`. El bucket de S3 se organizará de la siguiente forma para que el cliente web del alumno pueda consumirlo directamente [17]:
```text
s3://<AWS_BUCKET_NAME>/modelos/
                         └── [nombre_categoria]/
                               ├── labels.json (Mapeo de clases {"0": "UNO", "1": "DOS"})
                               └── tfjs_model/
                                     ├── model.json (Estructura de la red neuronal)
                                     └── shard1of1.bin (Pesos binarios optimizados)
```

---

## 💻 IMPLEMENTACIÓN SUGERIDA: CONTROLADOR DE CLOUD (`cloud_service.py`)

Crea o integra en tu backend un módulo de control de AWS que administre la lógica con `boto3`. Asegúrate de leer el archivo `.env` usando `dotenv`:

```python
import os
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

load_dotenv()

class LSPCloudService:
    def __init__(self):
        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )

    def verify_local_model(self, category_name):
        """Verifica si el modelo local .keras existe."""
        path = os.path.join("data", "modelos", category_name, "model.keras")
        return os.path.exists(path)

    def check_cloud_status(self, category_name):
        """
        Consulta S3 para determinar el estado de sincronización.
        Retorna: 'NO_SUBIDO', 'PUBLICADO' o 'DESACTUALIZADO'
        """
        if not self.verify_local_model(category_name):
            return "SIN_MODELO_LOCAL"

        prefix = f"modelos/{category_name}/tfjs_model/model.json"
        local_model_path = os.path.join("data", "modelos", category_name, "model.keras")
        
        try:
            # Verificar si existe el archivo en S3
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' not in response:
                return "NO_SUBIDO"
            
            # Si existe, comparar marcas de tiempo para ver si se reentrenó en local
            s3_meta = response['Contents'][0]
            s3_time = s3_meta['LastModified'].timestamp()
            local_time = os.path.getmtime(local_model_path)
            
            # Si el modelo local es más nuevo por más de 5 segundos, está desactualizado en la nube
            if local_time > (s3_time + 5):
                return "DESACTUALIZADO"
                
            return "PUBLICADO"
            
        except Exception as e:
            print(f"[CLOUD] Error al verificar estado en S3: {str(e)}")
            return "ERROR_CONEXION"

    def convert_to_tfjs(self, category_name):
        """Compila el modelo .keras nativo a formato comprimido TensorFlow.js"""
        model_dir = os.path.join("data", "modelos", category_name)
        keras_path = os.path.join(model_dir, "model.keras")
        tfjs_dir = os.path.join(model_dir, "tfjs_model")
        
        os.makedirs(tfjs_dir, exist_ok=True)
        
        # Comando para invocar el compilador oficial de TFJS en el entorno local
        command = [
            "tensorflowjs_converter",
            "--input_format", "keras",
            keras_path,
            tfjs_dir
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("[CLOUD] Conversión exitosa a TensorFlow.js")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[CLOUD] Fallo en la conversión de TFJS: {e.stderr}")
            raise RuntimeError(f"Error del conversor TensorFlow.js: {e.stderr}")

    def upload_category_model(self, category_name):
        """Sube la carpeta tfjs_model y el archivo labels.json a S3"""
        model_dir = os.path.join("data", "modelos", category_name)
        labels_path = os.path.join(model_dir, "labels.json")
        tfjs_dir = os.path.join(model_dir, "tfjs_model")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError("No se encontró el archivo de etiquetas labels.json")
            
        # 1. Asegurar la conversión de TFJS antes de subir [22]
        self.convert_to_tfjs(category_name)
        
        try:
            # 2. Subir labels.json
            s3_labels_key = f"modelos/{category_name}/labels.json"
            self.s3_client.upload_file(labels_path, self.bucket_name, s3_labels_key)
            
            # 3. Subir todos los archivos compilados de TFJS (model.json y fragmentos shard*)
            for file_name in os.listdir(tfjs_dir):
                local_file = os.path.join(tfjs_dir, file_name)
                s3_key = f"modelos/{category_name}/tfjs_model/{file_name}"
                self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
                
            print(f"[CLOUD] Sincronización exitosa de: {category_name}")
            return True
        except Exception as e:
            raise RuntimeError(f"Fallo en la carga de archivos a AWS: {str(e)}")

    def delete_category_model(self, category_name):
        """Elimina todos los archivos del modelo en S3 correspondientes a la categoría"""
        prefix = f"modelos/{category_name}/"
        try:
            # Listar todos los objetos bajo el prefijo
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    delete_us = [{'Key': obj['Key']} for obj in page['Contents']]
                    self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': delete_us})
            print(f"[CLOUD] Eliminada categoría {category_name} de S3.")
            return True
        except Exception as e:
            raise RuntimeError(f"Error al borrar el modelo de S3: {str(e)}")
```

---

## 🎨 INTERFAZ EN FLET: PESTAÑA DE GESTIÓN DE NUBE
Debes integrar una tercera pestaña llamada **"Nube / AWS"** o integrar estos controles de forma compacta en tu administrador. El diseño debe respetar la paleta escolar celeste, blanca y azul académico, adaptándose perfectamente al viewport sin desbordamientos.

### Requisitos visuales de la ventana de sincronización:
1.  **Encabezado de Nube:** Un banner con un ícono de nube (`ft.icons.CLOUD_SYNC`) en color celeste `#4A90E2` con el título: `"PANEL DE SINCRONIZACIÓN Y DESPLIEGUE CLOUD"`.
2.  **Tabla de Categorías (`ft.DataTable` o `ft.ListView`):**
    *   **Categoría:** Texto en negrita azul académico (`#1A365D`).
    *   **Estado en PC:** Icono de check verde si existe `model.keras` localmente; cruz roja si no hay modelo entrenado aún.
    *   **Estado en la Nube:** Un Badge visual e interactivo:
        *   ☁️ **"No Publicado"** (Fondo gris suave `#ECEFF1`, texto gris oscuro)
        *   🚀 **"Sincronizado"** (Fondo verde pedagógico claro `#E8F5E9`, texto verde oscuro `#2E7D32`)
        *   ⚠️ **"Desactualizado"** (Fondo amarillo/ámbar claro `#FFF3E0`, texto ámbar `#E65100` - indica que el modelo local fue reentrenado después de subirlo).
3.  **Botones de Acción Dinámicos por Fila:**
    *   **Subir / Actualizar (`ft.IconButton` o `ft.ElevatedButton`):** Dispara la conversión y carga en S3. Cambia visualmente a spinner de carga mientras se ejecuta.
    *   **Eliminar (`ft.IconButton`):** Icono de basura en color rojo amigable `#E25C5C` para borrar la categoría en la nube.
4.  **Consola Inferior de Progreso:** Una barra de carga lineal celeste (`ft.ProgressBar`) que se active durante la transferencia, acompañada de un texto indicador de transferencia en tiempo real.

---

## 🛡️ SISTEMA DE EXCEPCIONES Y SEGURIDAD CON POPUP

Toda llamada que involucre comunicación de red con AWS debe ejecutarse en hilos asíncronos para evitar bloquear la UI de Flet. Además, debe estar fuertemente encapsulada en bloques seguros:
1.  **Captura de Excepciones:** Si el usuario no tiene internet, si las credenciales en `.env` son incorrectas, o si falla la conversión de TensorFlow.js, el hilo **no debe colapsar**.
2.  **Popup Emergente de Diagnóstico (`ft.AlertDialog`):**
    *   Usa un cuadro de diálogo con borde celeste `#D1E4F8` e icono de alerta.
    *   Muestra un mensaje descriptivo y amigable del error (ej. *"No se pudo conectar a AWS. Verifique su conexión a Internet o las claves en su archivo .env"*), ofreciendo un botón celeste de "Entendido" para cerrar.

---

## 🤖 REGLAS DE EJECUCIÓN AUTÓNOMA DE ANTIGRAVITY
1.  Analiza la arquitectura local e integra la clase `LSPCloudService` en el archivo que mejor corresponda o crea `src/cloud_service.py` inyectándola de forma limpia en tu aplicación.
2.  Rediseña el menú principal para añadir la pestaña de **Nube** o los botones de sincronización interactivos según las especificaciones estéticas.
3.  Verifica que al subir un modelo se ejecute correctamente el binario de conversión de TensorFlow.js y se guarden las fragmentaciones en la carpeta temporal local antes de transferirlas.
4.  Realiza pruebas en caliente en tu terminal local corriendo `flet run src/app.py` para asegurar que el listado de S3 y los botones actualicen el estado visual dinámicamente y de manera segura.
