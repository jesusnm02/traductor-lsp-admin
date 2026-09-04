import os
import json
import time
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError, ClientError, EndpointConnectionError
from dotenv import load_dotenv

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELOS_DIR = os.path.join(DATA_DIR, "modelos")

# Cargar variables de entorno desde la raíz del proyecto
load_dotenv(os.path.join(ROOT_DIR, ".env"))

class LSPCloudService:
    """
    Servicio de sincronización y despliegue en la nube (AWS S3) y conversión
    a formato optimizado web TensorFlow.js (model.json + shard*.bin).
    """
    def __init__(self, modelos_dir=None):
        self.modelos_dir = modelos_dir or MODELOS_DIR
        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        self.s3_client = None
        self._init_client()

    def _init_client(self):
        """Inicializa el cliente de S3 con las credenciales cargadas."""
        if self.access_key and self.secret_key and self.bucket_name:
            try:
                self.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name=self.region_name
                )
            except Exception as e:
                print(f"[CLOUD] Error inicializando cliente S3: {e}")
                self.s3_client = None
        else:
            print("[CLOUD] Credenciales de AWS no configuradas o incompletas en .env")
            self.s3_client = None

    def get_local_model_path(self, category_name: str) -> str:
        """Retorna la ruta al archivo de modelo local si existe."""
        cat_dir = os.path.join(self.modelos_dir, category_name)
        keras_path = os.path.join(cat_dir, "model.keras")
        if os.path.exists(keras_path) and os.path.getsize(keras_path) > 0:
            return keras_path
        h5_path = os.path.join(cat_dir, "model.h5")
        if os.path.exists(h5_path) and os.path.getsize(h5_path) > 0:
            return h5_path
        return None

    def verify_local_model(self, category_name: str) -> bool:
        """Verifica si el modelo local existe físicamente y tiene tamaño > 0."""
        return self.get_local_model_path(category_name) is not None

    def check_cloud_status(self, category_name: str) -> str:
        """
        Consulta S3 para determinar el estado de sincronización.
        Retorna:
          - 'SIN_MODELO_LOCAL': No existe model.keras localmente.
          - 'NO_SUBIDO': Existe en local pero no está publicado en S3.
          - 'DESACTUALIZADO': El modelo local fue reentrenado posteriormente al publicado.
          - 'PUBLICADO': Sincronizado y actualizado en la nube.
          - 'ERROR_CONEXION': Fallo de red o credenciales inválidas.
        """
        local_model_path = self.get_local_model_path(category_name)
        if not local_model_path:
            return "SIN_MODELO_LOCAL"

        if not self.s3_client:
            self._init_client()
            if not self.s3_client:
                return "ERROR_CONEXION"

        prefix = f"modelos/{category_name}/tfjs_model/model.json"
        
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' not in response or len(response['Contents']) == 0:
                return "NO_SUBIDO"
            
            s3_meta = response['Contents'][0]
            s3_time = s3_meta['LastModified'].timestamp()
            local_time = os.path.getmtime(local_model_path)
            
            # Si el modelo local es más nuevo por más de 5 segundos, está desactualizado en la nube
            if local_time > (s3_time + 5):
                return "DESACTUALIZADO"
                
            return "PUBLICADO"
            
        except (NoCredentialsError, ClientError, EndpointConnectionError) as e:
            print(f"[CLOUD] Error al verificar estado en S3 para '{category_name}': {e}")
            return "ERROR_CONEXION"
        except Exception as e:
            print(f"[CLOUD] Error inesperado en S3: {e}")
            return "ERROR_CONEXION"

    def convert_to_tfjs(self, category_name: str) -> bool:
        """
        Compila el modelo .keras nativo a formato optimizado para web TensorFlow.js
        (model.json y fragmentos shard*.bin).
        Utiliza conversión Python in-process con fallback a comando CLI.
        """
        model_dir = os.path.join(self.modelos_dir, category_name)
        keras_path = self.get_local_model_path(category_name)
        if not keras_path:
            raise FileNotFoundError(f"No se encontró el modelo local para '{category_name}'.")

        tfjs_dir = os.path.join(model_dir, "tfjs_model")
        os.makedirs(tfjs_dir, exist_ok=True)

        # 1. Intentar conversión nativa con biblioteca tensorflowjs
        try:
            import tensorflow as tf
            from tensorflowjs.converters import save_keras_model
            model = tf.keras.models.load_model(keras_path)
            save_keras_model(model, tfjs_dir)
            print(f"[CLOUD] Conversión in-process exitosa a TFJS en: {tfjs_dir}")
            return True
        except Exception as py_err:
            print(f"[CLOUD] Conversión in-process falló ({py_err}), probando CLI...")

        # 2. Fallback mediante ejecutable CLI tensorflowjs_converter
        command = [
            "tensorflowjs_converter",
            "--input_format", "keras",
            keras_path,
            tfjs_dir
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("[CLOUD] Conversión exitosa a TensorFlow.js vía CLI")
            return True
        except Exception as cli_err:
            print(f"[CLOUD] Fallo en la conversión de TFJS vía CLI: {cli_err}")
            raise RuntimeError(f"Error del conversor TensorFlow.js: No se pudo generar model.json. Detalle: {cli_err}")

    def upload_category_model(self, category_name: str, progress_callback=None) -> bool:
        """
        Convierte a TFJS y sube la carpeta tfjs_model y el archivo labels.json a S3.
        Permite notificar el progreso a través de progress_callback(pct, text).
        """
        if not self.s3_client:
            self._init_client()
            if not self.s3_client:
                raise ConnectionError("No se pudo conectar a AWS S3. Verifique las credenciales en .env.")

        model_dir = os.path.join(self.modelos_dir, category_name)
        labels_path = os.path.join(model_dir, "labels.json")
        tfjs_dir = os.path.join(model_dir, "tfjs_model")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"No se encontró el archivo de etiquetas: {labels_path}")

        # 1. Conversión previa a TFJS
        if progress_callback:
            progress_callback(0.2, f"Convirtiendo '{category_name}' a TensorFlow.js...")
        self.convert_to_tfjs(category_name)

        if not os.path.exists(tfjs_dir) or len(os.listdir(tfjs_dir)) == 0:
            raise FileNotFoundError("La carpeta tfjs_model está vacía tras la conversión.")

        # 2. Subir labels.json
        if progress_callback:
            progress_callback(0.4, f"Subiendo diccionario de etiquetas labels.json...")
        s3_labels_key = f"modelos/{category_name}/labels.json"
        try:
            self.s3_client.upload_file(labels_path, self.bucket_name, s3_labels_key)
        except Exception as e:
            raise RuntimeError(f"Error subiendo labels.json a AWS: {e}")

        # 3. Subir todos los fragmentos compilados de TFJS
        tfjs_files = os.listdir(tfjs_dir)
        total_files = len(tfjs_files)
        for i, file_name in enumerate(tfjs_files):
            local_file = os.path.join(tfjs_dir, file_name)
            s3_key = f"modelos/{category_name}/tfjs_model/{file_name}"
            pct = 0.5 + (0.5 * ((i + 1) / total_files))
            if progress_callback:
                progress_callback(pct, f"Subiendo binario web: {file_name} ({i+1}/{total_files})...")
            try:
                self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
            except Exception as e:
                raise RuntimeError(f"Error subiendo {file_name} a AWS: {e}")

        if progress_callback:
            progress_callback(1.0, f"¡Sincronización completada con éxito en AWS S3!")
        print(f"[CLOUD] Sincronización exitosa de: {category_name} -> s3://{self.bucket_name}/modelos/{category_name}/")
        return True

    def delete_category_model(self, category_name: str, progress_callback=None) -> bool:
        """Elimina todos los archivos del modelo en S3 correspondientes a la categoría."""
        if not self.s3_client:
            self._init_client()
            if not self.s3_client:
                raise ConnectionError("No se pudo conectar a AWS S3. Verifique las credenciales en .env.")

        prefix = f"modelos/{category_name}/"
        if progress_callback:
            progress_callback(0.3, f"Buscando objetos de '{category_name}' en S3...")

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            deleted_count = 0
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page and len(page['Contents']) > 0:
                    delete_us = [{'Key': obj['Key']} for obj in page['Contents']]
                    self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': delete_us})
                    deleted_count += len(delete_us)

            if progress_callback:
                progress_callback(1.0, f"Modelo '{category_name}' eliminado de S3 ({deleted_count} archivos eliminados).")
            print(f"[CLOUD] Eliminada categoría {category_name} de S3 ({deleted_count} archivos).")
            return True
        except Exception as e:
            raise RuntimeError(f"Error al borrar el modelo de S3: {str(e)}")

    def list_all_cloud_categories(self) -> list:
        """Lista todas las categorías que existen en S3 bajo el prefijo modelos/"""
        if not self.s3_client:
            self._init_client()
            if not self.s3_client:
                return []

        categories = set()
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix="modelos/", Delimiter="/"):
                if 'CommonPrefixes' in page:
                    for cp in page['CommonPrefixes']:
                        # prefix es "modelos/<cat>/"
                        parts = cp['Prefix'].strip("/").split("/")
                        if len(parts) >= 2:
                            categories.add(parts[1])
            return sorted(list(categories))
        except Exception as e:
            print(f"[CLOUD] Error listando categorías en S3: {e}")
            return []
