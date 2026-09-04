import os
import json
import shutil
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MUESTRAS_DIR = os.path.join(DATA_DIR, "muestras")
MODELOS_DIR = os.path.join(DATA_DIR, "modelos")

os.makedirs(MUESTRAS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

class LSPDataManager:
    def __init__(self, base_dir=None):
        """
        Gestor de persistencia de datos y metadatos para el traductor LSP.
        Usa rutas absolutas para no verse afectado por el CWD de Flet.
        """
        if base_dir is None:
            self.base_dir = MUESTRAS_DIR
        elif not os.path.isabs(base_dir):
            self.base_dir = os.path.join(ROOT_DIR, base_dir)
        else:
            self.base_dir = base_dir
            
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_category_dir(self, category_name: str) -> str:
        """Retorna la ruta del directorio físico de una categoría."""
        return os.path.join(self.base_dir, category_name.lower().strip())

    def _get_word_dir(self, category_name: str, word: str) -> str:
        """Retorna la ruta del directorio de una palabra específica dentro de una categoría."""
        return os.path.join(self._get_category_dir(category_name), word.lower().strip())

    def _load_metadata(self, category_name: str) -> dict:
        """Carga el archivo metadata.json de una categoría."""
        category_dir = self._get_category_dir(category_name)
        metadata_path = os.path.join(category_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        # Estructura por defecto si no existe o está corrupto
        return {
            "category": category_name.lower().strip(),
            "training_type": "hands_body_and_face_37",
            "input_shape": [30, 255],
            "words": []
        }

    def _save_metadata(self, category_name: str, metadata: dict):
        """Guarda el archivo metadata.json para una categoría."""
        category_dir = self._get_category_dir(category_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        metadata_path = os.path.join(category_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def create_category(self, category_name: str) -> bool:
        """Crea una nueva categoría física de entrenamiento con su metadata.json."""
        clean_name = category_name.lower().strip()
        if not clean_name:
            return False
        category_dir = self._get_category_dir(clean_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            metadata = self._load_metadata(clean_name)
            metadata["category"] = clean_name
            self._save_metadata(clean_name, metadata)
            return True
        return False

    def rename_category(self, old_name: str, new_name: str) -> bool:
        """
        Renombra una categoría existente actualizando su carpeta física y metadatos.
        """
        clean_old = old_name.lower().strip()
        clean_new = new_name.lower().strip()
        if not clean_old or not clean_new or clean_old == clean_new:
            return False
        old_dir = self._get_category_dir(clean_old)
        new_dir = self._get_category_dir(clean_new)
        if not os.path.exists(old_dir):
            return False
        if os.path.exists(new_dir):
            raise ValueError(f"La categoría destino '{clean_new.upper()}' ya existe.")
        
        metadata = self._load_metadata(clean_old)
        metadata["category"] = clean_new
        os.rename(old_dir, new_dir)
        self._save_metadata(clean_new, metadata)
        return True

    def delete_category(self, category_name: str) -> bool:
        """
        Regla de negocio: Valida si la categoría contiene palabras.
        Si la lista de palabras no está vacía, lanza ValueError.
        Si está vacía, elimina físicamente la carpeta y metadata.json.
        """
        clean_name = category_name.lower().strip()
        words = self.get_words_in_category(clean_name)
        if words and len(words) > 0:
            raise ValueError("Debe eliminar todas las palabras antes de eliminar la categoría")
        
        category_dir = self._get_category_dir(clean_name)
        if os.path.exists(category_dir):
            shutil.rmtree(category_dir)
            return True
        return False

    def get_categories(self) -> list:
        """Retorna la lista de todas las categorías registradas físicamente."""
        if not os.path.exists(self.base_dir):
            return []
        categories = []
        for d in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, d)):
                categories.append(d)
        return sorted(categories)

    def get_words_in_category(self, category_name: str) -> list:
        """Retorna las palabras registradas en los metadatos de una categoría."""
        metadata = self._load_metadata(category_name)
        return metadata.get("words", [])

    def add_word_to_category(self, category_name: str, word: str) -> bool:
        """Agrega una palabra a los metadatos y crea su carpeta de almacenamiento."""
        word_clean = word.lower().strip()
        if not word_clean:
            return False
        category_dir = self._get_category_dir(category_name)
        if not os.path.exists(category_dir):
            self.create_category(category_name)

        metadata = self._load_metadata(category_name)
        word_dir = self._get_word_dir(category_name, word_clean)
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)

        if word_clean not in metadata["words"]:
            metadata["words"].append(word_clean)
            self._save_metadata(category_name, metadata)
            return True
        return False

    def delete_word(self, category_name: str, word: str) -> bool:
        """
        Elimina físicamente la carpeta de la palabra con todas sus muestras
        y la retira de los metadatos de la categoría. Si existía un modelo entrenado
        con esta palabra, retira el modelo desactualizado para forzar regeneración.
        """
        word_clean = word.lower().strip()
        word_dir = self._get_word_dir(category_name, word_clean)
        if os.path.exists(word_dir):
            shutil.rmtree(word_dir)

        metadata = self._load_metadata(category_name)
        removed = False
        if word_clean in metadata.get("words", []):
            metadata["words"].remove(word_clean)
            self._save_metadata(category_name, metadata)
            removed = True

        # Sincronización de negocio: si había un modelo que incluía esta palabra, retirarlo
        cat_clean = category_name.lower().strip()
        model_cat_dir = os.path.join(MODELOS_DIR, cat_clean)
        labels_file = os.path.join(model_cat_dir, "labels.json")
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    curr_labels = json.load(f)
                label_vals = [str(v).lower().strip() for v in curr_labels.values()] if isinstance(curr_labels, dict) else [str(v).lower().strip() for v in curr_labels]
                if word_clean in label_vals:
                    model_keras = os.path.join(model_cat_dir, "model.keras")
                    if os.path.exists(model_keras):
                        os.remove(model_keras)
                    if os.path.exists(labels_file):
                        os.remove(labels_file)
                    print(f"[DATA] Modelo desactualizado de '{category_name}' retirado tras eliminar la palabra '{word}'.")
            except Exception as ex:
                print(f"[DATA] Advertencia al sincronizar modelo tras borrado: {ex}")

        return removed

    def save_sequence(self, category_name: str, word: str, sequence_data) -> str:
        """
        Guarda un vector de secuencia normalizada como un archivo CSV.
        
        Args:
            category_name (str): Nombre de la categoría.
            word (str): Palabra grabada.
            sequence_data (list o np.ndarray): Matriz de tamaño (30, 159).
        """
        word_clean = word.lower().strip()
        word_dir = self._get_word_dir(category_name, word_clean)
        if not os.path.exists(word_dir):
            self.add_word_to_category(category_name, word_clean)

        seq_array = np.array(sequence_data, dtype=np.float32)
        if seq_array.ndim != 2 or seq_array.shape[0] < 5:
            raise ValueError(f"Dimensión de secuencia inválida. Esperado (Frames, N), recibido {seq_array.shape}")

        idx = 0
        while True:
            file_name = f"seq_{idx}.csv"
            file_path = os.path.join(word_dir, file_name)
            if not os.path.exists(file_path):
                break
            idx += 1

        df = pd.DataFrame(seq_array)
        df.to_csv(file_path, index=False)
        return file_path

    def get_sample_files(self, category_name: str, word: str) -> list:
        """
        Retorna la lista de archivos de muestra .csv para una palabra específica
        junto con información de filas, columnas y tamaño en KB.
        """
        word_dir = self._get_word_dir(category_name, word)
        if not os.path.exists(word_dir):
            return []

        files = [f for f in os.listdir(word_dir) if f.endswith('.csv')]
        def _sort_key(f_name):
            try:
                return int(f_name.replace("seq_", "").replace(".csv", ""))
            except Exception:
                return f_name
        files.sort(key=_sort_key)

        samples_info = []
        for file_name in files:
            file_path = os.path.join(word_dir, file_name)
            rows = 0
            cols = 0
            size_kb = 0.0
            try:
                size_kb = os.path.getsize(file_path) / 1024.0
                df = pd.read_csv(file_path)
                rows, cols = df.shape
            except Exception:
                pass

            samples_info.append({
                "filename": file_name,
                "filepath": file_path,
                "rows": rows,
                "cols": cols,
                "size_kb": round(size_kb, 1),
                "is_valid": bool(rows >= 20 and cols in [255, 159])
            })

        return samples_info

    def delete_sample_file(self, file_path: str) -> bool:
        """
        Elimina físicamente un archivo de muestra individual y reorganiza correlativamente los archivos restantes.
        """
        if os.path.exists(file_path):
            word_dir = os.path.dirname(file_path)
            os.remove(file_path)
            self._reindex_samples(word_dir)
            return True
        return False

    def delete_all_samples_for_word(self, category_name: str, word: str) -> int:
        """
        Elimina todas las muestras (.csv) registradas para una palabra específica.
        """
        word_dir = self._get_word_dir(category_name, word)
        if not os.path.exists(word_dir):
            return 0

        files = [f for f in os.listdir(word_dir) if f.endswith('.csv')]
        count = 0
        for f in files:
            p = os.path.join(word_dir, f)
            try:
                os.remove(p)
                count += 1
            except Exception:
                pass
        return count

    def _reindex_samples(self, word_dir: str):
        """Reorganiza correlativamente los archivos seq_0.csv, seq_1.csv... evitando huecos."""
        if not os.path.exists(word_dir):
            return
        files = [f for f in os.listdir(word_dir) if f.endswith('.csv')]
        def _sort_key(f_name):
            try:
                return int(f_name.replace("seq_", "").replace(".csv", ""))
            except Exception:
                return f_name
        files.sort(key=_sort_key)

        for new_idx, old_name in enumerate(files):
            new_name = f"seq_{new_idx}.csv"
            if old_name != new_name:
                old_p = os.path.join(word_dir, old_name)
                new_p = os.path.join(word_dir, new_name)
                try:
                    os.rename(old_p, new_p)
                except Exception:
                    pass

    def load_dataset_for_training(self, category_name: str, target_frames: int = None):
        """
        Carga el conjunto histórico de coordenadas para reentrenamiento completo.
        
        Returns:
            X (np.ndarray): Tensor de entrada con forma (N_muestras, target_frames, N_features)
            y (np.ndarray): Etiquetas numéricas (N_muestras,)
            label_map (dict): Diccionario de mapeo {etiqueta_numerica: "palabra"}
        """
        metadata = self._load_metadata(category_name)
        words = metadata.get("words", [])
        
        if len(words) < 2:
            raise ValueError("Se requieren al menos 2 palabras registradas en la categoría para realizar el entrenamiento.")

        label_map = {idx: word for idx, word in enumerate(words)}
        reverse_map = {word: idx for idx, word in enumerate(words)}

        X_list = []
        y_list = []

        # Detectar longitud esperada si no se especifica
        expected_len = target_frames

        for word in words:
            word_dir = self._get_word_dir(category_name, word)
            if not os.path.exists(word_dir):
                continue

            files = [f for f in os.listdir(word_dir) if f.endswith('.csv')]
            for file_name in files:
                file_path = os.path.join(word_dir, file_name)
                try:
                    df = pd.read_csv(file_path)
                    data = df.to_numpy(dtype=np.float32)
                    if data.ndim == 2:
                        if expected_len is None:
                            expected_len = data.shape[0]
                        if data.shape[0] == expected_len:
                            X_list.append(data)
                            y_list.append(reverse_map[word])
                        else:
                            print(f"Advertencia: Archivo {file_path} tiene longitud {data.shape[0]} != {expected_len}, omitiendo.")
                except Exception as e:
                    print(f"Error cargando {file_path}: {str(e)}")

        if len(X_list) == 0:
            raise ValueError(f"No se encontraron muestras válidas para entrenar la categoría '{category_name}'.")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        return X, y, label_map

# Alias de compatibilidad
LSPDatasetManager = LSPDataManager
