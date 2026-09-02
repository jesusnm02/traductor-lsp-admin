# Directrices de Refactorización de Software: Traductor LSP Local (Windows)
**Para:** Antigravity (Agente de Programación Autónoma)  
**Autor:** Arquitecto de Software Principal  
**Estado del Proyecto:** Refactorización Crítica y Expansión de Características Core  

---

## 1. Contexto y Misión Principal

El sistema actual del **Traductor de Lenguaje de Señas Peruano (LSP)** para administración local corre bajo **Python 3.10/3.11** en un entorno **Windows**. Actualmente, el archivo `src/app.py` se ha vuelto monolítico, acoplando la lógica de la interfaz gráfica (Flet), el procesamiento de visión por computadora (OpenCV y MediaPipe Holistic) y el entrenamiento de la red LSTM (TensorFlow/Keras).

Esto está generando bloqueos críticos en el hilo principal de renderizado de la UI, problemas de latencia de cámara y dolores de cabeza en mantenimiento. 

**Tu misión es:**
1. **Desacoplar quirúrgicamente** el código existente en una arquitectura modular limpia de 4 componentes dentro de `src/`.
2. **Resolver los cuellos de botella de hardware y concurrencia** en Windows.
3. **Implementar nuevas reglas de negocio y funcionalidades de seguridad** (como la detección de cámara obstruida y el CRUD avanzado de categorías con validaciones).

---

## FASE 1: Refactorización Modular (Arquitectura)

Debes reestructurar el directorio del proyecto moviendo la lógica de `src/app.py` a módulos especializados dentro de `src/`. No dupliques código; distribuye las responsabilidades siguiendo el **Principio de Responsabilidad Única (SRP)**.

### 1. Módulo: `src/data_manager.py` (Capa de Persistencia)
Este módulo debe encapsular toda la interacción con el sistema de archivos (`data_historica/`) y el archivo de configuración `metadata.json`.
* **Clase requerida:** `LSPDataManager`
* **Responsabilidades:**
  * Crear directorios físicos para categorías y palabras.
  * Leer, guardar y actualizar los archivos de metadatos (`metadata.json`) de cada categoría.
  * Escribir secuencias normalizadas en archivos CSV correlativos (`seq_X.csv`) de dimensiones `(30, 159)`.
  * Eliminar físicamente carpetas de palabras.
  * **Nuevo Requerimiento (Regla de Negocio):** Implementar un método `delete_category(category_name)` que valide primero si la lista de palabras en esa categoría está vacía. Si no lo está, debe lanzar una excepción `ValueError("Debe eliminar todas las palabras antes de eliminar la categoría")`. Si está vacía, debe borrar físicamente el directorio de la categoría y su archivo `metadata.json`.

### 2. Módulo: `src/vision_service.py` (Capa de Procesamiento)
Este módulo debe encargarse exclusivamente del flujo de la cámara de OpenCV y del procesamiento con MediaPipe.
* **Clase requerida:** `LSPVisionService`
* **Responsabilidades:**
  * Encapsular la instancia de `LSPNormalizer` (MediaPipe Holistic con 53 landmarks).
  * Controlar la captura de video de la cámara web.
  * **Nuevo Requerimiento (Detección de Cámara Tapada):** En cada lectura de frame, calcula el brillo promedio de la imagen usando `cv2.mean(frame)`. Si el promedio es inferior a **15.0**, activa una bandera booleana interna `is_camera_obstructed = True`. De lo contrario, `False`.
  * Dibujar los esqueletos y puntos clave de MediaPipe sobre el frame.
  * Mantener el búfer temporal de 30 frames cuando esté activa la bandera de grabación, retornando los datos al finalizar a través de un callback.

### 3. Módulo: `src/ui_components.py` (Capa de Presentación)
Este módulo debe definir las vistas y componentes gráficos modulares utilizando Flet. No debe contener llamadas directas de entrenamiento o persistencia de archivos, sino interactuar con los otros servicios a través de callbacks.
* **Funciones/Clases requeridas:**
  * `build_sidebar_panel()`: Construye el panel izquierdo para la gestión de categorías, ingreso de palabras y listas visuales.
  * `build_camera_panel()`: Construye el panel derecho con el visor de video, botones de cámara, y estado del entrenamiento.
  * **Nuevo Requerimiento (CRUD Categorías):** Junto al Dropdown de categorías, implementa dos botones de control: un botón de lápiz (Editar nombre de categoría) y un botón de basurero (Eliminar categoría).

### 4. Módulo: `src/app.py` (Punto de Entrada)
Este archivo debe ser ultraligero (menos de 50 líneas). Debe actuar únicamente como el **orquestador principal**.
* **Responsabilidades:**
  * Inicializar la ventana principal de Flet (`ft.Page`).
  * Instanciar `LSPDataManager`, `LSPVisionService` y `LSPTrainer`.
  * Ensamblar las vistas obtenidas de `src/ui_components.py` pasándoles las referencias de los servicios.
  * Configurar la desconexión segura (`page.on_disconnect`) para liberar la cámara y MediaPipe.

---

## FASE 2: Rendimiento y Asincronía

### 1. Solución de Lag de Cámara en Windows
En `src/vision_service.py`, al inicializar la captura de video, debes obligar a OpenCV a utilizar la API **DirectShow** (`cv2.CAP_DSHOW`), la cual elimina el retraso de inicio de 3 a 5 segundos típico de Windows:
```python
self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```
Configura de forma inmediata las dimensiones físicas a `640x480` a 30 FPS para balancear la carga computacional de MediaPipe en la CPU local:
```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### 2. Ejecución Multihilo Concurrente (No-Blocking UI)
Para evitar que Flet muestre la ventana congelada o en estado *"No responde"*, debes implementar concurrencia explícita:
* **Hilo de Cámara:** La lectura secuencial de fotogramas, inferencia de MediaPipe y cálculo de brillo promedio debe ejecutarse en un bucle continuo dentro de un hilo secundario:
  ```python
  self.camera_thread = threading.Thread(target=self._capture_loop, daemon=True)
  self.camera_thread.start()
  ```
* **Hilo de Entrenamiento:** El proceso de entrenamiento de la red LSTM (que invoca a TensorFlow y compila a TF.js) tarda varios segundos. Debes envolver la llamada al método de entrenamiento en un hilo secundario asíncrono para que la UI de Flet siga respondiendo y actualice una barra de carga circular:
  ```python
  threading.Thread(target=self._async_train_workflow, daemon=True).start()
  ```

---

## FASE 3: Nuevas Funcionalidades Core

### 1. Detección Inteligente de Cámara Obstruida o Tapada
* En el bucle de lectura de frames en `src/vision_service.py`, calcula la media de los canales de color en el frame actual:
  ```python
  mean_brightness = cv2.mean(frame)[0] # Canal B/G/R aproximado
  ```
* Si `mean_brightness < 15.0`, significa que la cámara está tapada, que no hay iluminación o que el dispositivo está obstruido.
* Envía esta información a la capa de UI. En Flet, muestra un texto grande en color rojo parpadeante o un banner de advertencia visual en la pantalla de la cámara que diga: **"⚠️ ADVERTENCIA: Cámara obstruida o iluminación insuficiente."** de manera dinámica.

### 2. Regla de Negocio de Eliminación de Categorías
* **Diseño del botón en UI:** Al hacer clic en el botón de eliminación junto al selector de categorías:
  1. Captura la categoría activa.
  2. Llama al método `db_manager.get_words_in_category(selected_category)`.
  3. Si la lista contiene al menos una palabra, la interfaz de Flet **debe bloquear el borrado** y lanzar una notificación emergente (`ft.SnackBar`):
     ```python
     page.show_snack_bar(
         ft.SnackBar(
             content=ft.Text("Debe eliminar todas las palabras antes de eliminar la categoría"),
             bgcolor=ft.colors.RED_800
         )
     )
     ```
  4. Si la lista está completamente vacía (no hay palabras asignadas en `metadata.json`), procede a llamar a `data_manager.delete_category(selected_category)`, borra físicamente la carpeta, limpia el dropdown y muestra un `SnackBar` verde de éxito: `"Categoría eliminada correctamente"`.

---

## Guía de Verificación y Pruebas para Antigravity

Ejecuta tu flujo de prueba secuencial de la siguiente manera:

1. **Instalación de Entorno:**
   Asegúrate de que no haya dependencias conflictivas de sistemas Unix. Ejecuta en PowerShell:
   ```powershell
   pip install -r requirements.txt
   ```
2. **Ejecución de Pruebas:**
   Inicia la aplicación de escritorio usando el intérprete de Flet:
   ```powershell
   flet run src/app.py
   ```
3. **Depuración Iterativa de la API de Flet:**
   Si la consola arroja errores como:
   * `TypeError: Dropdown.on_change must be a callable...`
   * `AttributeError: Page object has no attribute 'window_width'...`
   Debes buscar el error en el archivo correspondiente (`src/app.py` o `src/ui_components.py`), corregirlo ajustando el parámetro a la sintaxis oficial de la versión instalada de Flet, y relanzar el entorno. No te detengas hasta que la aplicación abra de manera fluida y limpia, sin alertas de consola.

**¡Procede con la refactorización modular de inmediato!**
