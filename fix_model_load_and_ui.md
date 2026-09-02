# Instrucciones para Corregir el Error de Carga de Modelo y Optimizar la UI en Flet

**DIRIGIDO A:** Antigravity (Agente de Programación y Optimización Autónoma)  
**OBJETIVO:** Solucionar el error crítico de decodificación UTF-8 al cargar el modelo de señas, corregir la concurrencia en la página de pruebas, y rediseñar la interfaz visual para que todos los elementos (deslizadores, consola de estado y cámara) sean perfectamente visibles en cualquier tamaño de pantalla.

---

### 🚨 PARTE 1: DIAGNÓSTICO Y CORRECCIÓN DEL ERROR CRÍTICO DE CARGA

#### 1. El Error de Decodificación UTF-8
**Síntoma:** La consola o la pantalla muestra: `Error cargando modelo: 'utf-8' codec can't decode byte 0xfa...`

**Causa Raíz:**
Este error ocurre físicamente cuando el código intenta abrir el archivo binario del modelo (`model.keras` o `model.h5`) utilizando un lector de texto plano, como `open(path, 'r', encoding='utf-8')` o al intentar pasar el archivo del modelo a un parseador de JSON (`json.load`). 
*   El byte `0xfa` (o firmas similares) es característico de archivos comprimidos o binarios (los archivos `.keras` son en realidad archivos zip comprimidos y los `.h5` son binarios HDF5).
*   **El cruce de rutas:** El error ocurre porque en `tester_service.py` o en la sección de carga de modelo de la UI, se ha intercambiado la ruta del archivo de etiquetas (`labels.json`) con la ruta del modelo binario, o bien se está llamando a una función de lectura de texto sobre el archivo del modelo.

#### 🛠️ Instrucciones de Corrección para Antigravity:
1.  **Inspecciona `tester_service.py` y `app.py`:** Busca cualquier bloque de código similar a:
    ```python
    # ¡ERROR! Esto causa el fallo de decodificación si model_path apunta al binario
    with open(model_path, 'r', encoding='utf-8') as f:
        ...
    ```
2.  **Separa estrictamente la carga de archivos:**
    *   **El Modelo Binario:** Debe cargarse **únicamente** con la API de Keras:  
        `self.model = tf.keras.models.load_model(model_path)`
    *   **Las Etiquetas (Labels):** Deben cargarse desde el archivo JSON independiente:  
        `with open(labels_path, 'r', encoding='utf-8') as f: self.labels = json.load(f)`
3.  **Verifica las extensiones y rutas antes de cargar:** Añade una validación defensiva en tu cargador:
    ```python
    if not model_path.endswith('.keras') and not model_path.endswith('.h5'):
        raise ValueError(f"Ruta de modelo inválida: {model_path}. Debe ser un archivo .keras o .h5")
    ```

---

### 🚨 PARTE 2: CORRECCIÓN DE LA CONCURRENCIA EN MODO "PRUEBAS"

**Síntoma:** Al estar en la página de pruebas, decir *"recopila"* activa el flujo de grabación, lo cual contamina el dataset con datos de prueba erróneos.
**Causa:** El hilo de escucha de Vosk (`voice_service.py`) no discrimina en qué pantalla o estado se encuentra el usuario.

#### 🛠️ Instrucciones de Corrección para Antigravity:
1.  En `voice_service.py`, añade un flag de control o propiedad de estado: `self.allow_voice_trigger = False`.
2.  En `app.py`, cuando el usuario cambie de pestaña (Navigation/Tabs):
    *   Si entra a la pestaña **"Captura/Entrenamiento"**: Cambia el flag a `True` (`voice_service.allow_voice_trigger = True`).
    *   Si entra a la pestaña **"Pruebas en Vivo"** o cualquier otra: Fuerza el flag a `False` (`voice_service.allow_voice_trigger = False`).
3.  En el bucle de escucha de Vosk, antes de disparar la grabación, valida:
    ```python
    if self.allow_voice_trigger and "recopila" in text:
        # Disparar callback seguro de grabación
    ```

---

### 🎨 PARTE 3: REDISEÑO DE UI - AJUSTE DE TAMAÑO Y FORMATO ESCOLAR

**Síntoma:** El deslizador de tiempo de espera, la configuración de frames y los mensajes flotantes de estado (consola inferior) quedan fuera de la pantalla (se van hacia abajo) o se enciman, impidiendo una experiencia de usuario cómoda.

#### 🛠️ Instrucciones de Rediseño de UI para Antigravity:

1.  **Habilita el Desplazamiento Automático (Scroll):**
    Asegúrate de configurar la página para que si la pantalla es pequeña, se active un scrollbar suave de Flutter en lugar de cortar el diseño:
    ```python
    page.scroll = ft.ScrollMode.AUTO
    ```

2.  **Optimiza las Dimensiones y Distribución Visual (Layout):**
    *   **Ancho/Alto Inicial:** Configura la ventana con dimensiones amplias y adaptables para pantallas estándar de laptops escolares:
        ```python
        page.window_width = 1280
        page.window_height = 820
        ```
    *   **Reduce Alturas Fijas Excesivas:** El `ListView` de palabras (`words_listview`) no debe tener una altura fija de `300`. Modifícalo a un máximo de `200` o `220` pixeles. Esto liberará espacio vertical inmediato y subirá el resto de controles.

3.  **Reorganiza la Consola de Estado (Mensajes Flotantes):**
    En lugar de colocar la barra de mensajes flotantes/status al fondo del panel derecho (debajo del monitor de video), **colócala directamente debajo del título principal de la aplicación** (como una franja horizontal o Banner superior permanente) o **encapsúlala dentro de la misma tarjeta del monitor de la cámara** en la esquina superior. Esto garantizará que el docente la vea de inmediato sin importar el scroll.

    ```python
    # Diseño sugerido para la barra de estado superior fija
    status_banner = ft.Container(
        content=ft.Row([
            ft.Icon(ft.icons.INFO_OUTLINE, color=ft.colors.BLUE_700),
            status_text,
            training_progress
        ], alignment=ft.MainAxisAlignment.START),
        bgcolor="#EBF4FF", # Celeste pastel
        border=ft.border.all(1, "#D1E4F8"),
        border_radius=8,
        padding=10,
        margin=ft.margin.only(bottom=10)
    )
    ```

4.  **Estructura Limpia de Configuración (Sliders):**
    Agrupa los deslizadores (Tiempo de espera, cantidad de frames) en una tarjeta horizontal compacta (`ft.Card` o `ft.Container` con fondo blanco y borde celeste) justo debajo del selector de categorías. No los dejes sueltos para que no consuman espacio vertical innecesario.

    ```python
    config_card = ft.Container(
        content=ft.Column([
            ft.Text("Configuración de Captura", weight=ft.FontWeight.BOLD, color="#1A365D", size=14),
            ft.Row([
                ft.Column([
                    ft.Text("Espera (s):", size=12),
                    slider_delay # Deslizador compacto
                ], expand=True),
                ft.Column([
                    ft.Text("Muestras (frames):", size=12),
                    slider_frames # Deslizador compacto
                ], expand=True)
            ], spacing=10)
        ]),
        bgcolor="#FFFFFF",
        border=ft.border.all(1, "#D1E4F8"),
        border_radius=10,
        padding=12,
        margin=ft.margin.only(bottom=15)
    )
    ```

5.  **Alineación del Monitor de Video:**
    Asegura que el contenedor de `camera_view` tenga un tamaño de visualización controlado (ej. `width=480`, `height=360`) para que conviva perfectamente con la barra de traducción y la síntesis de voz en pantalla.

---

### 🤖 INSTRUCCIONES DE EJECUCIÓN AUTÓNOMA PARA ANTIGRAVITY
1.  **Explora el código fuente** en tu espacio de trabajo local para localizar las variables y las funciones de carga de modelo en `tester_service.py` y `app.py`.
2.  **Corrige las rutas cruzadas** asegurando que la carga del JSON y de Keras sean totalmente independientes y no generen el error de decodificación UTF-8.
3.  **Rediseña la UI de Flet** siguiendo el esquema de colores celestes y blancos, añadiendo el scroll automático y acortando los componentes de lista para garantizar visibilidad al 100%.
4.  **Ejecuta el sistema** con `flet run src/app.py` y verifica la consola y pantalla en caliente hasta que cargue de forma impecable.
