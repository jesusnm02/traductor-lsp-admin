# Instrucciones de Refactorización y Diseño Escolar para Antigravity (v3)

**DIRIGIDO A:** Antigravity (Agente de Programación y Optimización Autónoma)  
**ROL:** Ingeniero de Software Principal & Especialista en Interfaces de Usuario  
**PROYECTO:** Sistema de Escritorio para el Traductor de Lengua de Señas Peruana (LSP)  

---

### 🚨 OBJETIVO DE ESTA ITERACIÓN
Debes modificar de forma autónoma el codebase actual para solucionar el molesto parpadeo de la cámara web en la interfaz de Flet, cambiar la activación por voz a manos libres con la palabra clave **"recopila"** (descartando "enciéndete"), implementar un panel de configuración dinámica de captura para el docente y rediseñar la interfaz gráfica completa utilizando una **estética escolar limpia, basada en color celeste y blanco**.

---

### 🛠️ TAREAS TÉCNICAS E IMPERATIVAS

#### 1. Solución al Parpadeo de la Cámara (Flickering en Flet)
*   **El problema:** El parpadeo ocurre porque el hilo de la cámara llama a `page.update()` de forma global a una tasa de 30 veces por segundo, lo que obliga a Flet a redibujar toda la interfaz de la aplicación, interrumpiendo el flujo del websocket.
*   **La solución:** 
    1.  En el callback de actualización del cuadro, **NUNCA** llames a `page.update()`. En su lugar, llama únicamente a la actualización del control de imagen de forma quirúrgica: `camera_view.update()`.
    2.  Estabiliza la tasa de refresco a exactamente **25 FPS** utilizando un retardo preciso de `time.sleep(0.04)` en el bucle principal de OpenCV. Esto aliviará la cola del renderizador de Flutter y eliminará el parpadeo por completo.

#### 2. Actualización de Activación por Voz (Vosk)
*   En `voice_service.py`, elimina la detección de la palabra `\"enciéndete\"`.
*   Implementa la detección de la palabra clave **`\"recopila\"`**.
*   Al escuchar `\"recopila\"`, el servicio de voz debe gatillar asíncronamente el flujo de preparación y posterior grabación de muestras para la palabra que esté seleccionada actualmente en la interfaz.

#### 3. Configuración Dinámica para el Docente (Variables de Captura)
Debes agregar un nuevo panel de configuración en `ui_components.py` que permita al docente modificar en tiempo real las siguientes reglas de negocio:
*   **Tiempo de Espera / Posicionamiento (Segundos):** Un control numérico o deslizante (`ft.Slider`) con un rango de `1.0` a `5.0` segundos (por defecto `3.0s`). Define cuánto dura la fase de preparación antes de que empiece a grabar.
*   **Cantidad de Frames por Seña:** Un control numérico (`ft.Slider` o `ft.TextField`) con rango de `20` a `60` frames (por defecto `30`). Define el tamaño exacto de la secuencia temporal que guardará el archivo `.csv` y que consumirá el entrenador.

Asegúrate de pasar estas variables de configuración dinámicamente desde la UI a la clase de control de `vision_service.py` para que la máquina de estados se adapte en caliente a los nuevos valores configurados por el docente.

#### 4. Rediseño Visual Escolar (Celeste y Blanco)
Debes rediseñar por completo la paleta cromática de Flet para darle un aspecto amigable, limpio y educativo:
*   **Tema de la página:** Configura `page.theme_mode = ft.ThemeMode.LIGHT` para cambiar a un fondo blanco/claro muy limpio.
*   **Paleta de Colores Escolar:**
    *   **Fondo de la aplicación (`page.bgcolor`):** `#F4F8FA` (un blanco/gris azulado sumamente suave).
    *   **Color Primario (Celeste Escolar):** `#4A90E2` o `#87CEEB`.
    *   **Contenedores y Tarjetas de Control:** Color de fondo blanco puro (`#FFFFFF`) con bordes curvos muy sutiles (`border_radius=12`) y bordes de color celeste claro (`#D1E4F8`).
    *   **Textos y Etiquetas:** Azul oscuro académico (`#1A365D`) para garantizar un excelente contraste de lectura.
    *   **Botón REC / Grabar:** Rojo amigable (`#E25C5C`) con texto blanco.
    *   **Barra de Estado:** Un banner azul pastel agradable (`#EBF4FF`) con tipografía clara y contrastada.

---

### 💻 EJEMPLO DE CÓDIGO DE IMPLEMENTACIÓN SUGERIDO (.md)

Asegúrate de estructurar el bucle del hilo de la cámara y la máquina de estados adaptativa de la siguiente manera:

```python
# Modificación sugerida para el bucle de visión en vision_service.py
class VisionService:
    def __init__(self, normalizer, dataset_manager):
        self.normalizer = normalizer
        self.dataset_manager = dataset_manager
        
        # Parámetros dinámicos configurables desde la UI
        self.pre_recording_delay = 3.0  # Tiempo de posicionamiento (segundos)
        self.target_frames = 30         # Cantidad de frames por muestra
        
        self.state = "INACTIVO"         # Máquina de estados: INACTIVO, PREPARACION, GRABACION, FIN
        # ... inicializaciones de hilos ...

    def update_params(self, delay, frames):
        """Permite al docente alterar los parámetros en caliente desde Flet"""
        self.pre_recording_delay = float(delay)
        self.target_frames = int(frames)

    def _vision_loop(self):
        # Inicializar MediaPipe una sola vez FUERA del bucle para evitar lag masivo
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic.Holistic(model_complexity=1)
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Optimización Windows DirectShow
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_frame_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            # ... procesamiento de MediaPipe y normalización del cuello ...
            
            # --- MANEJO SEGURO DE ACTUALIZACIÓN DE IMAGEN PARA EVITAR PARPADEO ---
            if self.frame_callback:
                # Convertir frame a base64
                _, buffer = cv2.imencode('.jpg', frame)
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # EJECUTAR DE FORMA SEGURA EN EL HILO DE FLET
                # Solo actualizamos el control 'camera_view', no toda la página
                self.frame_callback(img_b64)
                
            # Control estricto de FPS (~25 FPS) para eliminar el flickering
            time.sleep(0.04)
            
        cap.release()
        mp_holistic.close()
```

---

### 🤖 REGLAS DE EJECUCIÓN AUTÓNOMA
1.  Revisa los archivos `app.py`, `ui_components.py`, `vision_service.py` y `voice_service.py` para aplicar los cambios de forma coordinada.
2.  Prueba el resultado con `flet run src/app.py`. Si ocurre alguna excepción de Flet debido a la versión o propiedades de renderizado, corrígela de inmediato editando los componentes correspondientes.

---

### 📂 DIRECTRICES DE ACCESO AL CÓDIGO FUENTE
**Tienes acceso total y directo a todo el sistema de archivos de este proyecto de escritorio en tu entorno de desarrollo.** No necesitas que el usuario inserte manualmente ningún bloque de código en estas instrucciones. 

Analiza directamente el repositorio local de trabajo:
*   Examina y modifica los archivos correspondientes dentro de la carpeta `/src/` (incluyendo `app.py`, `ui_components.py`, `vision_service.py`, `voice_service.py` y `normalizer.py`).
*   Aplica las correcciones lógicas del refresco de la cámara, la palabra clave del motor de Vosk y el nuevo diseño visual de la paleta claro/celeste de manera quirúrgica sobre el codebase existente.
*   Utiliza tus herramientas de consola locales para correr el sistema, verificar excepciones y validar la fluidez antes de reportar la finalización de tus tareas.
