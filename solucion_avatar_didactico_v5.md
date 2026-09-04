# Directrices de Ingeniería V5: Estabilización de Hilos Flet, Animación Facial Dinámica y Rediseño de Avatar de Caricatura (Estilo Avatar.png)

Este documento técnico de nivel senior es el manual definitivo para que el agente autónomo **Antigravity** resuelva los problemas de congelamiento de la aplicación al mover la ventana, estabilice el flujo de video y refactore el algoritmo de OpenCV para renderizar un avatar de caricatura inclusivo con alta expresividad, manos de tipo guante grueso y movimientos faciales en tiempo real, basándose en la imagen **`Avatar.png`** ubicada en la raíz del proyecto.

---

## 🔍 DIAGNÓSTICO DE INGENIERÍA Y TRATAMIENTO DE ERRORES

### 1. El Problema del Congelamiento al Mover la Ventana (Lags y Bloqueos de Hilos)
*   **Causa Raíz:** En Flet, la comunicación entre el código Python y el motor gráfico de la interfaz de usuario se realiza mediante un canal interno de WebSockets. Cuando el docente arrastra, mueve o maximiza la ventana de la aplicación de escritorio, el sistema operativo interrumpe temporalmente el bucle gráfico para gestionar los eventos del sistema de ventanas (como `WM_MOVE` o `WM_SIZE`). Si tu hilo secundario de OpenCV/MediaPipe está enviando frames en un bucle cerrado sin ningún retraso o "respiro" (`time.sleep`), satura la cola de WebSockets de Flet, lo que provoca un desbordamiento del búfer, pérdida de sincronización y congela permanentemente la transmisión de la cámara o cierra la aplicación.
*   **La Solución de Concurrencia:** 
    1.  **Limitación Estricta de FPS (Throttling):** Limitar el bucle de la cámara a un máximo de **25-30 FPS** insertando un `time.sleep(0.04)` al final de cada iteración. Esto reduce el consumo de CPU en un 60% y previene la saturación del socket.
    2.  **No llamar a `page.update()`:** Bajo ninguna circunstancia se debe ejecutar `page.update()` dentro del bucle de la cámara. Solo se debe actualizar el control de imagen específico mediante `self.camera_image.update()`.
    3.  **Ejecución Segura en Hilo Cooperativo:** Correr la captura dentro de un hilo secundario de Python marcado como `daemon=True` para que se destruya instantáneamente si el programa principal se cierra, evitando hilos huérfanos.

### 2. Error de Vista Previa: `TypeError: Image.__init__() got an unexpected keyword argument`
*   **Causa Raíz:** Flet cambió la firma de inicialización de `ft.Image`. Pasar un string directo de Base64 mediante el argumento `src_base64` en el constructor de algunas versiones produce una excepción fatal que bloquea la previsualización del avatar didáctico.
*   **La Solución:** Crear la imagen asignando el string Base64 a través del protocolo universal de Data URI en el parámetro `src`, o instanciar el control vacío y asignarle el valor de la propiedad `.src_base64` en la línea siguiente.

---

## 🎨 REDISEÑO DEL AVATAR PEDAGÓGICO DE CARICATURA (BASADO EN `Avatar.png`)

Para evitar que los niños con discapacidades (auditivas, intelectuales o de atención) sufran de **sobrecarga cognitiva o fatiga visual** al ver una fría estructura alámbrica de MediaPipe (el esqueleto cibernético), rediseñaremos el dibujo para imitar el aspecto de **`Avatar.png`** (ubicado en la raíz del proyecto: `./Avatar.png`), caracterizado por rasgos amigables, contornos limpios de caricatura y colores sólidos de alto contraste.

El algoritmo de OpenCV tomará las coordenadas tridimensionales de MediaPipe Holistic (Manos, Rostro, Torso) y las proyectará sobre un lienzo celeste pastel `#F4F8FA` (RGB `(250, 248, 244)`) aplicando las siguientes especificaciones:

### A. Rostro y Cabello Realistas (Inspirado en Avatar.png)
*   **Forma del Rostro (Piel Cálida):** Obtén los landmarks del contorno facial externo de MediaPipe Face Mesh (puntos del `FACEMESH_CONTOUR` como `10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109`). Dibuja un polígono cerrado relleno con color piel cálido (RGB `(198, 226, 252)` o `#FCE2C6`).
*   **Cabello de Caricatura:** Usando los landmarks superiores del rostro (puntos del cuero cabelludo `10, 338, 297, 332` y `109, 67, 103`), proyecta un polígono ampliado hacia arriba (añadiendo un offset de 25-40 píxeles para dar volumen al peinado) y rellénalo de color marrón oscuro o negro, simulando un cabello estilizado.

### B. Animación Facial Dinámica en Tiempo Real
La gesticulación es una parte fundamental de la gramática de la Lengua de Señas Peruana (LSP). El avatar debe imitar en tiempo real los movimientos faciales del docente:

1.  **Cejas Dinámicas:**
    *   **Puntos de control:** Ceja izquierda (`70, 63, 105, 66, 107`) y ceja derecha (`336, 296, 334, 293, 300`).
    *   **Dibujo:** Traza dos líneas curvas gruesas (`thickness=4` o `5`) de color negro que copien con total precisión la inclinación, altura y curvatura de las cejas del docente cuando este gesticule (ej. al asombrarse o hacer una pregunta).
2.  **Ojos Grandes y Expresivos:**
    *   **Puntos de control:** Calcula el centro geométrico de los ojos usando los landmarks perióculo (ej. `159` para el izquierdo y `386` para el derecho).
    *   **Dibujo:** Dibuja dos círculos grandes de color azul marino `#1A365D` (radio 12-15 px) para simular los iris. Para darles una mirada viva, empática y de dibujo animado, dibuja un pequeño círculo blanco descentrado (radio 3-4 px) en la esquina superior derecha de cada ojo para recrear el **brillo de la mirada**.
3.  **Boca y Labios en Movimiento:**
    *   **Puntos de control:** Contorno de labios exteriores (puntos del bucle `61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78`).
    *   **Dibujo:** Dibuja un polígono relleno de color rojo amigable `#E25C5C` (RGB `(92, 92, 226)`) que trace la forma exacta de los labios.
    *   **Apertura Bucal:** Si la distancia entre el labio superior interno (`13`) y el labio inferior interno (`14`) supera un umbral mínimo (el docente abre la boca), dibuja un polígono interno de color negro/marrón oscuro para simular la cavidad bucal abierta y gesticulando en tiempo real.

### C. Manos Tipo Guante Grueso (Sin "Efecto Palito")
*   **El Problema:** El renderizado actual dibuja las articulaciones de la mano como finas líneas ("palitos") de 1 o 2 píxeles de grosor que dificultan la percepción espacial de los dedos de la mano del avatar en niños con baja visión o problemas de enfoque.
*   **La Solución de Alto Contraste:**
    1.  **El contorno de la palma:** Dibuja un polígono cerrado relleno en color celeste `#4A90E2` que conecte los puntos de la base de la palma (puntos `0, 1, 2, 5, 9, 13, 17` y de vuelta a `0`). Esto le dará una base sólida de mano tridimensional.
    2.  **Dedos de Caricatura:** Dibuja las conexiones de las articulaciones de los dedos utilizando líneas ultra gruesas (`thickness=12` o `14`) con extremos redondeados (`lineType=cv2.LINE_AA` para suavizado) en color celeste pedagógico `#4A90E2` (RGB `(226, 144, 74)`).
    3.  **Nudillos Redondos:** En cada uno de los 21 landmarks de la mano, dibuja un círculo relleno de color blanco puro (`#FFFFFF`) con un radio de `8-10 px` y un contorno celeste de `2 px`. Esto imitará perfectamente un **guante amigable de caricatura (tipo Mickey Mouse o dibujo animado)**, mejorando drásticamente el contraste visual para el aprendizaje.

### D. Ropa / Camiseta Escolar
*   **Dibujo:** Traza un trapezoide sólido de color negro o azul escolar que nazca desde la base del cuello, se extienda hacia los extremos de los hombros de MediaPipe Pose y caiga verticalmente hasta el final del frame del video, simulando la playera de `Avatar.png`.

---

## 📐 ESPECIFICACIONES DE INTERFAZ DE DIÁLOGO (MÓDULO DE ESCRITORIO)

1.  **Alineación de Botones en Flet:** Asegurar que los botones de captura dentro del modal se organicen en un contenedor horizontal `ft.Row` de ancho fijo (`width=620`) y con la propiedad `alignment=ft.MainAxisAlignment.SPACE_EVENLY` para que se distribuyan armónicamente en cualquier resolución de pantalla sin solaparse.
2.  **Centrado de Ventana:** Ejecutar `page.window_center()` inmediatamente después de definir las dimensiones iniciales de la ventana en `src/app.py` para asegurar que el programa abra exactamente en el centro del monitor del docente.

---

## 📢 PROMPT DE EJECUCIÓN DIRECTA PARA ANTIGRAVITY

```markdown
# PROMPT DE CORRECCIÓN: ESTABILIDAD DE HILOS, EVITAR LAGS Y AVATAR ANIMADO (ESTILO AVATAR.PNG)

**OBJETIVO:** Resolver definitivamente el congelamiento al arrastrar la ventana, corregir el bug de la previsualización de imágenes, y actualizar el dibujo de OpenCV para crear un avatar de caricatura amigable, inspirado en `Avatar.png` (en la raíz del proyecto), con cejas, boca y ojos que se mueven en tiempo real, y manos tipo guante grueso.

---

### 1. ESTABILIZACIÓN DEL BUCLE DE CÁMARA (EVITAR CONGELAMIENTOS)
*   **Acción:** Abre `src/ui_components.py` y localiza el bucle de captura de video del avatar (`while self.recording_active:`).
*   **Ajustes:**
    1.  Agrega un retardo preciso de `time.sleep(0.04)` dentro del bucle. Esto limitará la tasa de actualización a un máximo de 25 FPS para que la comunicación WebSocket de Flet no se sature al mover la ventana de escritorio.
    2.  No utilices `page.update()` dentro del bucle de la cámara; actualiza únicamente el control de imagen asignando el frame en Base64 a `self.camera_image.src_base64` y llamando inmediatamente a `self.camera_image.update()`.
    3.  Asegúrate de que el hilo de captura se inicie como un daemon thread (`daemon=True`) para prevenir hilos zombies al cerrar el diálogo o mover la aplicación.

---

### 2. CORRECCIÓN DEL VISOR DE VISTA PREVIA (`Image.__init__()`)
*   **Acción:** Reemplaza cualquier inicialización de Flet que use el argumento directo `src_base64=...` por la asignación de la propiedad estándar `.src_base64` en una línea posterior o utilizando el esquema Data URI Base64 en el parámetro `src`:
    ```python
    self.preview_image = ft.Image(width=320, height=240)
    self.preview_image.src_base64 = base64_str
    ```

---

### 3. REDISEÑO DEL AVATAR INCLUSIVO (ESTILO CARICATURA `Avatar.png`)
*   **Acción:** Refactora el método de renderizado de la cámara en OpenCV para que dibuje sobre un lienzo celeste sólido (`#F4F8FA` / RGB `(250, 248, 244)`) con la fisonomía y el estilo de la imagen `Avatar.png` (ubicada en la raíz del proyecto):
    *   **Rostro y Cabello:** Dibuja el óvalo del rostro con un polígono de color piel cálido (`#FCE2C6`). Dibuja un cabello marrón o negro relleno sobre la parte superior del rostro con un offset vertical para dar volumen.
    *   **Cejas Dinámicas:** Traza dos líneas curvas negras en las cejas que sigan de manera dinámica la inclinación y el movimiento vertical del docente.
    *   **Ojos Grandes y Vivos:** Dibuja dos círculos azul marino `#1A365D` en la posición de las pupilas, e introduce un círculo blanco muy pequeño en el cuadrante superior derecho de cada ojo para simular un "brillo de mirada animada".
    *   **Boca Dinámica:** Dibuja un polígono rojo para los labios externos. Si la distancia entre los labios internos es mayor a un umbral (boca abierta), dibuja un polígono negro interno que simule la gesticulación y apertura de la boca en tiempo real.
    *   **Camiseta:** Traza un trapezoide sólido negro que cubra el torso y hombros de la persona, eliminando por completo el fondo real de la cámara web.
    *   **Manos Tipo Guante:** 
        1. Dibuja un polígono relleno celeste `#4A90E2` que una la base de la palma (puntos 0, 1, 2, 5, 9, 13, 17).
        2. Dibuja los dedos uniendo los landmarks con líneas muy gruesas (`thickness=12`) en celeste `#4A90E2`.
        3. Dibuja círculos rellenos de color blanco de `8 px` con contorno celeste en las 21 articulaciones para simular un guante de caricatura amigable para los niños.
```
