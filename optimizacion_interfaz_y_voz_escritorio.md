# Directrices de Optimización: Estandarización de Cámaras, Ajuste de Bloque Predictivo y Comandos de Voz

Este documento técnico actúa como la hoja de ruta final para que el agente autónomos **Antigravity** aplique los retoques de diseño, estandarización de interfaces y control manos libres por voz en la aplicación de escritorio (`traductor-lsp-admin`).

---

## 🎨 1. ESTANDARIZACIÓN DEL ESTILO DE LA CÁMARA (NUBE, ENTRENAMIENTO Y TESTER)

Para lograr una identidad visual uniforme y profesional en toda la aplicación, el recuadro de la cámara web debe verse **exactamente igual** en las tres pestañas principales del sistema: **Sincronización Nube/AWS, Entrenamiento y Tester**.

### 📐 Especificación del Contenedor de la Cámara en Flet:
*   **Contenedor Principal (`ft.Container`):**
    *   `width`: `640` (Ancho uniforme estándar de resolución).
    *   `height`: `480` (Alto uniforme estándar de resolución).
    *   `border_radius`: `12` (Bordes curvos idénticos para suavizar la interfaz).
    *   `border`: `ft.border.all(3, "#4A90E2")` (Borde azul celeste escolar que encuadra la toma).
    *   `bgcolor`: `#1A365D` (Fondo azul marino oscuro para cuando la cámara esté apagada).
    *   `content`: Un control `ft.Image` centrado para proyectar el flujo de bytes/Base64.
*   **Estado Apagado:** Cuando la cámara no esté activa, debe mostrar un icono grande de cámara apagada (`ft.icons.VIDEOCAM_OFF_ROUNDED`, color `#D1E4F8`, size `48`) y una etiqueta de texto clara: `"Cámara Desactivada. Presiona 'Prender Cámara' para iniciar."`

---

## 🧱 2. AJUSTE DE ANCHO Y PREVENCIÓN DE DESBORDAMIENTO (BLOQUE MODELO PREDICTIVO)

En la sección del **Modelo Predictivo**, algunas etiquetas de texto y botones se desbordan o se cortan horizontalmente. Para solucionar este problema de visualización y asegurar que todo encaje en el espacio disponible:

*   **Ajuste de Textos (`ft.Text`):**
    *   Agrega la propiedad `overflow=ft.TextOverflow.ELLIPSIS` o `max_lines=2` en las etiquetas de descripción técnica largas.
    *   Reduce levemente el tamaño de fuente (`size=13` o `size=14` en lugar de `16` o `18`) para que los estados y métricas (ej: "Pérdida/Loss", "Precisión/Accuracy") se lean sin partir la interfaz.
*   **Estructura del Contenedor:**
    *   Usa controles contenedores flexibles con `ft.Row(wrap=True)` para que, si un botón o texto no cabe en el ancho asignado, salte automáticamente a la siguiente línea de forma equilibrada en lugar de romper el margen de la tarjeta.
    *   Asegura un padding interno controlado de `padding=15` y define un `width` máximo restrictivo para evitar expansiones erráticas.

---

## 🎛️ 3. BOTÓN DE HABILITAR/DESHABILITAR CAPTURA EN AWS/S3

En la columna derecha del bloque de gestión de recursos de S3, se debe integrar un control manual de energía para la cámara web:

*   **Botón "Prender Cámara" / "Apagar Cámara":**
    *   Un botón de acción directa (`ft.ElevatedButton` o `ft.IconButton`) con un estado dinámico.
    *   Si está apagado: Muestra `"Prender Cámara"` en color verde pedagógico (`#2E7D32`) con el icono `ft.icons.PLAY_ARROW_ROUNDED`.
    *   Si está encendido: Muestra `"Apagar Cámara"` en color rojo de advertencia (`#E25C5C`) con el icono `ft.icons.STOP_ROUNDED`.
    *   Al presionarlo, debe inicializar o detener asíncronamente el hilo del stream (`self.recording_active = True/False`) y liberar el hardware (`cap.release()`) de forma inmediata.

---

## 🎙️ 4. COMANDOS DE VOZ MANOS LIBRES (SPEECH RECOGNITION EN AWS/S3)

Para facilitar la experiencia de grabación del docente sin obligarlo a soltar las señas para usar el ratón, se implementará un reconocedor de voz de fondo en la pestaña de S3:

1.  **Hilo de Escucha en Segundo Plano:** Al encender la cámara en la pestaña de S3, se debe lanzar un hilo cooperativo asíncrono (`threading.Thread(target=self.listen_voice_commands, daemon=True)`) que utilice la librería `speech_recognition` configurada en idioma español (`es-PE` / `es-ES`).
2.  **Comandos Clave Soportados:**
    *   **"captura"** o **"foto"**: Dispara la función asíncrona de captura de pantalla, guarda el frame actual como un `.png` en local y actualiza la previsualización.
    *   **"grabar"** o **"graba"**: Inicia automáticamente la grabación del frame a un búfer local de frames para compilar el video o GIF.
    *   **"no grabes"** o **"detener"**: Detiene inmediatamente la recolección, guarda el GIF o MP4 local y actualiza el estado de sincronización.
3.  **Seguridad y Robustez:**
    *   Envuelve el reconocedor de voz inyectándolo en un bloque `try-except` para evitar crashes en caso de que no haya micrófono conectado o falle el acceso a la API local de Google.
    *   Utiliza una pequeña ventana de escucha corta (`timeout=1`, `phrase_time_limit=2`) para que la IA no bloquee el hilo gráfico esperando un comando infinito.

---

## 📢 PROMPT DE EJECUCIÓN DIRECTA PARA ANTIGRAVITY

```markdown
# PROMPT PARA EL AGENTE AUTÓNOMO ANTIGRAVITY: ESTANDARIZACIÓN DE CÁMARAS Y COMANDOS DE VOZ

**OBJETIVO:** Uniformar la interfaz de la cámara web en todas las pestañas, evitar desbordamientos de texto en el bloque predictivo, implementar el botón de encendido de la cámara en S3 y habilitar los comandos de voz manos libres.

---

### 1. ESTANDARIZACIÓN DE LA CÁMARA (AWS, ENTRENAMIENTO Y TESTER)
*   **Acción:** Reemplaza los contenedores de cámara en `src/ui_components.py` para las tres pestañas con una estructura unificada:
    *   Usa un `ft.Container` con un tamaño fijo de `width=640` y `height=480`, `border_radius=12`, borde de `3` de grosor en color `#4A90E2`, y fondo `#1A365D`.
    *   Cuando la cámara esté apagada, muestra el icono `ft.icons.VIDEOCAM_OFF_ROUNDED` (color `#D1E4F8`, tamaño `48`) y la etiqueta `"Cámara Desactivada. Presiona 'Prender Cámara' para iniciar."` en color blanco/celeste.

---

### 2. AJUSTE DE DISEÑO EN "MODELO PREDICTIVO" (PREVENIR DESBORDAMIENTOS)
*   **Acción:** Corrige las etiquetas de texto de este bloque en `src/ui_components.py`:
    *   Usa la propiedad `overflow=ft.TextOverflow.ELLIPSIS` o `max_lines=2` en textos extensos.
    *   Reduce las fuentes ligeramente (`size=13` o `size=14`).
    *   Configura el contenedor de botones de este bloque usando `ft.Row(wrap=True)` para que salten de línea automáticamente si no caben en el ancho de la tarjeta.

---

### 3. INTEGRACIÓN DEL BOTÓN DE CÁMARA EN AWS/S3
*   **Acción:** En la columna derecha del panel de S3, agrega el botón dinámico de prender/apagar cámara:
    *   Cuando esté apagado: Botón en verde (`#2E7D32`) con el texto `"Prender Cámara"`.
    *   Cuando esté encendido: Botón en rojo (`#E25C5C`) con el texto `"Apagar Cámara"`.
    *   Asegura que apagar la cámara libere el dispositivo (`cap.release()`) de forma asíncrona y segura.

---

### 4. IMPLEMENTACIÓN DE COMANDOS DE VOZ EN S3
*   **Acción:** Importa `speech_recognition` de manera segura en `src/ui_components.py`.
*   **Lógica:** Lanza un hilo daemon secundario cuando se encienda la cámara en la pestaña de S3. Escucha comandos breves en español:
    *   Si detecta `"captura"` o `"foto"`, llama a la función de tomar captura.
    *   Si detecta `"grabar"` o `"graba"`, inicia el buffer de grabación.
    *   Si detecta `"no grabes"` o `"detener"`, detiene la grabación y compila el recurso didáctico local.
    *   Captura cualquier excepción o ausencia de micrófono con `try-except` para que no afecte el renderizado de Flet.
```
