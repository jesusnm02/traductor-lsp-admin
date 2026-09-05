# Parche de Optimización Final: Distribución de Anchos, Consolidación de Switch de Voz y Cabello del Avatar v2

Este documento técnico de nivel senior es el manual definitivo para que el agente autónomo **Antigravity** aplique los últimos refinamientos visuales y funcionales en la aplicación de escritorio en Flet (`traductor-lsp-admin`).

---

## 📐 1. OPTIMIZACIONES DE ANCHO EN LA INTERFAZ (FLET)

### A. Pestaña de Tester: Bloque "ESPERANDO SEÑA" (Feedback en Tiempo Real)
*   **Problema:** El bloque visual que muestra el estado de la predicción ("ESPERANDO SEÑA" o la seña traducida en verde) tiene un ancho muy acotado que desaprovecha el espacio horizontal de la pestaña de pruebas.
*   **Ajuste:** Modifica el contenedor (`ft.Container` o `ft.Card`) del feedback para que ocupe todo el ancho disponible. 
*   **Sintaxis en Flet:**
    ```python
    # Establecer la expansión horizontal al máximo sin romper el diseño de columnas
    self.feedback_container.expand = True
    self.feedback_container.width = None # Permite que Flet calcule el ancho dinámicamente
    # Si está dentro de una fila o columna, asegúrate de que el control secundario tenga:
    # alignment=ft.alignment.center o se expanda según el layout.
    ```

### B. Pestaña de AWS: Bloque "Controles Manos Libres"
*   **Problema:** Los controles de voz manos libres están en un contenedor angosto, lo que hace que los elementos queden demasiado ajustados.
*   **Ajuste:** Expande este contenedor para ocupar un ancho de banda mayor (por ejemplo, `width=450` o `expand=True` dentro de su fila correspondiente), equilibrando el espacio con la cámara web pero sin desplazar ni romper el layout del reproductor de video de la derecha.

---

## 🎙️ 2. CONSOLIDACIÓN DE SWITCHES DE VOZ (UN SOLO BOTÓN DE ACTIVACIÓN)

*   **Problema:** Usar dos controles deslizantes (switches) independientes para "Voz Captura" y "Voz Grabación" resulta confuso y redundante para el usuario.
*   **La Solución:** Consolidar todo el subsistema de voz en **un único Switch universal** titulado `"Activar Comandos de Voz"`.
*   **Comportamiento del Hilo de Voz (`speech_recognition`):**
    *   Cuando este switch único esté activo (`True`), el hilo secundario asíncrono escuchará de forma continua el micrófono.
    *   El motor de reconocimiento procesará las tres palabras clave en español de forma unificada:
        1.  **"captura"** o **"foto"** -> Llama automáticamente a la función de tomar captura instantánea (`self.tomar_captura()`).
        2.  **"grabar"** o **"graba"** -> Llama automáticamente a la función de iniciar grabación de video (`self.iniciar_grabacion()`).
        3.  **"detener"** o **"no grabes"** -> Llama automáticamente a la función de parar grabación y compilar localmente (`self.detener_grabacion()`).
    *   Si el switch se desactiva (`False`), el hilo de escucha de voz se suspende por completo, liberando el micrófono del sistema.

---

## 🎨 3. RETORNO AL CABELLO COMPLETO Y NATURAL EN EL AVATAR

*   **Problema:** Los "palitos" de pelo en la cabeza del avatar hacían que se viera demasiado simplificado. Se requiere volver a un estilo de cabello normal, completo y estilizado que tenga armonía y se parezca al diseño de `Avatar.png`.
*   **Ajuste in OpenCV:**
    *   En lugar de trazar líneas individuales sueltas sobre el cráneo, vuelve a proyectar un **polígono cerrado y relleno** de color negro o marrón oscuro.
    *   **Lógica de dibujo:** Toma el límite superior de las cejas (landmarks `109, 67, 103` y `336, 296, 334`) y el punto más alto de la cabeza (landmark `10`). Aplica un offset vertical hacia arriba de `30-40px` para darle volumen y cuerpo al peinado, y rellena el polígono usando `cv2.fillPoly`. Esto le dará al personaje un cabello real, tupido y sumamente amigable.

---

## 📢 PROMPT DE EJECUCIÓN DIRECTA PARA ANTIGRAVITY

```markdown
# PROMPT DE AJUSTES FINALES DE DISEÑO, VOZ Y AVATAR EN ESCRITORIO

**DIRIGIDO A:** Antigravity
**OBJETIVO:** Ajustar los anchos de los bloques en Tester y AWS, unificar el control de voz en un solo Switch y restaurar el peinado completo del avatar.

---\n### 1. AJUSTES DE ANCHO EN LA INTERFAZ
*   **Tester (Esperando Seña):** Modifica el contenedor del bloque de predicción ("ESPERANDO SEÑA") en la pestaña del Tester para que se expanda horizontalmente (`expand=True` o eliminando el ancho fijo) ocupando todo el ancho útil de su sección sin romper los componentes laterales.
*   **AWS (Controles Manos Libres):** Amplía la anchura del panel de "Controles Manos Libres" en el panel derecho para que se distribuya con mayor soltura, evitando que se encimen los textos o botones.

### 2. CONSOLIDACIÓN DE SWITCH DE VOZ
*   Abre `src/ui_components.py` y reemplaza los dos switches anteriores de voz por **un solo switch** de control titulado `"Activar Comandos de Voz"` (`self.switch_comandos_voz`).
*   Configura el hilo asíncrono de reconocimiento de voz de modo que, si el switch único está encendido, procese simultáneamente e indistintamente las órdenes de **"captura"**, **"grabar"** y **"detener" / "no grabes"**, mapeándolas a sus funciones de ejecución respectivas.

### 3. RESTAURAR CABELLO NATURAL DEL AVATAR
*   Abre el script de renderizado de OpenCV para el avatar.
*   Elimina el dibujo de "palitos" en la cabeza.
*   Vuelve a trazar un polígono relleno y cerrado (`cv2.fillPoly`) de color marrón o negro utilizando los puntos superiores de FaceMesh con un offset de elevación para recrear un peinado completo, natural y estilizado como el de `Avatar.png`.
```
