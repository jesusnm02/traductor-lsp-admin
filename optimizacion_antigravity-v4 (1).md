# Instrucciones de Refactorización, Control de Concurrencia y Rediseño Escolar para Antigravity (v4)

**DIRIGIDO A:** Antigravity (Agente de Programación y Optimización Autónoma)  
**ROL:** Ingeniero de Software Principal & Arquitecto de Interfaces en Flet  
**PROYECTO:** Sistema de Escritorio para el Traductor de Lengua de Señas Peruana (LSP)  

---

### 🚨 OBJETIVO DE ESTA ITERACIÓN  
Debes corregir con urgencia múltiples fallos críticos de UI y de concurrencia reportados durante las pruebas locales:
1.  **Resolver el Crash de Flet (`RuntimeError: An attempt to fetch destroyed session.`):** Ocurre porque hilos secundarios intentan interactuar con componentes de una sesión de Flet que ya fue cerrada o reiniciada por el usuario.
2.  **Habilitar y Mostrar Correctamente la Pantalla de Pruebas:** Resolver el problema por el cual la pestaña de inferencia/pruebas en vivo no es visible o no renderiza el traductor.
3.  **Rediseñar la Distribución (Layout):** Mover los deslizadores de configuración (tiempo de espera, cantidad de frames) a la parte inferior de la pantalla, configurar la lista de palabras como un contenedor con scroll vertical de alta fluidez ("tipo slider") y rediseñar con una estética escolar limpia (Celeste y Blanco).

---

### 🛠️ TAREAS TÉCNICAS E IMPERATIVAS

#### 1. Corrección del Crash: "An attempt to fetch destroyed session"
*   **La Causa:** En `ui_components.py` (líneas como la 551 y 553), llamadas de fondo asíncronas ejecutan `self.btn_generate_cnn.update()` o `self.page.update()`. Si el usuario cierra la app o la sesión se desconecta, Flet destruye la sesión física en memoria. Al intentar actualizar un control huérfano, Python arroja una excepción fatal.
*   **La Solución:**
    1.  **Bloques de Captura Seguros:** Envuelve **todas y cada una** de las llamadas `.update()` y `page.update()` dentro de los hilos de fondo o callbacks asíncronos en bloques `try-except` dedicados:
        ```python
        try:
            if self.page and self.page.is_active:
                self.btn_generate_cnn.update()
        except RuntimeError as e:
            if "destroyed session" in str(e):
                print("[CONCURRENCIA] Sesión cerrada. Abortando actualización de control.")
                return  # Salir limpiamente del hilo o callback
        ```
    2.  **Apagado Coordinado en `on_disconnect`:** Asegúrate de que `app.py` use la directiva `page.on_disconnect` para ordenar la detención inmediata de todos los bucles infinitos en `vision_service.py` y `voice_service.py` (`self.is_running = False`).

#### 2. Implementación de una Navegación por Pestañas Visible y Fluida
*   Para garantizar la visibilidad total de la pantalla de pruebas, implementa un control de pestañas nativo (`ft.Tabs`) en `app.py` que organice la aplicación de forma limpia:
    *   **Pestaña 1 (Captura y Entrenamiento):** Contiene la grilla de grabación de señas.
    *   **Pestaña 2 (Prueba en Vivo del Traductor):** Muestra el monitor de la cámara, el cargador de modelos `.keras` / `.h5` de la categoría activa, un texto gigante de predicción de la palabra y síntesis de voz offline con `pyttsx3`.
*   Asegúrate de que al cambiar de pestaña, se apague y encienda la cámara correspondientemente para evitar colisiones de hardware.

#### 3. Estructura y Rediseño Visual (Celeste y Blanco)
Ajusta la interfaz gráfica para que sea intuitiva y ergonómica para el docente:
*   **Deslizadores al Fondo:** Mueve los controles de *Tiempo de Espera* y *Cantidad de Muestras* a la parte inferior del panel izquierdo (debajo de la lista de palabras).
*   **Lista de Palabras Desplazable ("Scroll / Slider Vertical"):** Reemplaza cualquier contenedor de tamaño estático por un control `ft.Container` de altura máxima fija (ej. `height=240`) con bordes curvos celestes y establece su propiedad interna como un `ft.ListView` con scroll adaptativo para permitir que el docente deslice cómodamente hacia arriba y hacia abajo cuando tenga decenas de palabras registradas:
    ```python
    words_scrollable_container = ft.Container(
        content=ft.ListView(
            expand=True,
            scroll=ft.ScrollMode.ALWAYS, # Forzar barra de scroll vertical
            spacing=10
        ),
        height=240, # Altura controlada para que no empuje el resto de elementos
        border=ft.border.all(1, "#D1E4F8"),
        border_radius=12,
        bgcolor="#FFFFFF",
        padding=10
    )
    ```
*   **Alineación del Feed de Cámara:** Asegúrate de que el monitor de video de OpenCV esté centrado y que los mensajes de estado no se desplacen por debajo del área visible de la pantalla. Configura `page.scroll = ft.ScrollMode.AUTO` como red de seguridad global.
*   **Paleta Escolar:**
    *   Modo de pantalla: `page.theme_mode = ft.ThemeMode.LIGHT`
    *   Fondo de app: `#F4F8FA` (Blanco con tinte azulado suave)
    *   Paneles secundarios: Tarjetas `#FFFFFF` con bordes celestes `#D1E4F8` y esquinas redondeadas.
    *   Textos: Azul oscuro académico `#1A365D`.

---

### 💻 EJEMPLO DE DISEÑO COMPACTO Y SEGURO (Muestra .md)

Utiliza este patrón de diseño en `app.py` and `ui_components.py` para asegurar un flujo libre de deadlocks:

```python
# Ejemplo de manejo seguro en ui_components.py para el callback de UI
def update_ui_safely(control):
    """Actualiza un control de Flet capturando excepciones de sesión destruida."""
    try:
        if control and control.page:
            control.update()
    except RuntimeError as e:
        if "destroyed session" in str(e):
            # Ignorar de forma segura si la ventana fue cerrada por el docente
            pass

# Estructura compacta del Panel de Palabras (Scrollable) y Sliders en la parte inferior
panel_izquierdo = ft.Column([
    ft.Text("Vocabulario Registrado", size=16, color="#1A365D", weight=ft.FontWeight.BOLD),
    
    # Lista de palabras con scroll ("bajar y subir")
    words_scrollable_container,
    
    ft.Divider(height=10, color="#D1E4F8"),
    
    # Deslizadores reubicados en la parte inferior
    ft.Text("Configuración de Captura", size=14, color="#1A365D", weight=ft.FontWeight.BOLD),
    ft.Row([
        slider_tiempo_espera, # Slider de posicionamiento
        slider_cantidad_frames # Slider de longitud de secuencia (30, 45, etc.)
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
], spacing=15, expand=True)
```

---

### 🤖 INSTRUCCIONES DE EJECUCIÓN AUTÓNOMA PARA ANTIGRAVITY
1.  **Inspección del Repositorio:** Ingresa directamente a `src/` en el repositorio local de Windows.
2.  **Refactorización:** Modifica `app.py`, `ui_components.py`, `vision_service.py` y `tester_service.py` implementando las protecciones `try-except` de actualización de UI, el nuevo layout de pestañas, los deslizadores al fondo, y el contenedor de palabras con scrollbar forzado.
3.  **Prueba de Estabilidad:** Ejecuta `flet run src/app.py` y simula abrir, cambiar de pestaña ("Prueba" y "Grabación") y cerrar la aplicación para certificar que el error de sesión destruida no vuelva a aparecer en la consola.
