# Directrices de Corrección: Inferencia con Voz, Contexto de Voz y CRUD de Muestras

**DIRIGIDO A:** Antigravity (Agente de Programación y Optimización Autónoma)  
**ROL:** Principal Software Architect & Machine Learning Engineer  
**SISTEMA:** Administrador Local del Traductor LSP (Flet + OpenCV + TensorFlow + Vosk)

---

### 🚨 OBJETIVOS CRÍTICOS DE ESTA ITERACIÓN

Debes modificar el sistema de forma autónoma para solucionar tres fallos críticos identificados durante las pruebas reales con el docente:
1.  **Inferencia Muda e Invisible:** En la página de pruebas (`tester_service.py` / `ui_components.py`), al cargar el modelo de una categoría, las predicciones no se muestran de forma clara en pantalla ni se pronuncian por voz.
2.  **Fuga de Contexto de Voz:** El comando de voz **"recopila"** se activa accidentalmente estando en la sección de pruebas, corrompiendo el dataset. Debe restringirse estrictamente a la sección de captura.
3.  **Falta de CRUD Granular de Muestras ("Modificar Muestras"):** No existe forma de eliminar muestras de entrenamiento erróneas de manera individual sin borrar la palabra completa. Debes implementar una ventana emergente (`ft.AlertDialog`) que permita gestionar y depurar archivos `.csv` individuales antes de reentrenar.

---

### 🛠️ IMPLEMENTACIÓN PASO A PASO

#### 1. Inferencia en Tiempo Real con Voz (Text-to-Speech) y Letras Gigantes
En la sección de pruebas/inferencia (`tester_service.py` y el panel visual de pruebas de `ui_components.py`):
*   **Diseño de la UI de Salida:** Agrega debajo del visor de cámara un contenedor blanco puro con bordes redondeados celestes que contenga:
    *   Un texto gigante (`size=40`, `weight=ft.FontWeight.BOLD`, color `#1A365D`) que muestre la palabra traducida actual.
    *   Un indicador visual de confianza (por ejemplo, `ft.ProgressBar` o un porcentaje de probabilidad).
*   **Integración de Voz (Text-to-Speech):** Utiliza la librería **`pyttsx3`** para realizar la síntesis de voz de forma 100% offline y local en la computadora del docente (compatible con Windows).
*   **Algoritmo de Debounce e Inferencia:**
    1.  La ventana deslizante de 30 frames en `tester_service.py` analiza continuamente las coordenadas normalizadas por el cuello (37 puntos de cara, manos y torso).
    2.  Si el modelo de Keras predice una palabra con una probabilidad **mayor al 85%** de forma sostenida por al menos **10 frames consecutivos** (evitando parpadeos de predicción rápida):
        *   Actualiza el texto gigante en pantalla con la palabra traducida.
        *   Dispara un hilo secundario con `pyttsx3` para pronunciar la palabra en voz alta. 
        *   **Regla de bloqueo de repetición:** Guarda la última palabra pronunciada en una variable de estado. No vuelvas a pronunciar la misma palabra hasta que el modelo regrese al estado "sin seña" o detecte una palabra diferente por más de 1.5 segundos.

```python
# Ejemplo de implementación segura de TTS en segundo plano
import pyttsx3
import threading

def speak_word_offline(word):
    def tts_thread():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150) # Velocidad de voz natural
            engine.say(word)
            engine.runAndWait()
        except Exception as e:
            print(f"Error en TTS: {str(e)}")
            
    threading.Thread(target=tts_thread, daemon=True).start()
```

#### 2. Restricción del Comando de Voz "Recopila" por Contexto
Debes evitar que el micrófono de Vosk capture e inicie grabaciones de datos cuando el docente esté probando el traductor o cuando la cámara esté apagada:
*   En `voice_service.py` o en el callback receptor en `app.py`:
    *   Verifica que el estado actual de la pantalla/tab activa sea estrictamente la **"Pestaña de Grabación/Entrenamiento"**.
    *   Verifica que la cámara web esté encendida.
    *   Verifica que la máquina de estados de visión de `vision_service.py` esté en estado **`"INACTIVO"`** (esperando orden).
*   Si no se cumplen estas tres condiciones concurrentes de seguridad, el comando de voz **`"recopila"` debe ignorarse por completo**, emitiendo únicamente un log silencioso en consola para evitar corrupciones de datos.

#### 3. Modal de Gestión de Muestras ("Modificar Muestras")
Para dar un control total al docente sobre los datos grabados, debes implementar un modal interactivo:
*   **En la UI de Vocabulario:** Al lado de cada palabra en la lista principal, añade un botón con el icono de lista (`ft.icons.LIST_ALT` o `ft.icons.EDIT_NOTE`) titulado "Modificar Muestras".
*   **La Ventana Emergente (`ft.AlertDialog`):** Al hacer clic, se abre una ventana modal que realiza las siguientes operaciones en disco de forma dinámica:
    1.  Lee el directorio físico de la palabra (`data_historica/{categoria}/{palabra}/`).
    2.  Lista todos los archivos de coordenadas `.csv` existentes (ej. `seq_0.csv`, `seq_1.csv`, `seq_2.csv`...).
    3.  Muestra cada muestra en una fila con su número de secuencia, tamaño físico (validando que tenga 30 filas x 159 columnas) y un botón de eliminación individual (icono de basurero rojo).
    4.  Añade un botón destacado en la parte inferior: **"Eliminar Todas las Muestras"**, que vacíe por completo la carpeta.
*   **Lógica de Sincronización:**
    *   Al eliminar un archivo `.csv` individual mediante la UI del modal, llama a `data_manager.delete_sample_file(path)`.
    *   Renombra u organiza dinámicamente los archivos restantes para evitar huecos en la secuencia, o simplemente lee los archivos existentes al cargar la lista.
    *   Al cerrar el modal, actualiza automáticamente el contador de muestras en la pantalla principal de Flet (ej: `HOLA (12 muestras)`).

---

### 🤖 REGLAS DE CONTROL Y EJECUCIÓN PARA ANTIGRAVITY

1.  **Localización de Archivos:** Modifica directamente los módulos de tu espacio de trabajo de manera autónoma:
    *   `src/ui_components.py` para inyectar los botones de gestión de muestras y el diálogo de Flet.
    *   `src/data_manager.py` para añadir la lógica de borrado y listado de archivos `.csv` individuales.
    *   `src/voice_service.py` y `src/vision_service.py` para bloquear el comando de voz cuando el estado de la app no sea el adecuado.
    *   `src/tester_service.py` para incorporar la lógica de debouncing e inferencia por voz de `pyttsx3`.
2.  **Consistencia de Colores Escolares:** Todos los nuevos elementos visuales (el AlertDialog de muestras, la caja de texto gigante de traducción) deben mantener estrictamente la paleta clara celeste y blanca (`#F4F8FA` de fondo, `#4A90E2` celeste primario y `#1A365D` para textos académicos).
3.  **Validación en Caliente:** Al finalizar las modificaciones, ejecuta la aplicación utilizando `flet run src/app.py` en tu consola de Windows, interactúa con el modal, borra archivos, realiza una inferencia de prueba y comprueba que no ocurra ningún deadlock o error de hilos.

---
*Fin de las directrices de optimización.*
