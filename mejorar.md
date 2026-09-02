# Prompt de Actualización de Arquitectura: App de Escritorio Traductor LSP

## Contexto del Proyecto
Estamos desarrollando la aplicación de escritorio para la captura y entrenamiento de gestos del Lenguaje de Señas Peruana (LSP). La interfaz actual sufre bloqueos, falta de tiempos de preparación y necesita optimización en la captura de coordenadas para asegurar la escalabilidad del modelo.

## Objetivos de la Refactorización
Debes refactorizar el código actual implementando las siguientes 4 características clave. Genera el código correspondiente para cada módulo.

### 1. Separación de Hilos (Threading) para OpenCV y UI
**Problema:** La lectura de `cv2.VideoCapture` está bloqueando el hilo principal (Main Thread) de la interfaz gráfica, causando una pantalla negra hasta que ocurre un evento de UI (como escribir).
**Instrucción:** 
- Mueve el bucle de captura de video a un hilo secundario (Background Thread). 
- Utiliza el sistema de señales/eventos de la librería gráfica (ej. `pyqtSignal` si es PyQt, o `after` si es Tkinter) para enviar el frame ya procesado al hilo principal únicamente para su renderizado.
- Al presionar "Encender Cámara", el video debe fluir a 30 FPS sin congelar la ventana.

### 2. Máquina de Estados para Tiempos de Entrenamiento
**Problema:** El entrenamiento comienza instantáneamente sin dar tiempo al usuario para posicionarse.
**Instrucción:** 
Implementa una máquina de estados para el botón "Entrenar Categoría":
- **Estado 0 (Inactivo):** Esperando comando.
- **Estado 1 (Preparación):** Al activar, inicia un temporizador de 3 segundos. Dibuja un contador visual en el frame de video de OpenCV. No guardes datos de MediaPipe aún.
- **Estado 2 (Grabación):** Al finalizar los 3 segundos, cambia a grabación durante el tiempo estipulado (ej. 2-3 segundos) guardando los datos en disco o memoria.
- **Estado 3 (Fin):** Vuelve al Estado 0.

### 3. Integración de Vosk para Activación por Voz a Manos Libres
**Instrucción:**
- Agrega un control tipo Slider/Toggle en la UI para habilitar el "Modo Escucha".
- Cuando esté activo, levanta un hilo en segundo plano utilizando la librería `vosk` (modelo ligero en español) y `pyaudio`.
- El script debe escuchar el micrófono continuamente. Si detecta la palabra clave "enciéndete", debe disparar programáticamente el evento del botón "Entrenar Categoría" (iniciando el Estado 1 de preparación).

### 4. Normalización Espacial de Coordenadas (MediaPipe)
**Problema:** Las coordenadas de MediaPipe actuales son absolutas. Si el usuario se aleja o se acerca a la cámara, el modelo fallará en la inferencia por la diferencia de escalas.
**Instrucción:**
- Aplica normalización espacial a los puntos extraídos. 
- Selecciona un punto ancla estático del cuerpo (por ejemplo, el punto medio entre los hombros usando los landmarks de la pose de MediaPipe).
- Calcula la posición de todos los demás puntos (manos y rostro reducido) de manera relativa a ese punto ancla (Restando las coordenadas `x, y` del ancla a las coordenadas `x, y` del punto actual).
- Asegúrate de extraer solo los puntos faciales críticos (contorno de ojos, boca y nariz) para optimizar el rendimiento, ignorando el resto de la malla de 468 puntos.

## Resultado Esperado
Proporciona el código refactorizado o los fragmentos clave (`clases` o `funciones`) que integran el manejo de hilos, el bucle de cámara no bloqueante, la lógica del contador en pantalla y la función matemática de normalización del esqueleto.