# Prompt de Optimización y Normalización Geométrica para Antigravity

**DIRIGIDO A:** Antigravity (Agente de Programación y Optimización Autónoma)  
**ROL:** Ingeniero de Machine Learning & Visión Computacional Senior  
**CONTEXTO DEL PROYECTO:** Tesis de Traductor en Tiempo Real de Lengua de Señas Peruana (LSP) con procesamiento local de bajo consumo y cero costo de servidores.  

---

## 🚨 MISIÓN PRINCIPAL
Tu objetivo es eliminar por completo el cuello de botella que ralentiza el procesamiento de video en tiempo real de nuestra aplicación. Actualmente, la cámara y el renderizado presentan una latencia severa (caída drástica de FPS) debido al procesamiento ineficiente de los puntos faciales de MediaPipe Holistic.

Debes refactorizar de forma autónoma el pipeline de captura y procesamiento en el código fuente para lograr una tasa de refresco fluida (**mínimo de 30 FPS locales**) aplicando técnicas de optimización avanzadas a nivel de tesis y un sistema de normalización geométrica robusto basado en anclajes fisiológicos para dar soporte a la invarianza de escala (distancia del usuario).

---

## 🛠️ DIRECTRICES TÉCNICAS IMPERATIVAS (NIVEL TESIS)

### 1. Reducción Extrema del Feature Set Facial (Malla Minimalista)
* **El problema:** Procesar los 468 o 478 puntos de la malla facial completa de MediaPipe satura el procesador en entornos de ejecución locales, degradando drásticamente los FPS.
* **La solución:** Modifica el módulo de extracción para descartar sistemáticamente 431 o 441 puntos faciales redundantes. Extrae únicamente un subconjunto compacto de **37 puntos faciales clave** localizados en:
  * El contorno y comisuras de los labios (esencial para registrar la gesticulación oral y rasgos no manuales del LSP).
  * Los extremos exteriores e interiores de los ojos.
  * La curvatura y altura de las cejas.
* Esto reducirá la latencia de cómputo del estimador en más de un 70% y evitará el sobreajuste (*overfitting*) en el entrenamiento del modelo temporal downstream.

### 2. Normalización de Invarianza Espacial y de Escala (Fórmula de Anclaje en el Cuello)
Para asegurar que el sistema traduzca correctamente sin importar si el usuario se acerca o se aleja de la cámara, debes implementar la siguiente normalización geométrica basada en anclajes físicos, inspirada en las metodologías de normalización de puntos de interés para lengua de señas en tiempo real:

* **Anclaje de Origen (Traslación):** Define el punto medio del cuello (o en su defecto, el punto medio entre los hombros, `shoulder_mid`, como nodo de referencia $k = 0$) como el origen local de coordenadas $(0,0,0)$ de cada fotograma. Resta este vector a todas las coordenadas tridimensionales de la pose, cara y manos para anular las traslaciones absolutas del signante en el espacio físico.
* **Factor de Escala (Invarianza de Profundidad):** Calcula la norma euclidiana del vector que une el cuello (o el centro de los hombros) con el centro de la cabeza (punto medio de los ojos/nariz). Divide todas las coordenadas previamente trasladadas por este factor de escala.
* **Fórmula Matemática a Implementar:**
  $$P'_k = \frac{P_k - P_{\text{cuello}}}{\|P_{\text{cabeza}} - P_{\text{cuello}}\|_2}$$
  Donde:
  * $P_k$ es el vector de coordenadas crudo del landmark $k$.
  * $P_{\text{cuello}}$ es la coordenada del punto de anclaje (origen de traslación).
  * $\|P_{\text{cabeza}} - P_{\text{cuello}}\|_2$ es la distancia euclidiana de normalización (factor de escala).
* **Manos:** Para las manos, realiza adicionalmente una normalización local e independiente relativa a la muñeca correspondiente para aislar la morfología fina de los dedos de la traslación global de los brazos.

### 3. Optimización de Concurrencia y Parámetros de MediaPipe
* **Multithreading:** Asegúrate de delegar la captura de la cámara (OpenCV), la inferencia de landmarks de MediaPipe y el buffer de secuencias temporales a hilos independientes (`threading.Thread`), de modo que el renderizado de la interfaz visual jamás bloquee la pantalla del docente.
* **Ajuste de MediaPipe:** Configura la inicialización de MediaPipe Holistic con parámetros de velocidad óptimos:
  ```python
  model_complexity=0 # o 1, evitando complejidad 2 que está diseñada para imágenes estáticas pesadas
  refine_face_landmarks=False # No requerimos la malla de alta densidad para ojos
  min_detection_confidence=0.5
  min_tracking_confidence=0.5
  ```

---

## 🤖 REGLAS DE AUTONOMÍA Y FORMATO DE ENTREGA
1. **Autonomía Total:** Tienes libertad absoluta para examinar el codebase actual, reestructurar la lógica de los archivos, y reescribir las funciones de normalización. No necesitas pedir confirmaciones; tu métrica de éxito es que el sistema corra fluido a un alto FPS manteniendo la invarianza de escala.
2. **Formato Obligatorio:** Entrega cualquier bloque de código o script modificado estrictamente utilizando bloques de código en Markdown (especificando ````python ... ````) para asegurar una visualización impecable.

---

## 📂 SCRIPT ACTUAL A EVALUAR Y REFACTORIZAR

Por favor, analiza directamente los archivos src/vision_service.py y src/normalizer.py de este directorio e inyecta la normalización y reducción de puntos descrita anteriormente.
