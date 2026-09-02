import os
import json
import time
import queue
import threading
import unicodedata
import sounddevice as sd
import vosk

def _normalize_text(text: str) -> str:
    """Elimina tildes y convierte a minúsculas para matching tolerante."""
    if not text:
        return ""
    text_clean = unicodedata.normalize('NFD', text.lower())
    return ''.join(c for c in text_clean if unicodedata.category(c) != 'Mn')

class LSPVoiceService:
    """
    Servicio de reconocimiento de voz local continuo utilizando Vosk y micrófono en streaming.
    Detecta la palabra clave 'enciéndete' para activar la máquina de estados de entrenamiento
    a manos libres.
    """
    def __init__(self, trigger_callback=None, status_callback=None):
        self.trigger_callback = trigger_callback
        self.status_callback = status_callback
        
        self.model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()
        
        self.is_listening = False
        self.worker_thread = None
        self.stream = None
        
        self.last_trigger_time = 0.0
        self.cooldown_seconds = 4.0  # Evita múltiples disparos en ráfaga
        self.lock = threading.Lock()

    def _ensure_model(self):
        """Carga el modelo ligero en español desde caché local."""
        if self.model is None:
            if self.status_callback:
                self.status_callback("Cargando modelo de voz Vosk en español...")
            # Carga el modelo en español
            self.model = vosk.Model(lang="es")
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            self.recognizer.SetWords(True)

    def start(self):
        """Inicia la captura de micrófono y el bucle de reconocimiento en segundo plano."""
        with self.lock:
            if self.is_listening:
                return
            self.is_listening = True

        self.worker_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Detiene de forma segura el micrófono y el hilo de escucha."""
        with self.lock:
            self.is_listening = False

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

        if self.status_callback:
            self.status_callback("Reconocimiento de voz desactivado.")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback de sounddevice que encola los datos de audio crudo en bytes."""
        if status:
            pass
        if self.is_listening:
            self.audio_queue.put(bytes(indata))

    def _listen_loop(self):
        """Bucle principal de procesamiento de audio en hilo secundario."""
        try:
            self._ensure_model()
            if self.status_callback:
                self.status_callback("🎤 Micrófono activo. Diga 'Enciéndete' para iniciar.")

            # Abrir stream de audio PCM 16000Hz, 16-bit mono
            with sd.RawInputStream(
                samplerate=16000,
                blocksize=4000,
                dtype='int16',
                channels=1,
                callback=self._audio_callback
            ) as stream:
                self.stream = stream
                while self.is_listening:
                    try:
                        data = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if self.recognizer.AcceptWaveform(data):
                        res = json.loads(self.recognizer.Result())
                        recognized_text = _normalize_text(res.get("text", ""))
                        self._check_trigger(recognized_text)
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        partial_text = _normalize_text(partial.get("partial", ""))
                        self._check_trigger(partial_text)

        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error en micrófono: {str(e)}")
            self.stop()

    def _check_trigger(self, text: str):
        """Evalúa si el texto reconocido contiene la palabra clave 'enciéndete'."""
        if not text:
            return
        
        # Variantes fonéticas comunes capturadas por Vosk
        keywords = ["enciendete", "enciende te", "enciendete", "enciende", "enciendet"]
        
        if any(kw in text for kw in keywords):
            now = time.time()
            if now - self.last_trigger_time > self.cooldown_seconds:
                self.last_trigger_time = now
                if self.status_callback:
                    self.status_callback("🎤 ¡Palabra clave 'Enciéndete' detectada!")
                if self.trigger_callback:
                    # Ejecutar callback de activación en un hilo para no bloquear el audio
                    threading.Thread(target=self.trigger_callback, daemon=True).start()
                # Resetear el reconocedor para limpiar el buffer
                if self.recognizer:
                    self.recognizer.Reset()
