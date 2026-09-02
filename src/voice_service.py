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
    Servicio de reconocimiento de voz continuo no bloqueante utilizando Vosk y sounddevice.
    Diseñado para evitar deadlocks mediante threading.Event y despacho seguro con page.run_thread.
    """
    def __init__(self, page_ref=None, on_command_detected=None, status_callback=None):
        self.page = page_ref
        self.on_command_detected = on_command_detected
        self.status_callback = status_callback
        
        self.model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()
        
        self.is_running = threading.Event()
        self.thread = None
        self.stream = None
        
        self.last_trigger_time = 0.0
        self.cooldown_seconds = 4.0  # Cooldown para evitar ráfagas
        self.lock = threading.Lock()
        
        # Flag de control de contexto: solo activo en la pestaña de captura
        self.allow_voice_trigger = True

    def _ensure_model(self):
        """Carga el modelo ligero en español de forma perezosa (lazy) desde la caché local."""
        if self.model is None:
            self._notify_status("Cargando modelo de voz Vosk en español...")
            # Carga el modelo predescargado en español
            self.model = vosk.Model(lang="es")
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            self.recognizer.SetWords(True)

    def _notify_status(self, msg: str):
        if self.status_callback and self.is_running.is_set():
            try:
                if self.page and hasattr(self.page, "is_active") and not self.page.is_active:
                    self.stop()
                    return
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(self.status_callback, msg)
                else:
                    self.status_callback(msg)
            except RuntimeError as re:
                if "destroyed session" in str(re).lower():
                    self.stop()
                    return
            except Exception:
                pass

    def start(self):
        """Inicia la captura de audio en un hilo de fondo seguro."""
        if self.is_running.is_set():
            return
        self.is_running.set()
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Detiene de forma no bloqueante el micrófono y el hilo de escucha."""
        self.is_running.clear()

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            self.thread = None

        self._notify_status("Reconocimiento de voz desactivado.")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback del stream de sounddevice para encolar audio."""
        if self.is_running.is_set():
            self.audio_queue.put(bytes(indata))

    def _listen_loop(self):
        """Bucle de procesamiento de audio en hilo secundario."""
        try:
            self._ensure_model()
            self._notify_status("🎤 Micrófono activo. Diga 'Recopila' para iniciar.")

            with sd.RawInputStream(
                samplerate=16000,
                blocksize=4000,
                dtype='int16',
                channels=1,
                callback=self._audio_callback
            ) as stream:
                self.stream = stream
                while self.is_running.is_set():
                    try:
                        data = self.audio_queue.get(timeout=0.3)
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
            self._notify_status(f"Error en micrófono: {str(e)}")
            self.stop()

    def _check_trigger(self, text: str):
        """Evalúa si se detectó la palabra clave 'recopila' y despacha el evento de forma segura en la UI."""
        if not text:
            return
        
        # Validar contexto estricto: solo si allow_voice_trigger es True
        if not self.allow_voice_trigger:
            return

        keywords = ["recopila", "recopilar", "recopilalo", "recopilarlo", "recopilame"]
        if any(kw in text for kw in keywords):
            now = time.time()
            if now - self.last_trigger_time > self.cooldown_seconds:
                self.last_trigger_time = now
                self._notify_status("🎤 ¡Palabra clave 'Recopila' detectada!")
                
                if self.on_command_detected and self.is_running.is_set():
                    try:
                        if self.page and hasattr(self.page, "is_active") and not self.page.is_active:
                            self.stop()
                            return
                        if self.page and hasattr(self.page, "run_thread"):
                            self.page.run_thread(self.on_command_detected)
                        else:
                            threading.Thread(target=self.on_command_detected, daemon=True).start()
                    except RuntimeError as re:
                        if "destroyed session" in str(re).lower():
                            self.stop()
                            return
                    except Exception:
                        pass
                
                if self.recognizer:
                    self.recognizer.Reset()
