import os
import glob
import json
import threading
import flet as ft
from src.data_manager import LSPDataManager
from src.vision_service import LSPVisionService
from src.trainer import LSPTrainer
from src.voice_service import LSPVoiceService
from src.model_trainer import ModelTrainer
from src.tester_service import LiveTester

EMPTY_PIXEL_DATA = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# Paleta Cromática Escolar (Celeste y Blanco)
COLOR_BG_PAGE = "#F4F8FA"         # Fondo suave blanco/gris azulado
COLOR_CARD_BG = "#FFFFFF"         # Fondo de tarjetas blanco puro
COLOR_BORDER = "#D1E4F8"          # Borde celeste claro
COLOR_PRIMARY = "#4A90E2"         # Celeste escolar principal
COLOR_TEXT_TITLE = "#1A365D"      # Azul oscuro académico
COLOR_TEXT_BODY = "#2D3748"       # Gris carbón para texto general
COLOR_TEXT_MUTED = "#718096"      # Gris suave para texto secundario
COLOR_STATUS_BG = "#EBF4FF"       # Azul pastel suave para barra de estado
COLOR_REC_BTN = "#E25C5C"         # Rojo amigable para grabación
COLOR_SUCCESS = "#2E7D32"         # Verde pedagógico para éxito

def update_ui_safely(control):
    """
    Actualiza un control de Flet capturando excepciones de sesión destruida.
    Previene el fallo: RuntimeError: An attempt to fetch destroyed session.
    """
    try:
        if control and hasattr(control, "page") and control.page:
            control.update()
        elif control and hasattr(control, "update"):
            control.update()
    except RuntimeError as e:
        if "destroyed session" in str(e).lower():
            # Ignorar de forma segura si la ventana fue cerrada por el docente
            pass
    except Exception:
        pass

# Alias de compatibilidad
safe_update = update_ui_safely

def show_snack_bar(page: ft.Page, message: str, is_error: bool = False):
    """Muestra una notificación emergente estilizada con la paleta escolar sin crashear si se cierra."""
    def _show():
        try:
            if page and hasattr(page, "is_active") and not page.is_active:
                return
            sb = ft.SnackBar(
                content=ft.Text(message, color=ft.Colors.WHITE, weight=ft.FontWeight.W_500),
                bgcolor=COLOR_REC_BTN if is_error else COLOR_PRIMARY,
                open=True
            )
            if hasattr(page, "overlay"):
                page.overlay.append(sb)
            update_ui_safely(page)
        except Exception:
            pass

    if hasattr(page, "run_thread"):
        try:
            page.run_thread(_show)
        except Exception:
            pass
    else:
        _show()

class LSPUIController:
    """
    Controlador central de la interfaz escolar de administración para Traductor LSP (v4).
    - Carga estricta y defensiva de modelos Keras (.h5/.keras) y etiquetas JSON sin errores UTF-8.
    - Protección absoluta contra crashes por sesiones destruidas de Flet (update_ui_safely).
    - Navegación por pestañas nativas ft.Tabs visible y con prevención de colisiones de cámara.
    - Lista de palabras en contenedor con scroll vertical de alta fluidez (height=240).
    - Deslizadores de configuración reubicados en la parte inferior del panel izquierdo.
    """
    def __init__(self, page: ft.Page, data_manager: LSPDataManager, vision_service: LSPVisionService, trainer: LSPTrainer):
        self.page = page
        self.data_manager = data_manager
        self.vision_service = vision_service
        self.trainer = trainer
        self.model_trainer = ModelTrainer(sequence_length=30, features=255)

        # Referencia a pestañas para control de contexto
        self.tabs = None

        # Estados de la UI
        self.selected_category = None
        self.selected_word = None
        self.is_camera_active = False

        # Estado de validación en vivo
        self.live_tester = None
        self.is_testing = False

        # Referencia al diálogo modal de muestras
        self.current_samples_dialog = None
        self.current_samples_word = None
        self.samples_listview = ft.ListView(expand=True, spacing=6, padding=5)

        # Configurar comunicación con el servicio de visión
        self.vision_service.page = page
        self.vision_service.on_frame_callback = self.on_frame_update
        self.vision_service.on_state_changed = self.on_state_changed
        self.vision_service.recording_callback = self.on_recording_complete

        # Servicio de voz Vosk no bloqueante con comando 'recopila'
        self.voice_service = LSPVoiceService(
            page_ref=page,
            on_command_detected=self.on_voice_trigger_detected,
            status_callback=self.on_voice_status_update
        )

        # Construir controles visuales escolares optimizados
        self._init_controls()

        # Enlazar control de imagen directo para refresco quirúrgico
        self.vision_service.video_image_control = self.camera_view

    def _init_controls(self):
        # 1. Controles de Categoría
        self.new_category_input = ft.TextField(
            label="Nombre de Categoría",
            width=200,
            text_size=12,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_MUTED),
            color=COLOR_TEXT_TITLE,
            height=40
        )
        self.category_dropdown = ft.Dropdown(
            label="Categoría de Señas",
            hint_text="Seleccione categoría",
            width=250,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_MUTED),
            color=COLOR_TEXT_TITLE,
            height=40,
            on_select=lambda e: self.on_category_changed(e.control.value)
        )
        self.btn_edit_category = ft.IconButton(
            icon=ft.Icons.EDIT,
            tooltip="Editar nombre de categoría",
            icon_color=COLOR_PRIMARY,
            icon_size=20,
            on_click=self.show_edit_category_dialog
        )
        self.btn_delete_category = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            tooltip="Eliminar categoría",
            icon_color=COLOR_REC_BTN,
            icon_size=20,
            on_click=self.on_delete_category_clicked
        )

        # 2. Controles de Vocabulario
        self.new_word_input = ft.TextField(
            label="Nueva Palabra",
            width=200,
            text_size=12,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_MUTED),
            color=COLOR_TEXT_TITLE,
            height=40
        )
        # Lista de palabras con scroll siempre activo ("tipo slider vertical")
        self.words_listview = ft.ListView(
            expand=True,
            scroll=ft.ScrollMode.ALWAYS, # Forzar barra de scroll vertical
            spacing=10
        )

        # 3. Configuración para el Docente (Deslizadores al Fondo)
        self.slider_delay = ft.Slider(
            min=1.0,
            max=5.0,
            divisions=8,
            value=3.0,
            label="{value}s",
            active_color=COLOR_PRIMARY,
            on_change=self.on_teacher_params_changed
        )
        self.lbl_delay_val = ft.Text("3.0s", size=12, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)

        self.slider_frames = ft.Slider(
            min=20,
            max=60,
            divisions=8,
            value=30,
            label="{value} frames",
            active_color=COLOR_PRIMARY,
            on_change=self.on_teacher_params_changed
        )
        self.lbl_frames_val = ft.Text("30 frames", size=12, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)

        # 4. Visores de Cámara (Monitor centrado 480x360)
        self.camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=480,
            height=360,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain"
        )
        self.test_camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=480,
            height=360,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain"
        )

        self.warning_banner = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.AMBER_900, size=18),
                ft.Text("⚠️ ADVERTENCIA: Cámara obstruida o baja iluminación.",
                        color=ft.Colors.AMBER_900, weight=ft.FontWeight.BOLD, size=12)
            ], alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#FEF3C7",
            padding=6,
            border_radius=8,
            border=ft.Border.all(1, "#FCD34D"),
            visible=False
        )

        # 5. Modo Escucha por Voz con comando 'Recopila'
        self.switch_voice = ft.Switch(
            label="Modo Escucha (Vosk)",
            value=False,
            active_color=COLOR_PRIMARY,
            label_text_style=ft.TextStyle(color=COLOR_TEXT_TITLE, weight=ft.FontWeight.W_500, size=12),
            on_change=self.toggle_voice_mode,
            tooltip="Diga 'Recopila' para iniciar la preparación y grabación a manos libres"
        )
        self.voice_badge = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.MIC_OFF, size=15, color=COLOR_TEXT_MUTED),
                ft.Text("Voz inactiva", size=11, color=COLOR_TEXT_MUTED)
            ], spacing=4),
            padding=ft.Padding(6, 3, 6, 3),
            bgcolor="#F1F5F9",
            border_radius=6
        )

        # 6. Botones de Acción (Pestaña 1)
        self.btn_camera = ft.Button(
            content="Encender Cámara",
            icon=ft.Icons.VIDEOCAM,
            bgcolor=COLOR_PRIMARY,
            color=ft.Colors.WHITE,
            on_click=self.toggle_camera,
            width=180,
            height=40
        )
        self.btn_generate_cnn = ft.Button(
            content="Generar Modelo de Categoría",
            icon=ft.Icons.AUTO_AWESOME,
            bgcolor="#6366F1",
            color=ft.Colors.WHITE,
            on_click=self.run_cnn_training_flow,
            disabled=True,
            width=240,
            height=40,
            tooltip="Habilitado al tener al menos 30 muestras de cada palabra en la categoría"
        )

        # 7. Barra de Estado Escolar (Superior Permanente)
        self.status_text = ft.Text(
            value="Sistema Escolar Listo. Encienda la cámara para comenzar.",
            color=COLOR_TEXT_TITLE,
            size=13,
            weight=ft.FontWeight.W_500
        )
        self.training_progress = ft.ProgressRing(visible=False, width=16, height=16, stroke_width=2, color=COLOR_PRIMARY)

        # 8. Controles de la Pestaña 2: Inferencia en Tiempo Real con Voz y Letras Gigantes
        self.test_category_dropdown = ft.Dropdown(
            label="Modelo por Categoría",
            hint_text="Seleccione modelo binario",
            width=260,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            color=COLOR_TEXT_TITLE,
            height=40,
            on_select=self.on_test_category_selected
        )
        self.btn_toggle_test = ft.Button(
            content="Iniciar Prueba",
            icon=ft.Icons.PLAY_ARROW,
            bgcolor=COLOR_SUCCESS,
            color=ft.Colors.WHITE,
            on_click=self.toggle_live_test,
            width=180,
            height=40
        )
        # Tipografía gigante (size=36, color=#1A365D) para traducción clara
        self.lbl_prediction = ft.Text(
            value="Esperando seña...",
            size=36,
            weight=ft.FontWeight.BOLD,
            color=COLOR_TEXT_TITLE,
            text_align=ft.TextAlign.CENTER
        )
        self.lbl_confidence = ft.Text(
            value="Confianza: 0%",
            size=12,
            weight=ft.FontWeight.W_600,
            color=COLOR_PRIMARY
        )
        self.progress_bar_prediction = ft.ProgressBar(
            value=0.0,
            color=COLOR_PRIMARY,
            bgcolor=COLOR_STATUS_BG,
            height=8,
            border_radius=4,
            width=360
        )
        self.test_status_text = ft.Text(
            value="Seleccione un modelo (.h5/.keras) y pulse 'Iniciar Prueba' para validar señas en vivo.",
            size=12,
            color=COLOR_TEXT_MUTED
        )

        # Conectar controles de inferencia con vision_service
        self.vision_service.prediction_label_control = self.lbl_prediction
        self.vision_service.progress_bar_control = self.progress_bar_prediction
        self.vision_service.confidence_label_control = self.lbl_confidence

        # Cargar categorías y modelos iniciales
        self.load_categories_to_dropdown()
        self.load_trained_models_to_test_dropdown()

    # --- REFRESCO DE CÁMARA CON PROTECCIÓN CONTRA DESTROYED SESSION ---

    def on_frame_update(self, base64_image: str, is_obstructed: bool = False):
        data_src = f"data:image/jpeg;base64,{base64_image}"
        
        self.camera_view.src = data_src
        update_ui_safely(self.camera_view)

        if hasattr(self, "test_camera_view") and getattr(self.test_camera_view, "visible", True):
            self.test_camera_view.src = data_src
            update_ui_safely(self.test_camera_view)

        if self.warning_banner.visible != is_obstructed:
            self.warning_banner.visible = is_obstructed
            update_ui_safely(self.warning_banner)

    # --- CONFIGURACIÓN DINÁMICA DEL DOCENTE ---

    def on_teacher_params_changed(self, e):
        delay = float(self.slider_delay.value)
        frames = int(self.slider_frames.value)

        self.lbl_delay_val.value = f"{delay:.1f}s"
        self.lbl_frames_val.value = f"{frames} frames"
        
        self.vision_service.update_params(delay=delay, frames=frames)

        update_ui_safely(self.lbl_delay_val)
        update_ui_safely(self.lbl_frames_val)

    # --- TRANSICIONES DE LA MÁQUINA DE ESTADOS ---

    def on_state_changed(self, state: str, message: str = ""):
        if state == "Preparacion":
            self.status_text.value = f"⏱️ {message}"
            self.status_text.color = "#D97706"
            self.enable_ui_controls(False)
        elif state == "Grabacion":
            self.status_text.value = f"🔴 {message}"
            self.status_text.color = COLOR_REC_BTN
        elif state == "Inactivo":
            self.status_text.value = f"✅ {message}"
            self.status_text.color = COLOR_SUCCESS
            self.enable_ui_controls(True)
        elif state == "Fin":
            self.status_text.value = f"💾 {message}"
            self.status_text.color = COLOR_PRIMARY

        update_ui_safely(self.status_text)

    def on_recording_complete(self, category: str, word: str, sequence: list):
        try:
            file_path = self.data_manager.save_sequence(category, word, sequence)
            word_dir = self.data_manager._get_word_dir(category, word)
            num_muestras = len([f for f in os.listdir(word_dir) if f.endswith('.csv')])
            self.status_text.value = f"¡Muestra guardada! '{word.upper()}': {num_muestras} muestras."
            self.status_text.color = COLOR_SUCCESS
            show_snack_bar(self.page, f"Muestra #{num_muestras} registrada para '{word.upper()}'")
        except Exception as ex:
            self.status_text.value = f"Error al guardar la muestra: {str(ex)}"
            self.status_text.color = COLOR_REC_BTN
            show_snack_bar(self.page, f"Error al guardar muestra: {str(ex)}", is_error=True)

        self.enable_ui_controls(True)
        self.refresh_words_list()
        self.update_cnn_button_state()
        update_ui_safely(self.page)

    # --- RECONOCIMIENTO DE VOZ (VOSK) CON CONTROL ESTRICTO DE CONTEXTO ---

    def toggle_voice_mode(self, e):
        if self.switch_voice.value:
            if not self.is_camera_active:
                self.toggle_camera(None)

            self.voice_badge.content = ft.Row([
                ft.Icon(ft.Icons.MIC, size=15, color=COLOR_SUCCESS),
                ft.Text("Escuchando...", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.W_600)
            ], spacing=4)
            self.voice_badge.bgcolor = "#DCFCE7"
            self.voice_service.start()
            show_snack_bar(self.page, "Modo Escucha activo: diga 'Recopila' para iniciar grabación")
        else:
            self.voice_service.stop()
            self.voice_badge.content = ft.Row([
                ft.Icon(ft.Icons.MIC_OFF, size=15, color=COLOR_TEXT_MUTED),
                ft.Text("Voz inactiva", size=11, color=COLOR_TEXT_MUTED)
            ], spacing=4)
            self.voice_badge.bgcolor = "#F1F5F9"
            show_snack_bar(self.page, "Modo Escucha desactivado.")
        
        update_ui_safely(self.voice_badge)
        update_ui_safely(self.switch_voice)

    def on_voice_status_update(self, msg: str):
        self.status_text.value = msg
        update_ui_safely(self.status_text)

    def on_voice_trigger_detected(self):
        """
        REGLA DE SEGURIDAD ESTRICTA:
        El comando 'recopila' solo se ejecuta si:
        1. voice_service.allow_voice_trigger es True.
        2. La pestaña activa es Captura (índice 0).
        3. La cámara web está encendida.
        4. La máquina de visión está en estado 'Inactivo'.
        """
        if not getattr(self.voice_service, "allow_voice_trigger", True):
            print("[Voz] Comando 'recopila' ignorado: allow_voice_trigger está desactivado.")
            return

        if hasattr(self, "tabs") and self.tabs and getattr(self.tabs, "selected_index", 0) != 0:
            print("[Voz] Comando 'recopila' ignorado: fuera de la pestaña de Captura.")
            return

        if not self.is_camera_active:
            print("[Voz] Comando 'recopila' ignorado: la cámara web está apagada.")
            return

        if self.vision_service.current_state != "Inactivo":
            print(f"[Voz] Comando 'recopila' ignorado: visión en estado activo '{self.vision_service.current_state}'.")
            return

        target_word = None
        if self.selected_category and self.selected_word:
            target_word = self.selected_word
        elif self.selected_category:
            words = self.data_manager.get_words_in_category(self.selected_category)
            if words:
                target_word = words[0]
        else:
            cats = self.data_manager.get_categories()
            if cats:
                self.selected_category = cats[0]
                self.category_dropdown.value = cats[0]
                words = self.data_manager.get_words_in_category(cats[0])
                if words:
                    target_word = words[0]

        if not target_word:
            show_snack_bar(self.page, "Seleccione o cree una palabra antes de usar el comando de voz.", is_error=True)
            return

        self.selected_word = target_word
        show_snack_bar(self.page, f"🎤 ¡Comando 'Recopila' confirmado! Preparando captura para '{target_word.upper()}'")
        self.start_preparation_flow(target_word)

    # --- FLUJO DE PREPARACIÓN Y CAPTURA ---

    def start_preparation_flow(self, word: str):
        if not self.is_camera_active:
            self.toggle_camera(None)

        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría primero.", is_error=True)
            return

        self.selected_word = word
        self.enable_ui_controls(False)
        self.vision_service.start_preparation(self.selected_category, word)

    # --- GESTIÓN DE CATEGORÍAS Y PALABRAS ---

    def load_categories_to_dropdown(self):
        categories = self.data_manager.get_categories()
        self.category_dropdown.options = [ft.DropdownOption(key=cat, text=cat.upper()) for cat in categories]
        if self.selected_category and self.selected_category in categories:
            self.category_dropdown.value = self.selected_category
        elif categories:
            self.category_dropdown.value = None
        update_ui_safely(self.category_dropdown)

    def on_category_changed(self, category_val: str):
        if not category_val:
            return
        self.selected_category = category_val.lower().strip()
        self.selected_word = None
        self.status_text.value = f"Categoría activa: {category_val.upper()}"
        self.status_text.color = COLOR_TEXT_TITLE
        self.refresh_words_list()
        self.update_cnn_button_state()
        update_ui_safely(self.status_text)

    def update_cnn_button_state(self):
        if not self.selected_category:
            self.btn_generate_cnn.disabled = True
            self.btn_generate_cnn.tooltip = "Seleccione una categoría primero"
            update_ui_safely(self.btn_generate_cnn)
            return

        words = self.data_manager.get_words_in_category(self.selected_category)
        if len(words) < 2:
            self.btn_generate_cnn.disabled = True
            self.btn_generate_cnn.tooltip = f"Se requieren al menos 2 palabras (actual: {len(words)})"
            update_ui_safely(self.btn_generate_cnn)
            return

        missing = []
        for w in words:
            wdir = self.data_manager._get_word_dir(self.selected_category, w)
            count = len([f for f in os.listdir(wdir) if f.endswith('.csv')]) if os.path.exists(wdir) else 0
            if count < 30:
                missing.append(f"{w.upper()} ({count}/30)")

        if missing:
            self.btn_generate_cnn.disabled = True
            self.btn_generate_cnn.tooltip = f"Faltan muestras (mínimo 30 por palabra): {', '.join(missing)}"
        else:
            self.btn_generate_cnn.disabled = False
            self.btn_generate_cnn.tooltip = "¡Listo! Todas las palabras tienen 30+ muestras."

        try:
            if self.page and hasattr(self.page, "is_active") and self.page.is_active:
                update_ui_safely(self.btn_generate_cnn)
        except RuntimeError as e:
            if "destroyed session" in str(e).lower():
                print("[CONCURRENCIA] Sesión cerrada. Abortando actualización de control.")
                return

    def add_new_category(self, e):
        cat_name = self.new_category_input.value.strip()
        if not cat_name:
            show_snack_bar(self.page, "Ingrese un nombre para la categoría", is_error=True)
            return
        if self.data_manager.create_category(cat_name):
            show_snack_bar(self.page, f"Categoría '{cat_name.upper()}' creada correctamente")
            self.new_category_input.value = ""
            self.selected_category = cat_name.lower().strip()
            self.load_categories_to_dropdown()
            self.category_dropdown.value = self.selected_category
            self.refresh_words_list()
            self.update_cnn_button_state()
        else:
            show_snack_bar(self.page, "La categoría ya existe", is_error=True)

    def show_edit_category_dialog(self, e):
        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría primero para editar", is_error=True)
            return

        edit_input = ft.TextField(
            label="Nuevo Nombre de Categoría",
            value=self.selected_category.upper(),
            text_size=14,
            autofocus=True,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY
        )

        def save_rename(ev):
            new_name = edit_input.value.strip()
            if not new_name:
                show_snack_bar(self.page, "El nombre de categoría no puede estar vacío", is_error=True)
                return
            try:
                if self.data_manager.rename_category(self.selected_category, new_name):
                    old_name = self.selected_category
                    self.selected_category = new_name.lower().strip()
                    self.load_categories_to_dropdown()
                    self.category_dropdown.value = self.selected_category
                    self.refresh_words_list()
                    self.update_cnn_button_state()
                    self.load_trained_models_to_test_dropdown()
                    self.status_text.value = f"Categoría '{old_name.upper()}' renombrada a '{new_name.upper()}'."
                    self.status_text.color = COLOR_SUCCESS
                    show_snack_bar(self.page, "Categoría renombrada correctamente")
            except Exception as ex:
                show_snack_bar(self.page, str(ex), is_error=True)
            finally:
                if hasattr(self.page, "pop_dialog"):
                    self.page.pop_dialog()
                update_ui_safely(self.page)

        def cancel_dialog(ev):
            if hasattr(self.page, "pop_dialog"):
                self.page.pop_dialog()
            update_ui_safely(self.page)

        dialog = ft.AlertDialog(
            title=ft.Text("Editar Nombre de Categoría", color=COLOR_TEXT_TITLE, weight=ft.FontWeight.BOLD),
            content=edit_input,
            actions=[
                ft.TextButton(content="Cancelar", on_click=cancel_dialog),
                ft.Button(content="Guardar", on_click=save_rename, bgcolor=COLOR_PRIMARY, color=ft.Colors.WHITE)
            ]
        )
        if hasattr(self.page, "show_dialog"):
            self.page.show_dialog(dialog)
        update_ui_safely(self.page)

    def on_delete_category_clicked(self, e):
        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría primero para eliminar", is_error=True)
            return

        words = self.data_manager.get_words_in_category(self.selected_category)
        if words and len(words) > 0:
            show_snack_bar(
                self.page,
                "Debe eliminar todas las palabras antes de eliminar la categoría",
                is_error=True
            )
            return

        try:
            cat_deleted = self.selected_category
            self.data_manager.delete_category(self.selected_category)
            self.selected_category = None
            self.selected_word = None
            self.load_categories_to_dropdown()
            self.category_dropdown.value = None
            self.refresh_words_list()
            self.update_cnn_button_state()
            self.status_text.value = f"Categoría '{cat_deleted.upper()}' eliminada."
            self.status_text.color = COLOR_REC_BTN
            show_snack_bar(self.page, "Categoría eliminada correctamente")
        except Exception as ex:
            show_snack_bar(self.page, f"Error al eliminar categoría: {str(ex)}", is_error=True)

    def refresh_words_list(self):
        self.words_listview.controls.clear()
        if not self.selected_category:
            update_ui_safely(self.words_listview)
            return

        words = self.data_manager.get_words_in_category(self.selected_category)
        for word in words:
            word_dir = self.data_manager._get_word_dir(self.selected_category, word)
            samples_count = 0
            if os.path.exists(word_dir):
                samples_count = len([f for f in os.listdir(word_dir) if f.endswith('.csv')])

            is_complete = samples_count >= 30
            badge_color = COLOR_SUCCESS if is_complete else "#D97706"

            word_row = ft.Container(
                content=ft.Row([
                    ft.Text(f"{word.upper()} ({samples_count}/30)", expand=True, size=12, weight=ft.FontWeight.BOLD, color=badge_color),
                    # Botón Modificar Muestras
                    ft.IconButton(
                        icon=ft.Icons.LIST_ALT,
                        icon_color=COLOR_PRIMARY,
                        icon_size=18,
                        tooltip="Modificar Muestras (.csv individuales)",
                        on_click=lambda ev, w=word: self.open_samples_modal(w)
                    ),
                    # Botón Grabar
                    ft.IconButton(
                        icon=ft.Icons.RADIO_BUTTON_CHECKED,
                        icon_color=COLOR_REC_BTN,
                        icon_size=18,
                        tooltip="Grabar seña con preparación previa",
                        on_click=lambda ev, w=word: self.start_preparation_flow(w)
                    ),
                    # Botón Eliminar
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        icon_color=COLOR_REC_BTN,
                        icon_size=18,
                        tooltip="Eliminar palabra y todas sus muestras",
                        on_click=lambda ev, w=word: self.delete_word(w)
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=ft.Padding(8, 4, 8, 4),
                border=ft.Border.all(1, COLOR_BORDER),
                border_radius=8,
                bgcolor="#F8FAFC"
            )
            self.words_listview.controls.append(word_row)

        update_ui_safely(self.words_listview)

    def add_new_word(self, e):
        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría primero", is_error=True)
            return

        word_name = self.new_word_input.value.strip()
        if not word_name:
            show_snack_bar(self.page, "Ingrese un nombre de palabra", is_error=True)
            return

        if self.data_manager.add_word_to_category(self.selected_category, word_name):
            self.status_text.value = f"Palabra '{word_name.upper()}' agregada."
            self.status_text.color = COLOR_SUCCESS
            self.new_word_input.value = ""
            self.refresh_words_list()
            self.update_cnn_button_state()
        else:
            show_snack_bar(self.page, "La palabra ya existe en esta categoría", is_error=True)

    def delete_word(self, word: str):
        if self.data_manager.delete_word(self.selected_category, word):
            self.status_text.value = f"Palabra '{word.upper()}' eliminada del disco."
            self.status_text.color = COLOR_REC_BTN
            self.refresh_words_list()
            self.update_cnn_button_state()

    # --- MODAL DE GESTIÓN DE MUESTRAS (CRUD GRANULAR) ---

    def open_samples_modal(self, word: str):
        if not self.selected_category:
            return

        self.current_samples_word = word
        self._reload_samples_listview()

        def _on_close(e):
            if hasattr(self.page, "pop_dialog"):
                self.page.pop_dialog()
            self.refresh_words_list()
            self.update_cnn_button_state()

        def _on_delete_all(e):
            deleted = self.data_manager.delete_all_samples_for_word(self.selected_category, word)
            self._reload_samples_listview()
            self.refresh_words_list()
            self.update_cnn_button_state()
            show_snack_bar(self.page, f"Se eliminaron {deleted} muestras de '{word.upper()}'.")

        dialog = ft.AlertDialog(
            title=ft.Row([
                ft.Icon(ft.Icons.FOLDER_OPEN, color=COLOR_PRIMARY),
                ft.Text(f"Muestras de '{word.upper()}' ({self.selected_category.upper()})",
                        size=15, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
            ], spacing=8),
            content=ft.Container(
                content=self.samples_listview,
                width=500,
                height=320,
                bgcolor="#FFFFFF",
                border=ft.Border.all(1, COLOR_BORDER),
                border_radius=10,
                padding=6
            ),
            actions=[
                ft.Button(
                    content="Eliminar Todas las Muestras",
                    icon=ft.Icons.DELETE_SWEEP,
                    bgcolor=COLOR_REC_BTN,
                    color=ft.Colors.WHITE,
                    on_click=_on_delete_all
                ),
                ft.Button(
                    content="Cerrar",
                    bgcolor=COLOR_PRIMARY,
                    color=ft.Colors.WHITE,
                    on_click=_on_close
                )
            ]
        )

        self.current_samples_dialog = dialog
        if hasattr(self.page, "show_dialog"):
            self.page.show_dialog(dialog)
        update_ui_safely(self.page)

    def _reload_samples_listview(self):
        self.samples_listview.controls.clear()
        if not self.selected_category or not self.current_samples_word:
            return

        samples = self.data_manager.get_sample_files(self.selected_category, self.current_samples_word)

        if not samples:
            self.samples_listview.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.Icons.INBOX, size=36, color=COLOR_TEXT_MUTED),
                        ft.Text("No hay muestras grabadas para esta palabra.",
                                color=COLOR_TEXT_MUTED, size=12, weight=ft.FontWeight.W_500)
                    ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    height=180,
                    alignment=ft.Alignment.CENTER
                )
            )
        else:
            for idx, sample in enumerate(samples):
                badge_bg = "#DCFCE7" if sample["is_valid"] else "#FEF3C7"
                badge_fg = COLOR_SUCCESS if sample["is_valid"] else "#D97706"
                
                row_item = ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Text(f"Muestra #{idx + 1} ({sample['filename']})", size=12, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                            ft.Row([
                                ft.Container(
                                    content=ft.Text(f"{sample['rows']}f x {sample['cols']}c", size=10, color=badge_fg, weight=ft.FontWeight.W_600),
                                    bgcolor=badge_bg,
                                    border_radius=4,
                                    padding=ft.Padding(5, 2, 5, 2)
                                ),
                                ft.Text(f"{sample['size_kb']} KB", size=10, color=COLOR_TEXT_MUTED)
                            ], spacing=6)
                        ], spacing=2, expand=True),
                        ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color=COLOR_REC_BTN,
                            icon_size=18,
                            tooltip="Eliminar esta muestra individual",
                            on_click=lambda ev, p=sample["filepath"]: self.delete_single_sample_action(p)
                        )
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    bgcolor="#F8FAFC",
                    border=ft.Border.all(1, COLOR_BORDER),
                    border_radius=8,
                    padding=ft.Padding(8, 4, 8, 4)
                )
                self.samples_listview.controls.append(row_item)

        update_ui_safely(self.samples_listview)

    def delete_single_sample_action(self, file_path: str):
        if self.data_manager.delete_sample_file(file_path):
            self._reload_samples_listview()
            self.refresh_words_list()
            self.update_cnn_button_state()
            show_snack_bar(self.page, "Muestra eliminada correctamente.")

    def enable_ui_controls(self, enabled: bool):
        self.category_dropdown.disabled = not enabled
        self.btn_edit_category.disabled = not enabled
        self.btn_delete_category.disabled = not enabled
        self.new_category_input.disabled = not enabled
        self.new_word_input.disabled = not enabled
        self.words_listview.disabled = not enabled
        self.slider_delay.disabled = not enabled
        self.slider_frames.disabled = not enabled
        if enabled:
            self.update_cnn_button_state()
        else:
            self.btn_generate_cnn.disabled = True
        update_ui_safely(self.page)

    # --- ACCIONES DE CÁMARA ---

    def toggle_camera(self, e):
        if not self.is_camera_active:
            self.status_text.value = "Iniciando cámara web a 25 FPS estables..."
            update_ui_safely(self.status_text)
            
            self.vision_service.start()
            self.is_camera_active = True
            self.btn_camera.content = "Apagar Cámara"
            self.btn_camera.icon = ft.Icons.VIDEOCAM_OFF
            self.btn_camera.bgcolor = COLOR_REC_BTN
            self.status_text.value = "Cámara activa (25 FPS). Listo para operar."
            self.status_text.color = COLOR_SUCCESS
        else:
            self.vision_service.stop()
            self.is_camera_active = False
            self.btn_camera.content = "Encender Cámara"
            self.btn_camera.icon = ft.Icons.VIDEOCAM
            self.btn_camera.bgcolor = COLOR_PRIMARY
            self.camera_view.src = EMPTY_PIXEL_DATA
            self.test_camera_view.src = EMPTY_PIXEL_DATA
            self.warning_banner.visible = False
            self.status_text.value = "Cámara apagada."
            self.status_text.color = COLOR_TEXT_MUTED
        
        update_ui_safely(self.btn_camera)
        update_ui_safely(self.status_text)
        update_ui_safely(self.camera_view)
        update_ui_safely(self.test_camera_view)

    # --- MÓDULO DE ENTRENAMIENTO CNN 1D CON SAFE UPDATE ---

    def run_cnn_training_flow(self, e):
        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría para entrenar", is_error=True)
            return

        def _async_cnn_train():
            try:
                def _ui_start():
                    try:
                        self.training_progress.visible = True
                        self.status_text.value = f"Entrenando CNN 1D para '{self.selected_category.upper()}' (50 épocas)..."
                        self.status_text.color = COLOR_PRIMARY
                        self.enable_ui_controls(False)
                        update_ui_safely(self.page)
                    except RuntimeError as re:
                        if "destroyed session" in str(re).lower(): return

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                target_frames = self.vision_service.target_frames
                X, y, label_map = self.data_manager.load_dataset_for_training(self.selected_category, target_frames=target_frames)
                num_classes = len(label_map)

                model_path = self.model_trainer.build_and_train_cnn(
                    X_train=X,
                    y_train=y,
                    num_classes=num_classes,
                    category_name=self.selected_category,
                    label_map=label_map
                )

                def _ui_success():
                    try:
                        self.status_text.value = f"¡Modelo CNN generado con éxito! Guardado en: {model_path}"
                        self.status_text.color = COLOR_SUCCESS
                        show_snack_bar(self.page, f"¡Modelo CNN para '{self.selected_category.upper()}' generado con éxito!")
                        self.load_trained_models_to_test_dropdown()
                    except RuntimeError as re:
                        if "destroyed session" in str(re).lower(): return

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_success)

            except Exception as err:
                def _ui_error():
                    try:
                        self.status_text.value = f"Error en entrenamiento CNN: {str(err)}"
                        self.status_text.color = COLOR_REC_BTN
                        show_snack_bar(self.page, f"Error: {str(err)}", is_error=True)
                    except RuntimeError as re:
                        if "destroyed session" in str(re).lower(): return

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_error)
            finally:
                def _ui_finish():
                    try:
                        self.training_progress.visible = False
                        self.enable_ui_controls(True)
                        update_ui_safely(self.page)
                    except RuntimeError as re:
                        if "destroyed session" in str(re).lower(): return

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_finish)

        threading.Thread(target=_async_cnn_train, daemon=True).start()

    # --- MÓDULO DE PRUEBAS: CARGA SEGURA Y DEFENSIVA DE MODELOS ---

    def load_trained_models_to_test_dropdown(self):
        """Escanea 'modelos/' buscando exclusivamente archivos binarios .h5 y .keras (sin .json)."""
        os.makedirs('modelos', exist_ok=True)
        model_files = glob.glob('modelos/modelo_LSP_*.h5') + glob.glob('modelos/modelo_LSP_*.keras')
        
        categories = set()
        for mf in model_files:
            if mf.endswith('.json'):
                continue
            base = os.path.basename(mf)
            cat = base.replace("modelo_LSP_", "").replace(".h5", "").replace(".keras", "")
            if cat and not cat.endswith("_labels"):
                categories.add(cat)

        cat_list = sorted(list(categories))
        self.test_category_dropdown.options = [
            ft.DropdownOption(key=c, text=f"Categoría: {c.upper()}") for c in cat_list
        ]
        
        if cat_list and (not self.test_category_dropdown.value or self.test_category_dropdown.value not in cat_list):
            self.test_category_dropdown.value = cat_list[0]
        elif not cat_list:
            self.test_category_dropdown.value = None

        update_ui_safely(self.test_category_dropdown)

    def on_test_category_selected(self, e):
        cat = self.test_category_dropdown.value
        if cat:
            self.test_status_text.value = f"Modelo seleccionado: categoría {cat.upper()} listo para validar."
            update_ui_safely(self.test_status_text)

    def toggle_live_test(self, e):
        if not self.is_testing:
            cat = self.test_category_dropdown.value
            if not cat:
                show_snack_bar(self.page, "Debe seleccionar un modelo entrenado para probar", is_error=True)
                return

            # 1. Localizar estrictamente el modelo binario (.keras o .h5)
            model_path = f"modelos/modelo_LSP_{cat}.keras"
            if not os.path.exists(model_path):
                model_path = f"modelos/modelo_LSP_{cat}.h5"

            if not os.path.exists(model_path):
                show_snack_bar(self.page, f"No se encontró el archivo del modelo binario: {model_path}", is_error=True)
                return

            # Validación defensiva de formato binario
            if not (model_path.endswith('.keras') or model_path.endswith('.h5')):
                raise ValueError(f"Ruta de modelo inválida: {model_path}. Debe ser un archivo .keras o .h5")

            # 2. Cargar etiquetas estrictamente del JSON independiente
            labels_path = f"modelos/modelo_LSP_{cat}_labels.json"
            labels = []
            if os.path.exists(labels_path):
                try:
                    with open(labels_path, 'r', encoding='utf-8') as f:
                        label_map = json.load(f)
                        if isinstance(label_map, dict):
                            labels = [label_map[str(i)] if str(i) in label_map else label_map[i] for i in range(len(label_map))]
                        elif isinstance(label_map, list):
                            labels = label_map
                except Exception as ex_json:
                    print(f"Error leyendo {labels_path}: {ex_json}")
                    labels = self.data_manager.get_words_in_category(cat)
            else:
                labels = self.data_manager.get_words_in_category(cat)

            try:
                # 3. Inicializar LiveTester con modelo binario y labels
                self.live_tester = LiveTester(model_path=model_path, labels=labels, page_ref=self.page)
                self.live_tester.start()

                # Desactivar trigger de voz durante pruebas
                self.voice_service.allow_voice_trigger = False

                # Vincular controles con vision_service
                self.vision_service.live_tester = self.live_tester
                self.vision_service.prediction_label_control = self.lbl_prediction
                self.vision_service.progress_bar_control = self.progress_bar_prediction
                self.vision_service.confidence_label_control = self.lbl_confidence

                if not self.is_camera_active:
                    self.toggle_camera(None)

                self.is_testing = True
                self.btn_toggle_test.content = "Detener Prueba"
                self.btn_toggle_test.icon = ft.Icons.STOP
                self.btn_toggle_test.bgcolor = COLOR_REC_BTN
                self.lbl_prediction.value = "Esperando seña..."
                self.lbl_prediction.color = COLOR_TEXT_TITLE
                self.lbl_confidence.value = "Confianza: 0%"
                self.progress_bar_prediction.value = 0.0
                self.test_status_text.value = f"🔴 Prueba en vivo activa: {cat.upper()} ({len(labels)} clases). Voz activada."
                self.test_status_text.color = COLOR_SUCCESS
                show_snack_bar(self.page, f"Prueba iniciada para '{cat.upper()}'. Realice señas frente a la cámara.")

            except Exception as ex:
                show_snack_bar(self.page, f"Error cargando modelo: {str(ex)}", is_error=True)

        else:
            if self.live_tester:
                self.live_tester.stop()
            self.vision_service.live_tester = None
            self.vision_service.prediction_label_control = None

            # Restaurar trigger de voz si estamos en pestaña 0
            if hasattr(self, "tabs") and self.tabs and getattr(self.tabs, "selected_index", 0) == 0:
                self.voice_service.allow_voice_trigger = True

            self.is_testing = False
            self.btn_toggle_test.content = "Iniciar Prueba"
            self.btn_toggle_test.icon = ft.Icons.PLAY_ARROW
            self.btn_toggle_test.bgcolor = COLOR_SUCCESS
            self.lbl_prediction.value = "Prueba detenida."
            self.lbl_prediction.color = COLOR_TEXT_MUTED
            self.lbl_confidence.value = "Confianza: 0%"
            self.progress_bar_prediction.value = 0.0
            self.test_status_text.value = "Prueba detenida. Puede cambiar de modelo o reiniciar."
            self.test_status_text.color = COLOR_TEXT_MUTED
            show_snack_bar(self.page, "Prueba en vivo detenida.")

        update_ui_safely(self.btn_toggle_test)
        update_ui_safely(self.lbl_prediction)
        update_ui_safely(self.lbl_confidence)
        update_ui_safely(self.progress_bar_prediction)
        update_ui_safely(self.test_status_text)

    def close(self):
        """Detiene de forma segura todos los servicios y bucles en segundo plano."""
        try:
            if self.voice_service:
                self.voice_service.stop()
        except Exception:
            pass
        try:
            if self.live_tester:
                self.live_tester.stop()
        except Exception:
            pass
        try:
            if self.vision_service:
                self.vision_service.stop()
        except Exception:
            pass

# --- CONSTRUCTORES DE VISTAS ESCOLARES (PESTAÑAS v4) ---

def build_training_view(controller: LSPUIController) -> ft.Container:
    """
    Construye la vista de Captura y Entrenamiento:
    - Panel izquierdo con Vocabulario en contenedor de scroll vertical fluido (height=240).
    - Deslizadores de configuración (Espera y Frames) ubicados ergonómicamente al fondo del panel.
    - Panel derecho con el monitor de cámara centrado (480x360).
    """
    
    # 1. Contenedor Scrollable de Palabras (Scroll Adaptativo Vertical Obligatorio)
    words_scrollable_container = ft.Container(
        content=controller.words_listview,
        height=240, # Altura controlada para no empujar el resto de elementos
        border=ft.Border.all(1, "#D1E4F8"),
        border_radius=12,
        bgcolor="#FFFFFF",
        padding=10
    )

    # 2. Deslizadores de Configuración reubicados en la parte inferior
    sliders_bottom_card = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.TUNE, size=15, color=COLOR_PRIMARY),
                ft.Text("Configuración de Captura (Docente)", weight=ft.FontWeight.BOLD, color="#1A365D", size=13),
            ], spacing=6),
            ft.Row([
                ft.Column([
                    ft.Row([ft.Text("Espera (s):", size=11, color=COLOR_TEXT_BODY), controller.lbl_delay_val], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    controller.slider_delay
                ], expand=True),
                ft.Column([
                    ft.Row([ft.Text("Muestras (frames):", size=11, color=COLOR_TEXT_BODY), controller.lbl_frames_val], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    controller.slider_frames
                ], expand=True)
            ], spacing=10)
        ], spacing=4),
        bgcolor="#FFFFFF",
        border=ft.Border.all(1, "#D1E4F8"),
        border_radius=10,
        padding=10
    )

    # Panel Izquierdo Completo con estructura solicitada en muestra .md
    panel_izquierdo = ft.Column([
        ft.Text("1. Categorías de Estudio", size=15, color="#1A365D", weight=ft.FontWeight.BOLD),
        ft.Row([
            controller.new_category_input,
            ft.IconButton(ft.Icons.ADD_CIRCLE, icon_color=COLOR_PRIMARY, on_click=controller.add_new_category, tooltip="Crear Categoría")
        ], spacing=4),
        ft.Row([
            controller.category_dropdown,
            controller.btn_edit_category,
            controller.btn_delete_category,
        ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
        
        ft.Divider(height=10, color="#D1E4F8"),
        
        ft.Text("2. Vocabulario Registrado", size=15, color="#1A365D", weight=ft.FontWeight.BOLD),
        ft.Row([
            controller.new_word_input,
            ft.IconButton(ft.Icons.ADD_TASK, icon_color=COLOR_PRIMARY, on_click=controller.add_new_word, tooltip="Agregar Palabra")
        ], spacing=4),
        
        # Lista de palabras con scroll ("bajar y subir")
        words_scrollable_container,
        
        ft.Divider(height=10, color="#D1E4F8"),
        
        # Deslizadores reubicados en la parte inferior
        sliders_bottom_card
    ], spacing=8, expand=True)

    sidebar = ft.Container(
        content=panel_izquierdo,
        width=460,
        bgcolor=COLOR_CARD_BG,
        padding=14,
        border_radius=12,
        border=ft.Border.all(1, COLOR_BORDER)
    )

    # Panel Derecho: Monitor de Video Controlado y Centrado (480x360)
    camera_panel = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.CAMERA_ALT, color=COLOR_PRIMARY, size=18),
                    ft.Text("Captura de Señas & Red Convolucional", size=14, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                ], spacing=6),
                ft.Row([controller.switch_voice, controller.voice_badge], spacing=4)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            # Monitor de video (480x360)
            ft.Container(
                content=ft.Column([
                    controller.warning_banner,
                    controller.camera_view
                ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                border=ft.Border.all(2, COLOR_BORDER),
                border_radius=12,
                bgcolor="#0F172A",
                padding=4,
                alignment=ft.Alignment.CENTER
            ),
            
            # Botones de Acción
            ft.Row([
                controller.btn_camera,
                controller.btn_generate_cnn
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=12)
        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        expand=True,
        bgcolor=COLOR_CARD_BG,
        padding=14,
        border_radius=12,
        border=ft.Border.all(1, COLOR_BORDER)
    )

    return ft.Container(
        content=ft.Row([sidebar, camera_panel], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START, spacing=14),
        padding=4
    )

def build_live_testing_view(controller: LSPUIController) -> ft.Container:
    """
    Construye la vista de Pruebas y Validación en Vivo (Traductor):
    - Selector de modelos entrenados de la carpeta modelos/.
    - Monitor de cámara web (480x360).
    - Tarjeta gigante blanca con la palabra traducida y barra de confianza.
    - Síntesis de voz offline con pyttsx3.
    """
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.ANALYTICS_OUTLINED, color=COLOR_PRIMARY, size=24),
                ft.Text("Validación de Señas en Vivo con Síntesis de Voz (pyttsx3)", size=16, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
            ], spacing=8),
            ft.Divider(height=8, color=COLOR_BORDER),
            
            # Fila de Controles y Selector de Modelos
            ft.Row([
                controller.test_category_dropdown,
                controller.btn_toggle_test,
                ft.IconButton(
                    icon=ft.Icons.REFRESH,
                    icon_color=COLOR_PRIMARY,
                    tooltip="Recargar modelos de la carpeta modelos/",
                    on_click=lambda e: controller.load_trained_models_to_test_dropdown()
                )
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
            
            ft.Row([
                # Monitor de video en vivo (480x360)
                ft.Container(
                    content=controller.test_camera_view,
                    border=ft.Border.all(2, COLOR_BORDER),
                    border_radius=12,
                    bgcolor="#0F172A",
                    padding=4,
                    alignment=ft.Alignment.CENTER
                ),
                
                # Panel de Salida: Tipografía Gigante y Síntesis de Voz
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.RECORD_VOICE_OVER, color=COLOR_PRIMARY, size=20),
                            ft.Text("Traducción en Tiempo Real", size=14, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
                        ], spacing=6),
                        
                        # Tarjeta Blanca Pura con Bordes Celestes
                        ft.Container(
                            content=ft.Column([
                                ft.Text("PALABRA DETECTADA", size=11, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY),
                                controller.lbl_prediction,  # Tipografía gigante
                                ft.Container(height=4),
                                controller.lbl_confidence,  # Porcentaje
                                controller.progress_bar_prediction,  # Barra celeste
                                ft.Container(height=4),
                                ft.Row([
                                    ft.Icon(ft.Icons.VOLUME_UP, size=15, color=COLOR_TEXT_MUTED),
                                    ft.Text("Voz activa (>85% sostenido en 10 frames)", size=11, color=COLOR_TEXT_MUTED)
                                ], spacing=4, alignment=ft.MainAxisAlignment.CENTER)
                            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
                            bgcolor="#FFFFFF",
                            border=ft.Border.all(2, COLOR_BORDER),
                            border_radius=12,
                            padding=16,
                            alignment=ft.Alignment.CENTER,
                            width=460,
                            height=240
                        ),
                        controller.test_status_text
                    ], spacing=10),
                    padding=6,
                    expand=True
                )
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START, spacing=14)
        ], spacing=10),
        padding=14,
        bgcolor=COLOR_CARD_BG,
        border_radius=12,
        border=ft.Border.all(1, COLOR_BORDER)
    )

def build_main_app_tabs(controller: LSPUIController) -> ft.Tabs:
    """
    Implementa el control de pestañas nativo ft.Tabs conectando TabBar y TabBarView.
    - Pestaña 1 (Captura y Entrenamiento): Grilla de grabación y gestión de señas.
    - Pestaña 2 (Prueba en Vivo del Traductor): Monitor, inferencia con letras gigantes y TTS.
    - Apaga la cámara automáticamente al cambiar de pestaña para prevenir colisiones de hardware.
    """
    view_training = build_training_view(controller)
    view_testing = build_live_testing_view(controller)

    tab1 = ft.Tab(label="Captura y Entrenamiento", icon=ft.Icons.SCHOOL)
    tab2 = ft.Tab(label="Prueba en Vivo del Traductor", icon=ft.Icons.FACT_CHECK)

    tab_bar = ft.TabBar(
        tabs=[tab1, tab2],
        divider_color=COLOR_BORDER,
        indicator_color=COLOR_PRIMARY,
        label_color=COLOR_PRIMARY,
        unselected_label_color=COLOR_TEXT_MUTED
    )

    tab_view = ft.TabBarView(
        expand=True,
        controls=[
            view_training,
            view_testing
        ]
    )

    def on_tab_change(e):
        try:
            new_idx = int(e.data) if hasattr(e, "data") and e.data is not None else tabs.selected_index
        except Exception:
            new_idx = getattr(tabs, "selected_index", 0)

        # 1. Apagar cámara si estaba activa para evitar colisiones de hardware entre pestañas
        if controller.is_camera_active:
            controller.toggle_camera(None)

        if controller.is_testing:
            controller.toggle_live_test(None)

        # 2. Control de contexto de voz y recarga de modelos
        if new_idx == 0:
            controller.voice_service.allow_voice_trigger = True
            print("[Tabs] Pestaña Captura activa: allow_voice_trigger = True")
        else:
            controller.voice_service.allow_voice_trigger = False
            controller.load_trained_models_to_test_dropdown()
            print("[Tabs] Pestaña Pruebas activa: allow_voice_trigger = False")

    tabs = ft.Tabs(
        length=2,
        height=680,
        content=ft.Column(expand=True, controls=[tab_bar, tab_view], spacing=6),
        on_change=on_tab_change
    )

    controller.tabs = tabs
    return tabs
