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

def show_snack_bar(page: ft.Page, message: str, is_error: bool = False):
    """Muestra una notificación emergente estilizada con la paleta escolar."""
    def _show():
        sb = ft.SnackBar(
            content=ft.Text(message, color=ft.Colors.WHITE, weight=ft.FontWeight.W_500),
            bgcolor=COLOR_REC_BTN if is_error else COLOR_PRIMARY,
            open=True
        )
        if hasattr(page, "overlay"):
            page.overlay.append(sb)
        try:
            page.update()
        except Exception:
            pass

    if hasattr(page, "run_thread"):
        page.run_thread(_show)
    else:
        _show()

class LSPUIController:
    """
    Controlador central de la interfaz escolar de administración para Traductor LSP.
    - Renderizado quirúrgico de cámara sin parpadeo (zero flickering).
    - Activación por voz con comando 'Recopila'.
    - Panel de configuración dinámica de captura para el docente.
    - Estética escolar celeste y blanco de alto contraste.
    """
    def __init__(self, page: ft.Page, data_manager: LSPDataManager, vision_service: LSPVisionService, trainer: LSPTrainer):
        self.page = page
        self.data_manager = data_manager
        self.vision_service = vision_service
        self.trainer = trainer
        self.model_trainer = ModelTrainer(sequence_length=30, features=255)

        # Estados de la UI
        self.selected_category = None
        self.selected_word = None
        self.is_camera_active = False

        # Estado de validación en vivo
        self.live_tester = None
        self.is_testing = False

        # Configurar comunicación segura con el servicio de visión
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

        # Construir componentes visuales escolares
        self._init_controls()

        # Enlazar control de imagen directo para refresco
        self.vision_service.video_image_control = self.camera_view

    def _init_controls(self):
        # 1. Controles de Categoría
        self.new_category_input = ft.TextField(
            label="Nombre de Categoría",
            width=210,
            text_size=13,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_MUTED),
            color=COLOR_TEXT_TITLE
        )
        self.category_dropdown = ft.Dropdown(
            label="Categoría de Señas",
            hint_text="Seleccione categoría",
            width=260,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_MUTED),
            color=COLOR_TEXT_TITLE,
            on_select=lambda e: self.on_category_changed(e.control.value)
        )
        self.btn_edit_category = ft.IconButton(
            icon=ft.Icons.EDIT,
            tooltip="Editar nombre de categoría",
            icon_color=COLOR_PRIMARY,
            on_click=self.show_edit_category_dialog
        )
        self.btn_delete_category = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            tooltip="Eliminar categoría",
            icon_color=COLOR_REC_BTN,
            on_click=self.on_delete_category_clicked
        )

        # 2. Controles de Vocabulario
        self.new_word_input = ft.TextField(
            label="Nueva Palabra",
            width=210,
            text_size=13,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_MUTED),
            color=COLOR_TEXT_TITLE
        )
        self.words_listview = ft.ListView(expand=True, spacing=8, padding=5)

        # 3. Visores de Cámara (Entrenamiento y Testing)
        self.camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=520,
            height=390,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain"
        )
        self.test_camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=520,
            height=390,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain"
        )

        self.warning_banner = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.AMBER_900),
                ft.Text("⚠️ ADVERTENCIA: Cámara obstruida o baja iluminación.",
                        color=ft.Colors.AMBER_900, weight=ft.FontWeight.BOLD, size=12)
            ], alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#FEF3C7",
            padding=8,
            border_radius=8,
            border=ft.Border.all(1, "#FCD34D"),
            visible=False
        )

        # 4. Modo Escucha por Voz con comando 'Recopila'
        self.switch_voice = ft.Switch(
            label="Modo Escucha (Vosk)",
            value=False,
            active_color=COLOR_PRIMARY,
            label_text_style=ft.TextStyle(color=COLOR_TEXT_TITLE, weight=ft.FontWeight.W_500, size=13),
            on_change=self.toggle_voice_mode,
            tooltip="Diga 'Recopila' para iniciar la preparación y grabación a manos libres"
        )
        self.voice_badge = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.MIC_OFF, size=16, color=COLOR_TEXT_MUTED),
                ft.Text("Voz inactiva", size=11, color=COLOR_TEXT_MUTED)
            ], spacing=4),
            padding=ft.Padding(8, 4, 8, 4),
            bgcolor="#F1F5F9",
            border_radius=6
        )

        # 5. Panel de Configuración Dinámica para el Docente (Sliding & Parameters)
        self.slider_delay = ft.Slider(
            min=1.0,
            max=5.0,
            divisions=8,
            value=3.0,
            label="{value}s",
            active_color=COLOR_PRIMARY,
            on_change=self.on_teacher_params_changed
        )
        self.lbl_delay_val = ft.Text("3.0s", size=13, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)

        self.slider_frames = ft.Slider(
            min=20,
            max=60,
            divisions=8,
            value=30,
            label="{value} frames",
            active_color=COLOR_PRIMARY,
            on_change=self.on_teacher_params_changed
        )
        self.lbl_frames_val = ft.Text("30 frames", size=13, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)

        # 6. Botones de Acción (Pestaña 1)
        self.btn_camera = ft.Button(
            content="Encender Cámara",
            icon=ft.Icons.VIDEOCAM,
            bgcolor=COLOR_PRIMARY,
            color=ft.Colors.WHITE,
            on_click=self.toggle_camera,
            width=200,
            height=44
        )
        self.btn_generate_cnn = ft.Button(
            content="Generar Modelo de Categoría",
            icon=ft.Icons.AUTO_AWESOME,
            bgcolor="#6366F1",
            color=ft.Colors.WHITE,
            on_click=self.run_cnn_training_flow,
            disabled=True,
            width=260,
            height=44,
            tooltip="Habilitado al tener al menos 30 muestras de cada palabra en la categoría"
        )

        # 7. Barra de Estado Escolar
        self.status_text = ft.Text(
            value="Sistema Escolar Listo. Encienda la cámara para comenzar.",
            color=COLOR_TEXT_TITLE,
            size=13,
            weight=ft.FontWeight.W_500
        )
        self.training_progress = ft.ProgressRing(visible=False, width=18, height=18, stroke_width=2, color=COLOR_PRIMARY)
        self.training_status_container = ft.Row([self.training_progress, self.status_text], alignment=ft.MainAxisAlignment.START)

        # 8. Controles de la Pestaña 2 (Pruebas y Validación en Vivo)
        self.test_category_dropdown = ft.Dropdown(
            label="Modelo por Categoría",
            hint_text="Seleccione un modelo entrenado",
            width=280,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            color=COLOR_TEXT_TITLE,
            on_select=self.on_test_category_selected
        )
        self.btn_toggle_test = ft.Button(
            content="Iniciar Prueba",
            icon=ft.Icons.PLAY_ARROW,
            bgcolor=COLOR_SUCCESS,
            color=ft.Colors.WHITE,
            on_click=self.toggle_live_test,
            width=200,
            height=44
        )
        self.lbl_prediction = ft.Text(
            value="Esperando seña...",
            size=28,
            weight=ft.FontWeight.BOLD,
            color=COLOR_PRIMARY
        )
        self.test_status_text = ft.Text(
            value="Seleccione un modelo (.h5) y pulse 'Iniciar Prueba' para validar señas en vivo.",
            size=13,
            color=COLOR_TEXT_MUTED
        )

        # Cargar categorías y modelos disponibles
        self.load_categories_to_dropdown()
        self.load_trained_models_to_test_dropdown()

    # --- REFRESCO QUIRÚRGICO DE LA CÁMARA (ZERO FLICKERING) ---

    def on_frame_update(self, base64_image: str, is_obstructed: bool = False):
        """
        Actualización dirigida exclusivamente sobre los componentes Image de Flet.
        NUNCA llama a page.update(), eliminando al 100% el parpadeo en pantalla.
        """
        data_src = f"data:image/jpeg;base64,{base64_image}"
        
        # 1. Refresco aislado del visor de captura
        self.camera_view.src = data_src
        try:
            self.camera_view.update()
        except Exception:
            pass

        # 2. Refresco aislado del visor de testing si está en uso
        if hasattr(self, "test_camera_view") and getattr(self.test_camera_view, "visible", True):
            self.test_camera_view.src = data_src
            try:
                self.test_camera_view.update()
            except Exception:
                pass

        # 3. Refresco condicional del banner únicamente al variar su estado
        if self.warning_banner.visible != is_obstructed:
            self.warning_banner.visible = is_obstructed
            try:
                self.warning_banner.update()
            except Exception:
                pass

    # --- CONFIGURACIÓN DINÁMICA DEL DOCENTE ---

    def on_teacher_params_changed(self, e):
        """Aplica en caliente las nuevas variables de tiempo de espera y frames por seña."""
        delay = float(self.slider_delay.value)
        frames = int(self.slider_frames.value)

        self.lbl_delay_val.value = f"{delay:.1f}s"
        self.lbl_frames_val.value = f"{frames} frames"
        
        # Actualizar máquina de visión en caliente
        self.vision_service.update_params(delay=delay, frames=frames)

        try:
            self.lbl_delay_val.update()
            self.lbl_frames_val.update()
        except Exception:
            self.page.update()

    # --- TRANSICIONES DE LA MÁQUINA DE ESTADOS (DESPACHADAS VÍA RUN_THREAD) ---

    def on_state_changed(self, state: str, message: str = ""):
        """Callback invocado por la máquina de estados desde page.run_thread."""
        if state == "Preparacion":
            self.status_text.value = f"⏱️ {message}"
            self.status_text.color = "#D97706"  # Ámbar suave
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

        try:
            self.status_text.update()
        except Exception:
            pass

    def on_recording_complete(self, category: str, word: str, sequence: list):
        """Persiste la secuencia en disco y actualiza la UI de manera no bloqueante."""
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
        try:
            self.page.update()
        except Exception:
            pass

    # --- RECONOCIMIENTO DE VOZ (VOSK) CON COMANDO 'RECOPILA' ---

    def toggle_voice_mode(self, e):
        """Activa o desactiva el hilo de escucha de micrófono con Vosk."""
        if self.switch_voice.value:
            if not self.is_camera_active:
                self.toggle_camera(None)

            self.voice_badge.content = ft.Row([
                ft.Icon(ft.Icons.MIC, size=16, color=COLOR_SUCCESS),
                ft.Text("Escuchando...", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.W_600)
            ], spacing=4)
            self.voice_badge.bgcolor = "#DCFCE7"
            self.voice_service.start()
            show_snack_bar(self.page, "Modo Escucha activo: diga 'Recopila' para iniciar grabación")
        else:
            self.voice_service.stop()
            self.voice_badge.content = ft.Row([
                ft.Icon(ft.Icons.MIC_OFF, size=16, color=COLOR_TEXT_MUTED),
                ft.Text("Voz inactiva", size=11, color=COLOR_TEXT_MUTED)
            ], spacing=4)
            self.voice_badge.bgcolor = "#F1F5F9"
            show_snack_bar(self.page, "Modo Escucha desactivado.")
        
        try:
            self.voice_badge.update()
            self.switch_voice.update()
        except Exception:
            self.page.update()

    def on_voice_status_update(self, msg: str):
        self.status_text.value = msg
        try:
            self.status_text.update()
        except Exception:
            pass

    def on_voice_trigger_detected(self):
        """Gatillado por Vosk al detectar 'recopila' de forma segura en page.run_thread."""
        if not self.is_camera_active:
            self.toggle_camera(None)

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
        show_snack_bar(self.page, f"🎤 ¡Comando 'Recopila' detectado! Prepárate para '{target_word.upper()}'")
        self.start_preparation_flow(target_word)

    # --- FLUJO DE PREPARACIÓN Y CAPTURA SIN ESPERA BLOQUEANTE ---

    def start_preparation_flow(self, word: str):
        """Inicia la máquina de estados sin bloqueos de tiempo en el hilo de la UI."""
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
        try:
            self.category_dropdown.update()
        except Exception:
            pass

    def on_category_changed(self, category_val: str):
        if not category_val:
            return
        self.selected_category = category_val.lower().strip()
        self.selected_word = None
        self.status_text.value = f"Categoría activa: {category_val.upper()}"
        self.status_text.color = COLOR_TEXT_TITLE
        self.refresh_words_list()
        self.update_cnn_button_state()

    def update_cnn_button_state(self):
        """Habilita el botón de entrenamiento CNN si todas las palabras tienen >= 30 muestras."""
        if not self.selected_category:
            self.btn_generate_cnn.disabled = True
            self.btn_generate_cnn.tooltip = "Seleccione una categoría primero"
            try: self.btn_generate_cnn.update()
            except Exception: pass
            return

        words = self.data_manager.get_words_in_category(self.selected_category)
        if len(words) < 2:
            self.btn_generate_cnn.disabled = True
            self.btn_generate_cnn.tooltip = f"Se requieren al menos 2 palabras (actual: {len(words)})"
            try: self.btn_generate_cnn.update()
            except Exception: pass
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
            self.btn_generate_cnn.update()
        except Exception:
            self.page.update()

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
                self.page.update()

        def cancel_dialog(ev):
            if hasattr(self.page, "pop_dialog"):
                self.page.pop_dialog()
            self.page.update()

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
        self.page.update()

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
            try: self.words_listview.update()
            except Exception: pass
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
                    ft.Text(f"{word.upper()} ({samples_count}/30)", expand=True, size=13, weight=ft.FontWeight.BOLD, color=badge_color),
                    ft.IconButton(
                        icon=ft.Icons.RADIO_BUTTON_CHECKED,
                        icon_color=COLOR_REC_BTN,
                        tooltip="Grabar seña con preparación previa",
                        on_click=lambda ev, w=word: self.start_preparation_flow(w)
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        icon_color=COLOR_REC_BTN,
                        tooltip="Eliminar palabra y sus muestras",
                        on_click=lambda ev, w=word: self.delete_word(w)
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=ft.Padding(10, 6, 10, 6),
                border=ft.Border.all(1, COLOR_BORDER),
                border_radius=8,
                bgcolor="#F8FAFC"
            )
            self.words_listview.controls.append(word_row)

        try:
            self.words_listview.update()
        except Exception:
            self.page.update()

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
        try:
            self.page.update()
        except Exception:
            pass

    # --- ACCIONES DE CÁMARA ---

    def toggle_camera(self, e):
        """Enciende o apaga la cámara web de forma asíncrona a 25 FPS estables."""
        if not self.is_camera_active:
            self.status_text.value = "Iniciando cámara web a 25 FPS estables..."
            try: self.status_text.update()
            except Exception: pass
            
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
        
        try:
            self.btn_camera.update()
            self.status_text.update()
            self.camera_view.update()
            self.test_camera_view.update()
        except Exception:
            self.page.update()

    # --- MÓDULO DE ENTRENAMIENTO CNN 1D ---

    def run_cnn_training_flow(self, e):
        """Entrena la CNN 1D espacio-temporal en un hilo secundario independiente con despacho seguro."""
        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría para entrenar", is_error=True)
            return

        def _async_cnn_train():
            try:
                def _ui_start():
                    self.training_progress.visible = True
                    self.status_text.value = f"Entrenando CNN 1D para '{self.selected_category.upper()}' (50 épocas)..."
                    self.status_text.color = COLOR_PRIMARY
                    self.enable_ui_controls(False)
                    self.page.update()

                self.page.run_thread(_ui_start)

                # Cargar dataset de la categoría
                target_frames = self.vision_service.target_frames
                X, y, label_map = self.data_manager.load_dataset_for_training(self.selected_category, target_frames=target_frames)
                num_classes = len(label_map)

                # Entrenar CNN 1D y exportar modelo .h5
                model_path = self.model_trainer.build_and_train_cnn(
                    X_train=X,
                    y_train=y,
                    num_classes=num_classes,
                    category_name=self.selected_category,
                    label_map=label_map
                )

                def _ui_success():
                    self.status_text.value = f"¡Modelo CNN generado con éxito! Guardado en: {model_path}"
                    self.status_text.color = COLOR_SUCCESS
                    show_snack_bar(self.page, f"¡Modelo CNN para '{self.selected_category.upper()}' generado con éxito!")
                    self.load_trained_models_to_test_dropdown()

                self.page.run_thread(_ui_success)

            except Exception as err:
                def _ui_error():
                    self.status_text.value = f"Error en entrenamiento CNN: {str(err)}"
                    self.status_text.color = COLOR_REC_BTN
                    show_snack_bar(self.page, f"Error: {str(err)}", is_error=True)

                self.page.run_thread(_ui_error)
            finally:
                def _ui_finish():
                    self.training_progress.visible = False
                    self.enable_ui_controls(True)
                    self.page.update()

                self.page.run_thread(_ui_finish)

        threading.Thread(target=_async_cnn_train, daemon=True).start()

    # --- MÓDULO DE PRUEBAS Y VALIDACIÓN EN VIVO (SLIDING WINDOW) ---

    def load_trained_models_to_test_dropdown(self):
        """Escanea el directorio 'modelos/' y puebla el dropdown de pruebas."""
        os.makedirs('modelos', exist_ok=True)
        model_files = glob.glob('modelos/modelo_LSP_*.h5')
        
        categories = []
        for mf in model_files:
            base = os.path.basename(mf)
            cat = base.replace("modelo_LSP_", "").replace(".h5", "")
            categories.append(cat)

        self.test_category_dropdown.options = [
            ft.DropdownOption(key=c, text=f"Categoría: {c.upper()}") for c in categories
        ]
        
        if categories and not self.test_category_dropdown.value:
            self.test_category_dropdown.value = categories[0]
        elif not categories:
            self.test_category_dropdown.value = None

        try:
            self.test_category_dropdown.update()
        except Exception:
            pass

    def on_test_category_selected(self, e):
        cat = self.test_category_dropdown.value
        if cat:
            self.test_status_text.value = f"Modelo seleccionado: modelo_LSP_{cat}.h5 listo para probar."
            try: self.test_status_text.update()
            except Exception: self.page.update()

    def toggle_live_test(self, e):
        """Inicia o detiene la prueba en tiempo real con sliding window sin bloquear Flet."""
        if not self.is_testing:
            cat = self.test_category_dropdown.value
            if not cat:
                show_snack_bar(self.page, "Debe seleccionar un modelo entrenado para probar", is_error=True)
                return

            model_path = f"modelos/modelo_LSP_{cat}.h5"
            labels_path = f"modelos/modelo_LSP_{cat}_labels.json"
            
            if not os.path.exists(model_path):
                show_snack_bar(self.page, f"No se encontró el archivo del modelo: {model_path}", is_error=True)
                return

            labels = []
            if os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    label_map = json.load(f)
                    labels = [label_map[str(i)] if str(i) in label_map else label_map[i] for i in range(len(label_map))]
            else:
                labels = self.data_manager.get_words_in_category(cat)

            try:
                self.live_tester = LiveTester(model_path, labels, page_ref=self.page)
                self.live_tester.start()

                # Vincular tester con el servicio de visión
                self.vision_service.live_tester = self.live_tester
                self.vision_service.prediction_label_control = self.lbl_prediction

                if not self.is_camera_active:
                    self.toggle_camera(None)

                self.is_testing = True
                self.btn_toggle_test.content = "Detener Prueba"
                self.btn_toggle_test.icon = ft.Icons.STOP
                self.btn_toggle_test.bgcolor = COLOR_REC_BTN
                self.lbl_prediction.value = "Esperando seña..."
                self.lbl_prediction.color = COLOR_PRIMARY
                self.test_status_text.value = f"🔴 Prueba en vivo activa: {cat.upper()} ({len(labels)} clases)."
                self.test_status_text.color = COLOR_SUCCESS
                show_snack_bar(self.page, f"Prueba en vivo iniciada para '{cat.upper()}'. Realice señas frente a la cámara.")

            except Exception as ex:
                show_snack_bar(self.page, f"Error cargando modelo: {str(ex)}", is_error=True)

        else:
            if self.live_tester:
                self.live_tester.stop()
            self.vision_service.live_tester = None
            self.vision_service.prediction_label_control = None

            self.is_testing = False
            self.btn_toggle_test.content = "Iniciar Prueba"
            self.btn_toggle_test.icon = ft.Icons.PLAY_ARROW
            self.btn_toggle_test.bgcolor = COLOR_SUCCESS
            self.lbl_prediction.value = "Prueba detenida."
            self.lbl_prediction.color = COLOR_TEXT_MUTED
            self.test_status_text.value = "Prueba detenida. Puede cambiar de modelo o reiniciar."
            self.test_status_text.color = COLOR_TEXT_MUTED
            show_snack_bar(self.page, "Prueba en vivo detenida.")

        try:
            self.btn_toggle_test.update()
            self.lbl_prediction.update()
            self.test_status_text.update()
        except Exception:
            self.page.update()

    def close(self):
        if self.voice_service:
            self.voice_service.stop()
        if self.live_tester:
            self.live_tester.stop()

# --- CONSTRUCTORES DE VISTAS ESCOLARES (PESTAÑAS) ---

def build_training_view(controller: LSPUIController) -> ft.Container:
    """Construye la vista principal con estética escolar (blanco y celeste)."""
    
    # 1. Panel Lateral Izquierdo: Gestión de Categorías y Vocabulario
    sidebar = ft.Container(
        content=ft.Column([
            ft.Text("1. Categorías de Estudio", size=15, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
            ft.Row([
                controller.new_category_input,
                ft.IconButton(ft.Icons.ADD_CIRCLE, icon_color=COLOR_PRIMARY, on_click=controller.add_new_category, tooltip="Crear Categoría")
            ], spacing=6),
            ft.Row([
                controller.category_dropdown,
                controller.btn_edit_category,
                controller.btn_delete_category,
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
            ft.Divider(height=10, color=COLOR_BORDER),
            
            ft.Text("2. Vocabulario de la Categoría", size=15, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
            ft.Row([
                controller.new_word_input,
                ft.IconButton(ft.Icons.ADD_TASK, icon_color=COLOR_PRIMARY, on_click=controller.add_new_word, tooltip="Agregar Palabra")
            ], spacing=6),
            ft.Container(
                content=controller.words_listview,
                height=240,
                border=ft.Border.all(1, COLOR_BORDER),
                border_radius=10,
                padding=6,
                bgcolor="#FFFFFF"
            ),
            
            # Panel de Configuración Dinámica para el Docente
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.TUNE, size=18, color=COLOR_PRIMARY),
                        ft.Text("Configuración de Captura (Docente)", size=13, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                    ], spacing=6),
                    ft.Row([
                        ft.Text("Espera previa:", size=12, color=COLOR_TEXT_BODY),
                        controller.lbl_delay_val,
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    controller.slider_delay,
                    ft.Row([
                        ft.Text("Frames por muestra:", size=12, color=COLOR_TEXT_BODY),
                        controller.lbl_frames_val,
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    controller.slider_frames,
                ], spacing=4),
                bgcolor=COLOR_STATUS_BG,
                border=ft.Border.all(1, COLOR_BORDER),
                border_radius=10,
                padding=10
            )
        ], spacing=10),
        width=460,
        bgcolor=COLOR_CARD_BG,
        padding=16,
        border_radius=12,
        border=ft.Border.all(1, COLOR_BORDER)
    )

    # 2. Panel Derecho: Cámara Web a 25 FPS y Acciones
    camera_panel = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.CAMERA_ALT, color=COLOR_PRIMARY, size=20),
                    ft.Text("Captura de Señas & Red Convolucional", size=15, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                ], spacing=6),
                ft.Row([controller.switch_voice, controller.voice_badge], spacing=6)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            # Monitor de video
            ft.Container(
                content=ft.Column([
                    controller.warning_banner,
                    controller.camera_view
                ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                border=ft.Border.all(2, COLOR_BORDER),
                border_radius=12,
                bgcolor="#0F172A",
                padding=6,
                alignment=ft.Alignment.CENTER
            ),
            
            # Panel de Botones (Cámara y Modelo CNN 1D)
            ft.Row([
                controller.btn_camera,
                controller.btn_generate_cnn
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=14),
            
            # Barra de Estado Escolar en Banner Azul Pastel
            ft.Container(
                content=controller.training_status_container,
                bgcolor=COLOR_STATUS_BG,
                border_radius=10,
                padding=10,
                border=ft.Border.all(1, COLOR_BORDER)
            )
        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
        expand=True,
        bgcolor=COLOR_CARD_BG,
        padding=16,
        border_radius=12,
        border=ft.Border.all(1, COLOR_BORDER)
    )

    return ft.Container(
        content=ft.Row([sidebar, camera_panel], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START, spacing=14),
        padding=6
    )

def build_live_testing_view(controller: LSPUIController) -> ft.Container:
    """Construye la pestaña de Pruebas y Validación en Vivo con la estética escolar."""
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.ANALYTICS_OUTLINED, color=COLOR_PRIMARY, size=26),
                ft.Text("Validación de Señas en Tiempo Real (Ventana Deslizante 25 FPS)", size=17, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
            ], spacing=10),
            ft.Divider(height=10, color=COLOR_BORDER),
            
            ft.Row([
                controller.test_category_dropdown,
                controller.btn_toggle_test,
                ft.IconButton(
                    icon=ft.Icons.REFRESH,
                    icon_color=COLOR_PRIMARY,
                    tooltip="Recargar modelos de la carpeta modelos/",
                    on_click=lambda e: controller.load_trained_models_to_test_dropdown()
                )
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=15),
            
            ft.Row([
                # Monitor de video en vivo
                ft.Container(
                    content=controller.test_camera_view,
                    border=ft.Border.all(2, COLOR_BORDER),
                    border_radius=12,
                    bgcolor="#0F172A",
                    padding=6,
                    alignment=ft.Alignment.CENTER
                ),
                
                # Tarjeta de predicción escolar
                ft.Container(
                    content=ft.Column([
                        ft.Text("Resultado de Inferencia", size=15, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                        ft.Container(
                            content=ft.Column([
                                ft.Icon(ft.Icons.RECORD_VOICE_OVER, size=46, color=COLOR_PRIMARY),
                                controller.lbl_prediction,
                                ft.Container(
                                    content=ft.Text("Umbral de confianza pedagógico: > 85%", size=12, color=COLOR_PRIMARY, weight=ft.FontWeight.W_500),
                                    bgcolor=COLOR_STATUS_BG,
                                    border_radius=6,
                                    padding=ft.Padding(10, 4, 10, 4),
                                    border=ft.Border.all(1, COLOR_BORDER)
                                )
                            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                            bgcolor="#F8FAFC",
                            border=ft.Border.all(2, COLOR_BORDER),
                            border_radius=12,
                            padding=25,
                            alignment=ft.Alignment.CENTER,
                            width=460,
                            height=240
                        ),
                        controller.test_status_text
                    ], spacing=12),
                    padding=10,
                    expand=True
                )
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START, spacing=18)
        ], spacing=14),
        padding=16,
        bgcolor=COLOR_CARD_BG,
        border_radius=12,
        border=ft.Border.all(1, COLOR_BORDER)
    )

def build_main_app_tabs(controller: LSPUIController) -> ft.Tabs:
    """Ensambla las pestañas principales de la aplicación con TabBar y TabBarView de Flet 0.86."""
    t1 = ft.Tab(label="Captura y Entrenamiento", icon=ft.Icons.SCHOOL)
    t2 = ft.Tab(label="Validación en Vivo (Testing)", icon=ft.Icons.FACT_CHECK)
    
    tab_bar = ft.TabBar(
        tabs=[t1, t2],
        divider_color=COLOR_BORDER,
        indicator_color=COLOR_PRIMARY,
        label_color=COLOR_PRIMARY,
        unselected_label_color=COLOR_TEXT_MUTED
    )
    tab_views = ft.TabBarView(
        controls=[
            build_training_view(controller),
            build_live_testing_view(controller)
        ],
        expand=True
    )
    
    return ft.Tabs(
        length=2,
        content=ft.Column([tab_bar, tab_views], expand=True, spacing=8),
        expand=True
    )
