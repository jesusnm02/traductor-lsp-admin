import os
import glob
import json
import threading
import time
import flet as ft
from src.data_manager import LSPDataManager, LSPDatasetManager
from src.vision_service import LSPVisionService
from src.model_trainer import ModelTrainer, LSPTrainer
from src.voice_service import LSPVoiceService
from src.tester_service import LSPTesterService, LiveTester, cargar_modelo_de_pruebas, show_error_popup
from src.cloud_service import LSPCloudService

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MUESTRAS_DIR = os.path.join(DATA_DIR, "muestras")
MODELOS_DIR = os.path.join(DATA_DIR, "modelos")

os.makedirs(MUESTRAS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

EMPTY_PIXEL_DATA = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

# =========================================================================
# PALETA CROMÁTICA ESCOLAR (STITCH UI DESIGN SYSTEM)
# =========================================================================
COLOR_BG_PAGE = "#F4F8FA"         # Fondo suave gris blanco-azulado
COLOR_CARD_BG = "#FFFFFF"         # Fondo de tarjetas blanco puro
COLOR_BORDER = "#D1E4F8"          # Borde celeste institucional
COLOR_PRIMARY = "#0A66C2"         # Azul/celeste de acción principal
COLOR_PRIMARY_LIGHT = "#EBF4FF"   # Fondo celeste pastel para insignias y banners
COLOR_TEXT_TITLE = "#1A365D"      # Azul marino académico para títulos
COLOR_TEXT_BODY = "#2D3748"       # Gris carbón para texto general
COLOR_TEXT_MUTED = "#64748B"      # Gris suave para metadatos y rangos
COLOR_REC_BTN = "#E25C5C"         # Rojo amigable para captura/peligro
COLOR_SUCCESS = "#2E7D32"         # Verde pedagógico para 30/30 completado
COLOR_AMBER = "#D97706"           # Ámbar/naranja para señas en progreso
COLOR_DARK_MONITOR = "#0B0F19"    # Fondo negro azulado para monitores de cámara

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
    Controlador central de la interfaz escolar de administración para Traductor LSP.
    Implementación píxel a píxel del diseño de referencia de Google Stitch:
    - Pestaña 1 (Captura/Entrenamiento): 3 tarjetas laterales estructuradas + monitor 640x440 con overlays HUD.
    - Pestaña 2 (Pruebas en Vivo): Sub-banner activo, 3 tarjetas laterales + tarjeta de traducción gigante con tipografía 45px.
    """
    def __init__(self, page: ft.Page, data_manager: LSPDataManager, vision_service: LSPVisionService, trainer=None, tester_service=None):
        self.page = page
        self.data_manager = data_manager if data_manager is not None else LSPDatasetManager(base_dir=MUESTRAS_DIR)
        self.vision_service = vision_service
        self.trainer = trainer if trainer is not None else LSPTrainer(self.data_manager, export_base_dir=MODELOS_DIR)
        self.model_trainer = ModelTrainer(dataset_manager=self.data_manager, sequence_length=30, features=255, export_base_dir=MODELOS_DIR)
        self.live_tester = tester_service if tester_service is not None else LSPTesterService(model_base_dir=MODELOS_DIR, page_ref=page)
        self.cloud_service = LSPCloudService(modelos_dir=MODELOS_DIR)

        # Referencias de control de pestañas y botones superiores de modo
        self.tabs = None
        self.btn_mode_train = None
        self.btn_mode_test = None
        self.btn_mode_cloud = None
        self.test_model_banner = None

        # Controles y estado de Sincronización en la Nube (AWS S3) y Recursos Didácticos
        self.cloud_categories_listview = ft.ListView(expand=True, spacing=6, padding=4)
        self.cloud_resources_listview = ft.ListView(expand=True, spacing=6, padding=4)
        self.cloud_model_card_container = ft.Container(padding=ft.Padding(10, 6, 10, 6))
        self.current_cloud_category = None
        self.cloud_progress_bar = ft.ProgressBar(value=0.0, color=COLOR_PRIMARY, bgcolor="#E2E8F0", height=6, border_radius=3, visible=False)
        self.lbl_cloud_status = ft.Text("Consola de Transferencia: Listo para sincronizar modelos y guías didácticas con AWS S3.", size=12, weight=ft.FontWeight.W_500, color=COLOR_TEXT_TITLE, expand=True)
        self.cloud_progress_ring = ft.ProgressRing(width=16, height=16, stroke_width=2, color=COLOR_PRIMARY, visible=False)
        self.cloud_statuses = {}

        # FilePicker para asociación de recursos didácticos
        self.file_picker = ft.FilePicker()
        self.file_picker.on_result = self.on_file_picker_result
        self._picking_target_category = None
        self._picking_target_word = None
        if hasattr(self.page, "overlay") and self.page.overlay is not None:
            if self.file_picker not in self.page.overlay:
                self.page.overlay.append(self.file_picker)

        # Estados de la UI
        self.selected_category = None
        self.selected_word = None
        self.is_camera_active = False
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
        # 1. Controles de Categoría (Card 1)
        self.new_category_input = ft.TextField(
            hint_text="Nueva categoría...",
            text_size=12,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            color=COLOR_TEXT_TITLE,
            height=38,
            expand=True,
            border_radius=8,
            content_padding=ft.Padding(10, 8, 10, 8)
        )
        self.category_dropdown = ft.Dropdown(
            hint_text="Seleccionar categoría activa...",
            text_size=12,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            color=COLOR_TEXT_TITLE,
            height=38,
            expand=True,
            border_radius=8,
            content_padding=ft.Padding(10, 0, 10, 0),
            on_select=lambda e: self.on_category_changed(e.control.value)
        )
        self.btn_edit_category = ft.IconButton(
            icon=ft.Icons.EDIT_OUTLINED,
            tooltip="Editar nombre de categoría",
            icon_color=COLOR_PRIMARY,
            icon_size=18,
            on_click=self.show_edit_category_dialog
        )
        self.btn_delete_category = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            tooltip="Eliminar categoría",
            icon_color=COLOR_REC_BTN,
            icon_size=18,
            on_click=self.on_delete_category_clicked
        )

        # 2. Controles de Vocabulario (Card 2)
        self.lbl_queue_count = ft.Text("0 Señas registradas", size=11, color=COLOR_TEXT_MUTED)
        self.new_word_input = ft.TextField(
            hint_text="Nueva palabra/seña...",
            text_size=12,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            color=COLOR_TEXT_TITLE,
            height=38,
            expand=True,
            border_radius=8,
            prefix_icon=ft.Icons.PAN_TOOL_ALT_OUTLINED,
            content_padding=ft.Padding(10, 8, 10, 8)
        )
        self.words_listview = ft.ListView(
            expand=True,
            scroll=ft.ScrollMode.ALWAYS,
            spacing=6,
            padding=2
        )

        # 3. Configuración para el Docente (Card 3 - Deslizadores al Fondo)
        self.slider_delay = ft.Slider(
            min=1.0,
            max=5.0,
            divisions=8,
            value=3.0,
            label="{value}s",
            active_color=COLOR_PRIMARY,
            on_change=self.on_teacher_params_changed
        )
        self.lbl_delay_val = ft.Text("3.0s", size=12, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY)

        self.slider_frames = ft.Slider(
            min=20,
            max=60,
            divisions=8,
            value=30,
            label="{value} frames",
            active_color=COLOR_PRIMARY,
            on_change=self.on_teacher_params_changed
        )
        self.lbl_frames_val = ft.Text("30", size=12, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY)

        # 4. Visores de Cámara (Monitores con estilo Stitch 640x360)
        self.camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=640,
            height=360,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain"
        )
        self.test_camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=640,
            height=310,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain"
        )

        self.warning_banner = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.AMBER_900, size=16),
                ft.Text("⚠️ ADVERTENCIA: Iluminación deficiente o cámara obstruida.",
                        color=ft.Colors.AMBER_900, weight=ft.FontWeight.BOLD, size=11)
            ], alignment=ft.MainAxisAlignment.CENTER),
            bgcolor="#FEF3C7",
            padding=4,
            border_radius=6,
            border=ft.Border.all(1, "#FCD34D"),
            visible=False
        )

        # Overlays HUD en el Monitor de Cámara (Stitch)
        self.lbl_hud_fps = ft.Text("FPS: 25.0", size=10, color="#38BDF8", weight=ft.FontWeight.BOLD)
        self.lbl_hud_conf = ft.Text("Confianza: 98.4%", size=10, color="#38BDF8", weight=ft.FontWeight.BOLD)
        self.lbl_hud_hand = ft.Text("Mano: Derecha", size=10, color="#38BDF8", weight=ft.FontWeight.BOLD)
        self.lbl_hud_status = ft.Text("ESTADO: • Listo para operar", size=10, color="#E2E8F0", weight=ft.FontWeight.W_500)
        self.lbl_hud_target = ft.Text("PATRÓN OBJETIVO: Seleccione seña", size=10, color="#F8FAFC", weight=ft.FontWeight.BOLD)

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
                ft.Icon(ft.Icons.MIC_OFF, size=14, color=COLOR_TEXT_MUTED),
                ft.Text("Voz inactiva", size=11, color=COLOR_TEXT_MUTED)
            ], spacing=4),
            padding=ft.Padding(6, 3, 6, 3),
            bgcolor="#F1F5F9",
            border_radius=6
        )

        # 5b. Modo Avatar de Privacidad (De-identificar Docente)
        self.switch_avatar = ft.Switch(
            label="Modo Avatar de Privacidad (De-identificar Docente)",
            value=False,
            active_color=COLOR_PRIMARY,
            label_text_style=ft.TextStyle(color=COLOR_TEXT_TITLE, weight=ft.FontWeight.W_600, size=11),
            on_change=self.toggle_privacy_avatar,
            tooltip="Descarta el video real y proyecta un títere vectorial escolar para proteger la identidad biométrica"
        )

        # 5c. Selector de Categoría para Tab 3 (Nube AWS)
        self.cloud_category_dropdown = ft.Dropdown(
            hint_text="Categoría activa en S3...",
            text_size=12,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            color=COLOR_TEXT_TITLE,
            height=38,
            width=240,
            border_radius=8,
            content_padding=ft.Padding(10, 0, 10, 0),
            on_select=lambda e: self.on_cloud_category_changed(e.control.value)
        )

        # 6. Botones de Acción (Pestaña 1)
        self.btn_camera = ft.Button(
            content="Encender Cámara",
            icon=ft.Icons.VIDEOCAM_OUTLINED,
            bgcolor="#F1F5F9",
            color=COLOR_TEXT_TITLE,
            on_click=self.toggle_camera,
            width=200,
            height=42
        )
        self.btn_generate_cnn = ft.Button(
            content="Entrenar Categoría",
            icon=ft.Icons.AUTO_AWESOME,
            bgcolor=COLOR_PRIMARY,
            color=ft.Colors.WHITE,
            on_click=self.run_cnn_training_flow,
            disabled=True,
            width=220,
            height=42,
            tooltip="Habilitado al tener al menos 30 muestras de cada palabra en la categoría"
        )

        # 7. Barra de Estado Escolar (Superior Permanente)
        self.status_text = ft.Text(
            value="Sistema listo. Micrófono activo (Escuchando comando: 'recopila').",
            color=COLOR_TEXT_TITLE,
            size=12,
            weight=ft.FontWeight.W_500
        )
        self.training_progress = ft.ProgressRing(visible=False, width=14, height=14, stroke_width=2, color=COLOR_PRIMARY)

        # 8. Sub-banner Dinámico de Modelo Activo en Modo Pruebas (Stitch test.png)
        self.lbl_active_model_subbanner = ft.Text(
            value="• Modelo Activo: Ninguno seleccionado. Elija una categoría para iniciar inferencia a 25.0 FPS.",
            size=12,
            color=COLOR_TEXT_TITLE,
            weight=ft.FontWeight.W_500
        )

        # 9. Controles de la Pestaña 2: Inferencia en Tiempo Real
        self.test_category_dropdown = ft.Dropdown(
            hint_text="Seleccione modelo de prueba...",
            text_size=12,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            color=COLOR_TEXT_TITLE,
            height=38,
            expand=True,
            border_radius=8,
            content_padding=ft.Padding(10, 0, 10, 0),
            on_select=self.on_test_category_selected
        )
        self.btn_toggle_test = ft.Button(
            content="Iniciar Prueba",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            bgcolor=COLOR_PRIMARY,
            color=ft.Colors.WHITE,
            on_click=self.toggle_live_test,
            height=38,
            expand=True
        )
        self.test_classes_listview = ft.ListView(
            expand=True,
            scroll=ft.ScrollMode.ALWAYS,
            spacing=6,
            padding=2
        )
        self.lbl_test_classes_count = ft.Text("0 Clases", size=11, color=COLOR_TEXT_MUTED)

        # Componentes de la Tarjeta Gigante de Traducción (Tipografía 45px - Stitch)
        self.lbl_prediction = ft.Text(
            value="ESPERANDO SEÑA...",
            size=45,
            weight=ft.FontWeight.BOLD,
            color=COLOR_TEXT_TITLE,
            text_align=ft.TextAlign.LEFT
        )
        self.lbl_prediction_badge = ft.Container(
            content=ft.Text("Detección Activa", size=11, color=COLOR_PRIMARY, weight=ft.FontWeight.BOLD),
            bgcolor=COLOR_PRIMARY_LIGHT,
            border_radius=6,
            padding=ft.Padding(8, 3, 8, 3)
        )
        self.lbl_confidence = ft.Text(
            value="0.0%",
            size=26,
            weight=ft.FontWeight.BOLD,
            color=COLOR_TEXT_TITLE
        )
        self.progress_bar_prediction = ft.ProgressBar(
            value=0.0,
            color=COLOR_PRIMARY,
            bgcolor="#E2E8F0",
            height=8,
            border_radius=4
        )
        self.lbl_detection_state = ft.Text("✓ Estado: Esperando inicio de prueba en vivo", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.W_500)
        self.test_status_text = ft.Text(
            value="Margen de Error: ± 0.04",
            size=11,
            color=COLOR_TEXT_MUTED
        )

        # Conectar controles de inferencia con vision_service
        self.vision_service.prediction_label_control = self.lbl_prediction
        self.vision_service.progress_bar_control = self.progress_bar_prediction
        self.vision_service.confidence_label_control = self.lbl_confidence

        # Cargar categorías y modelos iniciales
        self.load_categories_to_dropdown()
        self.load_trained_models_to_test_dropdown()

    # --- CONTROL DE PESTAÑAS Y MODOS SUPERIORES ---

    def switch_tab(self, index: int):
        """Conmuta limpiamente de pestaña actualizando los botones pill superiores y evitando colisiones de cámara."""
        if hasattr(self, "tabs") and self.tabs:
            self.tabs.selected_index = index
            update_ui_safely(self.tabs)

        # Actualizar estilo de píldoras superiores
        pills = [self.btn_mode_train, self.btn_mode_test, self.btn_mode_cloud]
        for i, p in enumerate(pills):
            if p:
                if i == index:
                    p.bgcolor = COLOR_PRIMARY
                    p.content.color = ft.Colors.WHITE
                    p.content.weight = ft.FontWeight.BOLD
                else:
                    p.bgcolor = ft.Colors.TRANSPARENT
                    p.content.color = COLOR_TEXT_MUTED
                    p.content.weight = ft.FontWeight.W_500
                update_ui_safely(p)

        if index == 0:
            if self.test_model_banner:
                self.test_model_banner.visible = False
                update_ui_safely(self.test_model_banner)
            self.voice_service.allow_voice_trigger = True
        elif index == 1:
            if self.test_model_banner:
                self.test_model_banner.visible = True
                update_ui_safely(self.test_model_banner)
            self.voice_service.allow_voice_trigger = False
            self.load_trained_models_to_test_dropdown()
        elif index == 2:
            if self.test_model_banner:
                self.test_model_banner.visible = False
                update_ui_safely(self.test_model_banner)
            self.voice_service.allow_voice_trigger = False
            self.refresh_cloud_table()

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
        self.lbl_frames_val.value = f"{frames}"
        
        self.vision_service.update_params(delay=delay, frames=frames)

        update_ui_safely(self.lbl_delay_val)
        update_ui_safely(self.lbl_frames_val)

    # --- TRANSICIONES DE LA MÁQUINA DE ESTADOS ---

    def on_state_changed(self, state: str, message: str = ""):
        if state == "Preparacion":
            self.status_text.value = f"⏱️ {message}"
            self.status_text.color = COLOR_AMBER
            self.lbl_hud_status.value = f"ESTADO: • {message}"
            self.enable_ui_controls(False)
        elif state == "Grabacion":
            self.status_text.value = f"🔴 {message}"
            self.status_text.color = COLOR_REC_BTN
            self.lbl_hud_status.value = f"ESTADO: • {message}"
        elif state == "Inactivo":
            self.status_text.value = f"✅ {message}"
            self.status_text.color = COLOR_SUCCESS
            self.lbl_hud_status.value = "ESTADO: • Sistema en reposo listo para operar"
            self.enable_ui_controls(True)
        elif state == "Fin":
            self.status_text.value = f"💾 {message}"
            self.status_text.color = COLOR_PRIMARY
            self.lbl_hud_status.value = f"ESTADO: • {message}"

        update_ui_safely(self.status_text)
        update_ui_safely(self.lbl_hud_status)

    def on_recording_complete(self, category: str, word: str, sequence: list):
        try:
            file_path = self.data_manager.save_sequence(category, word, sequence)
            word_dir = self.data_manager._get_word_dir(category, word)
            num_muestras = len([f for f in os.listdir(word_dir) if f.endswith('.csv')])
            self.status_text.value = f"¡Muestra guardada! '{word.upper()}': {num_muestras} muestras registradas."
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
                ft.Icon(ft.Icons.MIC, size=14, color=COLOR_SUCCESS),
                ft.Text("Escuchando...", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.W_600)
            ], spacing=4)
            self.voice_badge.bgcolor = "#DCFCE7"
            self.voice_service.start()
            show_snack_bar(self.page, "Modo Escucha activo: diga 'Recopila' para iniciar grabación")
        else:
            self.voice_service.stop()
            self.voice_badge.content = ft.Row([
                ft.Icon(ft.Icons.MIC_OFF, size=14, color=COLOR_TEXT_MUTED),
                ft.Text("Voz inactiva", size=11, color=COLOR_TEXT_MUTED)
            ], spacing=4)
            self.voice_badge.bgcolor = "#F1F5F9"
            show_snack_bar(self.page, "Modo Escucha desactivado.")
        
        update_ui_safely(self.voice_badge)
        update_ui_safely(self.switch_voice)

    def toggle_privacy_avatar(self, e):
        val = bool(self.switch_avatar.value)
        self.vision_service.set_privacy_avatar_mode(val)
        if val:
            show_snack_bar(self.page, "Modo Avatar de Privacidad ACTIVADO: El video real se descarta.")
        else:
            show_snack_bar(self.page, "Modo Avatar DESACTIVADO: Vista de cámara real restaurada.")
        update_ui_safely(self.switch_avatar)

    def on_voice_status_update(self, msg: str):
        self.status_text.value = msg
        update_ui_safely(self.status_text)

    def on_voice_trigger_detected(self):
        if not getattr(self.voice_service, "allow_voice_trigger", True):
            return

        if hasattr(self, "tabs") and self.tabs and getattr(self.tabs, "selected_index", 0) != 0:
            return

        if not self.is_camera_active:
            return

        if self.vision_service.current_state != "Inactivo":
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
        show_snack_bar(self.page, f"🎤 ¡Comando 'Recopila' detectado! Preparando '{target_word.upper()}'")
        self.start_preparation_flow(target_word)

    # --- FLUJO DE PREPARACIÓN Y CAPTURA ---

    def start_preparation_flow(self, word: str):
        if not self.is_camera_active:
            self.toggle_camera(None)

        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría primero.", is_error=True)
            return

        self.selected_word = word
        self.lbl_hud_target.value = f"PATRÓN OBJETIVO: {word.upper()}"
        update_ui_safely(self.lbl_hud_target)

        self.enable_ui_controls(False)
        self.vision_service.start_preparation(self.selected_category, word)

    # --- GESTIÓN DE CATEGORÍAS Y PALABRAS ---

    def load_categories_to_dropdown(self):
        categories = self.data_manager.get_categories()
        self.category_dropdown.options = [ft.dropdown.Option(text=cat.upper(), key=cat) for cat in categories]
        if self.selected_category and self.selected_category in categories:
            self.category_dropdown.value = self.selected_category
        elif categories:
            self.category_dropdown.value = categories[0]
            self.on_category_changed(categories[0])
        else:
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

        update_ui_safely(self.btn_generate_cnn)

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
            self.lbl_queue_count.value = "0 Señas registradas"
            update_ui_safely(self.lbl_queue_count)
            update_ui_safely(self.words_listview)
            return

        words = self.data_manager.get_words_in_category(self.selected_category)
        self.lbl_queue_count.value = f"{len(words)} Señas en cola"
        update_ui_safely(self.lbl_queue_count)

        for word in words:
            word_dir = self.data_manager._get_word_dir(self.selected_category, word)
            samples_count = 0
            if os.path.exists(word_dir):
                samples_count = len([f for f in os.listdir(word_dir) if f.endswith('.csv')])

            is_complete = samples_count >= 30
            is_active_word = (self.selected_word == word)

            # Estilo píxel a píxel según Stitch principal.png
            if is_complete:
                # 30/30 Completado: fondo blanco, check verde, texto verde
                item_bg = "#FFFFFF"
                item_border = COLOR_BORDER
                icon_indicator = ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED, color=COLOR_SUCCESS, size=18)
                word_text_color = COLOR_SUCCESS
                label_str = f"{word.upper()} ({samples_count}/30)"
                rec_btn = ft.IconButton(
                    icon=ft.Icons.VIDEOCAM_OUTLINED,
                    icon_color=COLOR_TEXT_MUTED,
                    icon_size=18,
                    tooltip="Grabar muestra adicional",
                    on_click=lambda ev, w=word: self.start_preparation_flow(w)
                )
            elif samples_count > 0 or is_active_word:
                # En progreso / Seleccionado: fondo amarillo claro, punto ámbar, botón REC rojo
                item_bg = "#FEF9C3"
                item_border = "#FDE047"
                icon_indicator = ft.Container(width=10, height=10, bgcolor=COLOR_AMBER, border_radius=5)
                word_text_color = "#B45309"
                label_str = f"{word.upper()} ({samples_count}/30 muestras)"
                rec_btn = ft.Container(
                    content=ft.Row([
                        ft.Container(width=7, height=7, bgcolor=ft.Colors.WHITE, border_radius=4),
                        ft.Text("REC", size=10, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)
                    ], spacing=4, alignment=ft.MainAxisAlignment.CENTER),
                    bgcolor=COLOR_REC_BTN,
                    border_radius=12,
                    padding=ft.Padding(8, 3, 8, 3),
                    on_click=lambda ev, w=word: self.start_preparation_flow(w),
                    tooltip="Iniciar grabación con cuenta regresiva"
                )
            else:
                # 0/30 Sin iniciar: círculo gris
                item_bg = "#FFFFFF"
                item_border = "#E2E8F0"
                icon_indicator = ft.Icon(ft.Icons.RADIO_BUTTON_UNCHECKED, color="#94A3B8", size=18)
                word_text_color = COLOR_TEXT_MUTED
                label_str = f"{word.upper()} (0/30)"
                rec_btn = ft.IconButton(
                    icon=ft.Icons.VIDEOCAM_OUTLINED,
                    icon_color=COLOR_TEXT_MUTED,
                    icon_size=18,
                    tooltip="Grabar muestra inicial",
                    on_click=lambda ev, w=word: self.start_preparation_flow(w)
                )

            word_row = ft.Container(
                content=ft.Row([
                    ft.Row([
                        icon_indicator,
                        ft.Text(label_str, size=12, weight=ft.FontWeight.BOLD, color=word_text_color)
                    ], spacing=8, expand=True),
                    ft.Row([
                        ft.IconButton(
                            icon=ft.Icons.DESCRIPTION_OUTLINED,
                            icon_color=COLOR_PRIMARY,
                            icon_size=17,
                            tooltip="Modificar Muestras (.csv individuales)",
                            on_click=lambda ev, w=word: self.open_samples_modal(w)
                        ),
                        rec_btn,
                        ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color=COLOR_REC_BTN,
                            icon_size=17,
                            tooltip="Eliminar palabra y sus muestras",
                            on_click=lambda ev, w=word: self.delete_word(w)
                        )
                    ], spacing=2)
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=ft.Padding(8, 5, 8, 5),
                border=ft.Border.all(1, item_border),
                border_radius=8,
                bgcolor=item_bg
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
            self.status_text.value = f"Palabra '{word.upper()}' eliminada del disco y modelo desactualizado retirado."
            self.status_text.color = COLOR_REC_BTN
            self.refresh_words_list()
            self.update_cnn_button_state()
            self.load_trained_models_to_test_dropdown()
            show_snack_bar(self.page, f"Palabra '{word.upper()}' eliminada. Modelos actualizados.")

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
                badge_fg = COLOR_SUCCESS if sample["is_valid"] else COLOR_AMBER
                
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
            self.btn_camera.icon = ft.Icons.VIDEOCAM_OFF_OUTLINED
            self.btn_camera.bgcolor = COLOR_REC_BTN
            self.btn_camera.color = ft.Colors.WHITE
            self.status_text.value = "Cámara activa (25 FPS). Listo para operar."
            self.status_text.color = COLOR_SUCCESS
        else:
            self.vision_service.stop()
            self.is_camera_active = False
            self.btn_camera.content = "Encender Cámara"
            self.btn_camera.icon = ft.Icons.VIDEOCAM_OUTLINED
            self.btn_camera.bgcolor = "#F1F5F9"
            self.btn_camera.color = COLOR_TEXT_TITLE
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
                        show_error_popup(self.page, "Error en Entrenamiento de Red CNN", f"Ocurrió un fallo durante el entrenamiento de '{self.selected_category}':\n\n{str(err)}")
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

    # --- MÓDULO DE PRUEBAS: CARGA SEGURA DESDE data/modelos/ ---

    def load_trained_models_to_test_dropdown(self):
        """
        Escanea la ruta absoluta 'data/modelos/' buscando categorías entrenadas válidas.
        Valida que existan tanto el modelo binario (.keras) como las etiquetas (labels.json),
        verificando únicamente tamaño físico > 0 y NUNCA abriendo el binario como texto.
        """
        categories = set()
        model_base_dir = MODELOS_DIR
        os.makedirs(model_base_dir, exist_ok=True)
        
        if os.path.exists(model_base_dir):
            for item in os.listdir(model_base_dir):
                dir_path = os.path.join(model_base_dir, item)
                if os.path.isdir(dir_path):
                    has_model = (os.path.exists(os.path.join(dir_path, "model.keras")) and os.path.getsize(os.path.join(dir_path, "model.keras")) > 0) or \
                                (os.path.exists(os.path.join(dir_path, "model.h5")) and os.path.getsize(os.path.join(dir_path, "model.h5")) > 0)
                    has_labels = os.path.exists(os.path.join(dir_path, "labels.json"))
                    if has_model and has_labels:
                        categories.add(item.lower().strip())

        cat_list = sorted(list(categories))
        self.test_category_dropdown.options = [
            ft.dropdown.Option(text=f"Categoría: {c.upper()}", key=c) for c in cat_list
        ]
        
        if cat_list and (not self.test_category_dropdown.value or self.test_category_dropdown.value not in cat_list):
            self.test_category_dropdown.value = cat_list[0]
            self.on_test_category_selected(None)
        elif not cat_list:
            self.test_category_dropdown.value = None

        update_ui_safely(self.test_category_dropdown)

    def on_test_category_selected(self, e):
        cat = self.test_category_dropdown.value
        if not cat:
            return

        category_dir = os.path.join(MODELOS_DIR, cat)
        candidate_paths = [
            os.path.join(category_dir, "model.keras"),
            os.path.join(category_dir, "model.h5"),
        ]
        valid_model = any(os.path.exists(cp) and os.path.getsize(cp) > 0 for cp in candidate_paths)

        # Cargar etiquetas en la lista lateral de señas de la Pestaña 2 (Stitch test.png)
        labels_path = os.path.join(category_dir, "labels.json")
        self.test_classes_listview.controls.clear()

        if valid_model and os.path.exists(labels_path):
            try:
                with open(labels_path, 'r', encoding='utf-8') as f:
                    label_map = json.load(f)
                
                classes = list(label_map.values()) if isinstance(label_map, dict) else list(label_map)
                self.lbl_test_classes_count.value = f"{len(classes)} Clases"

                for idx, cname in enumerate(classes):
                    num_badge = f"{idx + 1:02d}"
                    row_item = ft.Container(
                        content=ft.Row([
                            ft.Row([
                                ft.Container(
                                    content=ft.Text(num_badge, size=11, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY),
                                    bgcolor=COLOR_PRIMARY_LIGHT,
                                    border_radius=4,
                                    padding=ft.Padding(6, 3, 6, 3)
                                ),
                                ft.Column([
                                    ft.Text(cname.upper(), size=12, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                                    ft.Text("Gesto postulado", size=10, color=COLOR_TEXT_MUTED)
                                ], spacing=1)
                            ], spacing=8),
                            ft.Container(
                                content=ft.Text("✓ Listo", size=10, color=COLOR_TEXT_MUTED, weight=ft.FontWeight.W_500),
                                bgcolor="#F1F5F9",
                                border_radius=10,
                                padding=ft.Padding(6, 2, 6, 2)
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        bgcolor="#FFFFFF",
                        border=ft.Border.all(1, "#E2E8F0"),
                        border_radius=8,
                        padding=ft.Padding(8, 5, 8, 5)
                    )
                    self.test_classes_listview.controls.append(row_item)

                self.lbl_active_model_subbanner.value = f"• Modelo Activo: Categoría '{cat.upper()}' cargado con éxito. Procesando a 25.0 FPS local."
                self.lbl_prediction_badge.content = ft.Text(cat.upper(), size=11, color=COLOR_PRIMARY, weight=ft.FontWeight.BOLD)

            except Exception as ex:
                print(f"[TEST UI] Error leyendo clases: {ex}")

        update_ui_safely(self.lbl_test_classes_count)
        update_ui_safely(self.test_classes_listview)
        update_ui_safely(self.lbl_active_model_subbanner)
        update_ui_safely(self.lbl_prediction_badge)

    def toggle_live_test(self, e):
        if not self.is_testing:
            cat = self.test_category_dropdown.value
            if not cat:
                show_snack_bar(self.page, "Debe seleccionar un modelo entrenado para probar", is_error=True)
                return

            try:
                # Carga desacoplada desde ruta absoluta MODELOS_DIR
                if not hasattr(self, "live_tester") or self.live_tester is None:
                    self.live_tester = LSPTesterService(model_base_dir=MODELOS_DIR, page_ref=self.page)

                self.live_tester.load_trained_model(cat)
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
                self.btn_toggle_test.icon = ft.Icons.STOP_ROUNDED
                self.btn_toggle_test.bgcolor = COLOR_REC_BTN
                self.lbl_prediction.value = "ESPERANDO SEÑA..."
                self.lbl_confidence.value = "0.0%"
                self.progress_bar_prediction.value = 0.0
                self.lbl_detection_state.value = f"✓ Estado: Inferencia activa ({cat.upper()})"
                show_snack_bar(self.page, f"Prueba en vivo iniciada para '{cat.upper()}'. Realice señas frente a la cámara.")

            except Exception as ex:
                show_snack_bar(self.page, f"Error cargando modelo: {str(ex)}", is_error=True)
                show_error_popup(self.page, "Error al Cargar Modelo de Pruebas", f"No se pudo cargar el modelo para la categoría '{cat}':\n\n{str(ex)}")
                print(f"[ERROR CARGA MODELO] {str(ex)}")

        else:
            if self.live_tester:
                self.live_tester.stop()
            self.vision_service.live_tester = None
            self.vision_service.prediction_label_control = None

            if hasattr(self, "tabs") and self.tabs and getattr(self.tabs, "selected_index", 0) == 0:
                self.voice_service.allow_voice_trigger = True

            self.is_testing = False
            self.btn_toggle_test.content = "Iniciar Prueba"
            self.btn_toggle_test.icon = ft.Icons.PLAY_ARROW_ROUNDED
            self.btn_toggle_test.bgcolor = COLOR_PRIMARY
            self.lbl_prediction.value = "PRUEBA DETENIDA"
            self.lbl_confidence.value = "0.0%"
            self.progress_bar_prediction.value = 0.0
            self.lbl_detection_state.value = "✓ Estado: Prueba detenida"
            show_snack_bar(self.page, "Prueba en vivo detenida.")

        update_ui_safely(self.btn_toggle_test)
        update_ui_safely(self.lbl_prediction)
        update_ui_safely(self.lbl_confidence)
        update_ui_safely(self.progress_bar_prediction)
        update_ui_safely(self.lbl_detection_state)

    def trigger_tts_pronounce(self, e):
        """Pronuncia la palabra actual en voz alta con pyttsx3."""
        word = self.lbl_prediction.value
        if word and word not in ["ESPERANDO SEÑA...", "PRUEBA DETENIDA", ""]:
            from src.tester_service import speak_word_offline
            speak_word_offline(word)
            show_snack_bar(self.page, f"Pronunciando: {word}")

    # --- MÓDULO DE SINCRONIZACIÓN CLOUD (AWS S3 & TENSORFLOW.JS) ---

    # --- MÓDULO DE SINCRONIZACIÓN CLOUD Y GESTIÓN MULTIMEDIA (AWS S3) ---

    def on_cloud_category_changed(self, category_name: str):
        """Manejador de selección en el dropdown de categorías de la pestaña Nube."""
        if not category_name:
            return
        self.current_cloud_category = category_name.lower().strip()
        self.render_cloud_model_card(self.current_cloud_category)
        self.refresh_cloud_resources_table(self.current_cloud_category)

    def render_cloud_model_card(self, category_name: str):
        """Renderiza la tarjeta del modelo predictivo TensorFlow.js de la categoría activa."""
        cat = category_name.lower().strip()
        st = self.cloud_statuses.get(cat, self.cloud_service.check_cloud_status(cat))
        has_local = self.cloud_service.verify_local_model(cat)

        # 1. Badge Local (.keras)
        if has_local:
            local_badge = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.CHECK_CIRCLE, size=14, color=COLOR_SUCCESS),
                    ft.Text("Entrenado (.keras)", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.W_600)
                ], spacing=4),
                bgcolor="#DCFCE7",
                border_radius=6,
                padding=ft.Padding(8, 4, 8, 4)
            )
        else:
            local_badge = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.CANCEL_OUTLINED, size=14, color=COLOR_REC_BTN),
                    ft.Text("Sin Modelo Local", size=11, color=COLOR_REC_BTN, weight=ft.FontWeight.W_600)
                ], spacing=4),
                bgcolor="#FEE2E2",
                border_radius=6,
                padding=ft.Padding(8, 4, 8, 4)
            )

        # 2. Badge AWS S3 (TF.js)
        if st == "PUBLICADO":
            cloud_badge = ft.Container(
                content=ft.Row([
                    ft.Text("🚀", size=12),
                    ft.Text("Sincronizado en S3", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.BOLD)
                ], spacing=4),
                bgcolor="#E8F5E9",
                border_radius=6,
                padding=ft.Padding(8, 4, 8, 4)
            )
            sync_btn_text = "Actualizar en S3"
            sync_btn_bg = "#F1F5F9"
            sync_btn_fg = COLOR_PRIMARY
            delete_disabled = False
        elif st == "DESACTUALIZADO":
            cloud_badge = ft.Container(
                content=ft.Row([
                    ft.Text("⚠️", size=12),
                    ft.Text("Desactualizado en S3", size=11, color="#E65100", weight=ft.FontWeight.BOLD)
                ], spacing=4),
                bgcolor="#FFF3E0",
                border_radius=6,
                padding=ft.Padding(8, 4, 8, 4)
            )
            sync_btn_text = "Re-sincronizar"
            sync_btn_bg = COLOR_PRIMARY
            sync_btn_fg = ft.Colors.WHITE
            delete_disabled = False
        elif st == "SIN_MODELO_LOCAL":
            cloud_badge = ft.Container(
                content=ft.Row([
                    ft.Text("☁️", size=12),
                    ft.Text("Solo en S3", size=11, color=COLOR_TEXT_MUTED, weight=ft.FontWeight.W_600)
                ], spacing=4),
                bgcolor="#ECEFF1",
                border_radius=6,
                padding=ft.Padding(8, 4, 8, 4)
            )
            sync_btn_text = "Subir a S3"
            sync_btn_bg = "#E2E8F0"
            sync_btn_fg = COLOR_TEXT_MUTED
            delete_disabled = False
        elif st == "ERROR_CONEXION":
            cloud_badge = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.WIFI_OFF, size=14, color=COLOR_REC_BTN),
                    ft.Text("Error Conexión", size=11, color=COLOR_REC_BTN, weight=ft.FontWeight.BOLD)
                ], spacing=4),
                bgcolor="#FEE2E2",
                border_radius=6,
                padding=ft.Padding(8, 4, 8, 4)
            )
            sync_btn_text = "Reintentar"
            sync_btn_bg = COLOR_PRIMARY
            sync_btn_fg = ft.Colors.WHITE
            delete_disabled = True
        else:  # NO_SUBIDO
            cloud_badge = ft.Container(
                content=ft.Row([
                    ft.Text("☁️", size=12),
                    ft.Text("No Publicado", size=11, color="#455A64", weight=ft.FontWeight.BOLD)
                ], spacing=4),
                bgcolor="#ECEFF1",
                border_radius=6,
                padding=ft.Padding(8, 4, 8, 4)
            )
            sync_btn_text = "Compilar y Subir a S3"
            sync_btn_bg = COLOR_PRIMARY
            sync_btn_fg = ft.Colors.WHITE
            delete_disabled = True

        btn_upload = ft.Button(
            content=sync_btn_text,
            icon=ft.Icons.CLOUD_UPLOAD_OUTLINED,
            bgcolor=sync_btn_bg,
            color=sync_btn_fg,
            disabled=not has_local,
            on_click=lambda ev, c=cat: self.sync_category_to_cloud(c),
            height=36
        )

        btn_del_cloud = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            icon_color=COLOR_REC_BTN,
            icon_size=18,
            disabled=delete_disabled,
            tooltip="Eliminar modelo TensorFlow.js de AWS S3",
            on_click=lambda ev, c=cat: self.delete_category_from_cloud(c)
        )

        card_content = ft.Row([
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.AUTO_AWESOME, color=COLOR_PRIMARY, size=20),
                    ft.Column([
                        ft.Text(f"CATEGORÍA: {cat.upper()}", size=13, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                        ft.Text(f"modelos/{cat}/tfjs_model/ (model.json + shard*.bin)", size=10, color=COLOR_TEXT_MUTED)
                    ], spacing=1)
                ], spacing=8),
                width=260
            ),
            ft.Container(content=local_badge, width=170),
            ft.Container(content=cloud_badge, width=190),
            ft.Row([btn_upload, btn_del_cloud], spacing=6, expand=True)
        ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER)

        self.cloud_model_card_container.content = card_content
        update_ui_safely(self.cloud_model_card_container)

    def refresh_cloud_resources_table(self, category_name: str):
        """
        Consulta de forma asíncrona todos los recursos didácticos de la categoría en S3 y local,
        poblando interactivamente la lista de señas.
        """
        def _fetch_worker():
            try:
                cat_clean = category_name.lower().strip()
                words = list(self.data_manager.get_words_in_category(cat_clean))
                cloud_resources = self.cloud_service.list_cloud_resources_for_category(cat_clean)
                all_words = sorted(list(set(words).union(set(cloud_resources.keys()))))

                rows = []
                if not all_words:
                    rows.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Icon(ft.Icons.VIDEO_LIBRARY_OUTLINED, size=32, color=COLOR_TEXT_MUTED),
                                ft.Text(f"No hay señas registradas en la categoría '{cat_clean.upper()}'.", color=COLOR_TEXT_MUTED, size=12)
                            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                            height=140,
                            alignment=ft.Alignment.CENTER
                        )
                    )
                else:
                    for word in all_words:
                        # Muestras locales
                        word_dir = self.data_manager._get_word_dir(cat_clean, word)
                        samples_count = 0
                        if os.path.exists(word_dir):
                            samples_count = len([f for f in os.listdir(word_dir) if f.endswith('.csv')])

                        # Estado del recurso multimedia didáctico
                        res_status = self.cloud_service.check_resource_status(cat_clean, word, cloud_resources)
                        st = res_status["status"]
                        local_path = res_status["local_path"]
                        has_local = (local_path is not None)

                        # 1. Miniatura / Estado Local
                        if has_local:
                            filename = os.path.basename(local_path)
                            badge_local = ft.Container(
                                content=ft.Row([
                                    ft.Icon(ft.Icons.IMAGE, size=15, color=COLOR_SUCCESS),
                                    ft.Text(filename, size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.W_600)
                                ], spacing=4),
                                bgcolor="#DCFCE7",
                                border_radius=6,
                                padding=ft.Padding(8, 4, 8, 4),
                                tooltip=f"Archivo local: {local_path}"
                            )
                        else:
                            badge_local = ft.Container(
                                content=ft.Row([
                                    ft.Icon(ft.Icons.IMAGE_NOT_SUPPORTED_OUTLINED, size=15, color=COLOR_TEXT_MUTED),
                                    ft.Text("Sin guía local", size=11, color=COLOR_TEXT_MUTED)
                                ], spacing=4),
                                bgcolor="#F1F5F9",
                                border_radius=6,
                                padding=ft.Padding(8, 4, 8, 4)
                            )

                        # 2. Estado en S3
                        if st == "SINCRONIZADO":
                            badge_s3 = ft.Container(
                                content=ft.Row([
                                    ft.Text("🚀", size=12),
                                    ft.Text("Sincronizado", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.BOLD)
                                ], spacing=4),
                                bgcolor="#E8F5E9",
                                border_radius=6,
                                padding=ft.Padding(8, 4, 8, 4)
                            )
                        elif st == "PENDIENTE":
                            badge_s3 = ft.Container(
                                content=ft.Row([
                                    ft.Text("⚠️", size=12),
                                    ft.Text("Pendiente", size=11, color="#E65100", weight=ft.FontWeight.BOLD)
                                ], spacing=4),
                                bgcolor="#FFF3E0",
                                border_radius=6,
                                padding=ft.Padding(8, 4, 8, 4),
                                tooltip="El archivo local fue modificado o aún no se ha subido a S3"
                            )
                        else:  # NO_EN_S3
                            badge_s3 = ft.Container(
                                content=ft.Row([
                                    ft.Text("☁️", size=12),
                                    ft.Text("No en S3", size=11, color="#455A64", weight=ft.FontWeight.BOLD)
                                ], spacing=4),
                                bgcolor="#ECEFF1",
                                border_radius=6,
                                padding=ft.Padding(8, 4, 8, 4)
                            )

                        # 3. Acciones Compactas
                        btn_pick = ft.Button(
                            content="Cargar Archivo",
                            icon=ft.Icons.ATTACH_FILE,
                            bgcolor="#F1F5F9",
                            color=COLOR_PRIMARY,
                            height=32,
                            on_click=lambda ev, c=cat_clean, w=word: self.open_file_picker_for_word(c, w)
                        )
                        btn_upload_res = ft.Button(
                            content="Subir a S3",
                            icon=ft.Icons.CLOUD_UPLOAD_OUTLINED,
                            bgcolor=COLOR_PRIMARY if has_local else "#E2E8F0",
                            color=ft.Colors.WHITE if has_local else COLOR_TEXT_MUTED,
                            disabled=not has_local,
                            height=32,
                            on_click=lambda ev, c=cat_clean, w=word: self.upload_word_resource(c, w)
                        )
                        has_cloud_or_local = (st != "NO_EN_S3" or has_local)
                        btn_del_res = ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color=COLOR_REC_BTN,
                            icon_size=18,
                            disabled=not has_cloud_or_local,
                            tooltip="Eliminar recurso didáctico de S3 y local",
                            on_click=lambda ev, c=cat_clean, w=word: self.delete_word_resource(c, w)
                        )

                        row_container = ft.Container(
                            content=ft.Row([
                                ft.Container(
                                    content=ft.Row([
                                        ft.Icon(ft.Icons.TRANSLATE, color=COLOR_PRIMARY, size=16),
                                        ft.Column([
                                            ft.Text(word.upper(), size=12, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                                            ft.Text(f"{samples_count} muestras", size=10, color=COLOR_TEXT_MUTED)
                                        ], spacing=1)
                                    ], spacing=8),
                                    width=180
                                ),
                                ft.Container(content=badge_local, width=180),
                                ft.Container(content=badge_s3, width=180),
                                ft.Row([btn_pick, btn_upload_res, btn_del_res], spacing=6, expand=True)
                            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            bgcolor="#FFFFFF",
                            border=ft.Border.all(1, COLOR_BORDER),
                            border_radius=8,
                            padding=ft.Padding(12, 6, 12, 6)
                        )
                        rows.append(row_container)

                def _update_ui():
                    self.cloud_resources_listview.controls = rows
                    update_ui_safely(self.cloud_resources_listview)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_update_ui)
                else:
                    _update_ui()
            except Exception as ex:
                print(f"[CLOUD] Error cargando recursos didácticos: {ex}")

        threading.Thread(target=_fetch_worker, daemon=True).start()

    def open_file_picker_for_word(self, category: str, word: str):
        """Abre el explorador de archivos para asociar una guía visual didáctica a la seña."""
        self._picking_target_category = category
        self._picking_target_word = word
        try:
            self.file_picker.pick_files(
                dialog_title=f"Seleccionar Guía Visual para '{word.upper()}' (GIF, PNG, JPG, MP4)",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["gif", "png", "jpg", "jpeg", "mp4"],
                allow_multiple=False
            )
        except Exception as ex:
            show_snack_bar(self.page, f"Error abriendo explorador de archivos: {ex}", is_error=True)

    def on_file_picker_result(self, e):
        """Manejador asíncrono tras seleccionar un archivo multimedia didáctico."""
        if not e.files or len(e.files) == 0:
            return
        cat = getattr(self, "_picking_target_category", None)
        word = getattr(self, "_picking_target_word", None)
        if not cat or not word:
            return
        picked_file = e.files[0]
        picked_path = picked_file.path
        if not picked_path:
            return

        def _copy_worker():
            try:
                saved_path = self.cloud_service.save_local_resource(cat, word, picked_path)
                filename = os.path.basename(saved_path)
                show_snack_bar(self.page, f"Guía didáctica '{filename}' guardada para '{word.upper()}'.")
                self.refresh_cloud_resources_table(cat)
            except Exception as ex:
                show_snack_bar(self.page, f"Error al asociar archivo: {str(ex)}", is_error=True)

        threading.Thread(target=_copy_worker, daemon=True).start()

    def upload_word_resource(self, category: str, word: str):
        """Sube la guía multimedia didáctica de la seña a AWS S3 (recursos/{categoria}/{palabra}.[ext])."""
        def _upload_worker():
            try:
                def _ui_start():
                    self.cloud_progress_ring.visible = True
                    self.cloud_progress_bar.visible = True
                    self.cloud_progress_bar.value = None
                    self.lbl_cloud_status.value = f"Subiendo guía didáctica de '{word.upper()}' a AWS S3..."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                self.cloud_service.upload_resource(category, word)

                def _ui_done():
                    self.lbl_cloud_status.value = f"¡Guía didáctica de '{word.upper()}' sincronizada con éxito en AWS S3!"
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.value = 1.0
                    show_snack_bar(self.page, f"Guía didáctica de '{word.upper()}' subida a S3.")
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)
                    self.refresh_cloud_resources_table(category)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_done)
                else:
                    _ui_done()
            except Exception as ex:
                def _ui_err():
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.visible = False
                    self.lbl_cloud_status.value = f"Error al subir guía de '{word.upper()}': {str(ex)}"
                    show_snack_bar(self.page, f"Error al subir a S3: {str(ex)}", is_error=True)
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_err)
                else:
                    _ui_err()

        threading.Thread(target=_upload_worker, daemon=True).start()

    def delete_word_resource(self, category: str, word: str):
        """Elimina la guía multimedia didáctica de AWS S3 y del directorio local."""
        def _delete_worker():
            try:
                def _ui_start():
                    self.cloud_progress_ring.visible = True
                    self.cloud_progress_bar.visible = True
                    self.cloud_progress_bar.value = None
                    self.lbl_cloud_status.value = f"Eliminando guía didáctica de '{word.upper()}' de S3 y local..."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                self.cloud_service.delete_resource(category, word, delete_local=True)

                def _ui_done():
                    self.lbl_cloud_status.value = f"Guía didáctica de '{word.upper()}' eliminada de S3 y local."
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.value = 0.0
                    show_snack_bar(self.page, f"Guía de '{word.upper()}' eliminada.")
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)
                    self.refresh_cloud_resources_table(category)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_done)
                else:
                    _ui_done()
            except Exception as ex:
                def _ui_err():
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.visible = False
                    self.lbl_cloud_status.value = f"Error al eliminar guía: {str(ex)}"
                    show_snack_bar(self.page, f"Error al eliminar de S3: {str(ex)}", is_error=True)
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_err)
                else:
                    _ui_err()

        threading.Thread(target=_delete_worker, daemon=True).start()

    def refresh_cloud_table(self, e=None):
        """Escanea las categorías locales y de AWS S3, actualizando selector, modelo TF.js y recursos didácticos."""
        def _async_refresh():
            try:
                def _ui_start():
                    self.cloud_progress_ring.visible = True
                    self.lbl_cloud_status.value = "Consultando estados de sincronización en AWS S3..."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                local_cats = set(self.data_manager.get_categories())
                if os.path.exists(MODELOS_DIR):
                    for item in os.listdir(MODELOS_DIR):
                        if os.path.isdir(os.path.join(MODELOS_DIR, item)):
                            local_cats.add(item.lower().strip())

                cloud_cats = set(self.cloud_service.list_all_cloud_categories())
                all_cats = sorted(list(local_cats.union(cloud_cats)))

                statuses = {}
                for cat in all_cats:
                    statuses[cat] = self.cloud_service.check_cloud_status(cat)

                self.cloud_statuses = statuses

                current = getattr(self, "current_cloud_category", None)
                if not current or current not in all_cats:
                    if self.selected_category and self.selected_category in all_cats:
                        current = self.selected_category
                    elif all_cats:
                        current = all_cats[0]
                    else:
                        current = None
                self.current_cloud_category = current

                def _ui_populate():
                    self.cloud_category_dropdown.options = [
                        ft.dropdown.Option(text=c.upper(), key=c) for c in all_cats
                    ]
                    self.cloud_category_dropdown.value = current
                    update_ui_safely(self.cloud_category_dropdown)

                    if current:
                        self.render_cloud_model_card(current)
                        self.refresh_cloud_resources_table(current)
                    else:
                        self.cloud_model_card_container.content = ft.Container(
                            content=ft.Text("No hay categorías registradas en local ni en S3.", size=12, color=COLOR_TEXT_MUTED),
                            padding=10
                        )
                        self.cloud_resources_listview.controls.clear()
                        update_ui_safely(self.cloud_model_card_container)
                        update_ui_safely(self.cloud_resources_listview)

                    self.cloud_progress_ring.visible = False
                    self.lbl_cloud_status.value = f"Estados actualizados ({len(all_cats)} categorías inspeccionadas en AWS S3)."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_populate)
                else:
                    _ui_populate()

            except Exception as e:
                def _ui_fail():
                    self.cloud_progress_ring.visible = False
                    self.lbl_cloud_status.value = f"Fallo al consultar AWS S3: {str(e)}"
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.lbl_cloud_status)
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_fail)
                else:
                    _ui_fail()

        threading.Thread(target=_async_refresh, daemon=True).start()

    def sync_category_to_cloud(self, category_name: str):
        """Convierte el modelo a TF.js y lo sube con labels.json a AWS S3."""
        def _async_upload():
            try:
                def _ui_start():
                    self.cloud_progress_ring.visible = True
                    self.cloud_progress_bar.visible = True
                    self.cloud_progress_bar.value = 0.1
                    self.lbl_cloud_status.value = f"Iniciando compilación y sincronización de '{category_name.upper()}' con AWS S3..."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                def _prog(pct, msg):
                    def _ui_prog():
                        self.cloud_progress_bar.value = pct
                        self.lbl_cloud_status.value = msg
                        update_ui_safely(self.cloud_progress_bar)
                        update_ui_safely(self.lbl_cloud_status)
                    if self.page and hasattr(self.page, "run_thread"):
                        self.page.run_thread(_ui_prog)

                self.cloud_service.upload_category_model(category_name, progress_callback=_prog)

                def _ui_success():
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.value = 1.0
                    self.lbl_cloud_status.value = f"¡Modelo '{category_name.upper()}' desplegado exitosamente en AWS S3!"
                    show_snack_bar(self.page, f"¡'{category_name.upper()}' sincronizado con AWS S3 en formato TF.js!")
                    self.refresh_cloud_table()

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_success)

            except Exception as ex:
                def _ui_err():
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.visible = False
                    self.lbl_cloud_status.value = f"Error en sincronización Cloud: {str(ex)}"
                    show_snack_bar(self.page, f"Error AWS: {str(ex)}", is_error=True)
                    show_error_popup(
                        self.page,
                        "Fallo de Sincronización Cloud (AWS S3)",
                        f"No se pudo completar la transferencia a AWS S3 para '{category_name}':\n\n{str(ex)}"
                    )
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_err)

        threading.Thread(target=_async_upload, daemon=True).start()

    def delete_category_from_cloud(self, category_name: str):
        """Elimina todos los archivos del modelo en AWS S3."""
        def _async_delete():
            try:
                def _ui_start():
                    self.cloud_progress_ring.visible = True
                    self.cloud_progress_bar.visible = True
                    self.cloud_progress_bar.value = 0.3
                    self.lbl_cloud_status.value = f"Eliminando archivos de '{category_name.upper()}' de AWS S3..."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                def _prog(pct, msg):
                    def _ui_prog():
                        self.cloud_progress_bar.value = pct
                        self.lbl_cloud_status.value = msg
                        update_ui_safely(self.cloud_progress_bar)
                        update_ui_safely(self.lbl_cloud_status)
                    if self.page and hasattr(self.page, "run_thread"):
                        self.page.run_thread(_ui_prog)

                self.cloud_service.delete_category_model(category_name, progress_callback=_prog)

                def _ui_success():
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.visible = False
                    self.lbl_cloud_status.value = f"Modelo '{category_name.upper()}' eliminado de AWS S3."
                    show_snack_bar(self.page, f"Modelo '{category_name.upper()}' eliminado de S3.")
                    self.refresh_cloud_table()

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_success)

            except Exception as ex:
                def _ui_err():
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.visible = False
                    self.lbl_cloud_status.value = f"Error al eliminar en AWS: {str(ex)}"
                    show_snack_bar(self.page, f"Error AWS: {str(ex)}", is_error=True)
                    show_error_popup(
                        self.page,
                        "Fallo al Eliminar en AWS S3",
                        f"No se pudo eliminar el modelo en AWS S3 para '{category_name}':\n\n{str(ex)}"
                    )
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_err)

        threading.Thread(target=_async_delete, daemon=True).start()

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

# =========================================================================
# CONSTRUCTORES DE VISTAS ESCOLARES SEGÚN GOOGLE STITCH
# =========================================================================

def build_training_view(controller: LSPUIController) -> ft.Container:
    """
    Construye la vista de Captura y Entrenamiento (Stitch principal.png):
    - Columna izquierda: 3 tarjetas blancas con bordes celestes (Categorías, Vocabulario scrollable 230px, Sliders).
    - Columna derecha: Monitor de video 640x440 con overlays HUD MediaPipe + botones Encender Cámara y Entrenar.
    """
    # 1. Card 1: Categorías de Estudio
    card_categorias = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.FOLDER_OPEN_OUTLINED, color=COLOR_PRIMARY, size=18),
                    ft.Text("1. Categorías de Estudio", weight=ft.FontWeight.BOLD, size=13, color=COLOR_TEXT_TITLE)
                ], spacing=6),
                ft.Container(
                    content=ft.Text("GESTIÓN ACTIVA", size=10, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY),
                    bgcolor=COLOR_PRIMARY_LIGHT,
                    border_radius=4,
                    padding=ft.Padding(6, 2, 6, 2)
                )
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([
                controller.new_category_input,
                ft.IconButton(
                    icon=ft.Icons.ADD,
                    icon_color=ft.Colors.WHITE,
                    bgcolor=COLOR_PRIMARY,
                    icon_size=18,
                    on_click=controller.add_new_category,
                    tooltip="Crear Categoría",
                    width=38,
                    height=38
                )
            ], spacing=6),
            ft.Row([
                controller.category_dropdown,
                controller.btn_edit_category,
                controller.btn_delete_category,
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=2)
        ], spacing=8),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12
    )

    # 2. Card 2: Vocabulario Registrado con scroll restringido de exactamente 230px
    words_scrollable_container = ft.Container(
        content=controller.words_listview,
        height=230,  # Especificación de 230px
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=8,
        bgcolor="#FFFFFF",
        padding=6
    )

    card_vocabulario = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.FORMAT_LIST_BULLETED_ROUNDED, color=COLOR_PRIMARY, size=18),
                    ft.Text("2. Vocabulario Registrado", weight=ft.FontWeight.BOLD, size=13, color=COLOR_TEXT_TITLE)
                ], spacing=6),
                controller.lbl_queue_count
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([
                controller.new_word_input,
                ft.Button(
                    content="+ Agregar",
                    bgcolor=COLOR_PRIMARY,
                    color=ft.Colors.WHITE,
                    on_click=controller.add_new_word,
                    height=38
                )
            ], spacing=6),
            words_scrollable_container
        ], spacing=8),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12
    )

    # 3. Card 3: Configuración de Captura (Deslizadores al fondo)
    card_sliders = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.TUNE, size=16, color=COLOR_PRIMARY),
                    ft.Text("Configuración de Captura", weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE, size=13),
                ], spacing=6),
                ft.Icon(ft.Icons.SETTINGS_OUTLINED, size=16, color=COLOR_TEXT_MUTED)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([
                ft.Column([
                    ft.Row([
                        ft.Text("Espera:", size=11, color=COLOR_TEXT_BODY),
                        controller.lbl_delay_val,
                        ft.Text("(1.0 - 5.0)", size=10, color=COLOR_TEXT_MUTED)
                    ], spacing=4),
                    controller.slider_delay
                ], expand=True, spacing=2),
                ft.Column([
                    ft.Row([
                        ft.Text("Muestras:", size=11, color=COLOR_TEXT_BODY),
                        controller.lbl_frames_val,
                        ft.Text("(20 - 60)", size=10, color=COLOR_TEXT_MUTED)
                    ], spacing=4),
                    controller.slider_frames
                ], expand=True, spacing=2)
            ], spacing=10)
        ], spacing=4),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12
    )

    # Columna Izquierda Completa
    columna_izquierda = ft.Column([
        card_categorias,
        card_vocabulario,
        card_sliders
    ], spacing=10, width=410)

    # Panel Derecho: Monitor de Video Controlado con Overlays HUD (Stitch principal.png)
    monitor_camara = ft.Container(
        content=ft.Column([
            # Barra superior interna del monitor
            ft.Row([
                ft.Row([
                    ft.Container(width=7, height=7, bgcolor="#10B981", border_radius=4),
                    ft.Text("VISTA EN VIVO - MEDIAPIPE SKELETAL TRACKER", size=10, weight=ft.FontWeight.BOLD, color="#94A3B8")
                ], spacing=6),
                ft.Row([
                    ft.Container(
                        content=ft.Text("LSP MODEL v2.4", size=9, weight=ft.FontWeight.BOLD, color="#94A3B8"),
                        bgcolor="#1E293B",
                        border_radius=4,
                        padding=ft.Padding(5, 2, 5, 2)
                    ),
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.LOCK, size=10, color="#10B981"),
                            ft.Text("CALIBRADO", size=9, weight=ft.FontWeight.BOLD, color="#10B981")
                        ], spacing=3),
                        bgcolor="#064E3B",
                        border_radius=4,
                        padding=ft.Padding(5, 2, 5, 2)
                    )
                ], spacing=6)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            # Imagen de la Cámara
            ft.Container(
                content=ft.Column([
                    controller.warning_banner,
                    controller.camera_view
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
                alignment=ft.Alignment.CENTER
            ),

            # Barra inferior interna de HUD
            ft.Row([
                # HUD Izquierdo: Métricas y Estado
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            controller.lbl_hud_fps,
                            controller.lbl_hud_conf,
                            controller.lbl_hud_hand
                        ], spacing=8),
                        controller.lbl_hud_status
                    ], spacing=2),
                    bgcolor="#1E293B",
                    border_radius=6,
                    padding=ft.Padding(8, 4, 8, 4)
                ),
                # HUD Derecho: Patrón Objetivo
                ft.Container(
                    content=ft.Row([
                        controller.lbl_hud_target,
                        ft.Icon(ft.Icons.PAN_TOOL_ALT, size=14, color="#38BDF8")
                    ], spacing=4),
                    bgcolor="#1E293B",
                    border_radius=6,
                    padding=ft.Padding(8, 6, 8, 6)
                )
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        ], spacing=6),
        bgcolor=COLOR_DARK_MONITOR,
        border=ft.Border.all(2, COLOR_BORDER),
        border_radius=12,
        padding=10,
        width=640
    )

    columna_derecha = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Text("Monitor de Captura y Tracking 3D", size=14, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                ft.Row([controller.switch_voice, controller.voice_badge], spacing=4)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            monitor_camara,
            ft.Row([
                controller.btn_camera,
                controller.switch_avatar,
                controller.btn_generate_cnn
            ], alignment=ft.MainAxisAlignment.CENTER, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=14)
        ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=14,
        expand=True
    )

    return ft.Container(
        content=ft.Row([columna_izquierda, columna_derecha], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START, spacing=14),
        padding=0
    )

def build_live_testing_view(controller: LSPUIController) -> ft.Container:
    """
    Construye la vista de Pruebas y Validación en Vivo (Stitch test.png):
    - Columna izquierda: Selección de modelo + Lista de señas registradas + Info técnica.
    - Columna derecha: Monitor 640x310 con HUD de prueba + Tarjeta gigante con tipografía 45px y barra de confianza.
    """
    # 1. Card 1: Seleccionar Categoría
    card_select = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Container(
                        content=ft.Text("1", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        bgcolor=COLOR_PRIMARY,
                        border_radius=10,
                        width=20,
                        height=20,
                        alignment=ft.Alignment.CENTER
                    ),
                    ft.Text("Seleccionar Categoría", weight=ft.FontWeight.BOLD, size=13, color=COLOR_TEXT_TITLE)
                ], spacing=6),
                ft.Container(
                    content=ft.Text("v2.4 LSP", size=9, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED),
                    bgcolor="#F1F5F9",
                    border_radius=4,
                    padding=ft.Padding(5, 2, 5, 2)
                )
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            controller.test_category_dropdown,
            controller.btn_toggle_test
        ], spacing=8),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12
    )

    # 2. Card 2: Señas Registradas (con scroll)
    card_classes = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Container(
                        content=ft.Text("2", size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        bgcolor=COLOR_PRIMARY,
                        border_radius=10,
                        width=20,
                        height=20,
                        alignment=ft.Alignment.CENTER
                    ),
                    ft.Text("Señas Registradas", weight=ft.FontWeight.BOLD, size=13, color=COLOR_TEXT_TITLE)
                ], spacing=6),
                controller.lbl_test_classes_count
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Container(
                content=controller.test_classes_listview,
                height=230,
                border=ft.Border.all(1, COLOR_BORDER),
                border_radius=8,
                padding=6
            )
        ], spacing=8),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12
    )

    # 3. Card 3: Información Técnica del Modelo
    card_tech_info = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Text("Red Neuronal:", size=11, color=COLOR_TEXT_MUTED),
                ft.Text("CNN 1D + MediaPipe 3D", size=11, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([
                ft.Text("Latencia de Red:", size=11, color=COLOR_TEXT_MUTED),
                ft.Text("~40 ms", size=11, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Row([
                ft.Text("Voz Sintetizada:", size=11, color=COLOR_TEXT_MUTED),
                ft.Text("ES-PE (pyttsx3 Offline)", size=11, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        ], spacing=4),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=10
    )

    columna_izquierda_test = ft.Column([
        card_select,
        card_classes,
        card_tech_info
    ], spacing=10, width=340)

    # Panel Derecho: Monitor de Inferencia + Tarjeta Gigante
    monitor_test = ft.Container(
        content=ft.Column([
            # Header del monitor
            ft.Row([
                ft.Row([
                    ft.Container(width=7, height=7, bgcolor=COLOR_REC_BTN, border_radius=4),
                    ft.Text("CÁMARA DOCENTE EN VIVO", size=10, weight=ft.FontWeight.BOLD, color="#F8FAFC"),
                    ft.Container(
                        content=ft.Text("21 Landmarks 3D", size=9, color="#94A3B8"),
                        bgcolor="#1E293B",
                        border_radius=4,
                        padding=ft.Padding(5, 2, 5, 2)
                    )
                ], spacing=6),
                ft.Text("FPS: 25.0   Resolución: 640x480", size=10, color="#94A3B8")
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            # Imagen de video
            ft.Container(
                content=controller.test_camera_view,
                alignment=ft.Alignment.CENTER
            ),

            # Barra inferior del monitor
            ft.Row([
                ft.Row([
                    ft.Container(
                        content=ft.Text("Pausar Captura", size=10, color="#E2E8F0"),
                        bgcolor="#1E293B",
                        border_radius=4,
                        padding=ft.Padding(6, 3, 6, 3)
                    ),
                    ft.Container(
                        content=ft.Text("Ocultar Malla", size=10, color="#E2E8F0"),
                        bgcolor="#1E293B",
                        border_radius=4,
                        padding=ft.Padding(6, 3, 6, 3)
                    )
                ], spacing=6),
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.PAN_TOOL, size=12, color=COLOR_PRIMARY),
                        ft.Text("Mano en Cuadro: Derecha (Principal)", size=10, weight=ft.FontWeight.BOLD, color="#93C5FD")
                    ], spacing=4),
                    bgcolor="#1E293B",
                    border_radius=4,
                    padding=ft.Padding(6, 3, 6, 3)
                )
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        ], spacing=6),
        bgcolor=COLOR_DARK_MONITOR,
        border=ft.Border.all(2, COLOR_BORDER),
        border_radius=12,
        padding=10,
        width=640
    )

    # Tarjeta de Traducción Gigante (Tipografía 45px - Stitch test.png)
    tarjeta_gigante = ft.Container(
        content=ft.Column([
            # Fila de cabecera
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.TRANSLATE, color=COLOR_PRIMARY, size=18),
                    ft.Text("TRADUCCIÓN LSP EN TIEMPO REAL", size=12, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
                ], spacing=6),
                ft.Row([
                    ft.Button(
                        content="🔊 Pronunciar",
                        bgcolor=COLOR_PRIMARY_LIGHT,
                        color=COLOR_PRIMARY,
                        height=30,
                        on_click=controller.trigger_tts_pronounce
                    ),
                    ft.Container(
                        content=ft.Row([
                            ft.Container(width=6, height=6, bgcolor="#10B981", border_radius=3),
                            ft.Text("Detección Continua", size=10, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY)
                        ], spacing=4),
                        bgcolor=COLOR_PRIMARY_LIGHT,
                        border_radius=10,
                        padding=ft.Padding(8, 4, 8, 4)
                    )
                ], spacing=6)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

            # Palabra detectada en letras gigantes (size=45) y porcentaje
            ft.Row([
                ft.Row([
                    controller.lbl_prediction,
                    controller.lbl_prediction_badge
                ], spacing=8, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Column([
                    controller.lbl_confidence,
                    ft.Text("Nivel de Confianza", size=10, color=COLOR_TEXT_MUTED)
                ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.END)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),

            # Barra de progreso
            controller.progress_bar_prediction,

            # Pie de tarjeta
            ft.Row([
                controller.lbl_detection_state,
                controller.test_status_text
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        ], spacing=8),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(2, COLOR_BORDER),
        border_radius=12,
        padding=14,
        width=640
    )

    columna_derecha_test = ft.Container(
        content=ft.Column([
            monitor_test,
            tarjeta_gigante
        ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=14,
        expand=True
    )

    return ft.Container(
        content=ft.Row([columna_izquierda_test, columna_derecha_test], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START, spacing=14),
        padding=0
    )

def build_cloud_view(controller: LSPUIController) -> ft.Container:
    """
    Construye la vista de Sincronización Nube (AWS S3) y Gestión de Recursos Didácticos:
    1. Selector de Categoría Activa: Dropdown superior con botón Refrescar y Badge del Bucket.
    2. Panel de Modelo (Cabecera): Tarjeta compacta para compilar y sincronizar el modelo TF.js en S3.
    3. Sección: Gestión de Recursos Didácticos (Guías de Señas):
       Tabla interactiva de palabras con estado local (guia.gif / guia.png), badge de S3 y acciones.
    4. Consola de Transferencia Inferior: Barra de progreso en tiempo real y mensajes.
    """
    # 1. Encabezado de Nube con Selector de Categoría Activa
    card_header = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Container(
                    content=ft.Icon(ft.Icons.CLOUD_SYNC, color=COLOR_PRIMARY, size=24),
                    bgcolor=COLOR_PRIMARY_LIGHT,
                    border_radius=8,
                    padding=8
                ),
                ft.Column([
                    ft.Text("PANEL DE SINCRONIZACIÓN Y RECURSOS NUBE (AWS S3)", size=14, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                    ft.Text("Gestión unificada de modelos TensorFlow.js y guías visuales didácticas (GIF/Imagen)", size=11, color=COLOR_TEXT_MUTED)
                ], spacing=1)
            ], spacing=10),
            ft.Row([
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.STORAGE, size=14, color=COLOR_PRIMARY),
                        ft.Text(f"Bucket: {controller.cloud_service.bucket_name or 'No configurado'}", size=11, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
                    ], spacing=6),
                    bgcolor=COLOR_PRIMARY_LIGHT,
                    border_radius=8,
                    padding=ft.Padding(10, 6, 10, 6)
                ),
                controller.cloud_category_dropdown,
                ft.Button(
                    content="Refrescar Estados",
                    icon=ft.Icons.REFRESH,
                    bgcolor=COLOR_PRIMARY,
                    color=ft.Colors.WHITE,
                    on_click=controller.refresh_cloud_table,
                    height=38
                )
            ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12
    )

    # 2. Panel de Modelo (Cabecera)
    card_model = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.MEMORY, color=COLOR_PRIMARY, size=18),
                    ft.Text("1. Modelo Predictivo Web (TensorFlow.js)", size=13, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
                ], spacing=6),
                ft.Container(
                    content=ft.Text("MOTOR INFERENCIA WEB", size=9, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY),
                    bgcolor=COLOR_PRIMARY_LIGHT,
                    border_radius=4,
                    padding=ft.Padding(6, 2, 6, 2)
                )
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            controller.cloud_model_card_container
        ], spacing=8),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12
    )

    # 3. Sección: Gestión de Recursos Didácticos (Guías de Señas)
    table_header = ft.Container(
        content=ft.Row([
            ft.Container(ft.Text("PALABRA / SEÑA", size=11, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED), width=180),
            ft.Container(ft.Text("GUÍA LOCAL (DIDÁCTICA)", size=11, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED), width=180),
            ft.Container(ft.Text("ESTADO EN AWS S3", size=11, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED), width=180),
            ft.Container(ft.Text("ACCIONES DE GESTIÓN MULTIMEDIA", size=11, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED), expand=True)
        ], alignment=ft.MainAxisAlignment.START),
        padding=ft.Padding(12, 6, 12, 6),
        bgcolor="#F8FAFC",
        border_radius=8
    )

    card_resources = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.VIDEO_LIBRARY_OUTLINED, color=COLOR_PRIMARY, size=18),
                    ft.Text("2. Gestión de Recursos Didácticos (Guías de Señas)", size=13, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
                ], spacing=6),
                ft.Text("S3: recursos/{categoria}/{palabra}.[ext]", size=11, color=COLOR_TEXT_MUTED)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            table_header,
            ft.Container(
                content=controller.cloud_resources_listview,
                height=260,
                padding=2
            )
        ], spacing=8),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12,
        expand=True
    )

    # 4. Consola de Transferencia Inferior
    card_console = ft.Container(
        content=ft.Column([
            ft.Row([
                controller.cloud_progress_ring,
                controller.lbl_cloud_status
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            controller.cloud_progress_bar
        ], spacing=6),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=10,
        padding=10
    )

    return ft.Container(
        content=ft.Column([
            card_header,
            card_model,
            card_resources,
            card_console
        ], spacing=10, expand=True),
        padding=0
    )

def build_main_app_tabs(controller: LSPUIController) -> ft.Tabs:
    """
    Implementa el control de pestañas nativo ft.Tabs conectado a las tres vistas de la aplicación:
    1. Captura y Entrenamiento
    2. Prueba en Vivo del Traductor
    3. Nube / AWS
    """
    view_training = build_training_view(controller)
    view_testing = build_live_testing_view(controller)
    view_cloud = build_cloud_view(controller)

    tab1 = ft.Tab(label="Captura y Entrenamiento", icon=ft.Icons.SCHOOL)
    tab2 = ft.Tab(label="Prueba en Vivo del Traductor", icon=ft.Icons.FACT_CHECK)
    tab3 = ft.Tab(label="Nube / AWS", icon=ft.Icons.CLOUD_SYNC)

    tab_bar = ft.TabBar(
        tabs=[tab1, tab2, tab3],
        divider_color=COLOR_BORDER,
        indicator_color=COLOR_PRIMARY,
        label_color=COLOR_PRIMARY,
        unselected_label_color=COLOR_TEXT_MUTED
    )

    tab_view = ft.TabBarView(
        expand=True,
        controls=[
            view_training,
            view_testing,
            view_cloud
        ]
    )

    def on_tab_change(e):
        try:
            new_idx = int(e.data) if hasattr(e, "data") and e.data is not None else tabs.selected_index
        except Exception:
            new_idx = getattr(tabs, "selected_index", 0)

        controller.switch_tab(new_idx)

    tabs = ft.Tabs(
        length=3,
        height=680,
        content=ft.Column(expand=True, controls=[tab_bar, tab_view], spacing=6),
        on_change=on_tab_change
    )

    controller.tabs = tabs
    return tabs
