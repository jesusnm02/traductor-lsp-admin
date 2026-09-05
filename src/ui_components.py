import os
import sys
import glob
import json
import threading
import time
import base64
import cv2
import flet as ft
try:
    import speech_recognition as sr
except ImportError:
    sr = None
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

        # Asegurar overlay seguro para compatibilidad con controles de servicio (Flet 0.86+)
        if hasattr(self.page, "_overlay") and hasattr(self.page._overlay, "controls"):
            if not getattr(self.page, "_safe_overlay_installed", False):
                original_overlay = self.page._overlay.controls
                page_ref = self.page
                class SafeOverlayList(list):
                    def __init__(self, target):
                        super().__init__(target)
                        self._target = target
                        self._virtual_services = []

                    def append(self, item):
                        if isinstance(item, ft.FilePicker):
                            if item not in self._virtual_services:
                                self._virtual_services.append(item)
                            if hasattr(page_ref, "services") and item not in page_ref.services:
                                page_ref.services.append(item)
                            return
                        self._target.append(item)
                        super().append(item)

                    def __contains__(self, item):
                        return item in self._virtual_services or super().__contains__(item) or item in self._target

                self.page._overlay.controls = SafeOverlayList(original_overlay)
                self.page._safe_overlay_installed = True

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

    def create_camera_placeholder(self) -> ft.Container:
        """Contenedor negro minimalista para estado de cámara desactivada sin textos superpuestos."""
        return ft.Container(
            width=480,
            height=360,
            bgcolor=ft.Colors.BLACK,
            visible=False
        )

    def create_standard_camera_container(self, camera_img_or_placeholder, camera_img=None) -> ft.Container:
        """Crea el contenedor estandarizado 480x360 con borde celeste escolar #4A90E2 y fondo negro sólido."""
        img = camera_img if camera_img is not None else camera_img_or_placeholder
        return ft.Container(
            content=img,
            width=480,
            height=360,
            border_radius=12,
            border=ft.Border.all(3, "#4A90E2"),
            bgcolor=ft.Colors.BLACK,
            alignment=ft.Alignment.CENTER,
            clip_behavior=ft.ClipBehavior.HARD_EDGE
        )

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

        # 4. Visores de Cámara Estandarizados (480x360, borde #4A90E2, fondo negro minimalista)
        self.camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=480,
            height=360,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain",
            visible=True
        )
        self.test_camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=480,
            height=360,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain",
            visible=True
        )
        self.placeholder_train = self.create_camera_placeholder()
        self.placeholder_test = self.create_camera_placeholder()
        self.camera_container_train = self.create_standard_camera_container(self.placeholder_train, self.camera_view)
        self.camera_container_test = self.create_standard_camera_container(self.placeholder_test, self.test_camera_view)

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

        # 5b. Selector de Categoría para Tab 3 (Nube AWS)
        self.cloud_category_dropdown = ft.Dropdown(
            hint_text="Categoría activa en S3...",
            text_size=12,
            bgcolor="#FFFFFF",
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_PRIMARY,
            color=COLOR_TEXT_TITLE,
            height=38,
            width=220,
            border_radius=8,
            content_padding=ft.Padding(10, 0, 10, 0),
            on_select=lambda e: self.on_cloud_category_changed(e.control.value)
        )

        # 5c. Controles de Cámara de S3 y Comandos de Voz (Columna Derecha)
        self.cloud_active_category = None
        self.cloud_active_word = None
        self.is_cloud_recording = False
        self.cloud_recorded_frames = []
        self.voice_listener_running = False

        self.cloud_camera_image = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=480,
            height=360,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain",
            visible=True
        )
        self.placeholder_cloud = self.create_camera_placeholder()
        self.camera_container_cloud = self.create_standard_camera_container(self.placeholder_cloud, self.cloud_camera_image)
        self.switch_cloud_avatar = ft.Switch(
            label="Usar Avatar de Privacidad",
            value=True,
            active_color=COLOR_PRIMARY,
            label_text_style=ft.TextStyle(color=COLOR_TEXT_TITLE, weight=ft.FontWeight.W_600, size=11),
            on_change=self.toggle_cloud_privacy_avatar,
            tooltip="Activar para renderizar el títere de caricatura escolar, desactivar para usar cámara real"
        )
        self.btn_cloud_camera = ft.Button(
            content="Prender Cámara",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            bgcolor="#2E7D32",
            color=ft.Colors.WHITE,
            on_click=self.toggle_cloud_camera,
            height=36,
            width=180
        )
        self.btn_cloud_snapshot = ft.Button(
            content="Tomar Foto",
            icon=ft.Icons.CAMERA_ALT,
            bgcolor="#0284C7",
            color=ft.Colors.WHITE,
            on_click=self.cloud_take_photo_action,
            height=36,
            expand=True
        )
        self.btn_cloud_record = ft.Button(
            content="Grabar",
            icon=ft.Icons.RADIO_BUTTON_CHECKED,
            bgcolor="#EF4444",
            color=ft.Colors.WHITE,
            on_click=self.cloud_start_recording_action,
            height=36,
            expand=True
        )
        self.btn_cloud_stop = ft.Button(
            content="Detener",
            icon=ft.Icons.STOP,
            bgcolor="#64748B",
            color=ft.Colors.WHITE,
            on_click=self.cloud_stop_recording_action,
            height=36,
            disabled=True,
            expand=True
        )
        self.btn_cloud_preview = ft.Button(
            content="Vista Previa",
            icon=ft.Icons.VISIBILITY,
            bgcolor="#F1F5F9",
            color=COLOR_TEXT_TITLE,
            on_click=self.cloud_preview_action,
            height=36,
            expand=True
        )
        self.lbl_cloud_active_word = ft.Text(
            "Palabra: (Seleccione una de la tabla)",
            size=12,
            weight=ft.FontWeight.BOLD,
            color=COLOR_PRIMARY
        )
        self.lbl_cloud_cam_status = ft.Text(
            "Cámara lista. Seleccione una palabra y use los botones o comandos de voz.",
            size=11,
            color=COLOR_TEXT_MUTED
        )
        self.progress_cloud_compile = ft.ProgressBar(visible=False, color=COLOR_PRIMARY, height=4)
        self.lbl_voice_command_status = ft.Text(
            "🎙️ Micrófono Activo: Esperando comando ('captura', 'grabar', 'no grabes')...",
            size=11,
            color=COLOR_PRIMARY,
            weight=ft.FontWeight.W_500
        )

        # 5d. Selector unificado para Control de Voz (optimizacion_diseno_y_voz_v2.md)
        self.switch_comandos_voz = ft.Switch(
            label="Activar Comandos de Voz",
            value=False,
            active_color=COLOR_PRIMARY,
            label_text_style=ft.TextStyle(color=COLOR_TEXT_TITLE, size=12, weight=ft.FontWeight.W_600),
            on_change=self.toggle_comandos_voz
        )
        # Compatibilidad hacia atrás con selectores previos
        self.switch_voz_captura = self.switch_comandos_voz
        self.switch_voz_grabacion = self.switch_comandos_voz

        # Alias para compatibilidad con solucion_scroll_y_auto_update_escritorio.md y optimizacion_diseno_y_voz_v2.md
        self.camera_image_container = self.camera_container_cloud
        self.btn_capturar = self.btn_cloud_snapshot
        self.btn_grabar = self.btn_cloud_record
        self.btn_toggle_camara = self.btn_cloud_camera
        self.switch_avatar_mode = self.switch_cloud_avatar
        self.tomar_captura = self.cloud_take_photo_action
        self.iniciar_grabacion = self.cloud_start_recording_action
        self.detener_grabacion = self.cloud_stop_recording_action
        self.panel_derecho_camara = None
        self.feedback_container = None

        # 6. Botones de Acción (Pestaña 1)
        self.btn_camera = ft.Button(
            content="Prender Cámara",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            bgcolor="#2E7D32",
            color=ft.Colors.WHITE,
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
        self.btn_test_camera = ft.Button(
            content="Prender Cámara",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            bgcolor="#2E7D32",
            color=ft.Colors.WHITE,
            on_click=self.toggle_test_camera,
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
            if hasattr(self, "vision_service") and self.vision_service:
                self.vision_service.set_privacy_avatar_mode(False)
            self.stop_voice_commands_listener()
            if self.test_model_banner:
                self.test_model_banner.visible = False
                update_ui_safely(self.test_model_banner)
            self.voice_service.allow_voice_trigger = True
        elif index == 1:
            if hasattr(self, "vision_service") and self.vision_service:
                self.vision_service.set_privacy_avatar_mode(False)
            self.stop_voice_commands_listener()
            if self.test_model_banner:
                self.test_model_banner.visible = True
                update_ui_safely(self.test_model_banner)
            self.voice_service.allow_voice_trigger = False
            self.load_trained_models_to_test_dropdown()
        elif index == 2:
            if hasattr(self, "switch_cloud_avatar") and self.switch_cloud_avatar:
                self.vision_service.set_privacy_avatar_mode(bool(self.switch_cloud_avatar.value))
            if self.test_model_banner:
                self.test_model_banner.visible = False
                update_ui_safely(self.test_model_banner)
            self.voice_service.allow_voice_trigger = False
            self.refresh_cloud_table()
            if self.is_camera_active:
                self.start_voice_commands_listener()

    # --- REFRESCO DE CÁMARA CON PROTECCIÓN CONTRA DESTROYED SESSION ---

    def on_frame_update(self, base64_image: str, is_obstructed: bool = False):
        if not getattr(self, "is_camera_active", False):
            return

        data_src = f"data:image/jpeg;base64,{base64_image}"
        current_tab = getattr(self.tabs, "selected_index", 0) if hasattr(self, "tabs") and self.tabs else 0

        if current_tab == 0:
            if hasattr(self, "camera_view") and getattr(self.camera_view, "visible", False):
                self.camera_view.src = data_src
                update_ui_safely(self.camera_view)
        elif current_tab == 1:
            if hasattr(self, "test_camera_view") and getattr(self.test_camera_view, "visible", False):
                self.test_camera_view.src = data_src
                update_ui_safely(self.test_camera_view)
        elif current_tab == 2:
            if hasattr(self, "cloud_camera_image") and getattr(self.cloud_camera_image, "visible", False):
                self.cloud_camera_image.src = data_src
                update_ui_safely(self.cloud_camera_image)

        if getattr(self, "is_cloud_recording", False):
            raw_frame = getattr(self.vision_service, "last_raw_frame", None)
            if raw_frame is not None:
                self.cloud_recorded_frames.append(raw_frame.copy())
                cnt = len(self.cloud_recorded_frames)
                if cnt % 4 == 0:
                    self.lbl_cloud_cam_status.value = f"🔴 Grabando: {cnt} frames acumulados..."
                    update_ui_safely(self.lbl_cloud_cam_status)

        if hasattr(self, "warning_banner") and self.warning_banner.visible != is_obstructed:
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

    def toggle_privacy_avatar(self, e=None):
        if hasattr(self, "toggle_cloud_privacy_avatar"):
            self.toggle_cloud_privacy_avatar(e)

    def toggle_test_privacy_avatar(self, e=None):
        pass

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

    # --- ACCIONES DE CÁMARA Y CONTROL DE HARDWARE ---

    def release_all_camera_streams(self):
        """
        Libera de forma centralizada, segura y síncrona todas las capturas activas de la webcam (OpenCV VideoCapture)
        para evitar el error de hardware ocupado y pantalla negra al abrir modales o alternar vistas.
        """
        try:
            # 1. Detener sesión de inferencia en tiempo real si estaba activa
            if hasattr(self, "is_testing") and self.is_testing:
                if hasattr(self, "live_tester") and self.live_tester:
                    try:
                        self.live_tester.stop()
                    except Exception:
                        pass
                if hasattr(self, "vision_service") and self.vision_service:
                    self.vision_service.live_tester = None
                self.is_testing = False
                if hasattr(self, "btn_toggle_test") and self.btn_toggle_test:
                    self.btn_toggle_test.content = "Iniciar Prueba"
                    self.btn_toggle_test.icon = ft.Icons.PLAY_ARROW_ROUNDED
                    self.btn_toggle_test.bgcolor = COLOR_PRIMARY
                    update_ui_safely(self.btn_toggle_test)

            # 2. Detener hilo de comandos de voz si estaba en marcha
            if hasattr(self, "stop_voice_commands_listener"):
                self.stop_voice_commands_listener()

            self.is_cloud_recording = False

            # 3. Detener y liberar VideoCapture del servicio de visión principal
            if hasattr(self, "vision_service") and self.vision_service:
                self.vision_service.stop()

            # 4. Restablecer banderas de estado y sincronizar botones/placeholders
            self.is_camera_active = False
            self.sync_camera_buttons_and_placeholders(False)

            if hasattr(self, "warning_banner") and self.warning_banner:
                self.warning_banner.visible = False
                update_ui_safely(self.warning_banner)

            # 5. Pausa obligatoria para que el sistema operativo y DirectShow en Windows liberen el descriptor físico
            time.sleep(0.15)
        except Exception as ex:
            print(f"[CAM RELEASE] Error liberando cámaras: {ex}")

    def sync_camera_buttons_and_placeholders(self, is_active: bool):
        """Sincroniza el estado de los 3 botones de cámara y sus placeholders en todas las pestañas."""
        btn_text = "Apagar Cámara" if is_active else "Prender Cámara"
        btn_icon = ft.Icons.STOP_ROUNDED if is_active else ft.Icons.PLAY_ARROW_ROUNDED
        btn_color = "#E25C5C" if is_active else "#2E7D32"

        buttons = [getattr(self, "btn_camera", None),
                   getattr(self, "btn_test_camera", None),
                   getattr(self, "btn_cloud_camera", None)]

        for btn in buttons:
            if btn is not None:
                btn.content = btn_text
                btn.icon = btn_icon
                btn.bgcolor = btn_color
                btn.color = ft.Colors.WHITE
                update_ui_safely(btn)

        # Sincronizar visibilidad de placeholders y cámaras
        placeholders = [getattr(self, "placeholder_train", None),
                        getattr(self, "placeholder_test", None),
                        getattr(self, "placeholder_cloud", None)]
        for pl in placeholders:
            if pl is not None:
                pl.visible = not is_active
                update_ui_safely(pl)

        camera_views = [getattr(self, "camera_view", None),
                        getattr(self, "test_camera_view", None),
                        getattr(self, "cloud_camera_image", None)]
        for cv in camera_views:
            if cv is not None:
                cv.visible = is_active
                if not is_active:
                    cv.src = EMPTY_PIXEL_DATA
                update_ui_safely(cv)

    def toggle_test_camera(self, e=None):
        """Conmuta la cámara web directamente desde la pestaña de Tester."""
        self.toggle_camera(e)

    def toggle_camera(self, e=None):
        if not self.is_camera_active:
            self.status_text.value = "Iniciando cámara web a 25 FPS estables..."
            update_ui_safely(self.status_text)
            
            self.vision_service.start()
            self.is_camera_active = True
            self.sync_camera_buttons_and_placeholders(True)

            self.status_text.value = "Cámara activa (25 FPS). Listo para operar."
            self.status_text.color = COLOR_SUCCESS
            update_ui_safely(self.status_text)
            if hasattr(self, "tabs") and getattr(self.tabs, "selected_index", 0) == 2:
                self.start_voice_commands_listener()
        else:
            self.stop_voice_commands_listener()
            self.vision_service.stop()
            self.is_camera_active = False
            self.sync_camera_buttons_and_placeholders(False)

            if hasattr(self, "warning_banner") and self.warning_banner:
                self.warning_banner.visible = False
                update_ui_safely(self.warning_banner)
            self.status_text.value = "Cámara apagada."
            self.status_text.color = COLOR_TEXT_MUTED
            update_ui_safely(self.status_text)

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
                    ft.Text("Entrenado (.keras)", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.W_600, overflow=ft.TextOverflow.ELLIPSIS, max_lines=1)
                ], spacing=4),
                bgcolor="#DCFCE7",
                border_radius=6,
                padding=ft.Padding(8, 4, 8, 4)
            )
        else:
            local_badge = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.CANCEL_OUTLINED, size=14, color=COLOR_REC_BTN),
                    ft.Text("Sin Modelo Local", size=11, color=COLOR_REC_BTN, weight=ft.FontWeight.W_600, overflow=ft.TextOverflow.ELLIPSIS, max_lines=1)
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
                    ft.Text("Sincronizado en S3", size=11, color=COLOR_SUCCESS, weight=ft.FontWeight.BOLD, overflow=ft.TextOverflow.ELLIPSIS, max_lines=1)
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
                    ft.Text("Desactualizado en S3", size=11, color="#E65100", weight=ft.FontWeight.BOLD, overflow=ft.TextOverflow.ELLIPSIS, max_lines=1)
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
                    ft.Text("Solo en S3", size=11, color=COLOR_TEXT_MUTED, weight=ft.FontWeight.W_600, overflow=ft.TextOverflow.ELLIPSIS, max_lines=1)
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
                    ft.Text("Error Conexión", size=11, color=COLOR_REC_BTN, weight=ft.FontWeight.BOLD, overflow=ft.TextOverflow.ELLIPSIS, max_lines=1)
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
                    ft.Text("No Publicado", size=11, color="#455A64", weight=ft.FontWeight.BOLD, overflow=ft.TextOverflow.ELLIPSIS, max_lines=1)
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

        # Fila 1: Información de categoría y ruta técnica
        header_row = ft.Row([
            ft.Icon(ft.Icons.AUTO_AWESOME, color=COLOR_PRIMARY, size=18),
            ft.Column([
                ft.Text(
                    f"CATEGORÍA: {cat.upper()}",
                    size=13,
                    weight=ft.FontWeight.BOLD,
                    color=COLOR_TEXT_TITLE,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    max_lines=1
                ),
                ft.Text(
                    f"modelos/{cat}/tfjs_model/ (model.json + shard*.bin)",
                    size=10,
                    color=COLOR_TEXT_MUTED,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    max_lines=1
                )
            ], spacing=1, expand=True)
        ], spacing=8, expand=True)

        # Fila 2: Badges y Botones de Acción flexibles con salto de línea automático
        badges_actions_row = ft.Row(
            controls=[
                local_badge,
                cloud_badge,
                btn_upload,
                btn_del_cloud
            ],
            spacing=8,
            wrap=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER
        )

        card_content = ft.Column([
            header_row,
            badges_actions_row
        ], spacing=8)

        self.cloud_model_card_container.content = card_content
        update_ui_safely(self.cloud_model_card_container)

    def refresh_cloud_resources_table(self, category_name: str):
        """
        Consulta de forma asíncrona todos los recursos didácticos de la categoría en S3 y local,
        poblando interactivamente la lista de señas adaptada para la columna izquierda (600px).
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
                    if not self.cloud_active_word or self.cloud_active_word not in all_words:
                        self.cloud_active_category = cat_clean
                        self.cloud_active_word = all_words[0]
                        self.lbl_cloud_active_word.value = f"Palabra activa: {all_words[0].upper()} ({cat_clean.upper()})"
                        update_ui_safely(self.lbl_cloud_active_word)

                    for word in all_words:
                        is_selected = bool(self.cloud_active_word and word.lower().strip() == self.cloud_active_word.lower().strip())
                        # Muestras locales
                        word_dir = self.data_manager._get_word_dir(cat_clean, word)
                        samples_count = 0
                        if os.path.exists(word_dir):
                            samples_count = len([f for f in os.listdir(word_dir) if f.endswith('.csv')])

                        # Estado del recurso multimedia didáctico
                        res_status = self.cloud_service.check_resource_status(cat_clean, word, cloud_resources)
                        st = res_status["status"]
                        local_path = res_status["local_path"]
                        has_local = (local_path is not None and os.path.exists(local_path))
                        has_cloud = (st == "SINCRONIZADO" or st == "PENDIENTE")

                        # 1. Estado Local: 🟢 "Listo" / ⚫ "Sin Guía"
                        if has_local:
                            filename = os.path.basename(local_path)
                            badge_local = ft.Container(
                                content=ft.Row([
                                    ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED, size=13, color=COLOR_SUCCESS),
                                    ft.Text("Listo", size=10, color=COLOR_SUCCESS, weight=ft.FontWeight.BOLD),
                                    ft.Text(f"({filename.split('.')[-1].upper()})", size=9, color="#15803D")
                                ], spacing=3),
                                bgcolor="#DCFCE7",
                                border_radius=6,
                                padding=ft.Padding(6, 3, 6, 3),
                                tooltip=f"Archivo local: {local_path}"
                            )
                        else:
                            badge_local = ft.Container(
                                content=ft.Row([
                                    ft.Icon(ft.Icons.CIRCLE, size=8, color=COLOR_TEXT_MUTED),
                                    ft.Text("Sin Guía", size=10, color=COLOR_TEXT_MUTED, weight=ft.FontWeight.W_500)
                                ], spacing=4),
                                bgcolor="#F1F5F9",
                                border_radius=6,
                                padding=ft.Padding(6, 3, 6, 3)
                            )

                        # 2. Estado en S3: 🚀 "Sincronizado" / ⚠️ "Pendiente" / ☁️ "No en S3"
                        if st == "SINCRONIZADO":
                            badge_s3 = ft.Container(
                                content=ft.Row([
                                    ft.Text("🚀", size=11),
                                    ft.Text("Sincronizado", size=10, color=COLOR_SUCCESS, weight=ft.FontWeight.BOLD)
                                ], spacing=3),
                                bgcolor="#E8F5E9",
                                border_radius=6,
                                padding=ft.Padding(6, 3, 6, 3),
                                tooltip="Coincide con S3"
                            )
                        elif st == "PENDIENTE":
                            badge_s3 = ft.Container(
                                content=ft.Row([
                                    ft.Text("⚠️", size=11),
                                    ft.Text("Pendiente", size=10, color="#E65100", weight=ft.FontWeight.BOLD)
                                ], spacing=3),
                                bgcolor="#FFF3E0",
                                border_radius=6,
                                padding=ft.Padding(6, 3, 6, 3),
                                tooltip="Local modificado pendiente de subir a S3"
                            )
                        else:  # NO_EN_S3
                            badge_s3 = ft.Container(
                                content=ft.Row([
                                    ft.Text("☁️", size=11),
                                    ft.Text("No en S3", size=10, color="#455A64", weight=ft.FontWeight.BOLD)
                                ], spacing=3),
                                bgcolor="#ECEFF1",
                                border_radius=6,
                                padding=ft.Padding(6, 3, 6, 3),
                                tooltip="No subido a S3"
                            )

                        # 3. Controles de Fila Integrados
                        btn_select_capture = ft.Button(
                            content="Capturar",
                            icon=ft.Icons.VIDEOCAM,
                            bgcolor="#0284C7" if is_selected else COLOR_PRIMARY,
                            color=ft.Colors.WHITE,
                            height=30,
                            tooltip=f"Seleccionar '{word.upper()}' para captura en panel de cámara",
                            on_click=lambda ev, c=cat_clean, w=word: self.select_cloud_word_for_capture(c, w)
                        )

                        btn_view_local = ft.IconButton(
                            icon=ft.Icons.VISIBILITY,
                            icon_color="#0284C7" if has_local else COLOR_TEXT_MUTED,
                            icon_size=16,
                            disabled=not has_local,
                            tooltip=f"Ver recurso local ({filename if has_local else 'No disponible'})",
                            on_click=lambda ev, c=cat_clean, w=word: self.open_local_media_preview(c, w)
                        )

                        btn_upload_s3 = ft.IconButton(
                            icon=ft.Icons.CLOUD_UPLOAD,
                            icon_color=COLOR_SUCCESS if has_local else COLOR_TEXT_MUTED,
                            icon_size=16,
                            disabled=not has_local,
                            tooltip="Subir a AWS S3",
                            on_click=lambda ev, c=cat_clean, w=word: self.upload_word_resource(c, w)
                        )

                        btn_delete_s3 = ft.IconButton(
                            icon=ft.Icons.CLOUD_OFF,
                            icon_color="#E11D48" if has_cloud else COLOR_TEXT_MUTED,
                            icon_size=16,
                            disabled=not has_cloud,
                            tooltip="Eliminar de AWS S3",
                            on_click=lambda ev, c=cat_clean, w=word: self.delete_word_resource_s3(c, w)
                        )

                        btn_delete_local = ft.IconButton(
                            icon=ft.Icons.DELETE_OUTLINE,
                            icon_color="#DC2626" if has_local else COLOR_TEXT_MUTED,
                            icon_size=16,
                            disabled=not has_local,
                            tooltip="Eliminar archivo local",
                            on_click=lambda ev, c=cat_clean, w=word: self.delete_word_resource_local(c, w)
                        )

                        btn_pick = ft.IconButton(
                            icon=ft.Icons.ATTACH_FILE,
                            icon_color=COLOR_TEXT_MUTED,
                            icon_size=16,
                            tooltip="Importar archivo desde PC",
                            on_click=lambda ev, c=cat_clean, w=word: self.open_file_picker_for_word(c, w)
                        )

                        row_container = ft.Container(
                            content=ft.Row([
                                ft.Container(
                                    content=ft.Row([
                                        ft.Icon(ft.Icons.TRANSLATE, color=COLOR_PRIMARY, size=15),
                                        ft.Column([
                                            ft.Text(word.upper(), size=11, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                                            ft.Text(f"{samples_count} m.", size=9, color=COLOR_TEXT_MUTED)
                                        ], spacing=0)
                                    ], spacing=5),
                                    width=110
                                ),
                                ft.Container(content=badge_local, width=105),
                                ft.Container(content=badge_s3, width=105),
                                ft.Row([btn_select_capture, btn_view_local, btn_upload_s3, btn_delete_s3, btn_delete_local, btn_pick], spacing=2, expand=True)
                            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            bgcolor="#EFF6FF" if is_selected else "#FFFFFF",
                            border=ft.Border.all(1.5 if is_selected else 1, COLOR_PRIMARY if is_selected else COLOR_BORDER),
                            border_radius=8,
                            padding=ft.Padding(8, 4, 8, 4)
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

    def select_cloud_word_for_capture(self, category: str, word: str):
        """Selecciona la palabra activa para captura multimedia en la columna derecha de S3."""
        self.cloud_active_category = category.lower().strip()
        self.cloud_active_word = word.lower().strip()
        self.lbl_cloud_active_word.value = f"Palabra activa: {word.upper()} ({category.upper()})"
        self.lbl_cloud_cam_status.value = f"Listo para capturar seña para '{word.upper()}'."
        self.lbl_cloud_cam_status.color = COLOR_PRIMARY
        update_ui_safely(self.lbl_cloud_active_word)
        update_ui_safely(self.lbl_cloud_cam_status)
        show_snack_bar(self.page, f"Palabra activa: '{word.upper()}'. Listo para capturar en el monitor.")
        if not self.is_camera_active:
            self.toggle_cloud_camera()
        else:
            self.refresh_cloud_resources_table(category)

    def toggle_cloud_camera(self, e=None):
        """Conmuta la cámara web directamente dentro de la pestaña S3."""
        if not self.is_camera_active:
            self.release_all_camera_streams()
            time.sleep(0.1)

            if hasattr(self, "switch_cloud_avatar") and self.switch_cloud_avatar:
                self.vision_service.set_privacy_avatar_mode(bool(self.switch_cloud_avatar.value))

            self.vision_service.start()
            self.is_camera_active = True
            self.sync_camera_buttons_and_placeholders(True)

            self.lbl_cloud_cam_status.value = "Cámara activa (25 FPS). Listo para capturar."
            self.lbl_cloud_cam_status.color = COLOR_SUCCESS
            update_ui_safely(self.lbl_cloud_cam_status)

            self.start_voice_commands_listener()
        else:
            self.stop_voice_commands_listener()
            if self.is_cloud_recording:
                self.is_cloud_recording = False
                self.btn_cloud_record.disabled = False
                self.btn_cloud_stop.disabled = True
                self.btn_cloud_snapshot.disabled = False
                update_ui_safely(self.btn_cloud_record)
                update_ui_safely(self.btn_cloud_stop)
                update_ui_safely(self.btn_cloud_snapshot)

            self.vision_service.stop()
            self.is_camera_active = False
            self.sync_camera_buttons_and_placeholders(False)

            self.lbl_cloud_cam_status.value = "Cámara apagada."
            self.lbl_cloud_cam_status.color = COLOR_TEXT_MUTED
            update_ui_safely(self.lbl_cloud_cam_status)

    def toggle_cloud_privacy_avatar(self, e=None):
        """Conmuta entre cámara real y avatar pedagógico en la pestaña S3."""
        val = bool(self.switch_cloud_avatar.value)
        if hasattr(self, "vision_service") and self.vision_service:
            self.vision_service.set_privacy_avatar_mode(val)
        mode_str = "Avatar de Privacidad (Didáctico)" if val else "Cámara Real"
        self.lbl_cloud_cam_status.value = f"Modo visual: {mode_str}."
        update_ui_safely(self.lbl_cloud_cam_status)

    def toggle_comandos_voz(self, e=None):
        """Conmuta la escucha unificada de comandos de voz manos libres (optimizacion_diseno_y_voz_v2.md)."""
        val = bool(self.switch_comandos_voz.value)
        if hasattr(self, "switch_voz_captura") and self.switch_voz_captura is not self.switch_comandos_voz:
            self.switch_voz_captura.value = val
        if hasattr(self, "switch_voz_grabacion") and self.switch_voz_grabacion is not self.switch_comandos_voz:
            self.switch_voz_grabacion.value = val
        estado = "activados" if val else "desactivados"
        show_snack_bar(self.page, f"Comandos de voz {estado}.")
        update_ui_safely(self.switch_comandos_voz)
        if val:
            if getattr(self, "is_camera_active", False):
                self.start_voice_commands_listener()
            else:
                self.lbl_voice_command_status.value = "🎙️ Comandos de voz listos (se activarán al encender la cámara)."
                self.lbl_voice_command_status.color = COLOR_PRIMARY
                update_ui_safely(self.lbl_voice_command_status)
        else:
            self.stop_voice_commands_listener()

    def toggle_voice_capture(self, e=None):
        """Compatibilidad con selector de captura por voz."""
        self.toggle_comandos_voz(e)

    def toggle_voice_record(self, e=None):
        """Compatibilidad con selector de grabación por voz."""
        self.toggle_comandos_voz(e)

    def tomar_captura(self, e=None):
        """Ejecuta la captura de foto para el avatar."""
        return self.cloud_take_photo_action(e)

    def iniciar_grabacion(self, e=None):
        """Inicia la grabación de video para el avatar."""
        return self.cloud_start_recording_action(e)

    def detener_grabacion(self, e=None):
        """Detiene la grabación de video y compila el recurso."""
        return self.cloud_stop_recording_action(e)

    def ejecutar_captura_manual(self, e=None):
        """Alias para captura manual de foto."""
        self.tomar_captura(e)

    def ejecutar_grabacion_manual(self, e=None):
        """Alias para inicio de grabación de avatar."""
        self.iniciar_grabacion(e)

    def toggle_camara_stream(self, e=None):
        """Alias para encendido/apagado de cámara en panel S3."""
        self.toggle_cloud_camera(e)

    def toggle_avatar_mode(self, e=None):
        """Alias para conmutar modo avatar."""
        self.toggle_cloud_privacy_avatar(e)

    def cloud_take_photo_action(self, e=None):
        """Toma una foto estática PNG de la seña actual y la guarda localmente."""
        if not self.is_camera_active:
            show_snack_bar(self.page, "Debe encender la cámara antes de tomar una foto.", is_error=True)
            return
        if not self.cloud_active_word:
            show_snack_bar(self.page, "Seleccione primero una palabra de la tabla para capturar.", is_error=True)
            return

        frame = getattr(self.vision_service, "last_raw_frame", None)
        if frame is None:
            show_snack_bar(self.page, "No hay fotograma disponible para capturar.", is_error=True)
            return

        frame_to_save = frame.copy()
        cat = self.cloud_active_category or (self.cloud_category_dropdown.value or "").lower().strip()
        word = self.cloud_active_word

        def _snap_worker():
            try:
                saved_path = self.cloud_service.save_avatar_snapshot(cat, word, frame_to_save)
                filename = os.path.basename(saved_path)
                def _ui_done():
                    self.lbl_cloud_cam_status.value = f"✅ Foto '{filename}' guardada con éxito."
                    self.lbl_cloud_cam_status.color = COLOR_SUCCESS
                    update_ui_safely(self.lbl_cloud_cam_status)
                    show_snack_bar(self.page, f"Foto '{filename}' guardada para '{word.upper()}'.")
                    self.refresh_cloud_resources_table(cat)
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_done)
                else:
                    _ui_done()
            except Exception as ex:
                show_snack_bar(self.page, f"Error al guardar foto: {ex}", is_error=True)

        threading.Thread(target=_snap_worker, daemon=True).start()

    def cloud_start_recording_action(self, e=None):
        """Inicia la acumulación de frames para compilar un recurso GIF animado."""
        if not self.is_camera_active:
            show_snack_bar(self.page, "Debe encender la cámara antes de grabar.", is_error=True)
            return
        if not self.cloud_active_word:
            show_snack_bar(self.page, "Seleccione primero una palabra de la tabla para grabar.", is_error=True)
            return

        self.cloud_recorded_frames.clear()
        self.is_cloud_recording = True
        self.btn_cloud_record.disabled = True
        self.btn_cloud_stop.disabled = False
        self.btn_cloud_snapshot.disabled = True
        self.lbl_cloud_cam_status.value = f"🔴 GRABANDO SEÑA PARA '{self.cloud_active_word.upper()}'... Realice la seña."
        self.lbl_cloud_cam_status.color = "#EF4444"
        update_ui_safely(self.btn_cloud_record)
        update_ui_safely(self.btn_cloud_stop)
        update_ui_safely(self.btn_cloud_snapshot)
        update_ui_safely(self.lbl_cloud_cam_status)

    def cloud_stop_recording_action(self, e=None):
        """Detiene la grabación y compila el GIF animado en segundo plano."""
        if not getattr(self, "is_cloud_recording", False):
            return
        self.is_cloud_recording = False
        self.btn_cloud_stop.disabled = True
        self.btn_cloud_record.disabled = True
        self.btn_cloud_snapshot.disabled = True
        self.progress_cloud_compile.visible = True
        total_frames = len(self.cloud_recorded_frames)
        self.lbl_cloud_cam_status.value = f"⏳ Compilando animación GIF ({total_frames} frames)..."
        self.lbl_cloud_cam_status.color = COLOR_PRIMARY
        update_ui_safely(self.btn_cloud_stop)
        update_ui_safely(self.btn_cloud_record)
        update_ui_safely(self.btn_cloud_snapshot)
        update_ui_safely(self.progress_cloud_compile)
        update_ui_safely(self.lbl_cloud_cam_status)

        cat = self.cloud_active_category or (self.cloud_category_dropdown.value or "").lower().strip()
        word = self.cloud_active_word
        frames_to_save = list(self.cloud_recorded_frames)
        self.cloud_recorded_frames.clear()

        def _compile_worker():
            try:
                if not frames_to_save:
                    raise ValueError("No se capturaron frames durante la grabación.")
                saved_path = self.cloud_service.save_avatar_recording(cat, word, frames_to_save)
                filename = os.path.basename(saved_path)
                def _ui_done():
                    self.progress_cloud_compile.visible = False
                    self.btn_cloud_record.disabled = False
                    self.btn_cloud_snapshot.disabled = False
                    self.lbl_cloud_cam_status.value = f"✅ GIF '{filename}' guardado ({len(frames_to_save)} frames)."
                    self.lbl_cloud_cam_status.color = COLOR_SUCCESS
                    update_ui_safely(self.progress_cloud_compile)
                    update_ui_safely(self.btn_cloud_record)
                    update_ui_safely(self.btn_cloud_snapshot)
                    update_ui_safely(self.lbl_cloud_cam_status)
                    show_snack_bar(self.page, f"Guía didáctica '{filename}' guardada para '{word.upper()}'.")
                    self.refresh_cloud_resources_table(cat)
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_done)
                else:
                    _ui_done()
            except Exception as ex:
                def _ui_err():
                    self.progress_cloud_compile.visible = False
                    self.btn_cloud_record.disabled = False
                    self.btn_cloud_snapshot.disabled = False
                    self.lbl_cloud_cam_status.value = f"Error al compilar animación: {ex}"
                    self.lbl_cloud_cam_status.color = "#EF4444"
                    update_ui_safely(self.progress_cloud_compile)
                    update_ui_safely(self.btn_cloud_record)
                    update_ui_safely(self.btn_cloud_snapshot)
                    update_ui_safely(self.lbl_cloud_cam_status)
                    show_snack_bar(self.page, f"Error al compilar animación: {ex}", is_error=True)
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_err)
                else:
                    _ui_err()

        threading.Thread(target=_compile_worker, daemon=True).start()

    def cloud_preview_action(self, e=None):
        """Abre la vista previa del recurso multimedia activo."""
        if not self.cloud_active_word:
            show_snack_bar(self.page, "Seleccione primero una palabra de la tabla para ver su vista previa.", is_error=True)
            return
        cat = self.cloud_active_category or (self.cloud_category_dropdown.value or "").lower().strip()
        self.open_local_media_preview(cat, self.cloud_active_word)

    def start_voice_commands_listener(self):
        """Inicia el hilo cooperativo asíncrono de reconocimiento de voz usando speech_recognition."""
        if sr is None:
            self.lbl_voice_command_status.value = "⚠️ SpeechRecognition no instalado (Use botones manuales)"
            self.lbl_voice_command_status.color = ft.Colors.AMBER_800
            update_ui_safely(self.lbl_voice_command_status)
            return

        if hasattr(self, "switch_comandos_voz") and not getattr(self.switch_comandos_voz, "value", False):
            self.lbl_voice_command_status.value = "🎙️ Micrófono inactivo (Comandos de voz desactivados)."
            self.lbl_voice_command_status.color = COLOR_TEXT_MUTED
            update_ui_safely(self.lbl_voice_command_status)
            return

        if getattr(self, "voice_listener_running", False):
            return

        self.voice_listener_running = True
        self.lbl_voice_command_status.value = "🎙️ Micrófono Activo: Esperando comando ('captura', 'grabar', 'no grabes')..."
        self.lbl_voice_command_status.color = COLOR_PRIMARY
        update_ui_safely(self.lbl_voice_command_status)

        threading.Thread(target=self.listen_voice_commands, daemon=True).start()

    def listen_voice_commands(self):
        """Hilo en segundo plano para escuchar comandos de voz manos libres en español (speech_recognition)."""
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True

        try:
            with sr.Microphone() as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                except Exception:
                    pass

                while (getattr(self, "is_camera_active", False) and 
                       getattr(self, "voice_listener_running", False) and 
                       getattr(self.switch_comandos_voz, "value", True)):
                    try:
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                        command = ""
                        try:
                            command = recognizer.recognize_google(audio, language="es-PE").lower()
                        except Exception:
                            try:
                                command = recognizer.recognize_google(audio, language="es-ES").lower()
                            except Exception:
                                pass

                        if not command:
                            continue

                        if hasattr(self, "switch_comandos_voz") and not getattr(self.switch_comandos_voz, "value", True):
                            print("[VOICE COMMAND] Switch de voz desactivado durante la escucha.")
                            break

                        print(f"[VOICE COMMAND] Detectado: '{command}'")

                        def _notify(cmd_text):
                            self.lbl_voice_command_status.value = f"🎙️ Voz detectada: '{cmd_text}'"
                            self.lbl_voice_command_status.color = COLOR_SUCCESS
                            update_ui_safely(self.lbl_voice_command_status)

                        if "captura" in command or "foto" in command:
                            if self.page and hasattr(self.page, "run_thread"):
                                self.page.run_thread(_notify, "Foto / Captura")
                                self.page.run_thread(self.tomar_captura, None)
                            else:
                                _notify("Foto / Captura")
                                self.tomar_captura(None)
                        elif "grabar" in command or "graba" in command:
                            if self.page and hasattr(self.page, "run_thread"):
                                self.page.run_thread(_notify, "Grabar")
                                self.page.run_thread(self.iniciar_grabacion, None)
                            else:
                                _notify("Grabar")
                                self.iniciar_grabacion(None)
                        elif "no grabes" in command or "detener" in command or "alto" in command or "parar" in command:
                            if self.page and hasattr(self.page, "run_thread"):
                                self.page.run_thread(_notify, "Detener")
                                self.page.run_thread(self.detener_grabacion, None)
                            else:
                                _notify("Detener")
                                self.detener_grabacion(None)

                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        continue
                    except sr.RequestError as re:
                        print(f"[VOICE COMMAND] Error de conexión Google Speech: {re}")
                        time.sleep(1)
                    except Exception as loop_ex:
                        time.sleep(0.5)

        except Exception as ex:
            print(f"[VOICE COMMAND] Error o micrófono no disponible: {ex}")
            def _show_mic_err():
                self.lbl_voice_command_status.value = "⚠️ Micrófono no detectado (Use botones manuales)"
                self.lbl_voice_command_status.color = ft.Colors.AMBER_800
                update_ui_safely(self.lbl_voice_command_status)
            if self.page and hasattr(self.page, "run_thread"):
                self.page.run_thread(_show_mic_err)
            else:
                _show_mic_err()
        finally:
            self.voice_listener_running = False

    def stop_voice_commands_listener(self):
        """Detiene de forma cooperativa el hilo de reconocimiento de voz y libera el micrófono."""
        self.voice_listener_running = False
        if hasattr(self, "lbl_voice_command_status") and self.lbl_voice_command_status:
            self.lbl_voice_command_status.value = "🎙️ Micrófono inactivo (Comandos de voz desactivados)."
            self.lbl_voice_command_status.color = COLOR_TEXT_MUTED
            update_ui_safely(self.lbl_voice_command_status)

    def open_local_media_preview(self, category: str, word: str):
        """Muestra una ventana modal emergente con el GIF o imagen del avatar grabado localmente."""
        local_path = self.cloud_service.get_local_resource_path(category, word)
        if not local_path or not os.path.exists(local_path):
            show_snack_bar(self.page, f"No existe archivo local para '{word.upper()}'.", is_error=True)
            return

        try:
            with open(local_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")

            filename = os.path.basename(local_path)
            size_kb = os.path.getsize(local_path) / 1024.0

            ext = os.path.splitext(local_path)[1].lower().replace('.', '')
            mime = "image/gif" if ext == "gif" else ("image/jpeg" if ext in ["jpg", "jpeg"] else "image/png")
            data_uri = f"data:{mime};base64,{b64_data}"

            img_preview = ft.Image(
                src=data_uri,
                width=440,
                height=320,
                fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain",
                border_radius=8
            )
            try:
                img_preview.src_base64 = b64_data
            except Exception:
                pass

            def _close_preview(e=None):
                preview_dialog.open = False
                update_ui_safely(preview_dialog)
                if hasattr(self.page, "overlay") and preview_dialog in self.page.overlay:
                    try:
                        self.page.overlay.remove(preview_dialog)
                    except Exception:
                        pass

            preview_dialog = ft.AlertDialog(
                title=ft.Row([
                    ft.Container(
                        content=ft.Icon(ft.Icons.VISIBILITY, color=COLOR_PRIMARY, size=20),
                        bgcolor=COLOR_PRIMARY_LIGHT,
                        border_radius=6,
                        padding=6
                    ),
                    ft.Column([
                        ft.Text(f"VISTA PREVIA LOCAL: {word.upper()}", size=13, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                        ft.Text(f"{filename} • {size_kb:.1f} KB", size=10, color=COLOR_TEXT_MUTED)
                    ], spacing=1)
                ], spacing=8),
                content=ft.Container(
                    content=ft.Column([
                        ft.Container(
                            content=img_preview,
                            bgcolor="#F4F8FA",
                            border=ft.Border.all(1, COLOR_BORDER),
                            border_radius=8,
                            padding=6,
                            alignment=ft.Alignment.CENTER
                        ),
                        ft.Text(f"Ruta: {local_path}", size=9, color=COLOR_TEXT_MUTED, selectable=True)
                    ], spacing=6, tight=True),
                    width=460
                ),
                actions=[
                    ft.Button("Cerrar", on_click=_close_preview)
                ],
                actions_alignment=ft.MainAxisAlignment.END,
                modal=True,
                open=True
            )

            if hasattr(self.page, "overlay") and self.page.overlay is not None:
                self.page.overlay.append(preview_dialog)
                self.page.update()
        except Exception as ex:
            show_snack_bar(self.page, f"Error al abrir vista previa: {ex}", is_error=True)

    def open_file_picker_for_word(self, category: str, word: str):
        """Abre el explorador de archivos para asociar una guía visual didáctica a la seña."""
        self._picking_target_category = category
        self._picking_target_word = word

        def _open_native_picker():
            picked_path = None
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                picked_path = filedialog.askopenfilename(
                    title=f"Seleccionar Guía Visual para '{word.upper()}' (GIF, PNG, JPG, MP4)",
                    filetypes=[
                        ("Archivos Multimedia (*.gif, *.png, *.jpg, *.jpeg, *.mp4)", "*.gif;*.png;*.jpg;*.jpeg;*.mp4"),
                        ("Imágenes (*.gif, *.png, *.jpg, *.jpeg)", "*.gif;*.png;*.jpg;*.jpeg"),
                        ("Videos (*.mp4)", "*.mp4"),
                        ("Todos los archivos", "*.*")
                    ]
                )
                root.destroy()
            except Exception as ex:
                print(f"[PICKER] Advertencia diálogo nativo: {ex}")

            if picked_path:
                class _PickedFile:
                    def __init__(self, p):
                        self.path = p
                        self.name = os.path.basename(p)
                class _PickerEvent:
                    def __init__(self, files):
                        self.files = files

                self.on_file_picker_result(_PickerEvent([_PickedFile(picked_path)]))

        threading.Thread(target=_open_native_picker, daemon=True).start()

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
        """Sube la guía multimedia didáctica de la seña a AWS S3 (recursos/{categoria}/{palabra}_avatar.[ext])."""
        # Validación de existencia física previa
        local_path = self.cloud_service.get_local_resource_path(category, word)
        if not local_path or not os.path.exists(local_path):
            show_snack_bar(self.page, f"No existe archivo local para '{word.upper()}'. Graba o carga un avatar primero.", is_error=True)
            return

        def _upload_worker():
            try:
                def _ui_start():
                    self.cloud_progress_ring.visible = True
                    self.cloud_progress_bar.visible = True
                    self.cloud_progress_bar.value = None
                    self.lbl_cloud_status.value = f"Subiendo avatar didáctico de '{word.upper()}' a AWS S3..."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                self.cloud_service.upload_resource(category, word)

                def _ui_done():
                    self.lbl_cloud_status.value = f"¡Avatar didáctico de '{word.upper()}' sincronizado con éxito en AWS S3!"
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.value = 1.0
                    show_snack_bar(self.page, f"Avatar didáctico de '{word.upper()}' subido a S3.")
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

    def delete_word_resource_s3(self, category: str, word: str):
        """Elimina el recurso multimedia didáctico exclusivamente del bucket de AWS S3."""
        def _delete_s3_worker():
            try:
                def _ui_start():
                    self.cloud_progress_ring.visible = True
                    self.cloud_progress_bar.visible = True
                    self.cloud_progress_bar.value = None
                    self.lbl_cloud_status.value = f"Borrando recurso de '{word.upper()}' en AWS S3..."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                self.cloud_service.delete_resource_s3(category, word)

                def _ui_done():
                    self.lbl_cloud_status.value = f"Recurso de '{word.upper()}' eliminado de AWS S3."
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.value = 0.0
                    show_snack_bar(self.page, f"Recurso de '{word.upper()}' eliminado de S3.")
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
                    self.lbl_cloud_status.value = f"Error al eliminar de S3: {str(ex)}"
                    show_snack_bar(self.page, f"Error al eliminar de S3: {str(ex)}", is_error=True)
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)
                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_err)
                else:
                    _ui_err()

        threading.Thread(target=_delete_s3_worker, daemon=True).start()

    def delete_word_resource_local(self, category: str, word: str):
        """Elimina el archivo multimedia del avatar guardado en el disco local de la PC."""
        try:
            self.cloud_service.delete_resource_local(category, word)
            show_snack_bar(self.page, f"Archivo local de '{word.upper()}' eliminado de la PC.")
            self.refresh_cloud_resources_table(category)
        except Exception as ex:
            show_snack_bar(self.page, f"Error al eliminar archivo local: {ex}", is_error=True)

    def delete_word_resource(self, category: str, word: str):
        """Elimina la guía multimedia didáctica de AWS S3 y del directorio local (compatibilidad)."""
        def _delete_worker():
            try:
                def _ui_start():
                    self.cloud_progress_ring.visible = True
                    self.cloud_progress_bar.visible = True
                    self.cloud_progress_bar.value = None
                    self.lbl_cloud_status.value = f"Eliminando recurso didáctico de '{word.upper()}' de S3 y local..."
                    update_ui_safely(self.cloud_progress_ring)
                    update_ui_safely(self.cloud_progress_bar)
                    update_ui_safely(self.lbl_cloud_status)

                if self.page and hasattr(self.page, "run_thread"):
                    self.page.run_thread(_ui_start)

                self.cloud_service.delete_resource(category, word, delete_local=True)

                def _ui_done():
                    self.lbl_cloud_status.value = f"Recurso didáctico de '{word.upper()}' eliminado de S3 y local."
                    self.cloud_progress_ring.visible = False
                    self.cloud_progress_bar.value = 0.0
                    show_snack_bar(self.page, f"Recurso de '{word.upper()}' eliminado.")
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
                    show_snack_bar(self.page, f"Error al eliminar: {str(ex)}", is_error=True)
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

    # Panel Derecho: Monitor de Video Estandarizado (480x360) sin textos sobrepuestos
    camera_panel_train = ft.Column(
        controls=[
            controller.warning_banner,
            controller.camera_container_train,
            ft.Row(
                controls=[
                    controller.btn_camera,
                    controller.btn_generate_cnn
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=14
            )
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=15
    )

    columna_derecha = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Text("Monitor de Captura y Tracking 3D", size=14, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                ft.Row([controller.switch_voice, controller.voice_badge], spacing=4)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            camera_panel_train
        ], spacing=15, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=16,
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
    # 1. Card 1: Seleccionar Categoría y Controles de Cámara / Avatar
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
            ft.Row([
                controller.btn_toggle_test,
                controller.btn_test_camera
            ], spacing=6)
        ], spacing=8),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=10
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
                height=185,
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

    # Panel Derecho: Monitor de Inferencia Estandarizado (480x360) + Tarjeta Gigante
    camera_panel_test = ft.Column(
        controls=[
            controller.camera_container_test
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=12
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
        width=None,
        expand=True
    )
    controller.feedback_container = tarjeta_gigante

    columna_derecha_test = ft.Container(
        content=ft.Column([
            ft.Row([camera_panel_test], alignment=ft.MainAxisAlignment.CENTER),
            tarjeta_gigante
        ], spacing=12, horizontal_alignment=ft.CrossAxisAlignment.STRETCH),
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
    Layout estático de dos columnas (ft.Row):
    - Columna Izquierda (Ancho: 600px):
        1. Panel de Modelo Web TensorFlow.js
        2. Tabla interactiva de palabras y recursos didácticos
        3. Consola de sincronización y transferencia
    - Columna Derecha (Flexible, expand=True):
        Sección "Captura del Tutor Inteligente (Avatar / Persona)":
        1. Header con estado y palabra activa seleccionada
        2. Visor de video estático (controller.cloud_camera_image) en contenedor monitor oscuro
        3. Barra de progreso de compilación GIF
        4. Fila con botón Encender Cámara y Switch Avatar de Privacidad
        5. Fila de 4 botones distribuidos proporcionalmente: [Tomar Foto, Grabar, Detener, Ver Vista Previa]
        6. Indicador visual de escucha de comandos de voz en segundo plano
    """
    # Encabezado superior de Nube con selector y bucket
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

    # 1. Columna Izquierda (Ancho: 600px): Modelo + Recursos Didácticos + Consola
    card_model = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.MEMORY, color=COLOR_PRIMARY, size=18),
                    ft.Text(
                        "1. Modelo Predictivo Web (TensorFlow.js)",
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=COLOR_TEXT_TITLE,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        max_lines=1
                    )
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
        padding=15,
        width=600
    )

    table_header = ft.Container(
        content=ft.Row([
            ft.Container(ft.Text("PALABRA", size=10, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED), width=110),
            ft.Container(ft.Text("LOCAL", size=10, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED), width=105),
            ft.Container(ft.Text("AWS S3", size=10, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED), width=105),
            ft.Container(ft.Text("ACCIONES MULTIMEDIA", size=10, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MUTED), expand=True)
        ], alignment=ft.MainAxisAlignment.START),
        padding=ft.Padding(8, 6, 8, 6),
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
                ft.Text("S3: recursos/{cat}/{palabra}_avatar", size=10, color=COLOR_TEXT_MUTED)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            table_header,
            ft.Container(
                content=controller.cloud_resources_listview,
                height=260,
                padding=2
            )
        ], spacing=6),
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=12
    )

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

    columna_izquierda_s3 = ft.Column([
        card_model,
        card_resources,
        card_console
    ], spacing=10, width=600)

    # 2. Columna Derecha (Flexible): Captura del Tutor Inteligente con Scroll Adaptable
    container_switches_voz = ft.Container(
        width=480,
        padding=ft.Padding(16, 12, 16, 12),
        border=ft.Border.all(1, "#D1E4F8"),
        border_radius=10,
        bgcolor="#FFFFFF",
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.MIC, size=18, color=COLOR_PRIMARY),
                    ft.Text("Controles Manos Libres", size=13, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE),
                ], spacing=6),
                ft.Container(
                    content=ft.Text("Voz Activa", size=10, weight=ft.FontWeight.BOLD, color=COLOR_PRIMARY),
                    bgcolor=COLOR_PRIMARY_LIGHT,
                    border_radius=6,
                    padding=ft.Padding(6, 2, 6, 2)
                )
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            controller.switch_comandos_voz,
            ft.Text("Diga 'captura' / 'foto', 'grabar' o 'detener' / 'no grabes' para operar sin teclado.", size=11, color=COLOR_TEXT_MUTED)
        ], spacing=6)
    )

    panel_derecho_camara = ft.Column(
        scroll=ft.ScrollMode.AUTO,
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=12,
        controls=[
            # 1. Cabecera del Panel con palabra activa
            ft.Row([
                ft.Row([
                    ft.Icon(ft.Icons.VIDEOCAM_ROUNDED, color=COLOR_PRIMARY, size=20),
                    ft.Text("Grabador de Recursos Didácticos", size=14, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_TITLE)
                ], spacing=6),
                ft.Container(
                    content=controller.lbl_cloud_active_word,
                    bgcolor=COLOR_PRIMARY_LIGHT,
                    border_radius=6,
                    padding=ft.Padding(8, 4, 8, 4)
                )
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, width=480),

            # 2. Visor de Cámara Negro Minimalista (480x360)
            controller.camera_container_cloud,
            controller.progress_cloud_compile,

            # 3. Sliders de Habilitación de Comandos de Voz (Switches)
            container_switches_voz,

            # 4. Botonera de Acción Manual (Inmediatamente debajo)
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                controls=[
                    controller.btn_cloud_snapshot,
                    controller.btn_cloud_record,
                    controller.btn_cloud_stop,
                    controller.btn_cloud_preview
                ],
                spacing=8,
                width=480
            ),

            # 5. Botones de Control de Hardware
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    controller.btn_cloud_camera,
                    controller.switch_cloud_avatar
                ],
                width=480
            ),

            # 6. Indicadores de Estado
            ft.Row(
                controls=[controller.lbl_cloud_cam_status],
                alignment=ft.MainAxisAlignment.CENTER,
                width=480
            ),
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.MIC_NONE, size=16, color=COLOR_PRIMARY),
                    controller.lbl_voice_command_status
                ], spacing=6),
                bgcolor=COLOR_PRIMARY_LIGHT,
                border=ft.Border.all(1, "#BAE6FD"),
                border_radius=8,
                padding=ft.Padding(10, 6, 10, 6),
                width=480
            )
        ]
    )

    controller.panel_derecho_camara = panel_derecho_camara

    card_captura_s3 = ft.Container(
        content=panel_derecho_camara,
        bgcolor=COLOR_CARD_BG,
        border=ft.Border.all(1, COLOR_BORDER),
        border_radius=12,
        padding=14,
        expand=True
    )

    columna_derecha_s3 = ft.Container(
        content=card_captura_s3,
        expand=True
    )

    return ft.Container(
        content=ft.Column([
            card_header,
            ft.Row([
                columna_izquierda_s3,
                columna_derecha_s3
            ], spacing=14, expand=True, vertical_alignment=ft.CrossAxisAlignment.START)
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
