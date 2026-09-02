import os
import glob
import json
import time
import threading
import flet as ft
from src.data_manager import LSPDataManager
from src.vision_service import LSPVisionService, STATE_IDLE, STATE_PREPARATION, STATE_RECORDING, STATE_COMPLETE
from src.trainer import LSPTrainer
from src.voice_service import LSPVoiceService
from src.model_trainer import ModelTrainer
from src.tester_service import LiveTester

EMPTY_PIXEL_DATA = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

def show_snack_bar(page: ft.Page, message: str, is_error: bool = False):
    """Muestra una notificación emergente estilizada."""
    sb = ft.SnackBar(
        content=ft.Text(message, color=ft.Colors.WHITE, weight=ft.FontWeight.W_500),
        bgcolor=ft.Colors.RED_800 if is_error else ft.Colors.GREEN_800,
        open=True
    )
    if hasattr(page, "overlay"):
        page.overlay.append(sb)
    try:
        page.update()
    except Exception:
        pass

class LSPUIController:
    """
    Controlador central de la interfaz de usuario de administración para Traductor LSP.
    Gestiona la captura, la máquina de estados, el entrenamiento CNN 1D y la validación en vivo.
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

        # Configurar callbacks del servicio de visión
        self.vision_service.frame_callback = self.on_frame_update
        self.vision_service.recording_callback = self.on_recording_complete
        self.vision_service.state_callback = self.on_state_changed

        # Servicio de activación por voz (Vosk)
        self.voice_service = LSPVoiceService(
            trigger_callback=self.on_voice_trigger_detected,
            status_callback=self.on_voice_status_update
        )

        # Construir componentes visuales
        self._init_controls()

        # Conectar control de imagen directamente para el fix de latencia
        self.vision_service.video_image_control = self.camera_view

    def _init_controls(self):
        # 1. Controles de Categoría
        self.new_category_input = ft.TextField(label="Nombre de Categoría", width=200, text_size=14)
        self.category_dropdown = ft.Dropdown(
            label="Categoría de Señas",
            hint_text="Seleccione categoría",
            width=260,
            on_select=lambda e: self.on_category_changed(e.control.value)
        )
        self.btn_edit_category = ft.IconButton(
            icon=ft.Icons.EDIT,
            tooltip="Editar nombre de categoría",
            icon_color=ft.Colors.BLUE_300,
            on_click=self.show_edit_category_dialog
        )
        self.btn_delete_category = ft.IconButton(
            icon=ft.Icons.DELETE,
            tooltip="Eliminar categoría",
            icon_color=ft.Colors.RED_300,
            on_click=self.on_delete_category_clicked
        )

        # 2. Controles de Vocabulario
        self.new_word_input = ft.TextField(label="Nueva Palabra", width=200, text_size=14)
        self.words_listview = ft.ListView(expand=True, spacing=10, padding=10)

        # 3. Visores de Cámara (Entrenamiento y Testing)
        self.camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=500,
            height=375,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain"
        )
        self.test_camera_view = ft.Image(
            src=EMPTY_PIXEL_DATA,
            width=500,
            height=375,
            fit=ft.BoxFit.CONTAIN if hasattr(ft, "BoxFit") else "contain"
        )

        self.warning_banner = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, color=ft.Colors.YELLOW_ACCENT_400),
                ft.Text("⚠️ ADVERTENCIA: Cámara obstruida o iluminación insuficiente.",
                        color=ft.Colors.YELLOW_ACCENT_400, weight=ft.FontWeight.BOLD, size=13)
            ], alignment=ft.MainAxisAlignment.CENTER),
            bgcolor=ft.Colors.RED_900,
            padding=8,
            border_radius=6,
            visible=False
        )

        # 4. Modo Escucha por Voz (Vosk)
        self.switch_voice = ft.Switch(
            label="Modo Escucha (Vosk)",
            value=False,
            active_color=ft.Colors.GREEN_400,
            on_change=self.toggle_voice_mode,
            tooltip="Diga 'Enciéndete' para iniciar la preparación y grabación a manos libres"
        )
        self.voice_badge = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.MIC_OFF, size=18, color=ft.Colors.GREY_400),
                ft.Text("Voz inactiva", size=12, color=ft.Colors.GREY_400)
            ], spacing=5),
            padding=5
        )

        # 5. Botones de Acción (Pestaña 1)
        self.btn_camera = ft.Button(
            content="Encender Cámara",
            icon=ft.Icons.VIDEOCAM,
            bgcolor=ft.Colors.GREEN_900,
            color=ft.Colors.WHITE,
            on_click=self.toggle_camera,
            width=200,
            height=45
        )
        # Botón obligatorio: Generar Modelo de Categoría (CNN 1D)
        self.btn_generate_cnn = ft.Button(
            content="Generar Modelo de Categoría",
            icon=ft.Icons.AUTO_AWESOME,
            bgcolor=ft.Colors.DEEP_PURPLE_800,
            color=ft.Colors.WHITE,
            on_click=self.run_cnn_training_flow,
            disabled=True,
            width=250,
            height=45,
            tooltip="Habilitado al tener al menos 30 muestras de cada palabra en la categoría"
        )

        # 6. Indicadores de Estado y Progreso
        self.status_text = ft.Text(value="Sistema Listo. Encienda la cámara para comenzar.", color=ft.Colors.BLUE_200, size=14)
        self.training_progress = ft.ProgressRing(visible=False, width=20, height=20, stroke_width=2)
        self.training_status_container = ft.Row([self.training_progress, self.status_text], alignment=ft.MainAxisAlignment.START)

        # 7. Controles de la Pestaña 2 (Pruebas y Validación en Vivo)
        self.test_category_dropdown = ft.Dropdown(
            label="Modelo por Categoría",
            hint_text="Seleccione un modelo entrenado",
            width=280,
            on_select=self.on_test_category_selected
        )
        self.btn_toggle_test = ft.Button(
            content="Iniciar Prueba",
            icon=ft.Icons.PLAY_ARROW,
            bgcolor=ft.Colors.GREEN_800,
            color=ft.Colors.WHITE,
            on_click=self.toggle_live_test,
            width=200,
            height=45
        )
        self.lbl_prediction = ft.Text(
            value="Esperando seña...",
            size=30,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.GREEN_ACCENT_400
        )
        self.test_status_text = ft.Text(
            value="Seleccione un modelo (.h5) y pulse 'Iniciar Prueba' para validar señas en vivo.",
            size=14,
            color=ft.Colors.GREY_300
        )

        # Cargar categorías iniciales y modelos disponibles
        self.load_categories_to_dropdown()
        self.load_trained_models_to_test_dropdown()

    # --- FIX CRÍTICO DE CONGELAMIENTO (RENDERIZADO LIGERO Y ASÍNCRONO) ---

    def on_frame_update(self, base64_image: str, is_obstructed: bool = False):
        """
        Recibe el fotograma codificado y realiza el refresco explícito sobre el control Image.
        Evita page.update() global que congelaba la aplicación.
        """
        data_src = f"data:image/jpeg;base64,{base64_image}"
        
        # 1. Refresco del visor de la pestaña 1
        self.camera_view.src = data_src
        try:
            self.camera_view.update()
        except Exception:
            pass

        # 2. Refresco del visor de la pestaña 2 (si está visible)
        self.test_camera_view.src = data_src
        try:
            self.test_camera_view.update()
        except Exception:
            pass

        # 3. Banner de advertencia solo se actualiza al cambiar de estado
        if self.warning_banner.visible != is_obstructed:
            self.warning_banner.visible = is_obstructed
            try:
                self.warning_banner.update()
            except Exception:
                pass

    # --- CALLBACKS DE LA MÁQUINA DE ESTADOS ---

    def on_state_changed(self, state: int, message: str):
        """Callback ejecutado por las transiciones de la máquina de estados."""
        if state == STATE_PREPARATION:
            self.status_text.value = f"⏱️ {message}"
            self.status_text.color = ft.Colors.ORANGE_300
            self.enable_ui_controls(False)
        elif state == STATE_RECORDING:
            self.status_text.value = f"🔴 {message}"
            self.status_text.color = ft.Colors.RED_400
        elif state == STATE_IDLE:
            self.status_text.value = f"✅ {message}"
            self.status_text.color = ft.Colors.GREEN_400
            self.enable_ui_controls(True)
        try:
            self.status_text.update()
        except Exception:
            self.page.update()

    def on_recording_complete(self, category: str, word: str, sequence: list):
        """Recibe la secuencia de 30 frames y la persiste en disco."""
        try:
            file_path = self.data_manager.save_sequence(category, word, sequence)
            word_dir = self.data_manager._get_word_dir(category, word)
            num_muestras = len([f for f in os.listdir(word_dir) if f.endswith('.csv')])
            self.status_text.value = f"¡Muestra guardada! '{word.upper()}': {num_muestras} muestras."
            self.status_text.color = ft.Colors.GREEN_400
            show_snack_bar(self.page, f"Muestra #{num_muestras} guardada para '{word.upper()}'")
        except Exception as ex:
            self.status_text.value = f"Error al guardar la muestra: {str(ex)}"
            self.status_text.color = ft.Colors.RED_400
            show_snack_bar(self.page, f"Error al guardar muestra: {str(ex)}", is_error=True)

        self.enable_ui_controls(True)
        self.refresh_words_list()
        self.update_cnn_button_state()
        try:
            self.page.update()
        except Exception:
            pass

    # --- RECONOCIMIENTO DE VOZ (VOSK) ---

    def toggle_voice_mode(self, e):
        """Activa o desactiva el hilo de escucha de micrófono con Vosk."""
        if self.switch_voice.value:
            if not self.is_camera_active:
                self.toggle_camera(None)

            self.voice_badge.content = ft.Row([
                ft.Icon(ft.Icons.MIC, size=18, color=ft.Colors.GREEN_400),
                ft.Text("Escuchando...", size=12, color=ft.Colors.GREEN_400)
            ], spacing=5)
            self.voice_service.start()
            show_snack_bar(self.page, "Modo Escucha activo: diga 'Enciéndete' para iniciar grabación")
        else:
            self.voice_service.stop()
            self.voice_badge.content = ft.Row([
                ft.Icon(ft.Icons.MIC_OFF, size=18, color=ft.Colors.GREY_400),
                ft.Text("Voz inactiva", size=12, color=ft.Colors.GREY_400)
            ], spacing=5)
            show_snack_bar(self.page, "Modo Escucha desactivado.")
        self.page.update()

    def on_voice_status_update(self, msg: str):
        self.status_text.value = msg
        try:
            self.status_text.update()
        except Exception:
            pass

    def on_voice_trigger_detected(self):
        if not self.is_camera_active:
            self.toggle_camera(None)
            time.sleep(0.5)

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
        show_snack_bar(self.page, f"🎤 ¡Comando 'Enciéndete' detectado! Preparando captura para '{target_word.upper()}'")
        self.start_preparation_flow(target_word)

    # --- MÁQUINA DE ESTADOS: FLUJO DE PREPARACIÓN Y GRABACIÓN ---

    def start_preparation_flow(self, word: str):
        if not self.is_camera_active:
            self.toggle_camera(None)
            time.sleep(0.5)

        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría primero.", is_error=True)
            return

        self.selected_word = word
        self.enable_ui_controls(False)
        self.vision_service.start_preparation(self.selected_category, word)
        self.page.update()

    # --- GESTIÓN DE CATEGORÍAS Y PALABRAS ---

    def load_categories_to_dropdown(self):
        categories = self.data_manager.get_categories()
        self.category_dropdown.options = [ft.DropdownOption(key=cat, text=cat.upper()) for cat in categories]
        if self.selected_category and self.selected_category in categories:
            self.category_dropdown.value = self.selected_category
        elif categories:
            self.category_dropdown.value = None
        self.page.update()

    def on_category_changed(self, category_val: str):
        if not category_val:
            return
        self.selected_category = category_val.lower().strip()
        self.selected_word = None
        self.status_text.value = f"Categoría activa: {category_val.upper()}"
        self.status_text.color = ft.Colors.WHITE
        self.refresh_words_list()
        self.update_cnn_button_state()

    def update_cnn_button_state(self):
        """
        Habilita el botón 'Generar Modelo de Categoría' únicamente si cada palabra
        de la categoría tiene al menos 30 muestras grabadas.
        """
        if not self.selected_category:
            self.btn_generate_cnn.disabled = True
            self.btn_generate_cnn.tooltip = "Seleccione una categoría primero"
            self.page.update()
            return

        words = self.data_manager.get_words_in_category(self.selected_category)
        if len(words) < 2:
            self.btn_generate_cnn.disabled = True
            self.btn_generate_cnn.tooltip = f"Se requieren al menos 2 palabras (actual: {len(words)})"
            self.page.update()
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
            autofocus=True
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
                    self.status_text.color = ft.Colors.GREEN_400
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
            title=ft.Text("Editar Nombre de Categoría"),
            content=edit_input,
            actions=[
                ft.TextButton(content="Cancelar", on_click=cancel_dialog),
                ft.Button(content="Guardar", on_click=save_rename, bgcolor=ft.Colors.BLUE_800, color=ft.Colors.WHITE)
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
            self.status_text.color = ft.Colors.YELLOW_400
            show_snack_bar(self.page, "Categoría eliminada correctamente")
        except Exception as ex:
            show_snack_bar(self.page, f"Error al eliminar categoría: {str(ex)}", is_error=True)

    def refresh_words_list(self):
        self.words_listview.controls.clear()
        if not self.selected_category:
            self.page.update()
            return

        words = self.data_manager.get_words_in_category(self.selected_category)
        for word in words:
            word_dir = self.data_manager._get_word_dir(self.selected_category, word)
            samples_count = 0
            if os.path.exists(word_dir):
                samples_count = len([f for f in os.listdir(word_dir) if f.endswith('.csv')])

            is_complete = samples_count >= 30
            badge_color = ft.Colors.GREEN_400 if is_complete else ft.Colors.ORANGE_300

            word_row = ft.Container(
                content=ft.Row([
                    ft.Text(f"{word.upper()} ({samples_count}/30 muestras)", expand=True, size=14, weight=ft.FontWeight.BOLD, color=badge_color),
                    ft.IconButton(
                        icon=ft.Icons.FIBER_MANUAL_RECORD,
                        icon_color=ft.Colors.RED,
                        tooltip="Grabar seña con 3s de preparación previa",
                        on_click=lambda ev, w=word: self.start_preparation_flow(w)
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE,
                        icon_color=ft.Colors.RED_200,
                        tooltip="Eliminar palabra y sus muestras",
                        on_click=lambda ev, w=word: self.delete_word(w)
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=10,
                border=ft.Border.all(1, ft.Colors.OUTLINE),
                border_radius=8,
                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST
            )
            self.words_listview.controls.append(word_row)

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
            self.status_text.color = ft.Colors.GREEN_400
            self.new_word_input.value = ""
            self.refresh_words_list()
            self.update_cnn_button_state()
        else:
            show_snack_bar(self.page, "La palabra ya existe en esta categoría", is_error=True)

    def delete_word(self, word: str):
        if self.data_manager.delete_word(self.selected_category, word):
            self.status_text.value = f"Palabra '{word.upper()}' eliminada del disco."
            self.status_text.color = ft.Colors.YELLOW_400
            self.refresh_words_list()
            self.update_cnn_button_state()

    def enable_ui_controls(self, enabled: bool):
        self.category_dropdown.disabled = not enabled
        self.btn_edit_category.disabled = not enabled
        self.btn_delete_category.disabled = not enabled
        self.new_category_input.disabled = not enabled
        self.new_word_input.disabled = not enabled
        self.words_listview.disabled = not enabled
        if enabled:
            self.update_cnn_button_state()
        else:
            self.btn_generate_cnn.disabled = True
        self.page.update()

    # --- ACCIONES DE CÁMARA ---

    def toggle_camera(self, e):
        """Enciende o apaga la cámara web de forma asíncrona usando DirectShow."""
        if not self.is_camera_active:
            self.status_text.value = "Iniciando cámara y MediaPipe Holistic..."
            self.page.update()
            self.vision_service.start()
            self.is_camera_active = True
            self.btn_camera.content = "Apagar Cámara"
            self.btn_camera.icon = ft.Icons.VIDEOCAM_OFF
            self.btn_camera.bgcolor = ft.Colors.RED_900
            self.status_text.value = "Cámara activa. Listo para operar."
            self.status_text.color = ft.Colors.GREEN_400
        else:
            self.vision_service.stop()
            self.is_camera_active = False
            self.btn_camera.content = "Encender Cámara"
            self.btn_camera.icon = ft.Icons.VIDEOCAM
            self.btn_camera.bgcolor = ft.Colors.GREEN_900
            self.camera_view.src = EMPTY_PIXEL_DATA
            self.test_camera_view.src = EMPTY_PIXEL_DATA
            self.warning_banner.visible = False
            self.status_text.value = "Cámara apagada."
            self.status_text.color = ft.Colors.WHITE
        self.page.update()

    # --- MÓDULO DE ENTRENAMIENTO CNN 1D (MEJORA 2) ---

    def run_cnn_training_flow(self, e):
        """Entrena la CNN 1D espacio-temporal en un hilo secundario independiente."""
        if not self.selected_category:
            show_snack_bar(self.page, "Seleccione una categoría para entrenar", is_error=True)
            return

        def _async_cnn_train():
            try:
                self.training_progress.visible = True
                self.status_text.value = f"Entrenando CNN 1D para '{self.selected_category.upper()}' (50 épocas)..."
                self.status_text.color = ft.Colors.YELLOW_300
                self.enable_ui_controls(False)
                self.page.update()

                # Cargar dataset de la categoría
                X, y, label_map = self.data_manager.load_dataset_for_training(self.selected_category)
                num_classes = len(label_map)

                # Entrenar CNN 1D y exportar modelo .h5
                model_path = self.model_trainer.build_and_train_cnn(
                    X_train=X,
                    y_train=y,
                    num_classes=num_classes,
                    category_name=self.selected_category,
                    label_map=label_map
                )

                self.status_text.value = f"¡Modelo CNN generado con éxito! Guardado en: {model_path}"
                self.status_text.color = ft.Colors.GREEN_400
                show_snack_bar(self.page, f"¡Modelo CNN para '{self.selected_category.upper()}' generado con éxito!")

                # Actualizar listado de modelos disponibles para pruebas en vivo
                self.load_trained_models_to_test_dropdown()

            except Exception as err:
                self.status_text.value = f"Error en entrenamiento CNN: {str(err)}"
                self.status_text.color = ft.Colors.RED_400
                show_snack_bar(self.page, f"Error: {str(err)}", is_error=True)
            finally:
                self.training_progress.visible = False
                self.enable_ui_controls(True)
                self.page.update()

        threading.Thread(target=_async_cnn_train, daemon=True).start()

    # --- MÓDULO DE PRUEBAS Y VALIDACIÓN EN VIVO (MEJORA 2) ---

    def load_trained_models_to_test_dropdown(self):
        """Escanea el directorio 'modelos/' y puebla el dropdown de pruebas."""
        os.makedirs('modelos', exist_ok=True)
        model_files = glob.glob('modelos/modelo_LSP_*.h5')
        
        categories = []
        for mf in model_files:
            base = os.path.basename(mf)
            # Extraer category de modelo_LSP_{category}.h5
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
            self.page.update()

    def toggle_live_test(self, e):
        """Inicia o detiene la prueba en tiempo real con sliding window sobre la cámara."""
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

            # Cargar etiquetas
            labels = []
            if os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    label_map = json.load(f)
                    labels = [label_map[str(i)] if str(i) in label_map else label_map[i] for i in range(len(label_map))]
            else:
                labels = self.data_manager.get_words_in_category(cat)

            try:
                # Inicializar LiveTester
                self.live_tester = LiveTester(model_path, labels)
                self.live_tester.start()

                # Vincular tester con el servicio de visión
                self.vision_service.live_tester = self.live_tester
                self.vision_service.prediction_label_control = self.lbl_prediction

                # Asegurar cámara encendida
                if not self.is_camera_active:
                    self.toggle_camera(None)

                self.is_testing = True
                self.btn_toggle_test.content = "Detener Prueba"
                self.btn_toggle_test.icon = ft.Icons.STOP
                self.btn_toggle_test.bgcolor = ft.Colors.RED_800
                self.lbl_prediction.value = "Esperando seña..."
                self.lbl_prediction.color = ft.Colors.GREEN_ACCENT_400
                self.test_status_text.value = f"🔴 Prueba en vivo activa para: {cat.upper()} ({len(labels)} clases cargadas)."
                self.test_status_text.color = ft.Colors.GREEN_300
                show_snack_bar(self.page, f"Prueba en vivo iniciada para '{cat.upper()}'. Realice señas frente a la cámara.")

            except Exception as ex:
                show_snack_bar(self.page, f"Error cargando modelo: {str(ex)}", is_error=True)

        else:
            # Detener prueba
            if self.live_tester:
                self.live_tester.stop()
            self.vision_service.live_tester = None
            self.vision_service.prediction_label_control = None

            self.is_testing = False
            self.btn_toggle_test.content = "Iniciar Prueba"
            self.btn_toggle_test.icon = ft.Icons.PLAY_ARROW
            self.btn_toggle_test.bgcolor = ft.Colors.GREEN_800
            self.lbl_prediction.value = "Prueba detenida."
            self.lbl_prediction.color = ft.Colors.GREY_400
            self.test_status_text.value = "Prueba detenida. Puede cambiar de modelo o reiniciar."
            self.test_status_text.color = ft.Colors.GREY_300
            show_snack_bar(self.page, "Prueba en vivo detenida.")

        self.page.update()

    def close(self):
        if self.voice_service:
            self.voice_service.stop()
        if self.live_tester:
            self.live_tester.stop()

# --- CONSTRUCTORES DE VISTAS (PESTAÑAS) ---

def build_training_view(controller: LSPUIController) -> ft.Container:
    """Construye la vista principal de Gestión, Captura y Entrenamiento CNN 1D."""
    # Panel lateral izquierdo
    sidebar = ft.Container(
        content=ft.Column([
            ft.Text("1. Gestión de Categorías", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_200),
            ft.Row([
                controller.new_category_input,
                ft.IconButton(ft.Icons.ADD_BOX, on_click=controller.add_new_category, tooltip="Crear Categoría")
            ]),
            ft.Row([
                controller.category_dropdown,
                controller.btn_edit_category,
                controller.btn_delete_category,
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Divider(height=15),
            
            ft.Text("2. Vocabulario de la Categoría", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_200),
            ft.Row([
                controller.new_word_input,
                ft.IconButton(ft.Icons.ADD_CIRCLE, on_click=controller.add_new_word, tooltip="Agregar Palabra")
            ]),
            ft.Container(
                content=controller.words_listview,
                height=320,
                border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
                border_radius=10,
                padding=5
            ),
        ], spacing=12),
        width=450,
    )

    # Panel de cámara y acciones
    camera_panel = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Text("3. Captura de Movimiento & Modelo CNN", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_200),
                ft.Row([controller.switch_voice, controller.voice_badge], spacing=8)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            
            # Monitor de video y alerta de cámara obstruida
            ft.Container(
                content=ft.Column([
                    controller.warning_banner,
                    controller.camera_view
                ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                border=ft.Border.all(2, ft.Colors.BLUE_900),
                border_radius=12,
                bgcolor=ft.Colors.BLACK,
                padding=5,
                alignment=ft.Alignment.CENTER
            ),
            
            # Panel de Botones (Encender Cámara y Generar Modelo de Categoría)
            ft.Row([
                controller.btn_camera,
                controller.btn_generate_cnn
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=15),
            
            # Barra de Estado y Progreso
            ft.Container(
                content=controller.training_status_container,
                bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
                border_radius=8,
                padding=10,
                border=ft.Border.all(1, ft.Colors.OUTLINE)
            )
        ], spacing=12, alignment=ft.MainAxisAlignment.CENTER),
        expand=True,
        padding=10
    )

    return ft.Container(
        content=ft.Row([sidebar, camera_panel], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START),
        padding=10
    )

def build_live_testing_view(controller: LSPUIController) -> ft.Container:
    """Construye la nueva pestaña de Pruebas y Validación en Vivo con ventana deslizante."""
    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.ANALYTICS, color=ft.Colors.GREEN_400, size=26),
                ft.Text("Módulo de Pruebas y Validación en Vivo (Sliding Window)", size=18, weight=ft.FontWeight.BOLD),
            ], spacing=10),
            ft.Divider(height=10),
            
            # Barra de configuración de la prueba
            ft.Row([
                controller.test_category_dropdown,
                controller.btn_toggle_test,
                ft.IconButton(
                    icon=ft.Icons.REFRESH,
                    tooltip="Recargar modelos de la carpeta modelos/",
                    on_click=lambda e: controller.load_trained_models_to_test_dropdown()
                )
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=15),
            
            ft.Row([
                # Monitor de video en vivo
                ft.Container(
                    content=controller.test_camera_view,
                    border=ft.Border.all(2, ft.Colors.GREEN_900),
                    border_radius=12,
                    bgcolor=ft.Colors.BLACK,
                    padding=5,
                    alignment=ft.Alignment.CENTER
                ),
                
                # Tarjeta de visualización de predicciones
                ft.Container(
                    content=ft.Column([
                        ft.Text("Resultado de Inferencia en Tiempo Real", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_200),
                        ft.Container(
                            content=ft.Column([
                                ft.Icon(ft.Icons.RECORD_VOICE_OVER, size=48, color=ft.Colors.GREEN_ACCENT_400),
                                controller.lbl_prediction,
                                ft.Text("Umbral de confianza: > 85%", size=13, color=ft.Colors.GREY_400)
                            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=12),
                            bgcolor=ft.Colors.BLACK,
                            border=ft.Border.all(2, ft.Colors.GREEN_ACCENT_700),
                            border_radius=12,
                            padding=30,
                            alignment=ft.Alignment.CENTER,
                            width=450,
                            height=240
                        ),
                        controller.test_status_text
                    ], spacing=15),
                    padding=10,
                    expand=True
                )
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START, spacing=20)
        ], spacing=15),
        padding=15
    )

def build_main_app_tabs(controller: LSPUIController) -> ft.Tabs:
    """Ensambla las pestañas principales de la aplicación."""
    t1 = ft.Tab(label="Captura y Entrenamiento", icon=ft.Icons.APP_REGISTRATION)
    t2 = ft.Tab(label="Validación en Vivo (Testing)", icon=ft.Icons.ANALYTICS)
    
    tab_bar = ft.TabBar(tabs=[t1, t2], divider_color=ft.Colors.OUTLINE)
    tab_views = ft.TabBarView(
        controls=[
            build_training_view(controller),
            build_live_testing_view(controller)
        ],
        expand=True
    )
    
    return ft.Tabs(
        length=2,
        content=ft.Column([tab_bar, tab_views], expand=True, spacing=10),
        expand=True
    )
