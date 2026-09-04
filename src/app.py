import os
import sys
import asyncio
import atexit
import flet as ft

# Configurar la política nativa de event loop para Windows
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from src.data_manager import LSPDatasetManager
from src.model_trainer import LSPTrainer
from src.tester_service import LSPTesterService
from src.vision_service import LSPVisionService
from src.ui_components import LSPUIController, build_main_app_tabs

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MUESTRAS_DIR = os.path.join(DATA_DIR, "muestras")
MODELOS_DIR = os.path.join(DATA_DIR, "modelos")

os.makedirs(MUESTRAS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

def main(page: ft.Page):
    page.title = "Traductor LSP - Panel Escolar de Entrenamiento y Validación"
    # Rediseño visual escolar: Tema claro, fondo blanco-azulado y scroll automático
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = "#F4F8FA"
    page.padding = ft.Padding(16, 12, 16, 12)
    page.scroll = ft.ScrollMode.AUTO
    
    # Configuración de ventana centrada en el escritorio (solucion_avatar_didactico_v4.md)
    page.window_width = 1200
    page.window_height = 800
    page.window_resizable = False
    if hasattr(page, "window") and page.window:
        try:
            page.window.width = 1200
            page.window.height = 800
            page.window_resizable = False
            page.window.center()
        except Exception:
            pass
    if hasattr(page, "window_center"):
        try:
            page.window_center()
        except Exception:
            pass
    page.update()

    # =========================================================================
    # INICIALIZACIÓN UNIFICADA CON RUTAS ABSOLUTAS (solucion_rutas_y_limpieza.md)
    # =========================================================================
    # 1. El gestor de datos apunta a las muestras
    db_manager = LSPDatasetManager(base_dir=MUESTRAS_DIR)

    # 2. El entrenador apunta al gestor y guarda en la carpeta de modelos de data/
    trainer = LSPTrainer(db_manager, export_base_dir=MODELOS_DIR)

    # 3. El probador en vivo busca los modelos en la misma carpeta unificada
    tester_service = LSPTesterService(model_base_dir=MODELOS_DIR, page_ref=page)

    # Servicio de visión por computadora
    vision_service = LSPVisionService(page_ref=page)
    
    # Inicializar controlador de UI y ensamblar control nativo ft.Tabs
    ui_controller = LSPUIController(page, db_manager, vision_service, trainer, tester_service)
    tabs = build_main_app_tabs(ui_controller)

    # Registrar selector de archivos en el overlay de la página
    if hasattr(page, "overlay") and ui_controller.file_picker not in page.overlay:
        page.overlay.append(ui_controller.file_picker)

    # =========================================================================
    # COMPONENTES SUPERIORES DEL SISTEMA STITCH (principal.png / test.png)
    # =========================================================================

    # 1. Botones de Modo en la Cabecera (Entrenar / Test / Nube AWS)
    btn_mode_train = ft.Container(
        content=ft.Text("Entrenar", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        bgcolor="#0A66C2",
        border_radius=18,
        padding=ft.Padding(16, 5, 16, 5),
        on_click=lambda e: ui_controller.switch_tab(0)
    )
    btn_mode_test = ft.Container(
        content=ft.Text("Test", size=12, weight=ft.FontWeight.W_500, color="#64748B"),
        bgcolor=ft.Colors.TRANSPARENT,
        border_radius=18,
        padding=ft.Padding(16, 5, 16, 5),
        on_click=lambda e: ui_controller.switch_tab(1)
    )
    btn_mode_cloud = ft.Container(
        content=ft.Text("Nube AWS", size=12, weight=ft.FontWeight.W_500, color="#64748B"),
        bgcolor=ft.Colors.TRANSPARENT,
        border_radius=18,
        padding=ft.Padding(16, 5, 16, 5),
        on_click=lambda e: ui_controller.switch_tab(2)
    )

    ui_controller.btn_mode_train = btn_mode_train
    ui_controller.btn_mode_test = btn_mode_test
    ui_controller.btn_mode_cloud = btn_mode_cloud

    mode_switch_pill = ft.Container(
        content=ft.Row([btn_mode_train, btn_mode_test, btn_mode_cloud], spacing=2),
        bgcolor="#F1F5F9",
        border_radius=22,
        padding=2
    )

    # 2. Encabezado Institucional Completo
    header_row = ft.Row([
        # Logotipo Escolar + Título Institucional
        ft.Row([
            ft.Container(
                content=ft.Icon(ft.Icons.SCHOOL, color="#0A66C2", size=22),
                bgcolor="#EBF4FF",
                border_radius=8,
                padding=6
            ),
            ft.Text("TRADUCTOR DE LENGUA DE SEÑAS PERUANA (LSP)", size=17, weight=ft.FontWeight.BOLD, color="#1A365D"),
        ], spacing=10),

        # Conmutador de Modo + Badge Docente + Avatar de Usuario
        ft.Row([
            mode_switch_pill,
            ft.Container(width=1, height=20, bgcolor="#D1E4F8"),
            ft.Container(
                content=ft.Row([
                    ft.Container(width=7, height=7, bgcolor="#0A66C2", border_radius=4),
                    ft.Text("Módulo Docente", size=11, weight=ft.FontWeight.BOLD, color="#1A365D")
                ], spacing=6),
                bgcolor="#EBF4FF",
                border_radius=14,
                padding=ft.Padding(10, 5, 10, 5)
            ),
            ft.CircleAvatar(
                bgcolor="#0A66C2",
                radius=14,
                content=ft.Icon(ft.Icons.PERSON, color=ft.Colors.WHITE, size=15)
            )
        ], spacing=8)
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    # 3. Consola de Estado Superior Permanente (Visible de inmediato)
    status_banner = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Icon(ft.Icons.INFO_OUTLINE, color="#0A66C2", size=16),
                ui_controller.training_progress,
                ui_controller.status_text,
            ], spacing=8, expand=True),
            ft.Row([
                ft.Container(width=7, height=7, bgcolor="#0A66C2", border_radius=4),
                ft.Text("MOTOR IA ACTIVO", size=10, weight=ft.FontWeight.BOLD, color="#1A365D")
            ], spacing=6)
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        bgcolor="#EBF4FF",
        border=ft.Border.all(1, "#D1E4F8"),
        border_radius=8,
        padding=ft.Padding(12, 6, 12, 6),
        margin=ft.Margin(0, 0, 0, 2)
    )

    # 4. Sub-banner de Modelo Activo (Stitch test.png)
    test_model_banner = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Icon(ft.Icons.GRAPHIC_EQ, color="#0A66C2", size=16),
                ui_controller.lbl_active_model_subbanner,
            ], spacing=8, expand=True),
            ft.Row([
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.VOLUME_UP, color="#0A66C2", size=12),
                        ft.Text("TTS ACTIVO", size=9, weight=ft.FontWeight.BOLD, color="#0A66C2")
                    ], spacing=4),
                    bgcolor="#EBF4FF",
                    border_radius=10,
                    padding=ft.Padding(7, 3, 7, 3)
                ),
                ft.Container(
                    content=ft.Row([
                        ft.Container(width=5, height=5, bgcolor="#60A5FA", border_radius=3),
                        ft.Text("IA INFERENCIA ACTIVA", size=9, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)
                    ], spacing=4),
                    bgcolor="#0F172A",
                    border_radius=10,
                    padding=ft.Padding(7, 3, 7, 3)
                )
            ], spacing=6)
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
        bgcolor="#F8FAFC",
        border=ft.Border.all(1, "#D1E4F8"),
        border_radius=8,
        padding=ft.Padding(12, 5, 12, 5),
        margin=ft.Margin(0, 0, 0, 2),
        visible=False
    )
    ui_controller.test_model_banner = test_model_banner

    # Construir grilla principal de la aplicación con estética escolar Stitch
    page.add(
        ft.Column([
            header_row,
            status_banner,
            test_model_banner,
            tabs
        ], expand=True, spacing=4)
    )

    # Apagado coordinado en on_disconnect para detener bucles en segundo plano
    def on_disconnect(e=None):
        try:
            ui_controller.close()
            vision_service.stop()
            vision_service.close()
        except Exception:
            pass

    page.on_disconnect = on_disconnect
    page.on_close = on_disconnect
    if hasattr(page, "window") and page.window:
        page.window.on_event = lambda e: on_disconnect(e) if getattr(e, "data", "") == "close" else None
    atexit.register(on_disconnect)

if __name__ == "__main__":
    if hasattr(ft, "run"):
        ft.run(main)
    else:
        ft.app(target=main)