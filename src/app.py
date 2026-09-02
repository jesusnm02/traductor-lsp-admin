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

from src.data_manager import LSPDataManager
from src.vision_service import LSPVisionService
from src.trainer import LSPTrainer
from src.ui_components import LSPUIController, build_main_app_tabs

def main(page: ft.Page):
    page.title = "Traductor LSP - Panel Escolar de Entrenamiento y Validación"
    # Rediseño visual escolar: Tema claro, fondo blanco-azulado y scroll automático
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = "#F4F8FA"
    page.padding = 15
    page.scroll = ft.ScrollMode.AUTO
    
    if hasattr(page, "window") and page.window:
        page.window.width = 1280
        page.window.height = 840
        page.window.resizable = True
    else:
        page.window_width = 1280
        page.window_height = 840
        page.window_resizable = True

    # Inicializar componentes del Backend con despacho seguro a través de page
    data_manager = LSPDataManager()
    vision_service = LSPVisionService(page_ref=page)
    trainer = LSPTrainer(data_manager)
    
    # Inicializar controlador de UI y ensamblar control nativo ft.Tabs
    ui_controller = LSPUIController(page, data_manager, vision_service, trainer)
    tabs = build_main_app_tabs(ui_controller)

    # Banner superior permanente de estado (Visible de inmediato)
    status_banner = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.INFO_OUTLINE, color="#4A90E2", size=20),
            ui_controller.training_progress,
            ui_controller.status_text,
        ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
        bgcolor="#EBF4FF",
        border=ft.Border.all(1, "#D1E4F8"),
        border_radius=8,
        padding=ft.Padding(12, 8, 12, 8),
        margin=ft.Margin(0, 0, 0, 4)
    )

    # Construir grilla principal de la aplicación con estética escolar
    page.add(
        ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.SCHOOL, color="#4A90E2", size=32),
                ft.Text("TRADUCTOR DE LENGUA DE SEÑAS PERUANA (LSP)", size=20, weight=ft.FontWeight.BOLD, color="#1A365D"),
                ft.Container(
                    content=ft.Text("Módulo Docente", size=11, weight=ft.FontWeight.W_600, color="#4A90E2"),
                    bgcolor="#EBF4FF",
                    border_radius=8,
                    padding=ft.Padding(8, 3, 8, 3),
                    border=ft.Border.all(1, "#D1E4F8")
                )
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
            status_banner,
            tabs
        ], expand=True, spacing=6)
    )

    # Apagado coordinado en on_disconnect para detener bucles infinitos
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