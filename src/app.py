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
    page.title = "Traductor LSP - Administrador de Entrenamiento y Validación"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 15
    
    if hasattr(page, "window") and page.window:
        page.window.width = 1250
        page.window.height = 850
        page.window.resizable = True
    else:
        page.window_width = 1250
        page.window_height = 850
        page.window_resizable = True

    # Inicializar componentes del Backend
    data_manager = LSPDataManager()
    vision_service = LSPVisionService()
    trainer = LSPTrainer(data_manager)
    
    # Inicializar controlador de UI y ensamblar pestañas principales (Captura/Entrenamiento y Pruebas en Vivo)
    ui_controller = LSPUIController(page, data_manager, vision_service, trainer)
    tabs = build_main_app_tabs(ui_controller)

    # Construir grilla principal de la aplicación
    page.add(
        ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.ACCESSIBILITY_NEW, color=ft.Colors.BLUE, size=32),
                ft.Text("SISTEMA ADMINISTRADOR - TRADUCTOR LSP", size=22, weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.START),
            ft.Divider(height=10, color=ft.Colors.OUTLINE),
            tabs
        ], expand=True, spacing=10)
    )

    # Desconexión y liberación segura de recursos de hardware
    def on_disconnect(e=None):
        ui_controller.close()
        vision_service.close()

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