import os
import sys
import json
import threading
import webbrowser
import asyncio
import atexit
import flet as ft

# Configurar la política nativa de event loop para Windows
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

try:
    import requests
except ImportError:
    requests = None

# Constante de versión actual del software (solucion_scroll_y_auto_update_escritorio.md)
CURRENT_VERSION = "1.0.0"

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
def parse_semantic_version(v_str: str) -> list:
    """Convierte un string de versión como '1.2.0' o 'v1.0' en una lista de enteros para comparación."""
    try:
        clean = str(v_str).strip().lstrip("vV")
        parts = [int(x) for x in clean.split(".") if x.isdigit()]
        while len(parts) < 3:
            parts.append(0)
        return parts
    except Exception:
        return [0, 0, 0]

def check_for_updates_async(page: ft.Page, current_version: str = CURRENT_VERSION):
    """
    Ejecuta en segundo plano la verificación de versión en AWS S3.
    Si se detecta una nueva versión en version.json, despliega un diálogo
    de confirmación asíncrono para descargar automáticamente la nueva versión (.exe).
    """
    def _worker():
        try:
            bucket_name = os.getenv("AWS_BUCKET_NAME", "traductor-lsp-modelos-colegio")
            s3_url = os.getenv("S3_VERSION_URL", f"https://{bucket_name}.s3.amazonaws.com/version.json")
            
            data = None
            # 1. Consulta HTTP directa con requests
            if requests is not None:
                try:
                    resp = requests.get(s3_url, timeout=4)
                    if resp.status_code == 200:
                        data = resp.json()
                except Exception:
                    pass

            # 2. Si falló la consulta HTTP pública, intentar fallback mediante boto3
            if data is None:
                try:
                    import boto3
                    ak = os.getenv("AWS_ACCESS_KEY_ID")
                    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
                    region = os.getenv("AWS_REGION", "us-east-1")
                    if ak and sk and bucket_name:
                        s3_client = boto3.client(
                            "s3",
                            aws_access_key_id=ak,
                            aws_secret_access_key=sk,
                            region_name=region
                        )
                        s3_resp = s3_client.get_object(Bucket=bucket_name, Key="version.json")
                        content = s3_resp["Body"].read().decode("utf-8")
                        data = json.loads(content)
                except Exception:
                    pass

            if not data or not isinstance(data, dict):
                print("[AUTO-UPDATER] No se detectó versión remota en S3 o archivo inexistente.")
                return

            latest_version = str(data.get("latest_version", current_version)).strip()
            download_url = data.get("download_url", "")
            mandatory = bool(data.get("mandatory", False))
            changelog = data.get("changelog", "Mejoras generales en la interfaz y reconocimiento de señas.")

            if parse_semantic_version(latest_version) > parse_semantic_version(current_version):
                print(f"[AUTO-UPDATER] ¡Nueva versión detectada! v{latest_version} (Actual: v{current_version})")

                def _show_update_modal():
                    try:
                        def _on_download(e):
                            try:
                                if download_url:
                                    if hasattr(page, "launch_url"):
                                        page.launch_url(download_url)
                                    else:
                                        webbrowser.open(download_url)
                            except Exception as dl_err:
                                print(f"[AUTO-UPDATER] Error al abrir enlace de descarga: {dl_err}")
                                if download_url:
                                    webbrowser.open(download_url)

                            if hasattr(page, "close"):
                                page.close(dlg_update)
                            else:
                                dlg_update.open = False
                                page.update()

                        def _on_cancel(e):
                            if hasattr(page, "close"):
                                page.close(dlg_update)
                            else:
                                dlg_update.open = False
                                page.update()

                        actions = [
                            ft.Button(
                                content="Descargar e Instalar (.exe)",
                                icon=ft.Icons.DOWNLOAD_ROUNDED,
                                bgcolor="#0A66C2",
                                color=ft.Colors.WHITE,
                                on_click=_on_download
                            )
                        ]
                        if not mandatory:
                            actions.append(
                                ft.Button(
                                    content="Recordar Más Tarde",
                                    bgcolor="#F1F5F9",
                                    color="#1A365D",
                                    on_click=_on_cancel
                                )
                            )

                        dlg_update = ft.AlertDialog(
                            modal=mandatory,
                            title=ft.Row([
                                ft.Icon(ft.Icons.SYSTEM_UPDATE_ROUNDED, color="#0A66C2", size=24),
                                ft.Text("Actualización Disponible", weight=ft.FontWeight.BOLD, size=16, color="#1A365D")
                            ], spacing=8),
                            content=ft.Container(
                                width=440,
                                content=ft.Column([
                                    ft.Text(f"Se ha detectado una nueva versión del sistema Traductor LSP para Windows.", size=13, color="#2D3748"),
                                    ft.Container(height=4),
                                    ft.Container(
                                        content=ft.Row([
                                            ft.Column([
                                                ft.Text("Versión Instalada:", size=11, color="#64748B"),
                                                ft.Text(f"v{current_version}", size=13, weight=ft.FontWeight.BOLD, color="#1A365D"),
                                            ], spacing=1),
                                            ft.Icon(ft.Icons.ARROW_FORWARD, color="#0A66C2", size=18),
                                            ft.Column([
                                                ft.Text("Nueva Versión:", size=11, color="#64748B"),
                                                ft.Text(f"v{latest_version}", size=13, weight=ft.FontWeight.BOLD, color="#0A66C2"),
                                            ], spacing=1),
                                        ], alignment=ft.MainAxisAlignment.SPACE_AROUND, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                        padding=10,
                                        bgcolor="#F1F5F9",
                                        border_radius=8
                                    ),
                                    ft.Container(height=4),
                                    ft.Text("Novedades y Mejoras:", size=12, weight=ft.FontWeight.BOLD, color="#1A365D"),
                                    ft.Container(
                                        content=ft.Text(changelog, size=11, color="#475569"),
                                        padding=ft.Padding(8, 6, 8, 6),
                                        bgcolor="#F8FAFC",
                                        border=ft.Border.all(1, "#E2E8F0"),
                                        border_radius=6
                                    )
                                ], spacing=6, tight=True)
                            ),
                            actions=actions,
                            actions_alignment=ft.MainAxisAlignment.END
                        )

                        if hasattr(page, "open"):
                            page.open(dlg_update)
                        else:
                            page.dialog = dlg_update
                            dlg_update.open = True
                            page.update()

                    except Exception as modal_err:
                        print(f"[AUTO-UPDATER] Error al desplegar diálogo de actualización: {modal_err}")

                if hasattr(page, "run_thread"):
                    page.run_thread(_show_update_modal)
                else:
                    _show_update_modal()
            else:
                print(f"[AUTO-UPDATER] Sistema al día (v{current_version}).")

        except Exception as check_ex:
            print(f"[AUTO-UPDATER] Excepción en comprobación de versión: {check_ex}")

    threading.Thread(target=_worker, daemon=True).start()

def main(page: ft.Page):
    page.title = "Traductor LSP - Panel Escolar de Entrenamiento y Validación"
    # Rediseño visual escolar: Tema claro, fondo blanco-azulado y scroll automático
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = "#F4F8FA"
    page.padding = ft.Padding(16, 12, 16, 12)
    page.scroll = ft.ScrollMode.AUTO
    
    # Maximización nativa de ventana en Flet (solucion_definitiva_aws_y_comandos_voz.md)
    page.window_maximized = True
    page.window_resizable = True
    if hasattr(page, "window") and page.window:
        try:
            page.window.maximized = True
            page.window.resizable = True
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

    # Iniciar verificación asíncrona de versiones en AWS S3 sin congelar la interfaz
    check_for_updates_async(page)

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