"""
Web server CLI command for Face Sorter.

This module provides a Click command to start the FastAPI web server
for the Face Sorter UI.
"""

import click

import uvicorn

from face_sorter.config import get_settings


@click.command()
@click.option("--host", default=None, type=str, help="Server host")
@click.option("--port", default=None, type=int, help="Server port")
@click.option("--dev", is_flag=True, help="Enable development mode")
def web(host, port, dev):
    """
    Start the Face Sorter web UI.

    This command starts the FastAPI web server that provides the
    graphical user interface for face sorting operations.

    Example:
        face-sorter web
        face-sorter web --host 0.0.0.0 --port 8080 --dev
    """
    settings = get_settings()

    # Use provided values or fall back to settings
    host = host or settings.ui_host
    port = port or settings.ui_port
    reload = dev or settings.ui_reload
    log_level = settings.ui_log_level

    click.echo(f"Starting Face Sorter Web UI...")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Reload: {reload}")
    click.echo(f"  Log level: {log_level}")
    click.echo(f"  URL: http://{host}:{port}")
    click.echo("")

    uvicorn.run(
        "face_sorter.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )