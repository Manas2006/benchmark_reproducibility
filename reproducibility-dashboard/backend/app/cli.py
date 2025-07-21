import typer
import subprocess
import webbrowser
import time
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

app = typer.Typer()
console = Console()


@app.command()
def webui():
    """Launch the Reproducibility Dashboard web interface."""

    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    backend_dir = project_root / "backend"
    frontend_dir = project_root / "frontend"

    console.print(
        Panel.fit(
            "[bold blue]Reproducibility Dashboard[/bold blue]\n"
            "Launching full-stack web interface...",
            border_style="blue",
        )
    )

    # Check if directories exist
    if not backend_dir.exists():
        console.print("[red]Error: Backend directory not found[/red]")
        raise typer.Exit(1)

    if not frontend_dir.exists():
        console.print("[red]Error: Frontend directory not found[/red]")
        raise typer.Exit(1)

    # Start backend
    console.print("[yellow]Starting FastAPI backend...[/yellow]")
    backend_process = None
    try:
        backend_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait a moment for backend to start
        time.sleep(3)

        # Check if backend is running
        try:
            import requests

            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                console.print("[green]✓ Backend started successfully[/green]")
            else:
                console.print("[red]✗ Backend failed to start[/red]")
                raise typer.Exit(1)
        except ImportError:
            console.print(
                "[yellow]Warning: requests not available, skipping backend health check[/yellow]"
            )
        except Exception:
            console.print("[red]✗ Backend failed to start[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error starting backend: {e}[/red]")
        raise typer.Exit(1)

    # Start frontend
    console.print("[yellow]Starting React frontend...[/yellow]")
    frontend_process = None
    try:
        frontend_process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for frontend to start
        time.sleep(5)
        console.print("[green]✓ Frontend started successfully[/green]")

    except Exception as e:
        console.print(f"[red]Error starting frontend: {e}[/red]")
        if backend_process:
            backend_process.terminate()
        raise typer.Exit(1)

    # Open browser
    console.print("[yellow]Opening browser...[/yellow]")
    try:
        webbrowser.open("http://localhost:3000")
        console.print("[green]✓ Browser opened[/green]")
    except Exception as e:
        console.print(
            f"[yellow]Warning: Could not open browser automatically: {e}[/yellow]"
        )
        console.print("[yellow]Please open http://localhost:3000 manually[/yellow]")

    # Display success message
    console.print(
        Panel.fit(
            "[bold green]Dashboard Launched Successfully![/bold green]\n\n"
            "[bold]Access Points:[/bold]\n"
            "• Frontend: [link=http://localhost:3000]http://localhost:3000[/link]\n"
            "• Backend API: [link=http://localhost:8000]http://localhost:8000[/link]\n"
            "• API Docs: [link=http://localhost:8000/docs]http://localhost:8000/docs[/link]\n\n"
            "[bold]Usage:[/bold]\n"
            "1. Configure your experiment in the web interface\n"
            "2. Click 'Generate & Run' to start experiments\n"
            "3. Monitor progress in the 'Runs' tab\n"
            "4. View results in the 'Results' tab\n\n"
            "[bold]To stop the dashboard:[/bold] Press Ctrl+C",
            border_style="green",
        )
    )

    try:
        # Keep the processes running
        while True:
            time.sleep(1)

            # Check if processes are still running
            if backend_process and backend_process.poll() is not None:
                console.print("[red]Backend process stopped unexpectedly[/red]")
                break

            if frontend_process and frontend_process.poll() is not None:
                console.print("[red]Frontend process stopped unexpectedly[/red]")
                break

    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down dashboard...[/yellow]")

        # Terminate processes
        if backend_process:
            backend_process.terminate()
            console.print("[green]✓ Backend stopped[/green]")

        if frontend_process:
            frontend_process.terminate()
            console.print("[green]✓ Frontend stopped[/green]")

        console.print("[green]Dashboard shutdown complete[/green]")


@app.command()
def install():
    """Install dependencies for the reproducibility dashboard."""

    project_root = Path(__file__).parent.parent.parent
    backend_dir = project_root / "backend"
    frontend_dir = project_root / "frontend"

    console.print(
        Panel.fit(
            "[bold blue]Installing Reproducibility Dashboard Dependencies[/bold blue]",
            border_style="blue",
        )
    )

    # Install backend dependencies
    console.print("[yellow]Installing backend dependencies...[/yellow]")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=backend_dir,
            check=True,
        )
        console.print("[green]✓ Backend dependencies installed[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Failed to install backend dependencies: {e}[/red]")
        raise typer.Exit(1)

    # Install frontend dependencies
    console.print("[yellow]Installing frontend dependencies...[/yellow]")
    try:
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        console.print("[green]✓ Frontend dependencies installed[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Failed to install frontend dependencies: {e}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            "[bold green]Installation Complete![/bold green]\n\n"
            "You can now run the dashboard with:\n"
            "[bold]reproducibility-cli webui[/bold]\n"
            "or\n"
            "[bold]repro-cli webui[/bold]",
            border_style="green",
        )
    )


@app.command()
def test():
    """Run tests for the reproducibility dashboard."""

    project_root = Path(__file__).parent.parent.parent
    backend_dir = project_root / "backend"

    console.print(
        Panel.fit(
            "[bold blue]Running Reproducibility Dashboard Tests[/bold blue]",
            border_style="blue",
        )
    )

    # Run backend tests
    console.print("[yellow]Running backend tests...[/yellow]")
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/"], cwd=backend_dir, check=True
        )
        console.print("[green]✓ Backend tests passed[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Backend tests failed: {e}[/red]")
        raise typer.Exit(1)

    console.print(
        Panel.fit("[bold green]All Tests Passed![/bold green]", border_style="green")
    )


if __name__ == "__main__":
    app()
