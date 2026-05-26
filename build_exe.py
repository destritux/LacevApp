import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command):
    print(f"Executando: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Erro ao executar o comando. Código de retorno: {result.returncode}")
        return False
    return True

def main():
    print("=== LacevApp - Compilador de Executável (Cross-Platform) ===")
    
    # 1. Verificar instalação do PyInstaller no ambiente atual
    try:
        import PyInstaller
        print("PyInstaller detectado com sucesso.")
    except ImportError:
        print("PyInstaller não está instalado no ambiente Python atual.")
        install = input("Deseja instalar o PyInstaller agora usando pip? (s/n): ").strip().lower()
        if install == 's':
            success = run_command([sys.executable, "-m", "pip", "install", "pyinstaller"])
            if not success:
                print("Falha ao instalar o PyInstaller. Abortando.")
                sys.exit(1)
        else:
            print("PyInstaller é necessário para compilar o executável. Abortando.")
            sys.exit(1)

    # 2. Configurar separador de dados do PyInstaller de acordo com o Sistema Operacional
    current_os = platform.system()
    print(f"Sistema Operacional Detectado: {current_os}")
    
    # Separador de caminhos do PyInstaller: ';' no Windows, ':' no Linux/macOS
    sep = ';' if current_os == 'Windows' else ':'
    
    assets_dir = Path("assets")
    if not assets_dir.exists() or not assets_dir.is_dir():
        print("Erro: A pasta 'assets' não foi encontrada no diretório atual.")
        sys.exit(1)

    # 3. Construir o comando do PyInstaller
    # --onefile: Empacota em um único executável
    # --windowed: Não abre console CMD em segundo plano (específico para GUI)
    # --add-data: Inclui a pasta de assets (imagens/ícones) dentro do executável
    pyinstaller_cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=LacevApp",
        f"--add-data=assets{sep}assets",
        "main.py"
    ]
    
    # Se o ícone existir, adiciona o argumento do ícone
    icon_path = assets_dir / "lacev-App.ico"
    if icon_path.exists():
        pyinstaller_cmd.append(f"--icon={icon_path}")

    print("\nIniciando a compilação do executável com o PyInstaller...")
    success = run_command(pyinstaller_cmd)
    
    if success:
        print("\n=======================================================")
        print("Executável gerado com sucesso!")
        print("Você poderá encontrar o arquivo finalizado em:")
        print(f" -> {Path('dist').resolve()}")
        print("=======================================================")
    else:
        print("\nOcorreu um erro durante a compilação do executável.")
        sys.exit(1)

if __name__ == "__main__":
    main()
