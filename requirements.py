import os
import venv
import sys
import subprocess
from time import sleep

def create_venv():
    venv_dir_name = ".venv"
    venv_dir = os.path.abspath(venv_dir_name)
    
    print(f"\033[34m[1/2] Creating virtual environment in: {venv_dir}\033[0m")

    try:
        # Create the virtual environment
        # symlinks=False is generally safer on Windows to avoid permission errors
        venv.create(venv_dir, with_pip=True, clear=True, symlinks=(os.name != 'nt'))
        print(f"\033[32mSuccessfully created virtual environment: {venv_dir_name}\033[0m")

        # Determine the path to the python executable inside the new venv
        if sys.platform == "win32":
            python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            python_executable = os.path.join(venv_dir, "bin", "python")

        return python_executable

    except Exception as e:
        print(f"\033[31mError creating venv: {e}\033[0m")
        sys.exit(1)


def install_requirements(python_executable):
    if not os.path.exists("requirements.txt"):
        print(f"\033[31mError: 'requirements.txt' not found in {os.getcwd()}\033[0m")
        return

    print(f"\n\033[34m[2/2] Installing libraries using: {python_executable}\033[0m")
    
    print("Starting installation in", end=' ')
    for i in range(3, 0, -1):
        print(f"{i}...", end=' ', flush=True)
        sleep(0.5)
    print("\n")

    try:
        cmd = [python_executable, "-m", "pip", "install", "-r", "requirements.txt"]

        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              bufsize=1, universal_newlines=True) as p:
            # Print output line-by-line in real time
            for line in p.stdout:
                print(f"\033[36m{line.strip()}\033[0m")

        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)
        else:
            print(f"\n\033[32mInstallation successful!\033[0m")
            print(f"\nTo activate this environment, run:")
            if sys.platform == "win32":
                print(f"    .venv\\Scripts\\activate")
            else:
                print(f"    source .venv/bin/activate")

    except Exception as e:
        print(f"\n\033[31mInstallation failed: {e}\033[0m")


if __name__ == "__main__":
    venv_python = create_venv()
    install_requirements(venv_python)