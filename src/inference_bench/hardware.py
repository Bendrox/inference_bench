"""Description de la machine de bench : GPU (via nvidia-smi) et OS.

Tout ce qui répond à « sur quoi ce run a tourné » vit ici, pour que run.yaml
reste la trace complète et reproductible d'une expérience.
"""
import platform
import subprocess


def is_wsl() -> bool:
    """Sous WSL, le noyau Linux tourne sur Windows (release '...-microsoft-standard-WSL2')."""
    return "microsoft" in platform.release().lower()


def os_tag() -> str:
    """Sous-dossier de results/ : 'wsl' ou 'linux', pour ne pas mélanger les deux environnements."""
    return "wsl" if is_wsl() else "linux"


def get_gpu_info() -> dict:
    """Infos du 1er GPU via nvidia-smi (n'initialise pas CUDA dans ce process).
    Si nvidia-smi est absent ou échoue, retourne {"name": "unknown"} sans planter le bench."""
    cmd = ["nvidia-smi",
           "--query-gpu=name,memory.total,driver_version,enforced.power.limit,power.default_limit",
           "--format=csv,noheader,nounits"]
    try:
        sortie = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        name, mem, driver, power, power_default = [x.strip() for x in sortie.splitlines()[0].split(",")]
        return {"name": name, "memory_total_mib": int(mem), "driver": driver,
                "power_limit_w": float(power), "power_default_w": float(power_default)}
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return {"name": "unknown"}


def get_os_info() -> dict:
    """OS de la machine de bench, WSL signalé explicitement."""
    try:
        distro = platform.freedesktop_os_release().get("PRETTY_NAME", "unknown")
    except OSError:
        distro = "unknown"
    wsl = is_wsl()
    return {"host": "Windows + WSL" if wsl else platform.system(),
            "distro": distro, "kernel": platform.release(), "wsl": wsl}
