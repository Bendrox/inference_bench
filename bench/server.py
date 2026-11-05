import os, signal, subprocess, time, urllib.request, urllib.error
from pathlib import Path


from pathlib import Path
import subprocess

def start_server(serve_yaml: Path, log_path: Path) -> subprocess.Popen:
    """
    Lance un serveur vLLM en arrière-plan avec config YAML
    et journalise sa sortie dans un fichier de log.
    
    '-> subprocess.Popen' : indique que la fonction retourne un objet qui sert de "télécommande" 
    pour interagir avec le processus lancé permettant de le surveiller, de l'arrêter, ...
    """
    # Ouvre le fichier de log en mode écriture ("w") pour y stocker les flux du serveur.
    log = open(log_path, "w")
    
    # Lance le processus externe de manière asynchrone (le script Python continue son exécution).
    return subprocess.Popen(
        ["vllm", "serve", "--config", str(serve_yaml)],  # Commande système exécutée dans le terminal
        stdout=log,                                      # Redirige la sortie standard (les messages normaux) vers le fichier de log
        stderr=subprocess.STDOUT,                        # Fusionne le flux d'erreurs (stderr) avec stdout : 
                                                         # tous les messages d'erreur sont ainsi écrits dans le même fichier de log
        start_new_session=True                           # Isole le serveur dans un nouveau groupe de processus (le rend indépendant du script parent) et Gérer proprement les sous-processus cachés de vLLM
    )
    
def wait_ready(proc: subprocess.Popen, port: int, timeout: int = 600) -> None:
    debut=time.time()
    while time.time() - debut < timeout:
        if proc.poll() is not None:
            raise RuntimeError("Le serveur a planté, voir server.log")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    raise TimeoutError(f"Serveur pas prêt après {timeout}s")


def stop_server(proc: subprocess.Popen) -> None:
    """
    Arrête proprement un serveur lancé en arrière-plan ainsi que tout son groupe de processus.
    1. Demande un arrêt gracieux (SIGTERM) et attend jusqu'à 60 secondes.
    2. Si le serveur ne s'arrête pas à temps (Timeout), force l'arrêt (SIGKILL).
    """
    try:
        # Demande un arrêt propre (SIGTERM) à tout le groupe de processus.
        # On utilise 'os.killpg' (et non 'os.kill') car le serveur a été lancé avec 
        # 'start_new_session=True', ce qui crée un groupe indépendant. Cela permet 
        # d'arrêter aussi tous les sous-processus éventuels lancés par vLLM.
        os.killpg(proc.pid, signal.SIGTERM)
        
    except ProcessLookupError:
        # Si le processus n'existe déjà plus dans le système, 
        # on s'arrête là (rien à faire).
        return                                           

    try:
        # Étape 2 : On patiente (maximum 60 secondes) pour laisser le temps 
        # au serveur de nettoyer ses ressources et de se fermer proprement.
        proc.wait(timeout=60)
        
    except subprocess.TimeoutExpired:
        # Si le délai de 60 secondes est dépassé et que le serveur bloque toujours,
        # on passe à la manière forte : SIGKILL (arrêt immédiat et inévitable).
        os.killpg(proc.pid, signal.SIGKILL)
        
        # On s'assure que le système a bien terminé le processus avant de continuer.
        proc.wait()