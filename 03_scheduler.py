import schedule
import time
import subprocess
import sys
from datetime import datetime

def lancer_mise_a_jour():
    print(f"\n[{datetime.now().strftime('%d/%m/%Y %H:%M')}] Lancement automatique...")
    
    resultat = subprocess.run(
        [sys.executable, "02_update_data.py"],
        capture_output=True,
        text=True
    )
    
    if resultat.stdout:
        print(resultat.stdout)
    if resultat.stderr:
        print("Erreur :", resultat.stderr)
    
    print("Mise à jour terminée.")

# Chaque jour à 18h00 (après clôture S&P500)
schedule.every().day.at("18:00").do(lancer_mise_a_jour)

# Pour le Bitcoin (24h/24) : aussi à minuit
schedule.every().day.at("00:00").do(lancer_mise_a_jour)

print("=" * 50)
print("Planificateur démarré !")
print("Mises à jour : 18h00 et 00h00 chaque jour")
print("Appuie sur Ctrl+C pour arrêter")
print("=" * 50)

while True:
    schedule.run_pending()
    time.sleep(60)