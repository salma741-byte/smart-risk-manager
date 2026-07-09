import schedule
import time
import subprocess
import sys
from datetime import datetime

def lancer_mise_a_jour():
    print(f"\n[{datetime.now().strftime('%d/%m/%Y %H:%M')}] Lancement automatique...")

    # Etape 1 : mise a jour des donnees
    print("  Mise a jour des donnees...")
    r1 = subprocess.run(
        [sys.executable, "02_update_data.py"],
        capture_output=True, text=True
    )
    if r1.stdout: print(r1.stdout)
    if r1.stderr: print("Erreur update :", r1.stderr)

    # Etape 2 : re-entrainer le modele
    print("  Re-entrainement du modele...")
    r2 = subprocess.run(
        [sys.executable, "04_model.py"],
        capture_output=True, text=True
    )
    if r2.stdout: print(r2.stdout)
    if r2.stderr: print("Erreur modele :", r2.stderr)

    print(f"  Cycle termine a {datetime.now().strftime('%H:%M')}")

# S&P500 + VIX : apres cloture a 18h00
schedule.every().day.at("18:00").do(lancer_mise_a_jour)

# Bitcoin : tourne 24h/24 → mise a jour a minuit aussi
schedule.every().day.at("00:00").do(lancer_mise_a_jour)

print("=" * 50)
print("Planificateur demarre !")
print("Cycles : 18h00 et 00h00 chaque jour")
print("  1. Mise a jour SQLite")
print("  2. Re-entrainement modele")
print("Appuie sur Ctrl+C pour arreter")
print("=" * 50)

while True:
    schedule.run_pending()
    time.sleep(60)