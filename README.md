# Dictée Qwen

Dictée vocale locale en français pour Linux Mint/Cinnamon (X11) : transcription GPU via Qwen3-ASR, avec repli automatique sur whisper.cpp, overlay flottant et raccourcis clavier globaux pour démarrer/mettre en pause la dictée et copier le résultat. Pour les détails de conception complets, voir le document de design : [`docs/superpowers/specs/2026-10-30-dictee-vocale-globale-design.md`](docs/superpowers/specs/2026-10-30-dictee-vocale-globale-design.md).

## Installation

```bash
./setup.sh
uv run pytest  # vérifie que la logique interne fonctionne avant de lancer l'app
```

`setup.sh` installe les dépendances système (apt), crée l'environnement virtuel avec `uv venv --clear --system-site-packages` (nécessaire pour PyGObject/GTK), puis synchronise les dépendances Python avec `uv sync --dev`. Il propose aussi, en option, de cloner et compiler whisper.cpp avec support CUDA comme backend de secours.

Avec `uv`, il n'y a pas besoin de `source venv/bin/activate` comme avec un `venv` pip classique : chaque commande peut être lancée directement via `uv run <commande>`, qui active automatiquement l'environnement du projet. Si vous préférez une activation persistante façon pip, `source .venv/bin/activate` fonctionne aussi (le dossier s'appelle `.venv`, avec un point, et non `venv`) ; une fois activé, les commandes (`python3`, `pytest`, ...) s'exécutent normalement sans préfixe `uv run`.

## Lancement manuel

```bash
cd /chemin/vers/dictee-qwen  # exécuter depuis la racine du dépôt cloné
uv run python3 dictee_app.py config.toml
```

## Démarrage automatique à l'ouverture de session (Cinnamon)

Les raccourcis clavier globaux (démarrer/arrêter l'enregistrement, pause, copie) sont capturés directement par l'application via X11 — il n'y a donc **rien à configurer** dans *Paramètres système > Clavier*. La seule exigence est que `dictee_app.py` soit lancé automatiquement à l'ouverture de session.

Étapes :

1. Ouvrir *Paramètres système > Démarrage*.
2. Ajouter une entrée de démarrage personnalisée avec :
   - **Nom** : `Dictée Qwen`
   - **Commande** :
     ```bash
     uv run --project /chemin/vers/dictee-qwen python3 /chemin/vers/dictee-qwen/dictee_app.py /chemin/vers/dictee-qwen/config.toml
     ```
     L'option `--project` indique à `uv run` dans quel projet trouver l'environnement virtuel et le `uv.lock`, puisqu'une entrée de démarrage ne s'exécute pas forcément avec le dépôt comme répertoire de travail. **Remplacer les *trois* occurrences de `/chemin/vers/dictee-qwen` ci-dessus par le même chemin absolu réel du dépôt cloné** (après `--project`, dans le chemin du script, et dans le chemin de `config.toml`).
   - **Retard de démarrage** : quelques secondes (pour laisser le serveur X11 et la session graphique s'initialiser complètement).

## Checklist de tests manuels (non automatisés)

- [ ] Démarrage de l'app : la fenêtre flottante apparaît, le modèle se charge sans erreur (`torch.cuda.is_available()` vrai).
- [ ] Raccourci « toggle record » avec la fenêtre minimisée : l'enregistrement démarre.
- [ ] Dicter dans un navigateur avec le curseur dans un champ : le texte apparaît au fur et à mesure.
- [ ] Mettre le focus sur la fenêtre de l'app pendant la dictée : le texte n'est plus injecté, il reste dans la zone de texte de l'app.
- [ ] Bouton « Pause/Reprendre » et raccourci clavier correspondant : même effet dans les deux cas.
- [ ] Bouton « Copier » : le transcript complet se retrouve dans le presse-papiers (`xclip -o -selection clipboard`).
- [ ] Dictée longue (10-15 minutes) : aucune perte de données (comparer avec `/tmp/dictation_live.txt`).
- [ ] Débrancher le micro en cours de dictée : pas de plantage.
- [ ] Arrêter l'app (Ctrl+C ou fermeture de la fenêtre) : `/tmp/dictee_app.pid` est supprimé, `nvidia-smi` ne montre plus le processus en mémoire GPU.
- [ ] Relancer l'app juste après l'avoir arrêtée : pas de message « already running » bloquant à tort.
- [ ] Fermer la fenêtre (bouton X) : elle disparaît mais l'app continue de tourner (vérifier via `ps aux | grep dictee_app` ou `nvidia-smi` que le processus est toujours actif).
- [ ] Clic gauche sur l'icône du plateau système : la fenêtre réapparaît et revient au premier plan.
- [ ] Clic droit sur l'icône du plateau système : le menu affiche les bons libellés selon l'état courant (« Démarrer »/« Arrêter », « Mettre en pause »/« Reprendre » selon que l'app est idle/en enregistrement/en pause).
- [ ] « Quitter » depuis le menu du plateau système : l'app s'arrête proprement (`/tmp/dictee_app.pid` supprimé, plus de processus dans `nvidia-smi`).
