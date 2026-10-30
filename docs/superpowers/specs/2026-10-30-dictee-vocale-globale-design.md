# Dictée vocale globale sous Linux Mint — Design

## Contexte et objectif

Outil de dictée vocale locale (français), sans cloud, exploitant l'accélération
GPU CUDA d'une RTX 4060 Laptop (8 Go VRAM) sous Linux Mint (Cinnamon, X11).

Objectifs :
- Injecter le texte transcrit directement au niveau du curseur, dans
  n'importe quelle application (navigateur, éditeur, terminal, Duolingo…).
- Supporter des dictées longues (10–15 min) sans perte de données.
- Donner un retour visuel en temps réel pendant la dictée.
- Fonctionner entièrement en local avec accélération GPU.

Environnement : Linux Mint / Cinnamon / X11 (pas de support Wayland).

## Architecture globale

Un seul processus (`dictee_app.py`), lancé une fois (manuellement ou au
démarrage de session) et qui reste actif en arrière-plan, GUI visible ou
minimisée. Pas de processus séparés, pas de socket inter-process : tout
tourne dans un seul process avec des threads internes communiquant par
queues Python thread-safe.

Composants internes :

- **Listener de raccourcis globaux** (X11, `python-xlib`, `XGrabKey`) :
  capture les raccourcis configurés indépendamment du focus ou de l'état
  minimisé de la fenêtre. Ne fonctionne que sous X11 (pas Wayland).
- **Capture audio** (`sounddevice`) : alimente un buffer audio en continu
  quand l'enregistrement est actif.
- **VAD + transcription** : découpe le flux audio en segments par
  détection de silence (Silero VAD), envoie chaque segment complet au
  backend ASR actif, récupère le texte.
- **GUI (GTK3 / PyGObject)** : fenêtre flottante affichant le transcript en
  direct, avec boutons de contrôle. Mise à jour via `GLib.idle_add` depuis
  les autres threads.
- **Injecteur système** : à chaque segment transcrit, détecte la fenêtre
  active et décide d'injecter le texte ou non.
- **Fichier de secours** `/tmp/dictation_live.txt` : chaque segment y est
  ajouté immédiatement (append), pour récupération en cas de crash.

Un verrou d'instance (fichier PID dans `$XDG_RUNTIME_DIR` ou `/tmp`) empêche
le lancement de deux instances simultanées.

### Diagramme de flux

```
Raccourci clavier global (X11)  ─┐
Bouton GUI (Start/Stop/Pause)   ─┼─► file d'action ─► contrôleur d'état
                                  │                       │
                                  │                       ▼
                             Capture audio (sounddevice) ─► VAD (Silero)
                                                             │
                                                   segment de parole complet
                                                             │
                                                             ▼
                                              Backend ASR (Qwen3-ASR / whisper.cpp)
                                                             │
                                                        texte du segment
                                              ┌──────────────┼──────────────┐
                                              ▼              ▼              ▼
                                      Zone de texte GUI   Fichier de    Injection
                                      (toujours)          secours       (si fenêtre
                                                           (toujours)    active ≠ GUI)
```

## Backends ASR

**Backend principal : Qwen3-ASR**
- Modèle par défaut : `Qwen/Qwen3-ASR-1.7B-hf`, chargé via `transformers`
  (`AutoModelForMultimodalLM`, `dtype=torch.bfloat16`, `device="cuda"`).
  Empreinte VRAM ~3.4 Go, confortable sur les 8 Go disponibles.
- Alternative plus légère : `Qwen/Qwen3-ASR-0.6B-hf`, activable via
  `config.toml` (utile si VRAM contrainte ou besoin de latence plus faible).
- Langue forcée en français (`fr`).

**Backend de secours : whisper.cpp (CUDA)**
- Compilé avec `GGML_CUDA=1`, lancé en mode serveur (`whisper-server`,
  `large-v3-turbo`) au démarrage de l'app si sélectionné — le modèle reste
  chargé en VRAM, communication via HTTP local.
- Un seul backend actif à la fois (`config.toml`: `backend = "qwen-1.7b" |
  "qwen-0.6b" | "whisper"`), pas de double charge VRAM.

**Interface commune** (`asr_backends/base.py`) : `transcribe(audio: np.ndarray) -> str`,
implémentée par `qwen_backend.py` et `whisper_backend.py`.

## VAD et segmentation

- Silero VAD (CPU), découpe le flux audio continu en segments dès qu'une
  pause de parole dépasse un seuil configurable (défaut : 500 ms de
  silence).
- Paramètres réglables dans `config.toml` : seuil de silence, durée
  min/max d'un segment (éviter des segments trop courts ou trop longs).
- Chaque segment terminé est mis dans une queue vers le thread de
  transcription.

## GUI

- Fenêtre GTK3 flottante, toujours au premier plan (`set_keep_above(True)`),
  positionnée dans un coin de l'écran.
- Zone de texte scrollable affichant le transcript cumulé de la session en
  cours, mis à jour segment par segment.
- Indicateur de statut visuel : rouge = enregistrement, jaune = pause,
  gris = idle.
- Boutons : **Start/Stop**, **Pause/Reprendre**, **Copier** (copie tout le
  transcript actuel dans le presse-papiers sans l'injecter).
- Peut être minimisée sans arrêter l'app ni décharger le modèle de la
  VRAM — seule la fermeture complète de l'app (quitter) décharge le modèle.

## Raccourcis clavier

- Captés en interne par l'app via `python-xlib` (`XGrabKey`), donc
  fonctionnels même fenêtre minimisée ou sans focus. Aucune configuration
  dans *Paramètres système > Clavier* n'est nécessaire.
- Trois raccourcis par défaut, modifiables dans `config.toml` :
  - Toggle Start/Stop de l'enregistrement.
  - Toggle Pause/Reprendre.
  - Copier le transcript actuel dans le presse-papiers.
- Les raccourcis clavier et les boutons GUI déclenchent exactement les
  mêmes actions internes — totalement interchangeables à tout moment.

## Logique d'injection

À chaque segment transcrit :

1. Le texte est toujours ajouté à la zone de texte de la GUI et au fichier
   de secours `/tmp/dictation_live.txt`, quelle que soit la fenêtre active.
2. `xdotool getactivewindow` donne l'identifiant de la fenêtre active au
   moment présent.
3. Cet identifiant est comparé au titre/WM_CLASS fixe de la fenêtre de
   l'app (défini par la GUI). S'il correspond → pas d'injection, le texte
   reste uniquement dans la GUI (l'utilisateur utilisera le bouton
   **Copier** pour le récupérer ailleurs).
4. Sinon → le texte du segment est placé dans le presse-papiers
   (`xclip -selection clipboard`), puis `xdotool key --window <id> ctrl+v`
   simule le collage directement dans cette fenêtre précise (ciblée par id,
   pas juste "la fenêtre active" au moment du collage, qui pourrait avoir
   changé entre-temps).
5. Un espace est inséré entre deux segments injectés successifs, sauf si
   le segment suivant commence par un signe de ponctuation.
6. Le presse-papiers n'est pas sauvegardé/restauré : à la fin d'une
   session de dictée, il contient le dernier segment injecté. Limitation
   acceptée (décision utilisateur).

## Gestion d'erreurs et nettoyage

- Au démarrage, vérifications bloquantes avec message clair en cas
  d'échec : présence de `xdotool`/`xclip`, device audio d'entrée
  disponible, `torch.cuda.is_available()` vrai, session X11 (avertissement
  si `$XDG_SESSION_TYPE != x11`, car `xdotool`/`XGrabKey` ne fonctionnent
  pas sous Wayland).
- Chargement du modèle ASR dans un `try/except` : en cas d'OOM VRAM,
  message clair suggérant de basculer sur le modèle 0.6B ou sur le backend
  whisper.cpp.
- Échec de transcription d'un segment : loggé, n'interrompt pas la
  session en cours (l'enregistrement continue).
- Échec d'injection (fenêtre introuvable, `xdotool` en erreur) : loggé en
  avertissement ; le texte reste disponible dans la GUI et le fichier de
  secours — pas de crash.
- À l'arrêt (`SIGINT`/`SIGTERM` ou fermeture GUI) : arrêt du flux audio,
  `torch.cuda.empty_cache()`, arrêt du serveur `whisper-server` si lancé,
  suppression du fichier de verrou PID. Le fichier
  `/tmp/dictation_live.txt` n'est **pas** supprimé automatiquement (garde
  la trace pour récupération manuelle).

## Structure du projet

```
dictee-qwen/
├── dictee_app.py            # point d'entrée unique (GUI + logique + hotkeys)
├── dictee/
│   ├── audio_capture.py     # capture micro (sounddevice)
│   ├── vad_segmenter.py     # découpage par silence (Silero VAD)
│   ├── asr_backends/
│   │   ├── base.py          # interface commune transcribe(audio) -> str
│   │   ├── qwen_backend.py
│   │   └── whisper_backend.py
│   ├── injector.py          # détection fenêtre active + xclip/xdotool
│   ├── hotkeys.py           # grab clavier global (python-xlib)
│   └── gui.py               # fenêtre GTK3
├── config.toml               # backend choisi, langue, raccourcis, seuils VAD
├── setup.sh
├── requirements.txt
└── README.md
```

## Installation (`setup.sh`)

- Vérifie/installe les paquets système : `ffmpeg`, `xdotool`, `xclip`,
  `portaudio19-dev`, `python3-gi`/`gir1.2-gtk-3.0`, headers X11 pour
  `python-xlib`.
- Crée le `venv`, installe `torch` (roue `cu121` ou adaptée au driver
  NVIDIA détecté), `transformers`, `accelerate`, `sounddevice`, `numpy`,
  `silero-vad`, `python-xlib`, `PyGObject`.
- Propose en option de cloner et compiler `whisper.cpp` avec
  `GGML_CUDA=1` (seulement si le backend de secours est souhaité).
- Les poids `Qwen3-ASR-1.7B-hf` sont téléchargés au premier lancement par
  `transformers` (pas besoin de script dédié).

Le README documente comment ajouter `dictee_app.py` au démarrage
automatique de la session Cinnamon, pour qu'il tourne en arrière-plan dès
l'ouverture de session (nécessaire pour que les raccourcis globaux soient
disponibles).

## Stratégie de test

**Tests unitaires (pytest)**, avec mocks pour tout ce qui touche au
matériel/X11 :
- `vad_segmenter.py` : découpage correct de buffers audio synthétiques
  selon les seuils de silence configurés.
- `injector.py` : logique de jointure des segments (espace/ponctuation),
  et logique de décision "fenêtre active = GUI ou pas" (`xdotool` mocké).
- Chargement/validation de `config.toml` (valeurs par défaut, erreurs sur
  valeurs invalides).
- Contrat `asr_backends` : chaque backend respecte
  `transcribe(audio) -> str` (vérifié avec un backend factice en test).

**Tests d'intégration manuels** (checklist dans le README, non
automatisés), car nécessitant le vrai matériel :
- Capture micro réelle et inférence GPU réelle de bout en bout.
- Injection réelle dans différentes applications (navigateur, terminal,
  éditeur de texte).
- Raccourcis clavier globaux, y compris fenêtre minimisée.
- Dictée longue (10–15 min) sans perte de données ni fuite mémoire.
- Comportement en cas de débranchement du micro en cours de dictée.

## Décisions actées pendant le brainstorming

- Backend ASR principal : Qwen3-ASR (1.7B par défaut, 0.6B en option),
  whisper.cpp CUDA en secours — un seul actif à la fois.
- Pas de streaming "mot à mot" natif : découpage par VAD + transcription
  par segment, affiché/injecté au fur et à mesure.
- Modèle chargé une seule fois, processus persistant (pas de rechargement
  à chaque dictée).
- CLI et GUI fusionnés en un seul processus ; raccourcis clavier captés
  directement par l'app en X11, pas de configuration Cinnamon nécessaire.
- Pas de socket IPC : tout en interne via threads/queues.
- Détection automatique de la cible d'injection par fenêtre active (pas
  par méthode de lancement).
- Pas de sauvegarde/restauration du presse-papiers original (accepté
  comme limitation).
