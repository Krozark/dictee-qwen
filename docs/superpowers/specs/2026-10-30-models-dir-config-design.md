# Dossier de modèles configurable — Design

## Contexte et objectif

Le projet télécharge des modèles depuis trois sources différentes, chacune
avec son propre cache par défaut :
- Qwen3-ASR (via `transformers`/HuggingFace) → `~/.cache/huggingface`
- Silero VAD (via `torch.hub`) → `~/.cache/torch/hub`
- whisper.cpp (backend de secours) → `vendor/whisper.cpp/models/`

L'utilisateur veut pouvoir centraliser ces téléchargements dans un dossier
de son choix, configurable via `config.toml`.

## Nouveau champ de configuration

Ajout d'un champ optionnel à `Config` (`dictee/config.py`) :

```python
models_dir: str | None = None
```

- **Si absent/non défini** : comportement actuel inchangé — chaque outil
  garde son cache par défaut (HuggingFace, torch.hub, `vendor/whisper.cpp/models/`).
- **Si défini** (ex: `models_dir = "~/mes-modeles"`) : devient un dossier
  racine avec des sous-dossiers automatiques par outil, pour éviter que
  les différents caches ne se mélangent :
  - `${models_dir}/huggingface` → cache Qwen3-ASR
  - `${models_dir}/torch-hub` → cache Silero VAD
  - `${models_dir}/whisper-cpp` → modèles whisper.cpp

Le `~` dans la valeur TOML est étendu une seule fois, au chargement de la
config (`load_config`, dans `dictee/config.py`), via `os.path.expanduser`,
de sorte que tout le reste du code reçoit un chemin déjà prêt à l'emploi
(TOML ne fait pas d'expansion de `~` nativement).

Aucune validation d'existence n'est faite au chargement de la config
(cohérent avec le style actuel de `validate_config`, qui ne vérifie pas non
plus que `backup_file` est inscriptible) — les bibliothèques concernées
(`transformers`, `torch.hub`) créent elles-mêmes les dossiers nécessaires
au moment du téléchargement.

## Câblage côté Python

### `dictee/asr_backends/qwen_backend.py`

- Le type `Loader` et la fonction `default_loader` reçoivent un second
  paramètre : `default_loader(model_id: str, models_dir: str | None = None)`.
  Si `models_dir` est fourni, il est passé comme
  `cache_dir=f"{models_dir}/huggingface"` à `AutoProcessor.from_pretrained(...)`
  et `AutoModelForMultimodalLM.from_pretrained(...)`.
- `QwenBackend.__init__` reçoit un nouveau paramètre
  `models_dir: str | None = None`, transmis à l'appel `loader(model_id, models_dir)`.
- Les faux `loader` à un seul argument dans `tests/test_qwen_backend.py`
  sont mis à jour pour accepter ce second argument (`models_dir=None` par
  défaut) — changement mécanique, sans impact sur la logique des tests
  existants.

### `dictee/silero_vad.py`

- `load_silero_vad(sample_rate: int = 16000, models_dir: str | None = None)` :
  si `models_dir` est fourni, appelle `torch.hub.set_dir(f"{models_dir}/torch-hub")`
  avant `torch.hub.load(...)`.
- Cette fonction est déjà explicitement non couverte par les tests
  automatisés (vérifiée manuellement) ; aucun changement de test requis.

### `dictee_app.py`

- `build_asr_backend(config)` transmet `models_dir=config.models_dir` à
  `QwenBackend(...)`.
- L'appel à `load_silero_vad(...)` transmet aussi `models_dir=config.models_dir`.

## whisper.cpp et `setup.sh`

whisper.cpp est un outil externe (C++), son dossier de modèles ne peut pas
être redirigé via un paramètre de fonction Python. Approche :

- Dans `setup.sh`, après `uv sync --dev`, si l'utilisateur choisit de
  construire whisper.cpp, le script lit `models_dir` depuis `config.toml` via :
  ```bash
  uv run python3 -c "from dictee.config import load_config; print(load_config('config.toml').models_dir or '')"
  ```
- Si une valeur non vide est retournée, le script crée
  `${models_dir}/whisper-cpp/` puis remplace `vendor/whisper.cpp/models`
  par un lien symbolique vers ce dossier (`ln -s`), **avant** de lancer
  `./models/download-ggml-model.sh large-v3-turbo`. Le script de
  téléchargement de whisper.cpp fonctionne alors sans modification,
  écrivant en réalité dans le dossier centralisé via le lien symbolique.
- Si `models_dir` n'est pas renseigné, aucun changement : whisper.cpp
  garde `vendor/whisper.cpp/models/` comme avant.

## Documentation

`config.toml` (fichier de config par défaut livré avec le projet) reçoit
un commentaire expliquant le champ `models_dir`, sur le même modèle que
les commentaires déjà présents pour les autres champs (backend, hotkeys,
VAD, backup_file). Le champ lui-même reste commenté/absent par défaut
(comportement actuel préservé tant que l'utilisateur ne l'active pas).

## Tests

- `dictee/config.py` : tests pour l'expansion du `~` (ex:
  `models_dir = "~/mes-modeles"` devient un chemin absolu après
  `load_config`), et pour le cas par défaut (`models_dir` reste `None`
  si absent du fichier).
- `dictee/asr_backends/qwen_backend.py` : test que `default_loader`
  passe bien `cache_dir=f"{models_dir}/huggingface"` quand `models_dir`
  est fourni (en injectant un faux `AutoProcessor`/`AutoModelForMultimodalLM`,
  suivant le pattern déjà utilisé dans les tests existants), et qu'aucun
  `cache_dir` n'est passé quand `models_dir` est `None`.
- `dictee/silero_vad.py` : non testé automatiquement (cohérent avec
  l'existant — `load_silero_vad` nécessite un vrai `torch.hub`).
- `setup.sh` : vérifié manuellement (script shell, pas de suite de tests
  automatisée pour ce fichier dans le projet).

## Décisions actées pendant le brainstorming

- `models_dir` est optionnel ; absent = comportement actuel inchangé pour
  chaque outil (pas de valeur par défaut forcée par le projet).
- Un seul champ de config, avec sous-dossiers dérivés automatiquement par
  outil (`huggingface`, `torch-hub`, `whisper-cpp`) plutôt que trois
  champs séparés.
- Le `~` est étendu une seule fois, au chargement de la config.
- whisper.cpp est géré via un lien symbolique créé par `setup.sh`, pas via
  un paramètre Python (puisque c'est un processus externe).
