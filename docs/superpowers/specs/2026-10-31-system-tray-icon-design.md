# Icône de plateau système (minimisation) — Design

## Contexte et objectif

L'utilisateur veut pouvoir minimiser la fenêtre de dictée sans quitter
l'application (modèle ASR chargé, raccourcis clavier actifs), via une icône
dans le plateau système offrant les mêmes actions que les boutons de la
fenêtre principale.

## Choix technique

`Gtk.StatusIcon` (API GTK3 classique pour icône de plateau système).

- Pas de nouvelle dépendance système : fait partie de GTK3, déjà utilisé
  par le reste du projet.
- Fonctionne nativement sous Cinnamon/X11 (protocole XEmbed système tray
  encore supporté), à la différence de GNOME Shell qui l'a abandonné.
- Alternative écartée : `AyatanaAppIndicator3` (API plus moderne côté
  Ubuntu/elementary) — nécessiterait une dépendance système supplémentaire
  non garantie sur Mint, pour aucun bénéfice ici.
- `Gtk.StatusIcon` est officiellement dépréciée depuis GTK 3.14, mais
  reste pleinement fonctionnelle et c'est l'outil le plus simple pour la
  cible visée (Cinnamon).

## Comportement

- **Fermeture de la fenêtre** (bouton X) : au lieu de quitter l'app,
  cache la fenêtre dans le plateau système. L'app (modèle, raccourcis
  clavier, daemon) continue de tourner.
- **Clic gauche sur l'icône** : réaffiche la fenêtre et la ramène au
  premier plan (`present()`), qu'elle soit cachée ou simplement derrière
  d'autres fenêtres.
- **Clic droit sur l'icône** (ou menu contextuel GTK standard) : affiche
  un menu avec, dans l'ordre :
  1. Action démarrer/arrêter — label dynamique selon l'état courant
     (« Démarrer » / « Arrêter »), réutilise la fonction `button_labels`
     déjà existante (`dictee/gui.py`).
  2. Action pause/reprendre — label dynamique (« Mettre en pause » /
     « Reprendre »), même fonction `button_labels`.
  3. « Copier ».
  4. Séparateur.
  5. « Afficher la fenêtre » — réaffiche/ramène la fenêtre au premier plan
     (même action que le clic gauche sur l'icône).
  6. « Quitter » — déclenche la séquence d'arrêt complète de l'app (arrêt
     audio, libération GPU, relâche du verrou PID), la même que celle
     utilisée pour SIGINT/SIGTERM dans `dictee_app.py`. C'est le seul
     moyen de fermer complètement l'app depuis l'interface, puisque le
     bouton X ne quitte plus.
- **Icône** : fixe (ex: une icône de microphone générique), ne change pas
  visuellement selon l'état. L'état courant reste visible en ouvrant le
  menu (labels dynamiques) ou en réaffichant la fenêtre (zone de statut
  déjà existante).

## Implémentation

### `dictee/gui.py`

Ajout d'une nouvelle classe `TrayIcon`, à côté de `DictationWindow`
(même fichier — les deux classes GTK sont étroitement liées, utilisent les
mêmes imports et la même fonction `button_labels`, et le fichier reste de
taille raisonnable) :

```python
class TrayIcon:
    def __init__(self, *, get_state, on_toggle_record, on_toggle_pause,
                 on_copy, on_show_window, on_quit):
        ...
```

- `get_state: Callable[[], str]` est injecté (pas de dépendance directe
  sur `Controller` ni sur `DictationWindow`), suivant le même principe
  d'injection de dépendances utilisé partout ailleurs dans le projet
  (`Controller`, `DictationWindow`). Appelé au moment de construire le
  menu (clic droit), pour obtenir les labels à jour via `button_labels`.
- `on_show_window: Callable[[], None]` est appelé à la fois par le clic
  gauche sur l'icône et par l'item de menu « Afficher la fenêtre ».
- Construction du menu à chaque ouverture (pas de cache), pour garantir
  que les labels reflètent toujours l'état courant.

`DictationWindow.__init__` : remplace la connexion actuelle
`self.connect("destroy", lambda _win: Gtk.main_quit())` par une connexion
au signal `delete-event` (déclenché par le clic sur le bouton X) qui cache
la fenêtre au lieu de la détruire :

```python
self.connect("delete-event", self._on_delete_event)

def _on_delete_event(self, _window, _event) -> bool:
    self.hide()
    return True  # empêche la destruction par défaut de la fenêtre
```

Le signal `destroy` reste connecté à `Gtk.main_quit()` comme filet de
sécurité (au cas où la fenêtre serait détruite par un autre chemin), mais
n'est plus le mécanisme normal de fermeture.

### `dictee_app.py`

- Construction de `TrayIcon` après que `controller`/`shutdown` existent
  (ordre exact à régler pendant l'implémentation — `shutdown` est
  actuellement défini après la construction de `controller`/`hotkeys`),
  avec :
  - `get_state=lambda: controller.state`
  - `on_toggle_record`/`on_toggle_pause`/`on_copy` : mêmes lambdas que
    celles déjà passées à `DictationWindow`.
  - `on_show_window=window.present`
  - `on_quit=shutdown` (la fonction de nettoyage déjà utilisée pour
    SIGINT/SIGTERM — garantit un arrêt cohérent quel que soit le
    déclencheur).

## Tests

- Pas de test automatisé pour `TrayIcon` lui-même (widget GTK, nécessite
  un affichage réel) — cohérent avec l'approche déjà adoptée pour
  `DictationWindow` (vérification manuelle uniquement).
- `button_labels` est déjà testé (Task 13/récent fix) et réutilisé
  sans modification — aucun nouveau test de logique pure nécessaire pour
  cette fonctionnalité.
- Vérification manuelle (ajout à la checklist du README) : fermer la
  fenêtre la cache (l'app continue de tourner) ; clic gauche sur l'icône
  réaffiche la fenêtre ; menu contextuel affiche les bons labels selon
  l'état ; « Quitter » arrête proprement l'app (vérifier
  `/tmp/dictee_app.pid` supprimé, `nvidia-smi` ne montre plus le
  processus).

## Décisions actées pendant le brainstorming

- Icône de plateau système (`Gtk.StatusIcon`), pas une barre fine en haut
  de l'écran.
- Fermeture de la fenêtre = cacher dans le plateau système, pas quitter.
- Icône fixe, pas de changement visuel selon l'état (seul le menu/la
  fenêtre reflètent l'état).
- Clic gauche = toujours réafficher/ramener au premier plan (pas de
  bascule affiché/caché).
