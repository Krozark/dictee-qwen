# System Tray Icon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user minimize the dictation window to a system tray icon instead of quitting the app, with a right-click menu offering the same actions as the main window's buttons (start/stop, pause/resume, copy), plus show-window and quit.

**Architecture:** A new `TrayIcon` class (added to the existing `dictee/gui.py`, alongside `DictationWindow`) wraps `Gtk.StatusIcon`, built the same dependency-injected way as `DictationWindow` and `Controller` (no direct reference to either — just injected callables and a `get_state` getter). `DictationWindow`'s window-close button now hides the window instead of destroying it. `dictee_app.py` wires the tray icon's actions to the same `Controller`/`shutdown` callables already used by the window's buttons and the global hotkeys.

**Tech Stack:** GTK3 (`Gtk.StatusIcon`, `Gtk.Menu`), same as the rest of the GUI. No new dependencies.

---

## Task 1: Hide the window instead of destroying it on close

**Files:**
- Modify: `dictee/gui.py:80` (inside `DictationWindow.__init__`)

This task has no automated test — GTK window-close behavior requires a real
X11 display, consistent with how the rest of `DictationWindow` is only
manually verified (per the plan that originally built it).

- [ ] **Step 1: Replace the `destroy`-only close handling**

In `dictee/gui.py`, find this line inside `DictationWindow.__init__` (currently the last statement before the end of `__init__`):

```python
        self.connect("destroy", lambda _win: Gtk.main_quit())
```

Replace it with:

```python
        self.connect("delete-event", self._on_delete_event)
        self.connect("destroy", lambda _win: Gtk.main_quit())
```

Then add a new method to `DictationWindow`, right after `__init__` (before `append_segment`):

```python
    def _on_delete_event(self, _window, _event) -> bool:
        self.hide()
        return True
```

`delete-event` fires when the user clicks the window's close (X) button. Returning `True` tells GTK the close request was handled and the window should NOT be destroyed — `self.hide()` just hides it instead, keeping the app (model, hotkeys, audio) running. The existing `destroy` → `Gtk.main_quit()` connection stays as a safety net for the (now rare) case where something explicitly destroys the window — it is no longer the normal close path.

- [ ] **Step 2: Verify the change syntax-checks and existing tests still pass**

Run:
```bash
uv run python3 -c "import ast; ast.parse(open('dictee/gui.py').read()); print('SYNTAX OK')"
uv run python3 -c "import dictee.gui; print('IMPORT OK')"
uv run pytest -v
```
Expected: `SYNTAX OK`, `IMPORT OK`, and the full test suite passes with the same count as before this change (no test exercises `DictationWindow` directly beyond the pure `status_label`/`button_labels` functions, so the count should be unchanged).

- [ ] **Step 3: Manual verification (if a real X11 display is available)**

If `$DISPLAY` is set, do a safe, limited smoke test — construct the window, show it, and simulate a close request WITHOUT entering a real `Gtk.main()` loop (to avoid any risk of hanging):

```bash
uv run python3 -c "
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gdk, Gtk
from dictee.gui import DictationWindow

win = DictationWindow(on_toggle_record=lambda: None, on_toggle_pause=lambda: None, on_copy=lambda: None)
win.show_all()
while Gtk.events_pending():
    Gtk.main_iteration()

event = Gdk.Event.new(Gdk.EventType.DELETE)
result = win.emit('delete-event', event)
print('delete-event handler returned:', result)
print('window visible after delete-event:', win.get_visible())
"
```
Expected output: `delete-event handler returned: True` and `window visible after delete-event: False` — confirming the window hid itself instead of being destroyed. If no display is available, skip this step and rely on Step 2's checks plus a visual code read.

- [ ] **Step 4: Commit**

```bash
git add dictee/gui.py
git commit -m "feat: hide the dictation window instead of closing the app when its close button is clicked"
```

---

## Task 2: Add the `TrayIcon` class

**Files:**
- Modify: `dictee/gui.py` (append a new class after `DictationWindow`)

No automated test for this task either — `Gtk.StatusIcon`/`Gtk.Menu` require
a real display and are manually verified, consistent with `DictationWindow`.

- [ ] **Step 1: Append the `TrayIcon` class to `dictee/gui.py`**

Add this at the end of the file, after `DictationWindow`'s last method (`_clear_transcript_on_main_thread`):

```python
class TrayIcon:
    def __init__(
        self,
        *,
        get_state,
        on_toggle_record,
        on_toggle_pause,
        on_copy,
        on_show_window,
        on_quit,
    ):
        self._get_state = get_state
        self._on_toggle_record = on_toggle_record
        self._on_toggle_pause = on_toggle_pause
        self._on_copy = on_copy
        self._on_show_window = on_show_window
        self._on_quit = on_quit

        self._status_icon = Gtk.StatusIcon()
        self._status_icon.set_from_icon_name("audio-input-microphone")
        self._status_icon.set_tooltip_text(APP_WINDOW_TITLE)
        self._status_icon.connect("activate", self._on_activate)
        self._status_icon.connect("popup-menu", self._on_popup_menu)

    def _on_activate(self, _status_icon) -> None:
        self._on_show_window()

    def _on_popup_menu(self, _status_icon, button, activate_time) -> None:
        menu = self._build_menu()
        menu.popup(None, None, Gtk.StatusIcon.position_menu, self._status_icon, button, activate_time)

    def _build_menu(self) -> Gtk.Menu:
        record_label, pause_label = button_labels(self._get_state())
        menu = Gtk.Menu()

        record_item = Gtk.MenuItem(label=record_label)
        record_item.connect("activate", lambda _item: self._on_toggle_record())
        menu.append(record_item)

        pause_item = Gtk.MenuItem(label=pause_label)
        pause_item.connect("activate", lambda _item: self._on_toggle_pause())
        menu.append(pause_item)

        copy_item = Gtk.MenuItem(label="Copier")
        copy_item.connect("activate", lambda _item: self._on_copy())
        menu.append(copy_item)

        menu.append(Gtk.SeparatorMenuItem())

        show_item = Gtk.MenuItem(label="Afficher la fenêtre")
        show_item.connect("activate", lambda _item: self._on_show_window())
        menu.append(show_item)

        quit_item = Gtk.MenuItem(label="Quitter")
        quit_item.connect("activate", lambda _item: self._on_quit())
        menu.append(quit_item)

        menu.show_all()
        return menu
```

Notes on this design, for context (no action needed, just understanding why it's shaped this way):
- `get_state` is an injected zero-argument callable (e.g. `lambda: controller.state`), not a direct reference to `Controller` — this matches the dependency-injection pattern already used by `Controller` and `DictationWindow` elsewhere in this codebase (no GUI class reaches into another component's internals).
- `_build_menu()` is called fresh every time the menu is opened (not cached), so `button_labels(self._get_state())` always reflects the current state at the moment the user right-clicks the icon.
- `on_show_window` is used by both the left-click (`activate` signal) and the "Afficher la fenêtre" menu item — both should do the same thing (bring the window to the front).
- `menu.popup(None, None, Gtk.StatusIcon.position_menu, self._status_icon, button, activate_time)` is the standard GTK3 pattern for positioning a popup menu relative to a `Gtk.StatusIcon`.

- [ ] **Step 2: Verify the change syntax-checks and existing tests still pass**

Run:
```bash
uv run python3 -c "import ast; ast.parse(open('dictee/gui.py').read()); print('SYNTAX OK')"
uv run python3 -c "import dictee.gui; print('IMPORT OK')"
uv run pytest -v
```
Expected: `SYNTAX OK`, `IMPORT OK`, same test count as before (no test touches `TrayIcon` — it's GTK widget code, manually verified).

- [ ] **Step 3: Manual verification (if a real X11 display is available)**

If `$DISPLAY` is set, construct a `TrayIcon` directly (without entering `Gtk.main()`) and confirm it doesn't raise and produces a usable menu:

```bash
uv run python3 -c "
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from dictee.gui import TrayIcon

calls = []
tray = TrayIcon(
    get_state=lambda: 'recording',
    on_toggle_record=lambda: calls.append('record'),
    on_toggle_pause=lambda: calls.append('pause'),
    on_copy=lambda: calls.append('copy'),
    on_show_window=lambda: calls.append('show'),
    on_quit=lambda: calls.append('quit'),
)
menu = tray._build_menu()
labels = [item.get_label() for item in menu.get_children() if isinstance(item, Gtk.MenuItem)]
print('menu labels:', labels)
"
```
Expected output: `menu labels: ['Arrêter', 'Mettre en pause', 'Copier', 'Afficher la fenêtre', 'Quitter']` (the record/pause labels reflect `get_state() == 'recording'` via `button_labels`, matching the dynamic labels already used on the main window's buttons). If no display is available, skip and rely on Step 2 plus a visual code read confirming `_build_menu`'s structure matches `button_labels`'s known return values for `"recording"`.

- [ ] **Step 4: Commit**

```bash
git add dictee/gui.py
git commit -m "feat: add a system tray icon with the same actions as the window's buttons"
```

---

## Task 3: Wire the tray icon into the app entrypoint

**Files:**
- Modify: `dictee_app.py`

No automated test — this is integration glue, consistent with how the rest
of `dictee_app.py` is verified (syntax/import checks plus the manual README
checklist), not unit tests.

- [ ] **Step 1: Import `TrayIcon`**

In `dictee_app.py`, find this line near the top:

```python
from dictee.gui import DictationWindow
```

Replace it with:

```python
from dictee.gui import DictationWindow, TrayIcon
```

- [ ] **Step 2: Construct the tray icon after `shutdown` is defined**

In `dictee_app.py`, find this block (currently right before the two `GLib.unix_signal_add` calls):

```python
    def shutdown(*_args):
        audio_capture.stop()
        asr_backend.close()
        if config.backend == "whisper":
            pass  # whisper-server lifecycle is managed outside this process for now
        lock.release()
        Gtk.main_quit()
        return False

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, shutdown)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, shutdown)
```

Replace it with:

```python
    def shutdown(*_args):
        audio_capture.stop()
        asr_backend.close()
        if config.backend == "whisper":
            pass  # whisper-server lifecycle is managed outside this process for now
        lock.release()
        Gtk.main_quit()
        return False

    tray_icon = TrayIcon(
        get_state=lambda: controller.state,
        on_toggle_record=lambda: controller.toggle_record() or audio_toggle(),
        on_toggle_pause=lambda: controller.toggle_pause() or audio_pause_toggle(),
        on_copy=lambda: controller.copy(),
        on_show_window=window.present,
        on_quit=shutdown,
    )

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, shutdown)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, shutdown)
```

`tray_icon` is kept as a local variable in `main()`, which stays alive for the whole program (since `main()` doesn't return until `Gtk.main()` exits), so there's no risk of it being garbage-collected while the app is running. Using `shutdown` (the same function wired to SIGINT/SIGTERM) as the tray menu's "Quitter" action guarantees the exact same clean-teardown sequence (stop audio, release GPU, release the PID lock) regardless of whether the app is quit via the tray menu, a signal, or in the future any other trigger.

- [ ] **Step 3: Verify the change syntax-checks and existing tests still pass**

Run:
```bash
uv run python3 -c "import ast; ast.parse(open('dictee_app.py').read()); print('SYNTAX OK')"
uv run python3 -c "import dictee_app; print('IMPORT OK')"
uv run pytest -v
```
Expected: `SYNTAX OK`, `IMPORT OK`, same test count as before (this file has no automated tests of its own).

- [ ] **Step 4: Commit**

```bash
git add dictee_app.py
git commit -m "feat: wire the system tray icon into the app entrypoint"
```

---

## Task 4: Document the new behavior in the manual test checklist

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add tray-icon checklist items**

In `README.md`, find the end of the "Checklist de tests manuels (non automatisés)" list — specifically the last existing item:

```markdown
- [ ] Relancer l'app juste après l'avoir arrêtée : pas de message « already running » bloquant à tort.
```

Add these four new items right after it, keeping the same list:

```markdown
- [ ] Fermer la fenêtre (bouton X) : elle disparaît mais l'app continue de tourner (vérifier via `ps aux | grep dictee_app` ou `nvidia-smi` que le processus est toujours actif).
- [ ] Clic gauche sur l'icône du plateau système : la fenêtre réapparaît et revient au premier plan.
- [ ] Clic droit sur l'icône du plateau système : le menu affiche les bons libellés selon l'état courant (« Démarrer »/« Arrêter », « Mettre en pause »/« Reprendre » selon que l'app est idle/en enregistrement/en pause).
- [ ] « Quitter » depuis le menu du plateau système : l'app s'arrête proprement (`/tmp/dictee_app.pid` supprimé, plus de processus dans `nvidia-smi`).
```

- [ ] **Step 2: Verify the file is still valid Markdown and nothing else was touched**

Run:
```bash
git diff README.md
```
Expected: the diff shows only the four new lines appended at the end of the checklist, nothing else changed.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add system tray icon items to the manual test checklist"
```

---

## Self-review notes

- **Spec coverage:** every behavior in the design doc maps to a task — hide-on-close (Task 1), tray icon + menu with dynamic labels + show/quit actions (Task 2), wiring into the real app including reusing `shutdown` for "Quitter" (Task 3), manual test checklist additions (Task 4). The design's explicit choice of `Gtk.StatusIcon` (not `AppIndicator3`) and "fixed icon, no per-state icon change" are both reflected directly in Task 2's code.
- **Type consistency:** `TrayIcon`'s constructor keyword names (`get_state`, `on_toggle_record`, `on_toggle_pause`, `on_copy`, `on_show_window`, `on_quit`) are used identically in Task 2 (the class definition) and Task 3 (the `dictee_app.py` call site) — no renaming drift. `button_labels` is reused exactly as already defined in `dictee/gui.py` (Task 2 does not redefine or duplicate it).
- **No placeholders:** every step has complete, runnable code. The two genuinely GTK-display-dependent pieces (Tasks 1 and 2) are explicitly marked as manually verified (with a concrete, safe-to-run verification script each), not left with a vague "test this" instruction, consistent with how `DictationWindow` itself was originally verified.
