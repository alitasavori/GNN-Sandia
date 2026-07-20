"""Interactive continue/stop pause between training eval checkpoints.

Used by MLP / PowerFlowMultiNet (and optionally DA-GPS) trainers after each
``eval_every`` boundary so the user can stop early and still get the same
final artifacts as a normal completion (best/last ckpt + report json).

Prompting modes
---------------
- **TTY** (``sys.stdin.isatty()``): blocking ``input()`` for ``c``/``continue``
  or ``s``/``stop`` (case-insensitive). Re-prompts on garbage.
- **Non-TTY** (Colab ``subprocess.Popen`` pipes, CI, etc.): write
  ``INTERACTIVE_PAUSE.txt`` under ``out_dir`` and poll for empty control files
  ``CONTINUE`` or ``STOP`` (also accepts ``INTERACTIVE_REPLY.txt`` with one line).
  Poll every few seconds; reprint a reminder about every 60s.

Enable via ``--interactive_pause`` or env ``TRAIN_INTERACTIVE=1``.
Disable with ``--no_interactive_pause``.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

CONTINUE_FILE = "CONTINUE"
STOP_FILE = "STOP"
INSTRUCTION_FILE = "INTERACTIVE_PAUSE.txt"
REPLY_FILE = "INTERACTIVE_REPLY.txt"

_POLL_INTERVAL_S = 3.0
_REMINDER_EVERY_S = 60.0

_CONTROL_NAMES = (
    CONTINUE_FILE,
    STOP_FILE,
    CONTINUE_FILE.lower(),
    STOP_FILE.lower(),
    REPLY_FILE,
    INSTRUCTION_FILE,
)


def interactive_pause_enabled(args) -> bool:
    """True when pause is requested and not explicitly disabled."""
    if bool(getattr(args, "no_interactive_pause", False)):
        return False
    if bool(getattr(args, "interactive_pause", False)):
        return True
    env = str(os.environ.get("TRAIN_INTERACTIVE", "") or "").strip().lower()
    return env in ("1", "true", "yes", "on")


def should_interactive_pause(epoch: int, args) -> bool:
    """Pause at ``eval_every`` boundaries (not the final epoch)."""
    if not interactive_pause_enabled(args):
        return False
    ee = max(1, int(getattr(args, "eval_every", 10)))
    ep = int(epoch)
    epochs = int(getattr(args, "epochs", ep))
    return ep % ee == 0 and ep < epochs


def stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def parse_continue_or_stop(text: str) -> str | None:
    """Return ``'continue'`` / ``'stop'``, or ``None`` if unrecognized."""
    t = str(text or "").strip().lower()
    if t in ("c", "continue"):
        return "continue"
    if t in ("s", "stop"):
        return "stop"
    return None


def _clear_control_artifacts(out_dir: Path, *, keep_instructions: bool = False) -> None:
    for name in _CONTROL_NAMES:
        if keep_instructions and name == INSTRUCTION_FILE:
            continue
        p = out_dir / name
        if not p.is_file():
            continue
        try:
            p.unlink()
        except OSError:
            pass


def _write_instructions(
    out_dir: Path,
    *,
    epoch: int,
    epochs: int,
    best_line: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / INSTRUCTION_FILE
    body = (
        f"Interactive pause at epoch {epoch}/{epochs}\n"
        f"{best_line}\n\n"
        "To continue training: create an empty file named CONTINUE in this folder\n"
        "  (or write one line 'continue' / 'c' into INTERACTIVE_REPLY.txt).\n\n"
        "To stop now (save final checkpoint + reports and exit):\n"
        "  create an empty file named STOP\n"
        "  (or write one line 'stop' / 's' into INTERACTIVE_REPLY.txt).\n\n"
        "In Colab: use the file browser on OUT_DIR, or in another cell:\n"
        f"  !touch {out_dir / STOP_FILE}\n"
        f"  !touch {out_dir / CONTINUE_FILE}\n"
    )
    path.write_text(body, encoding="utf-8")
    return path


def _poll_choice(out_dir: Path) -> str | None:
    for name, choice in (
        (CONTINUE_FILE, "continue"),
        (CONTINUE_FILE.lower(), "continue"),
        (STOP_FILE, "stop"),
        (STOP_FILE.lower(), "stop"),
    ):
        if (out_dir / name).is_file():
            return choice
    reply = out_dir / REPLY_FILE
    if reply.is_file():
        try:
            return parse_continue_or_stop(reply.read_text(encoding="utf-8"))
        except OSError:
            return None
    return None


def _print_banner(
    *,
    out_dir: Path,
    epoch: int,
    epochs: int,
    best_line: str,
    tty: bool,
) -> None:
    print("", flush=True)
    print(f"=== Interactive pause (epoch {epoch}/{epochs}) ===", flush=True)
    print(best_line, flush=True)
    print("[c] continue training", flush=True)
    print("[s] stop now — save final checkpoint + reports and exit", flush=True)
    print(f"OUT_DIR: {out_dir.resolve()}", flush=True)
    if tty:
        print("Type c/continue or s/stop, then Enter.", flush=True)
        print(
            f"(Colab/non-TTY fallback: create empty file {CONTINUE_FILE} or {STOP_FILE} under OUT_DIR)",
            flush=True,
        )
    else:
        print(
            "stdin is not a TTY (typical Colab subprocess). "
            f"Create empty file {CONTINUE_FILE} or {STOP_FILE} under OUT_DIR "
            f"(see {INSTRUCTION_FILE}).",
            flush=True,
        )
        print(
            f"  e.g.  !touch '{out_dir / STOP_FILE}'   or   !touch '{out_dir / CONTINUE_FILE}'",
            flush=True,
        )
    print("=" * 48, flush=True)


def ask_continue_or_stop(
    *,
    out_dir: Path | str,
    epoch: int,
    epochs: int,
    best_val: float | None = None,
    best_epoch: int | None = None,
    poll_interval_s: float = _POLL_INTERVAL_S,
    reminder_every_s: float = _REMINDER_EVERY_S,
) -> str:
    """Block until the user chooses continue or stop. Returns ``'continue'`` or ``'stop'``."""
    out = Path(out_dir)
    if best_val is not None and best_epoch is not None:
        best_line = f"best val={float(best_val):.6g} @ epoch {int(best_epoch)}"
    elif best_val is not None:
        best_line = f"best val={float(best_val):.6g}"
    else:
        best_line = "best val=(n/a)"

    _clear_control_artifacts(out, keep_instructions=False)
    instr = _write_instructions(out, epoch=int(epoch), epochs=int(epochs), best_line=best_line)
    tty = stdin_is_tty()
    _print_banner(out_dir=out, epoch=int(epoch), epochs=int(epochs), best_line=best_line, tty=tty)
    print(f"Wrote {instr.resolve()}", flush=True)

    if tty:
        while True:
            # Prefer a file drop if the user creates one while at the prompt.
            file_choice = _poll_choice(out)
            if file_choice is not None:
                _clear_control_artifacts(out, keep_instructions=False)
                print(f"Interactive pause: {file_choice} (via control file)", flush=True)
                return file_choice
            try:
                raw = input(">>> ").strip()
            except EOFError:
                print(
                    "stdin EOF — falling back to file poll under OUT_DIR "
                    f"({CONTINUE_FILE}/{STOP_FILE}).",
                    flush=True,
                )
                tty = False
                break
            choice = parse_continue_or_stop(raw)
            if choice is not None:
                _clear_control_artifacts(out, keep_instructions=False)
                print(f"Interactive pause: {choice}", flush=True)
                return choice
            print("Unrecognized — enter c/continue or s/stop.", flush=True)

    # File-poll path (Colab / pipes / EOF fallback).
    last_reminder = time.monotonic()
    interval = max(0.5, float(poll_interval_s))
    reminder = max(interval, float(reminder_every_s))
    while True:
        choice = _poll_choice(out)
        if choice is not None:
            _clear_control_artifacts(out, keep_instructions=False)
            print(f"Interactive pause: {choice} (via control file)", flush=True)
            return choice
        now = time.monotonic()
        if now - last_reminder >= reminder:
            print(
                f"[interactive pause] still waiting at epoch {epoch}/{epochs} — "
                f"create {CONTINUE_FILE} or {STOP_FILE} under {out.resolve()}",
                flush=True,
            )
            last_reminder = now
        time.sleep(interval)


def add_interactive_pause_args(parser) -> None:
    """Register ``--interactive_pause`` / ``--no_interactive_pause`` on an ArgumentParser."""
    parser.add_argument(
        "--interactive_pause",
        action="store_true",
        help="After each --eval_every checkpoint, ask continue/stop via TTY input or "
        "CONTINUE/STOP files under --out_dir (also enabled by TRAIN_INTERACTIVE=1).",
    )
    parser.add_argument(
        "--no_interactive_pause",
        action="store_true",
        help="Disable interactive pause even if --interactive_pause or TRAIN_INTERACTIVE is set.",
    )


__all__ = [
    "CONTINUE_FILE",
    "STOP_FILE",
    "INSTRUCTION_FILE",
    "REPLY_FILE",
    "add_interactive_pause_args",
    "ask_continue_or_stop",
    "interactive_pause_enabled",
    "parse_continue_or_stop",
    "should_interactive_pause",
    "stdin_is_tty",
]
