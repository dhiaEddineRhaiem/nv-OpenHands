"""Drop-in replacement for OpenHands v0.62.0's tmux-based ``BashSession``.

This module is a **behaviour-compatible** re-implementation of
``openhands/runtime/utils/bash.py`` that drives a single persistent ``bash``
attached to a **pty** (via :func:`pty.openpty`) instead of a tmux pane driven by
``libtmux``.  It is meant to be copied over
``openhands/runtime/utils/bash.py`` at setup time.

Motivation
----------
Under high concurrency, ``libtmux.Server().new_session(...)`` intermittently
raises ``LibTmuxException('server exited unexpectedly')``.  Those crashes are
counted as failed rollouts and floor the SWE-GRPO reward.  Removing tmux from
the hot path eliminates that failure mode entirely.

Correctness contract
---------------------
RL rewards depend on the parsed command output / exit-code being identical to
the original tmux implementation.  To guarantee that, we keep the original's
PS1-metadata protocol and reuse its output-extraction helpers **verbatim**
(``split_bash_commands``, ``escape_bash_special_chars``, ``_remove_command_prefix``,
``_get_command_output``, ``_combine_outputs_between_matches``,
``_handle_completed_command`` / ``_handle_nochange_timeout_command`` /
``_handle_hard_timeout_command`` and the whole ``execute`` state machine).  Only
the *transport* is swapped: tmux ``send-keys`` / ``capture-pane`` become pty
writes and a background reader thread.

The transport swap requires a handful of small, well-understood adaptations vs.
the tmux code, each documented inline where it occurs. These were validated
against real bash 5.2 over a pty on the deployment host:

1. **Single-line PS1** (see the ``PS1`` class attribute). The upstream
   multi-line metadata prompt makes bash/readline redraw the prompt on a real
   pty, emitting a duplicate metadata block *before* the command output and
   causing empty extraction. Collapsing it to one physical line removes the
   redraw. The JSON fields are unchanged, so parsed metadata is identical.
2. **Echo off** (termios ``ECHO`` cleared before spawning bash in ``initialize``,
   re-asserted with ``stty -echo``). With echo suppressed the captured stream is
   just the command's real output + the metadata prompt -- no typed-command echo
   to strip -- and, crucially, the init command's literal PS1 string is not
   echoed as a bogus metadata block (which otherwise floods the log with
   "Failed to parse PS1 metadata" warnings). ``_remove_command_prefix`` stays as
   a safety net in case a command re-enables echo. ``initialize`` also disables
   readline completion (``bind 'set disable-completion on'``) so a literal TAB in
   a heredoc self-inserts instead of triggering filename completion (which would
   corrupt written files), and disables history expansion (``set +H``).
3. **CR stripped at read time** (in the reader thread). A pty emits CRLF; the
   whole buffer is kept CR-free so every marker search (especially
   ``_clear_screen``'s ``rfind`` of the PS1 markers, which previously failed on
   ``\r\n`` and wiped the buffer, injecting a spurious prompt) is reliable.
   ``_get_pane_content`` additionally strips ANSI escape sequences so the text
   matches tmux's ``capture-pane -p``.
4. ``execute``'s completion check is gated on ``output_changed_since_command``.
   tmux ``capture-pane`` is synchronous, so the original never observes the
   pre-command prompt after sending; our reader thread is asynchronous, so a
   completion check on the very first poll could otherwise match the *previous*
   prompt (which ends in the PS1 END marker) and return empty output.  This is
   exactly the guard OpenHands' own pty backend
   (``software-agent-sdk`` ``terminal_session.py``) adds for the same reason.
5. The ``endswith(PS1_END)`` completion fallback is additionally gated on the
   initial prompt having genuinely scrolled off. With echo off there is no
   command echo after the kept prompt block, so a bare ``endswith`` would fire on
   that STALE block the instant a signal handler emits a newline -- completing an
   interrupt/timeout recovery on the old prompt with empty output and the wrong
   exit code. Interrupts and normal commands instead complete on a NEW prompt
   block (count increase); the fallback is reserved for true history scroll-off.

The tmux memory monitor (``TmuxMemoryMonitor``) is intentionally **dropped**;
see the note in ``initialize``.

The pty mechanics (openpty + a persistent interactive bash, a non-blocking
master fd, and a background reader thread) mirror OpenHands' V1
``SubprocessTerminal``.
"""

import fcntl
import json
import os
import shutil
import pty
import re
import select
import shlex
import signal
import struct
import termios
import threading
import time
from collections import deque
from enum import Enum
from typing import Any

import bashlex
import psutil

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction
from openhands.events.observation import ErrorObservation
from openhands.events.observation.commands import (
    CMD_OUTPUT_PS1_BEGIN,
    CMD_OUTPUT_PS1_END,
    CmdOutputMetadata,
    CmdOutputObservation,
)
from openhands.runtime.utils.bash_constants import TIMEOUT_MESSAGE_TEMPLATE
from openhands.runtime.utils.command_blacklist import check_command_blacklist
from openhands.utils.shutdown_listener import should_continue

RUNTIME_USERNAME = os.getenv("RUNTIME_USERNAME")
SU_TO_USER = os.getenv("SU_TO_USER", "true").lower() in (
    "1",
    "true",
    "t",
    "yes",
    "y",
    "on",
)

# Byte written to the pty to submit a line (Enter). bash's line discipline maps
# it to a carriage return internally; keeping it '\n' matches typed input.
ENTER = b"\n"

# Matches ANSI control sequences so that the text we hand to the PS1 parser is
# escape-free, exactly like tmux's ``capture-pane -p`` output.  Covers CSI
# sequences (colours, cursor moves, bracketed-paste toggles), OSC sequences
# (title setting, terminated by BEL or ST) and the two-character C1 escapes.
_ANSI_ESCAPE_RE = re.compile(
    r"""
    \x1b
    (?:
        \[ [0-?]* [ -/]* [@-~]           # CSI ... final byte
      | \] .*? (?: \x07 | \x1b\\ )        # OSC ... (BEL | ST)
      | [@-Z\\-_]                         # 7-bit C1 (Fe), e.g. ESC c, ESC =
    )
    """,
    re.VERBOSE | re.DOTALL,
)


def split_bash_commands(commands: str) -> list[str]:
    # NOTE: copied verbatim from the original bash.py so command-splitting
    # behaviour (and therefore the "cannot execute multiple commands" path) is
    # byte-for-byte identical.
    if not commands.strip():
        return [""]
    try:
        parsed = bashlex.parse(commands)
    except (
        bashlex.errors.ParsingError,
        NotImplementedError,
        TypeError,
        AttributeError,
    ):
        # Added AttributeError to catch 'str' object has no attribute 'kind' error (issue #8369)
        logger.debug(
            f"Failed to parse bash commands\n"
            f"[input]: {commands}\n"
            f"The original command will be returned as is.",
            exc_info=True,
        )
        # If parsing fails, return the original commands
        return [commands]

    result: list[str] = []
    last_end = 0

    for node in parsed:
        start, end = node.pos

        # Include any text between the last command and this one
        if start > last_end:
            between = commands[last_end:start]
            logger.debug(f"BASH PARSING between: {between}")
            if result:
                result[-1] += between.rstrip()
            elif between.strip():
                # THIS SHOULD NOT HAPPEN
                result.append(between.rstrip())

        # Extract the command, preserving original formatting
        command = commands[start:end].rstrip()
        logger.debug(f"BASH PARSING command: {command}")
        result.append(command)

        last_end = end

    # Add any remaining text after the last command to the last command
    remaining = commands[last_end:].rstrip()
    logger.debug(f"BASH PARSING remaining: {remaining}")
    if last_end < len(commands) and result:
        result[-1] += remaining
        logger.debug(f"BASH PARSING result[-1] += remaining: {result[-1]}")
    elif last_end < len(commands):
        if remaining:
            result.append(remaining)
            logger.debug(f"BASH PARSING result.append(remaining): {result[-1]}")
    return result


def escape_bash_special_chars(command: str) -> str:
    r"""Escapes characters that have different interpretations in bash vs python.
    Specifically handles escape sequences like \;, \|, \&, etc.

    NOTE: copied verbatim from the original bash.py.  OpenHands' own pty backend
    applies this exact transform before writing to the pty (see
    ``terminal_session.py``), so keeping it here yields identical bytes on the
    wire as the tmux path did.
    """
    if command.strip() == "":
        return ""

    try:
        parts = []
        last_pos = 0

        def visit_node(node: Any) -> None:
            nonlocal last_pos
            if (
                node.kind == "redirect"
                and hasattr(node, "heredoc")
                and node.heredoc is not None
            ):
                # We're entering a heredoc - preserve everything as-is until we see EOF
                # Store the heredoc end marker (usually 'EOF' but could be different)
                between = command[last_pos : node.pos[0]]
                parts.append(between)
                # Add the heredoc start marker
                parts.append(command[node.pos[0] : node.heredoc.pos[0]])
                # Add the heredoc content as-is
                parts.append(command[node.heredoc.pos[0] : node.heredoc.pos[1]])
                last_pos = node.pos[1]
                return

            if node.kind == "word":
                # Get the raw text between the last position and current word
                between = command[last_pos : node.pos[0]]
                word_text = command[node.pos[0] : node.pos[1]]

                # Add the between text, escaping special characters
                between = re.sub(r"\\([;&|><])", r"\\\\\1", between)
                parts.append(between)

                # Check if word_text is a quoted string or command substitution
                if (
                    (word_text.startswith('"') and word_text.endswith('"'))
                    or (word_text.startswith("'") and word_text.endswith("'"))
                    or (word_text.startswith("$(") and word_text.endswith(")"))
                    or (word_text.startswith("`") and word_text.endswith("`"))
                ):
                    # Preserve quoted strings, command substitutions, and heredoc content as-is
                    parts.append(word_text)
                else:
                    # Escape special chars in unquoted text
                    word_text = re.sub(r"\\([;&|><])", r"\\\\\1", word_text)
                    parts.append(word_text)

                last_pos = node.pos[1]
                return

            # Visit child nodes
            if hasattr(node, "parts"):
                for part in node.parts:
                    visit_node(part)

        # Process all nodes in the AST
        nodes = list(bashlex.parse(command))
        for node in nodes:
            between = command[last_pos : node.pos[0]]
            between = re.sub(r"\\([;&|><])", r"\\\\\1", between)
            parts.append(between)
            last_pos = node.pos[0]
            visit_node(node)

        # Handle any remaining text after the last word
        remaining = command[last_pos:]
        parts.append(remaining)
        return "".join(parts)
    except (bashlex.errors.ParsingError, NotImplementedError, TypeError):
        logger.debug(
            f"Failed to parse bash commands for special characters escape\n"
            f"[input]: {command}\n"
            f"The original command will be returned as is.",
            exc_info=True,
        )
        return command


class BashCommandStatus(Enum):
    CONTINUE = "continue"
    COMPLETED = "completed"
    NO_CHANGE_TIMEOUT = "no_change_timeout"
    HARD_TIMEOUT = "hard_timeout"


def _remove_command_prefix(command_output: str, command: str) -> str:
    return command_output.lstrip().removeprefix(command.lstrip()).lstrip()


class BashSession:
    """Persistent bash session over a pty (no tmux / no libtmux).

    Public interface matches the original tmux ``BashSession`` exactly:
    ``__init__(work_dir, username=None, no_change_timeout_seconds=30,
    max_memory_mb=None)``, ``initialize()``, ``execute(action)``, ``close()``,
    and the ``cwd`` property.
    """

    POLL_INTERVAL = 0.05
    HISTORY_LIMIT = 10_000

    # SINGLE-LINE PS1 metadata prompt.
    #
    # The upstream ``CmdOutputMetadata.to_ps1_prompt()`` embeds newlines in the
    # prompt string. On a real pty that multi-line prompt makes bash/readline
    # *redraw* the prompt after accepting the input line, so a single command
    # produces TWO metadata blocks (an intermediate re-render before the output
    # plus the real one after it). The tmux capture never saw that because tmux
    # rendered to its own virtual screen. The duplicate intermediate block made
    # ``execute()`` complete early and extract an empty region (empty rewards).
    #
    # Collapsing the prompt to a single physical line removes the redraw, so
    # exactly one metadata block is emitted per prompt. The JSON *fields* are
    # byte-identical to ``to_ps1_prompt()`` (same keys/values), so
    # ``from_ps1_match`` yields identical metadata and the parsed observation is
    # unchanged -- only the prompt whitespace differs, and that is never part of
    # the observation content.
    PS1 = (
        "\n"
        + CMD_OUTPUT_PS1_BEGIN.strip()
        + json.dumps(
            {
                "pid": "$!",
                "exit_code": "$?",
                "username": r"\u",
                "hostname": r"\h",
                "working_dir": r"$(pwd)",
                "py_interpreter_path": r'$(which python 2>/dev/null || echo "")',
            }
        ).replace('"', r"\"")
        + CMD_OUTPUT_PS1_END.strip()
        + "\n"
    )

    def __init__(
        self,
        work_dir: str,
        username: str | None = None,
        no_change_timeout_seconds: int = 30,
        max_memory_mb: int | None = None,
    ):
        self.NO_CHANGE_TIMEOUT_SECONDS = no_change_timeout_seconds
        self.work_dir = work_dir
        self.username = username
        self._initialized = False
        self.max_memory_mb = max_memory_mb
        # Kept for attribute-compatibility with callers/tests that reference it.
        # The tmux memory monitor is dropped in the pty backend (see initialize).
        self.memory_monitor = None

        # pty transport state
        self.process = None
        self._master_fd: int | None = None
        self._reader_thread: threading.Thread | None = None
        # Line-per-item circular buffer (mirrors tmux history-limit truncation).
        # Slightly larger than HISTORY_LIMIT to match tmux keeping ~10,001 lines.
        self._buffer: deque[str] = deque(maxlen=self.HISTORY_LIMIT + 50)
        self._buffer_lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        # Build the shell command exactly like the original tmux backend so the
        # runtime user / login semantics are unchanged.
        _shell_command = "/bin/bash"
        if SU_TO_USER and self.username in list(
            filter(None, [RUNTIME_USERNAME, "root", "openhands"])
        ):
            # This starts a login shell for the given user (interactive on a pty).
            _shell_command = f"su {self.username} -"
            shell_argv = shlex.split(_shell_command)
        else:
            # ``-i`` forces interactive mode so PROMPT_COMMAND/PS1 take effect.
            # (A bash whose stdin is a pty is already interactive; -i just makes
            # this explicit and matches OpenHands' own pty backend.)
            shell_argv = ["/bin/bash", "-i"]
        # NEMO_NET_ISOLATE=1 (set by the Gym when `isolate_agent_network` is on): run the
        # model's shell in a PRIVATE network namespace. The action server (this process)
        # keeps the container's network so the Gym can still reach it; everything the model
        # runs through this session gets "Network is unreachable" for anything but loopback.
        # Why (SWE RL audit 2026-11-20): the training containers had full internet, and the
        # policy learned to `git clone` the upstream repo, `pip download` newer releases
        # and even fetch the SWE-smith dataset (gold patches) from Hugging Face, besides
        # fork-storming a CPython build with `make -j`. Loopback is brought up so local
        # test servers keep working (needs CAP_NET_ADMIN in the container: the Gym stops
        # dropping it when this mode is on), then the shell drops SYS_ADMIN/NET_ADMIN/
        # SYS_PTRACE/SYS_MODULE so model code cannot `nsenter` back into the host namespace.
        # Verified in two task images (apptainer 1.4.5): github/pypi unreachable, 127.0.0.1
        # bind+connect fine; `unshare`/`setpriv` come from util-linux, present in both.
        if os.environ.get("NEMO_NET_ISOLATE") == "1" and shutil.which("unshare"):
            lo_up = (
                "ip link set lo up 2>/dev/null || ifconfig lo up 2>/dev/null || "
                "python3 -c 'import socket,fcntl,struct;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
                "fcntl.ioctl(s,0x8914,struct.pack(\"16sH14s\",b\"lo\",0x41,b\"\\0\"*14))' 2>/dev/null; "
            )
            drop = (
                "exec setpriv --bounding-set=-sys_admin,-net_admin,-sys_ptrace,-sys_module "
                if shutil.which("setpriv")
                else "exec "
            )
            inner = lo_up + drop + " ".join(shlex.quote(a) for a in shell_argv)
            shell_argv = ["unshare", "-n", "--", "/bin/sh", "-c", inner]
            logger.info("NEMO_NET_ISOLATE=1: bash session runs in a private network namespace")

        # NOTE (memory limit): the original ran a ``TmuxMemoryMonitor`` thread
        # bounded by ``TMUX_MEMORY_LIMIT``. That machinery is tmux-specific and
        # is intentionally NOT ported here to minimise risk (the original's
        # ``prlimit``-based path was already commented out / FIXME). ``max_memory_mb``
        # is therefore accepted but not enforced. If per-session memory capping is
        # needed later, add a lightweight psutil monitor over ``self.process`` pgid.
        logger.debug(
            f"Initializing pty bash session in {self.work_dir} with command: {shell_argv}"
        )

        # Create the pty. The child gets the slave; we keep the master.
        master_fd, slave_fd = pty.openpty()

        # Give the pty a large window (matching the original tmux x=1000, y=1000)
        # so readline does not wrap long command echoes, which would otherwise
        # perturb the captured text vs. the tmux baseline.
        try:
            winsize = struct.pack("HHHH", 1000, 1000, 0, 0)  # rows, cols, xpix, ypix
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            logger.debug("Failed to set pty winsize; continuing", exc_info=True)

        # Disable terminal echo BEFORE bash starts, so the initialization command
        # (which contains the literal PS1 metadata string) is not echoed back into
        # the stream. Without this the echoed literal PS1 -- unexpanded, with
        # \"-escaped quotes -- is scanned as a bogus metadata block and floods the
        # log with "Failed to parse PS1 metadata" warnings during init. Echo stays
        # off for real commands too, giving clean output extraction. (A command
        # that re-enables echo is still handled: _remove_command_prefix strips a
        # single echoed command line.)
        try:
            _attrs = termios.tcgetattr(slave_fd)
            _attrs[3] &= ~termios.ECHO  # lflag
            termios.tcsetattr(slave_fd, termios.TCSANOW, _attrs)
        except Exception:
            logger.debug("Failed to disable pty echo; continuing", exc_info=True)

        import subprocess  # local import: only needed to spawn the shell

        try:
            self.process = subprocess.Popen(
                shell_argv,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=self.work_dir,
                env=os.environ.copy(),
                text=False,  # bytes I/O
                bufsize=0,
                start_new_session=True,  # own session/pgid for signal handling
                close_fds=True,
            )
        finally:
            # Parent must close its copy of the slave fd.
            try:
                os.close(slave_fd)
            except OSError:
                pass

        self._master_fd = master_fd
        # Non-blocking master so both the reader thread and _write_all can use
        # select() for readiness / backpressure.
        flags = fcntl.fcntl(self._master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self._master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Background reader: continuously drains the pty into the line buffer.
        self._reader_thread = threading.Thread(
            target=self._read_output_continuously, daemon=True
        )
        self._reader_thread.start()

        self._initialized = True

        # Configure bash to use our PS1 metadata prompt and disable PS2, and turn
        # OFF terminal echo. With echo off, the command the agent typed is not
        # echoed back into the pty stream, so the captured content is just the
        # command's real output followed by the metadata prompt -- no echo to
        # strip and no chance of a duplicated echo confusing extraction.
        # (_remove_command_prefix stays as a safety net in case a command
        # re-enables echo via stty/reset.)
        # Also make the interactive shell treat our sent bytes literally:
        #   * ``bind 'set disable-completion on'`` -> TAB self-inserts instead of
        #     triggering filename completion. Without this, a literal TAB in a
        #     heredoc (Go, Makefiles, indented code) is completed against the cwd
        #     and the written file is corrupted.
        #   * ``set +H`` -> disable history expansion so a literal '!' is not
        #     rewritten (matches OpenHands' own pty backend).
        self._send_keys_checked(
            f"export PROMPT_COMMAND='export PS1=\"{self.PS1}\"'; export PS2=\"\"; "
            f"stty -echo 2>/dev/null; bind 'set disable-completion on' 2>/dev/null; "
            f"set +H 2>/dev/null",
            enter=True,
        )

        # Wait until the metadata prompt is actually rendered before clearing,
        # so the first real command sees exactly one initial PS1 block. (tmux was
        # synchronous; the pty reader is not, so we must wait explicitly.)
        if not self._wait_for_ps1(timeout=10.0):
            logger.warning(
                "PS1 metadata prompt did not appear during bash initialization"
            )
        self._clear_screen()

        # Store the last command state for interactive input handling.
        self.prev_status: BashCommandStatus | None = None
        self.prev_output: str = ""
        self._closed = False
        logger.debug(f"Bash session initialized with work dir: {self.work_dir}")

        # Maintain the current working directory (updated from PS1 metadata).
        self._cwd = os.path.abspath(self.work_dir)
        self._initialized = True

    def __del__(self) -> None:
        """Ensure the session is closed when the object is destroyed."""
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Clean up the pty session and child process."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.process is not None:
                # Try a graceful exit first.
                try:
                    self._write_all(b"exit\n")
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    # Escalate to the process group.
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                        self.process.wait(timeout=1)
                    except Exception:
                        try:
                            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        except Exception:
                            pass
        except Exception:
            logger.debug("Error while closing pty bash session", exc_info=True)
        finally:
            # Closing the master fd makes the reader thread's read() fail/EOF.
            if self._master_fd is not None:
                try:
                    os.close(self._master_fd)
                except OSError:
                    pass
                self._master_fd = None
            if self._reader_thread is not None and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1)
            self.process = None

    @property
    def cwd(self) -> str:
        return self._cwd

    # ------------------------------------------------------------------ #
    # pty I/O core
    # ------------------------------------------------------------------ #
    def _read_output_continuously(self) -> None:
        """Drain the pty master into the line buffer until the shell exits."""
        fd = self._master_fd
        if fd is None:
            return
        try:
            while True:
                if self.process is not None and self.process.poll() is not None:
                    break
                if self._master_fd is None:
                    break
                try:
                    r, _, _ = select.select([fd], [], [], 0.1)
                except (OSError, ValueError):
                    break
                if not r:
                    continue
                try:
                    chunk = os.read(fd, 4096)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    break
                if not chunk:
                    break  # EOF
                # Strip carriage returns at read time. A pty emits CRLF; tmux's
                # capture-pane returned CR-free text. Removing CR here also means
                # the whole buffer (and every marker search in it) is clean '\n'
                # text -- critical for _clear_screen's rfind of the PS1 markers,
                # which previously failed on '\r\n' and wiped the buffer.
                text = chunk.decode("utf-8", errors="replace").replace("\r", "")
                with self._buffer_lock:
                    self._add_text_to_buffer(text)
        except Exception:
            logger.debug("pty reader thread terminated", exc_info=True)

    def _add_text_to_buffer(self, text: str) -> None:
        """Append text to the buffer, keeping one logical line per deque item.

        Mirrors OpenHands' V1 SubprocessTerminal so the maxlen truncation drops
        whole oldest lines, matching tmux's history-limit behaviour.
        Caller must hold ``self._buffer_lock``.
        """
        if self._buffer and not self._buffer[-1].endswith("\n"):
            combined_text = self._buffer[-1] + text
            self._buffer.pop()
        else:
            combined_text = text

        lines = combined_text.split("\n")
        for line in lines[:-1]:
            self._buffer.append(line + "\n")
        if lines[-1]:  # keep trailing partial line (if any)
            self._buffer.append(lines[-1])

    def _write_all(self, data: bytes) -> None:
        """Write all bytes to the pty master, respecting flow control.

        On Linux the pty write correctly returns EAGAIN / short-writes when the
        line-discipline input buffer is full (e.g. large heredocs), so we loop
        with select() until every byte is delivered. This is what lets big
        multi-line file writes go through without a manual pacing delay.
        """
        if self._master_fd is None:
            raise RuntimeError("pty bash session is not initialized")
        view = memoryview(data)
        total = 0
        n = len(data)
        while total < n:
            try:
                _, w, _ = select.select([], [self._master_fd], [], 1.0)
            except (OSError, ValueError):
                raise RuntimeError("pty master is not writable")
            if not w:
                continue
            try:
                written = os.write(self._master_fd, view[total:])
                total += written
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as e:
                logger.error(f"Failed to write to pty: {e}")
                raise

    def _is_special_key(self, command: str) -> bool:
        """Check if the command is a special key (e.g. ``C-c``)."""
        # Special keys are of the form C-<key>
        _command = command.strip()
        return _command.startswith("C-") and len(_command) == 3

    def _encode_keys(self, keys: str) -> bytes:
        """Translate a key spec into the bytes to write to the pty.

        - ``C-<x>`` -> the corresponding control byte (Ctrl+X), so Ctrl+C
          delivers SIGINT via the pty line discipline, matching tmux ``send-keys``.
        - anything else -> the literal text, with newlines normalised to ``\\n``.
        """
        if self._is_special_key(keys):
            ch = keys.strip()[-1].upper()
            return bytes([ord(ch) & 0x1F])
        raw = keys.encode("utf-8", "ignore")
        # Normalise line endings so embedded newlines (heredocs, multi-line
        # commands) submit lines exactly as typed input would.
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return raw

    def _send_keys_checked(self, keys: str, enter: bool = True) -> None:
        """Send keys to the pty; append Enter when requested.

        Replaces the tmux ``send-keys`` helper of the same name. For a special
        key (``C-c``/``C-z``/``C-d``) the caller passes ``enter=False`` and we
        emit only the control byte, exactly like the original.
        """
        payload = self._encode_keys(keys)
        if payload:
            self._write_all(payload)
        if enter:
            self._write_all(ENTER)

    def _wait_for_ps1(self, timeout: float = 10.0) -> bool:
        """Block until a PS1 metadata block is visible (init-time only)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if CmdOutputMetadata.matches_ps1_metadata(self._get_pane_content()):
                return True
            time.sleep(0.05)
        return False

    def _get_pane_content(self) -> str:
        """Return the captured session output, normalised like tmux capture-pane.

        Replaces tmux ``capture-pane -J -pS -``.  We strip carriage returns and
        ANSI escape sequences (tmux renders these to its virtual screen and its
        capture is escape-free) and rstrip each line -- matching the original's
        ``"\\n".join(line.rstrip() for line in capture_pane.stdout)`` so the PS1
        regex, prompt detection and command-echo removal behave identically.
        """
        with self._buffer_lock:
            raw = "".join(self._buffer)
        raw = raw.replace("\r", "")
        raw = _ANSI_ESCAPE_RE.sub("", raw)
        content = "\n".join(line.rstrip() for line in raw.split("\n"))
        return content

    def _clear_screen(self) -> None:
        """Reset the buffer to just the most recent PS1 block.

        Equivalent to the original's tmux ``C-l`` + ``clear-history``: afterwards
        ``_get_pane_content`` returns only the current prompt, so the next
        command's ``initial_ps1_count`` is 1.  We keep everything from the last
        ``###PS1JSON###`` marker onward (its trailing ``###PS1END###`` line is what
        prompt-detection keys off).  If no prompt is buffered yet, nudge bash to
        render one.
        """
        need_prompt_nudge = False
        # Use the bare markers (no surrounding newlines). CMD_OUTPUT_PS1_BEGIN is
        # '\n###PS1JSON###\n', but our single-line prompt is '###PS1JSON###{...}',
        # so we must match on the stripped token, not the newline-wrapped one.
        begin = CMD_OUTPUT_PS1_BEGIN.strip()  # '###PS1JSON###'
        end = CMD_OUTPUT_PS1_END.strip()  # '###PS1END###'
        with self._buffer_lock:
            if not self._buffer:
                need_prompt_nudge = True
            else:
                # CR is already gone (stripped at read time); strip ANSI so the
                # marker search is reliable.
                data = _ANSI_ESCAPE_RE.sub("", "".join(self._buffer))
                start_idx = data.rfind(begin)
                end_idx = data.rfind(end)
                if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                    # Keep only the most recent metadata block; prepend a newline
                    # so it stays on its own line for the ^ anchor in the PS1 regex.
                    tail = "\n" + data[start_idx:]
                    self._buffer.clear()
                    self._buffer.append(tail)
                else:
                    self._buffer.clear()
                    need_prompt_nudge = True
        if need_prompt_nudge:
            try:
                self._write_all(ENTER)  # ask bash to re-render a prompt
            except Exception:
                pass

    def _ready_for_next_command(self) -> None:
        """Reset the content buffer for a new command."""
        self._clear_screen()

    def _find_shell_proc(self, root_proc: psutil.Process) -> psutil.Process:
        """Walk down through single-child shell wrappers (e.g. su -> bash) to
        find the actual shell process. Returns the deepest shell in the chain.

        NOTE: copied verbatim from the original bash.py.
        """
        shell_proc = root_proc
        while True:
            kids = shell_proc.children(recursive=False)
            if not kids:
                break
            first = kids[0]
            try:
                name = first.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if len(kids) == 1 and name in ("bash", "sh", "zsh", "fish", "dash"):
                shell_proc = first
            else:
                break
        return shell_proc

    def _kill_command_processes(self) -> bool:
        """SIGKILL the foreground command processes, keeping the shell alive.

        pty analog of the original's ``_kill_pane_processes``: instead of looking
        up the tmux pane pid, we start from ``self.process`` (the ``bash``/``su``
        we spawned), walk down to the real shell and kill only its children.
        """
        try:
            if self.process is None:
                return False
            root_pid = self.process.pid
            print(f"[BASH_SIGNAL] Root PID: {root_pid}", flush=True)

            proc = psutil.Process(root_pid)
            shell_proc = self._find_shell_proc(proc)

            targets = shell_proc.children(recursive=True)
            if not targets:
                print(
                    f"[BASH_SIGNAL] Shell PID {shell_proc.pid} has no command children",
                    flush=True,
                )
                return False

            print(
                f"[BASH_SIGNAL] Shell: {shell_proc.pid} ({shell_proc.name()}) -> killing "
                f"{len(targets)} processes: {[(t.pid, t.name()) for t in targets]}",
                flush=True,
            )
            for target in targets:
                try:
                    target.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            psutil.wait_procs(targets, timeout=3)
            print("[BASH_SIGNAL] Kill complete", flush=True)
            return True
        except (psutil.NoSuchProcess, ValueError) as e:
            print(f"[BASH_SIGNAL] Kill failed: {type(e).__name__}: {e}", flush=True)
            return False
        except Exception as e:
            print(
                f"[BASH_SIGNAL] Kill failed unexpectedly: {type(e).__name__}: {e}",
                flush=True,
            )
            return False

    # ------------------------------------------------------------------ #
    # Output extraction (copied verbatim from the original -- operates on
    # captured text + session state, transport-agnostic).
    # ------------------------------------------------------------------ #
    def _get_command_output(
        self,
        command: str,
        raw_command_output: str,
        metadata: CmdOutputMetadata,
        continue_prefix: str = "",
    ) -> str:
        """Get the command output with the previous command output removed."""
        # remove the previous command output from the new output if any
        if self.prev_output:
            command_output = raw_command_output.removeprefix(self.prev_output)
            metadata.prefix = continue_prefix
        else:
            command_output = raw_command_output
        self.prev_output = raw_command_output  # update current command output anyway
        command_output = _remove_command_prefix(command_output, command)
        return command_output.rstrip()

    def _handle_completed_command(
        self,
        command: str,
        pane_content: str,
        ps1_matches: list[re.Match],
        hidden: bool,
    ) -> CmdOutputObservation:
        is_special_key = self._is_special_key(command)
        assert len(ps1_matches) >= 1, (
            f"Expected at least one PS1 metadata block, but got {len(ps1_matches)}.\n"
            f"---FULL OUTPUT---\n{pane_content!r}\n---END OF OUTPUT---"
        )
        metadata = CmdOutputMetadata.from_ps1_match(ps1_matches[-1])

        # Special case where the previous command output is truncated due to history limit
        # We should get the content BEFORE the last PS1 prompt
        get_content_before_last_match = bool(len(ps1_matches) == 1)

        # Update the current working directory if it has changed
        if metadata.working_dir != self._cwd and metadata.working_dir:
            logger.debug(
                f"directory_changed: {self._cwd}; {metadata.working_dir}; {command}"
            )
            self._cwd = metadata.working_dir

        logger.debug(f"COMMAND OUTPUT: {pane_content}")
        # Extract the command output between the two PS1 prompts
        raw_command_output = self._combine_outputs_between_matches(
            pane_content,
            ps1_matches,
            get_content_before_last_match=get_content_before_last_match,
        )

        if get_content_before_last_match:
            # Count the number of lines in the truncated output
            num_lines = len(raw_command_output.splitlines())
            metadata.prefix = f"[Previous command outputs are truncated. Showing the last {num_lines} lines of the output below.]\n"

        metadata.suffix = (
            f"\n[The command completed with exit code {metadata.exit_code}.]"
            if not is_special_key
            else f"\n[The command completed with exit code {metadata.exit_code}. CTRL+{command[-1].upper()} was sent.]"
        )
        command_output = self._get_command_output(
            command,
            raw_command_output,
            metadata,
        )
        self.prev_status = BashCommandStatus.COMPLETED
        self.prev_output = ""  # Reset previous command output
        self._ready_for_next_command()
        return CmdOutputObservation(
            content=command_output,
            command=command,
            metadata=metadata,
            hidden=hidden,
        )

    def _handle_nochange_timeout_command(
        self,
        command: str,
        pane_content: str,
        ps1_matches: list[re.Match],
    ) -> CmdOutputObservation:
        self.prev_status = BashCommandStatus.NO_CHANGE_TIMEOUT
        if len(ps1_matches) != 1:
            logger.warning(
                "Expected exactly one PS1 metadata block BEFORE the execution of a command, "
                f"but got {len(ps1_matches)} PS1 metadata blocks:\n---\n{pane_content!r}\n---"
            )
        raw_command_output = self._combine_outputs_between_matches(
            pane_content, ps1_matches
        )
        metadata = CmdOutputMetadata()  # No metadata available
        metadata.suffix = (
            f"\n[The command has no new output after {self.NO_CHANGE_TIMEOUT_SECONDS} seconds. "
            f"{TIMEOUT_MESSAGE_TEMPLATE}]"
        )
        command_output = self._get_command_output(
            command,
            raw_command_output,
            metadata,
            continue_prefix="[Below is the output of the previous command.]\n",
        )
        return CmdOutputObservation(
            content=command_output,
            command=command,
            metadata=metadata,
        )

    def _handle_hard_timeout_command(
        self,
        command: str,
        pane_content: str,
        ps1_matches: list[re.Match],
        timeout: float,
    ) -> CmdOutputObservation:
        self.prev_status = BashCommandStatus.HARD_TIMEOUT
        if len(ps1_matches) != 1:
            logger.warning(
                "Expected exactly one PS1 metadata block BEFORE the execution of a command, "
                f"but got {len(ps1_matches)} PS1 metadata blocks:\n---\n{pane_content!r}\n---"
            )
        raw_command_output = self._combine_outputs_between_matches(
            pane_content, ps1_matches
        )
        metadata = CmdOutputMetadata()  # No metadata available
        metadata.suffix = (
            f"\n[The command timed out after {timeout} seconds. "
            f"{TIMEOUT_MESSAGE_TEMPLATE}]"
        )
        command_output = self._get_command_output(
            command,
            raw_command_output,
            metadata,
            continue_prefix="[Below is the output of the previous command.]\n",
        )

        return CmdOutputObservation(
            command=command,
            content=command_output,
            metadata=metadata,
        )

    def _combine_outputs_between_matches(
        self,
        pane_content: str,
        ps1_matches: list[re.Match],
        get_content_before_last_match: bool = False,
    ) -> str:
        """Combine all outputs between PS1 matches (verbatim from original)."""
        if len(ps1_matches) == 1:
            if get_content_before_last_match:
                # The command output is the content before the last PS1 prompt
                return pane_content[: ps1_matches[0].start()]
            else:
                # The command output is the content after the last PS1 prompt
                return pane_content[ps1_matches[0].end() + 1 :]
        elif len(ps1_matches) == 0:
            return pane_content
        combined_output = ""
        for i in range(len(ps1_matches) - 1):
            # Extract content between current and next PS1 prompt
            output_segment = pane_content[
                ps1_matches[i].end() + 1 : ps1_matches[i + 1].start()
            ]
            combined_output += output_segment + "\n"
        # Add the content after the last PS1 prompt
        combined_output += pane_content[ps1_matches[-1].end() + 1 :]
        logger.debug(f"COMBINED OUTPUT: {combined_output}")
        return combined_output

    # ------------------------------------------------------------------ #
    # Command execution -- state machine kept faithful to the original.
    # ------------------------------------------------------------------ #
    def execute(self, action: CmdRunAction) -> CmdOutputObservation | ErrorObservation:
        """Execute a command in the bash session."""
        if not self._initialized:
            raise RuntimeError("Bash session is not initialized")

        # Strip the command of any leading/trailing whitespace
        logger.debug(f"RECEIVED ACTION: {action}")
        command = action.command.strip()
        is_input: bool = action.is_input

        print(f"COMMAND: {command}")
        print(f"IS INPUT: {is_input}")
        print("================================================")

        # If the previous command is not completed, we need to check if the command is empty
        if self.prev_status not in {
            BashCommandStatus.CONTINUE,
            BashCommandStatus.NO_CHANGE_TIMEOUT,
            BashCommandStatus.HARD_TIMEOUT,
        }:
            if command == "":
                return CmdOutputObservation(
                    content="ERROR: No previous running command to retrieve logs from.",
                    command="",
                    metadata=CmdOutputMetadata(),
                )
            if is_input:
                return CmdOutputObservation(
                    content="ERROR: No previous running command to interact with.",
                    command="",
                    metadata=CmdOutputMetadata(),
                )

        # Check if the command is a single command or multiple commands
        splited_commands = split_bash_commands(command)
        if len(splited_commands) > 1:
            # Extracted to a local so the f-string contains no backslash
            # (keeps the error text byte-identical to the original while
            # remaining compilable on Python < 3.12 as well).
            _provided_commands = "\n".join(
                f"({i + 1}) {cmd}" for i, cmd in enumerate(splited_commands)
            )
            return ErrorObservation(
                content=(
                    "ERROR: Cannot execute multiple commands at once.\n"
                    "Please run each command separately OR chain them into a single command via && or ;\n"
                    f"Provided commands:\n{_provided_commands}"
                )
            )

        # Check if the command is blacklisted (only for non-input commands)
        if not is_input and command:
            blacklist_result = check_command_blacklist(command)
            if blacklist_result.is_blocked:
                logger.warning(f"Command blocked by blacklist: {command!r}")
                return ErrorObservation(
                    content=blacklist_result.feedback,
                    error_id="COMMAND_BLACKLISTED",
                )
            if "kill" in command or "rm" in command:
                print(
                    f"[POSSIBLE BLACKLIST]Command {command} bypassed blacklist check",
                    flush=True,
                )

        # Get initial state before sending command
        initial_pane_output = self._get_pane_content()
        initial_ps1_matches = CmdOutputMetadata.matches_ps1_metadata(
            initial_pane_output
        )
        initial_ps1_count = len(initial_ps1_matches)
        logger.debug(f"Initial PS1 count: {initial_ps1_count}")

        start_time = time.time()
        last_change_time = start_time
        last_pane_output = (
            initial_pane_output  # Use initial output as the starting point
        )

        # When prev command is still running, and we are trying to send a new command
        if (
            self.prev_status
            in {
                BashCommandStatus.HARD_TIMEOUT,
                BashCommandStatus.NO_CHANGE_TIMEOUT,
            }
            and not last_pane_output.rstrip().endswith(
                CMD_OUTPUT_PS1_END.strip()
            )  # prev command is not completed
            # NOTE: .strip() (not .rstrip()) -- our single-line prompt ends with
            # '}###PS1END###' (no '\n' before the marker), whereas
            # CMD_OUTPUT_PS1_END.rstrip() keeps a leading '\n'. .strip() matches
            # both the single-line and the original multi-line prompt shapes.
            and not is_input
            and command != ""  # not input and not empty command
        ):
            _ps1_matches = CmdOutputMetadata.matches_ps1_metadata(last_pane_output)
            # Use initial_ps1_matches if _ps1_matches is empty, otherwise use _ps1_matches
            # This handles the case where the prompt might be scrolled off screen but existed before
            current_matches_for_output = (
                _ps1_matches if _ps1_matches else initial_ps1_matches
            )
            raw_command_output = self._combine_outputs_between_matches(
                last_pane_output, current_matches_for_output
            )
            metadata = CmdOutputMetadata()  # No metadata available
            metadata.suffix = (
                f'\n[Your command "{command}" is NOT executed. '
                "The previous command is still running - You CANNOT send new commands until the previous command is completed. "
                "By setting `is_input` to `true`, you can interact with the current process: "
                f"{TIMEOUT_MESSAGE_TEMPLATE}]"
            )
            logger.debug(f"PREVIOUS COMMAND OUTPUT: {raw_command_output}")
            command_output = self._get_command_output(
                command,
                raw_command_output,
                metadata,
                continue_prefix="[Below is the output of the previous command.]\n",
            )
            return CmdOutputObservation(
                command=command,
                content=command_output,
                metadata=metadata,
                hidden=getattr(action, "hidden", False),
            )

        # Send actual command/inputs to the pty
        is_interrupt_signal = (
            is_input
            and self._is_special_key(command)
            and command.strip() in ("C-c", "C-z", "C-d")
        )
        sent_command = command != ""
        if command != "":
            is_special_key = self._is_special_key(command)
            if is_input:
                logger.debug(f"SENDING INPUT TO RUNNING PROCESS: {command!r}")
                self._send_keys_checked(command, enter=not is_special_key)
            else:
                # convert command to raw string
                command = escape_bash_special_chars(command)
                logger.debug(f"SENDING COMMAND: {command!r}")
                self._send_keys_checked(command, enter=not is_special_key)

        # For interrupt signals (C-c, C-z), use a shorter retry interval.
        # Many processes (e.g. pytest, long-running test suites) need multiple
        # SIGINTs: the first triggers graceful shutdown, the second aborts.
        SIGNAL_RETRY_INTERVAL = 1  # seconds between resending the signal
        MAX_SIGNAL_RETRIES = 5
        signal_retry_count = 0
        last_signal_send_time = time.time()

        # Loop until the command completes or times out
        while should_continue():
            _start_time = time.time()
            logger.debug(f"GETTING PANE CONTENT at {_start_time}")
            cur_pane_output = self._get_pane_content()
            logger.debug(
                f"PANE CONTENT GOT after {time.time() - _start_time:.2f} seconds"
            )
            cur_pane_lines = cur_pane_output.split("\n")
            if len(cur_pane_lines) <= 20:
                logger.debug("PANE_CONTENT: {cur_pane_output}")
            else:
                logger.debug(f"BEGIN OF PANE CONTENT: {cur_pane_lines[:10]}")
                logger.debug(f"END OF PANE CONTENT: {cur_pane_lines[-10:]}")
            ps1_matches = CmdOutputMetadata.matches_ps1_metadata(cur_pane_output)
            current_ps1_count = len(ps1_matches)

            # pty adaptation: only treat the session as "changed since we sent
            # the command" once new bytes have arrived. The tmux backend read
            # synchronously so it never observed the pre-command prompt; our
            # reader thread is async, so without this guard the first poll could
            # match the previous prompt (which ends in PS1_END) and wrongly
            # return empty output. Matches OpenHands' own pty terminal_session.
            output_changed_since_command = cur_pane_output != initial_pane_output

            if cur_pane_output != last_pane_output:
                last_pane_output = cur_pane_output
                last_change_time = time.time()
                logger.debug(f"CONTENT UPDATED DETECTED at {last_change_time}")

            # 1) Execution completed:
            # Condition 1: A new prompt has appeared since the command started.
            # Condition 2: The prompt count hasn't increased because the initial
            # prompt scrolled off the history buffer, BUT the current pane ends
            # with a prompt.
            #
            # pty-specific guard on Condition 2: because echo is OFF, there is no
            # command echo pushing content past the kept prompt block. That kept
            # block already ends with the PS1 END marker, so a bare
            # ``endswith(PS1_END)`` would fire the instant any stray byte (e.g. the
            # newline a signal handler emits) makes output_changed true -- before
            # the real new prompt renders -- completing on the STALE prompt with
            # empty output and the previous exit code. This was the interrupt/
            # timeout-recovery corruption. We therefore only trust Condition 2 when
            # the initial prompt has genuinely scrolled off (is no longer present in
            # the pane); otherwise completion must come from a NEW prompt block
            # (count increase), which is what actually happens on interrupt (the
            # post-interrupt prompt) and on every normal command.
            #
            # .strip() (not .rstrip()): our single-line prompt ends with
            # '}###PS1END###' with no '\n' before the marker, so the newline-prefixed
            # CMD_OUTPUT_PS1_END.rstrip() would never match.
            _initial_stripped = initial_pane_output.strip()
            initial_prompt_scrolled_off = bool(
                _initial_stripped and _initial_stripped not in cur_pane_output
            )
            if (not sent_command or output_changed_since_command) and (
                current_ps1_count > initial_ps1_count
                or (
                    cur_pane_output.rstrip().endswith(CMD_OUTPUT_PS1_END.strip())
                    and initial_prompt_scrolled_off
                )
            ):
                return self._handle_completed_command(
                    command,
                    pane_content=cur_pane_output,
                    ps1_matches=ps1_matches,
                    hidden=getattr(action, "hidden", False),
                )

            # Timeout checks should only trigger if a new prompt hasn't appeared yet.

            # For interrupt signals, resend if no change after SIGNAL_RETRY_INTERVAL.
            # After all retries are exhausted, escalate to SIGKILL via PID.
            if is_interrupt_signal:
                if signal_retry_count < MAX_SIGNAL_RETRIES:
                    time_since_last_signal = time.time() - last_signal_send_time
                    if time_since_last_signal >= SIGNAL_RETRY_INTERVAL:
                        signal_retry_count += 1
                        logger.info(
                            f"Signal retry: resending {command!r} "
                            f"(attempt {signal_retry_count + 1}/{MAX_SIGNAL_RETRIES + 1}, "
                            f"no response for {time_since_last_signal:.1f}s)"
                        )
                        self._send_keys_checked(command, enter=False)
                        last_signal_send_time = time.time()
                        last_change_time = time.time()
                elif signal_retry_count == MAX_SIGNAL_RETRIES:
                    # All signal retries exhausted — escalate to SIGKILL
                    signal_retry_count += 1  # prevent re-entry
                    print(f"[BASH_SIGNAL] All {MAX_SIGNAL_RETRIES} retries exhausted. Escalating to SIGKILL.", flush=True)
                    logger.info(
                        f"All {MAX_SIGNAL_RETRIES} signal retries exhausted. "
                        f"Escalating to SIGKILL via PID."
                    )
                    killed = self._kill_command_processes()
                    print(f"[BASH_SIGNAL] _kill_command_processes returned: {killed}", flush=True)
                    if killed:
                        last_change_time = time.time()

            # 2) Execution timed out since there's no change in output
            # for a while (self.NO_CHANGE_TIMEOUT_SECONDS)
            # We ignore this if the command is *blocking*
            time_since_last_change = time.time() - last_change_time
            logger.debug(
                f"CHECKING NO CHANGE TIMEOUT ({self.NO_CHANGE_TIMEOUT_SECONDS}s): elapsed {time_since_last_change}. Action blocking: {action.blocking}"
            )
            if (
                not action.blocking
                and time_since_last_change >= self.NO_CHANGE_TIMEOUT_SECONDS
            ):
                return self._handle_nochange_timeout_command(
                    command,
                    pane_content=cur_pane_output,
                    ps1_matches=ps1_matches,
                )

            # 3) Execution timed out due to hard timeout
            elapsed_time = time.time() - start_time
            logger.debug(
                f"CHECKING HARD TIMEOUT ({action.timeout}s): elapsed {elapsed_time:.2f}"
            )
            if action.timeout and elapsed_time >= action.timeout:
                logger.debug("Hard timeout triggered.")
                return self._handle_hard_timeout_command(
                    command,
                    pane_content=cur_pane_output,
                    ps1_matches=ps1_matches,
                    timeout=action.timeout,
                )

            logger.debug(f"SLEEPING for {self.POLL_INTERVAL} seconds for next poll")
            time.sleep(self.POLL_INTERVAL)
        raise RuntimeError("Bash session was likely interrupted...")
