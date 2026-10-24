#!/usr/bin/env python3
"""Whistle -> Needle 2 voice-agent console for the Waveshare ESP32-P4-ETH.

Adapted (in place, from a verbatim copy) of the upstream whistle-esp32
tools/needle_tui.py - the newest superset of the two upstream TUIs - for
p4/needle_demo, where ONE firmware runs BOTH models:

    hold BOOT and speak -> Whistle transcribes -> Needle 2 turns the
    request into a grammar-guaranteed tool call -> the LED action.

The display runs live the whole time (a voice turn can start at any
moment), so there is no typed REPL here: spoken requests just happen,
typed ones go through --send.

  --serial <port>    attach to the board (default watch mode, Ctrl-C quits)
  --reset            pulse RTS for a clean reboot, watch the whole boot
  --send "<text>"    send one typed request, exit when the turn ends
  --think 0|1        set reasoning display off/on (persisted on the board)
  --whistle          follow p4/whistle_demo instead (STT-only firmware)

Protocol (p4/needle_demo), one message per line:
  EVT model source=sd|flash ...          weights loaded
  EVT ready model=whistle depth=...      STT up
  EVT ready model=needle2 layers=...     LLM up
  EVT priming i/n PCT elapsed=Ns eta=Ns  one-time prefix prime at boot
  EVT prefix tokens=N ...                prime finished
  EVT bench tokens=N ms=M ms_per_tok=P tps=T
  EVT mic_channel=L                      push-to-talk mic slot
  p | ping -> EVT pong                   liveness probe (no side effect)
  s | status                             recap the summary lines above
  -- voice turn --
  EVT recording | EVT released held_ms=N
  EVT clip ms=N peak_dbfs=F floor_dbfs=F
  EVT no_speech | EVT too_short
  EVT speech ms=N frames=N
  EVT lang xx
  TXT <transcript>
  EVT time total_ms=.. encode_ms=.. decode_ms=.. audio_ms=N
  EVT prof <phase> N ms                  mel stem encoder cross_kv decoder
  -- LLM turn (voice or typed) --
  EVT reading i/n
  EVT prefill tokens=N ms=M tps=T sink=S
  TOK <text>                             literal "\\n" for newline
  CONF <float>
  EVT done tokens=N ms=M tps=T
  ACT led color=.. mode=.. duration=..   |   ACT none
  EVT think=0|1 | EVT depth=N
  ERR <reason>

A turn ends at ACT/ERR, not at "EVT done" - the ACT line reports what the
device actually decided.

Usage (from the repo root, with the project venv):
  .venv/bin/python p4/tools/needle_tui.py --serial /dev/ttyACM1
  .venv/bin/python p4/tools/needle_tui.py --serial /dev/ttyACM1 --reset
  .venv/bin/python p4/tools/needle_tui.py --serial /dev/ttyACM1 \
      --send "make the light yellow for three seconds"
  .venv/bin/python p4/tools/needle_tui.py --serial /dev/ttyACM1 --whistle
"""
import argparse
import json
import queue
import re
import sys
import threading
import time

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CONSOLE = Console()

SWATCH = {
    "red": "#ff3b30", "green": "#34c759", "blue": "#0a84ff",
    "yellow": "#ffd60a", "purple": "#af52de", "white": "#f2f2f7",
    "cyan": "#32ade6", "magenta": "#ff2d55", "pink": "#ff6482",
    "orange": "#ff9f0a", "off": "#3a3a3c",
}


# ------------------------------------------------------------------ backend


class Backend:
    def send(self, prompt):
        raise NotImplementedError

    def lines(self):
        raise NotImplementedError

    def close(self):
        pass


class SerialBackend(Backend):
    """Link to the ESP32-P4-ETH over its WCH CH343P USB-UART bridge.

    pyserial's default open asserts DTR+RTS at once; through the board's
    auto-reset circuit that glitches the chip into download mode.  Open
    with both lines low instead.  reset=True pulses RTS alone, which
    reboots the board cleanly (hold no buttons) - handy to watch a full
    boot: model load, the one-time prime, the bench, the READY banner.
    """

    def __init__(self, port, baud=115200, reset=False):
        import serial
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, baud, 0.2
        self.s.dtr, self.s.rts = False, False
        self.s.open()
        if reset:
            self.reset()
        self.name = "ESP32-P4 " + port.split("/")[-1]
        self.q = queue.Queue()
        self._stop = False
        threading.Thread(target=self._reader, daemon=True).start()

    def reset(self):
        """Pulse RTS alone: a clean reboot (hold no buttons)."""
        self.s.rts = True
        time.sleep(0.15)
        self.s.rts = False

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                data = self.s.read(512)
            except Exception:
                return
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.q.put(line.decode("utf-8", "replace").rstrip("\r"))

    def send(self, prompt):
        self.s.write((prompt + "\n").encode())
        self.s.flush()

    def lines(self, idle_timeout=180):
        """Yield until the turn completes, or the board goes quiet."""
        last = time.time()
        while time.time() - last < idle_timeout:
            try:
                line = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            last = time.time()
            yield line
            if line.startswith(("ACT ", "ERR ")):
                return

    def close(self):
        self._stop = True
        try:
            self.s.close()
        except Exception:
            pass


def _kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def ensure_board(backend, sess=None):
    """Check the console is alive; reboot the board if it is not.

    Opening this USB-UART bridge can latch the chip into download mode -
    dead silent.  The firmware answers a bare "p" with "EVT pong" (no side
    effects).  No reply and no traffic at all means latched: pulse RTS for
    a clean reboot and let the caller watch the boot.  Lines seen while
    probing are fed to sess; any traffic means the board is alive (mid
    boot), even without a pong yet.
    """
    backend.send("p")
    deadline = time.time() + 1.5
    saw_traffic = False
    while time.time() < deadline:
        try:
            line = backend.q.get(timeout=0.2)
        except queue.Empty:
            continue
        saw_traffic = True
        if line.startswith("EVT pong"):
            # Alive: ask for a recap so a TUI attaching after boot can fill
            # its stats panel ("s" = status on both firmwares).
            backend.send("s")
            return True
        if sess is not None:
            sess.feed(line)
    if saw_traffic:
        return True
    backend.reset()
    return False


LANG_NAMES = {"en": "English", "de": "Deutsch", "fr": "Français", "es": "Español",
              "it": "Italiano", "nl": "Nederlands", "pl": "Polski"}


# ------------------------------------------------------------- agent session


class AgentSession:
    """State of the merged firmware: boot, then voice or typed turns."""

    def __init__(self):
        # board-level state
        self.status = "booting"
        self.st_info = {}       # EVT ready model=whistle ...
        self.llm_info = {}      # EVT ready model=needle2 ...
        self.bench = None       # EVT bench ...
        self.mic = None         # EVT mic_channel=L
        self.think = False       # firmware default (recap confirms)
        self.priming = None     # (i, n, pct, eta) during the boot prime
        self.last = ("", 0.0)   # last protocol line seen, and when
        # per-turn state
        self.voice = False      # True when the turn came from the microphone
        self.prompt = ""        # typed prompt (--send)
        self.text = None        # TXT transcript (voice)
        self.lang = None
        self.clip = {}
        self.speech = {}
        self.time = {}
        self.prof = {}
        self.raw = ""
        self.reading = None     # (i, n) during prefill
        self.prefill = None
        self.done = None
        self.conf = None
        self.action = None
        self.error = None
        self.history = []       # (label, action) of earlier turns

    def new_turn(self, voice, prompt=""):
        label = self.text if self.voice else (self.prompt or None)
        if label is not None or self.action is not None:
            self.history = ([(label, self.action)] + self.history)[:4]
        self.voice, self.prompt = voice, prompt
        self.text, self.lang = None, None
        self.clip, self.speech = {}, {}
        self.time, self.prof = {}, {}
        self.raw = ""
        self.reading = self.prefill = self.done = None
        self.conf = self.action = self.error = None

    def feed(self, line):
        if line and not line.startswith(("EVT =", "EVT  READY", "EVT  e.g")):
            self.last = (line, time.time())   # skip banner art
        if line.startswith("EVT priming "):
            m = re.match(r"EVT priming (\d+)/(\d+)\s+(\d+)%.*eta=(\d+)s", line)
            if m:
                self.priming = tuple(int(g) for g in m.groups())
                self.status = "priming"
        elif line.startswith("EVT prefix"):
            self.priming = None
        elif line.startswith("EVT ready"):
            info = _kv(line[len("EVT ready "):])
            if info.get("model") == "whistle":
                self.st_info = info
            elif info.get("model") == "needle2":
                self.llm_info = info
        elif line.startswith("EVT bench"):
            self.bench = _kv(line[len("EVT bench "):])
        elif line.startswith("EVT mic_channel="):
            self.mic = line.split("=")[1][:1]
        elif line.startswith("EVT depth="):
            self.st_info["depth"] = line.split("=")[1].split()[0]
        elif line.startswith("EVT think="):
            self.think = line.strip().endswith("1")
        elif line.startswith("EVT  READY"):
            self.status, self.error = "ready", None
        # -- voice turn --
        elif line.startswith("EVT recording"):
            self.new_turn(True)
            self.status = "recording"
        elif line.startswith("EVT released"):
            self.status = "transcribing"
        elif line.startswith("EVT clip"):
            self.clip = _kv(line[len("EVT clip "):])
        elif line.startswith(("EVT no_speech", "EVT too_short")):
            self.status = "nothing heard"
        elif line.startswith("EVT speech"):
            self.speech = _kv(line[len("EVT speech "):])
        elif line.startswith("EVT lang "):
            self.lang = line.split()[2]
        elif line.startswith("TXT"):
            self.text = line[4:] if len(line) > 4 else ""
        elif line.startswith("EVT time"):
            self.time = _kv(line)
        elif line.startswith("EVT prof"):
            parts = line.split()
            if len(parts) >= 4:
                self.prof[parts[2]] = parts[3]
        # -- LLM turn (voice or typed) --
        elif line.startswith("EVT reading "):
            m = re.match(r"EVT reading (\d+)/(\d+)", line)
            if m:
                self.reading = (int(m.group(1)), int(m.group(2)))
                self.status = "reading prompt"
        elif line.startswith("EVT prefill"):
            self.prefill = _kv(line)
            self.reading = None
            self.status = "generating"
        elif line.startswith("TOK "):
            self.raw += line[4:].replace("\\n", "\n")
            self.status = "generating"
        elif line.startswith("CONF "):
            try:
                self.conf = float(line[5:])
            except ValueError:
                pass
        elif line.startswith("EVT done"):
            self.done = _kv(line)
            self.status = "acting"
        elif line.startswith("ACT "):
            self.action = _kv(line[4:]) if "=" in line else {}
            self.status = "done" if self.action else "no match"
        elif line.startswith("ERR "):
            self.error = line[4:]
            self.status = "error"

    def think_text(self):
        m = re.search(r"<think>(.*?)(</think>|$)", self.raw, re.S)
        return m.group(1).strip() if m else ""

    def call_json(self):
        m = re.search(r"<tool_call>(.*?)(</tool_call>|$)", self.raw, re.S)
        if m:
            text = m.group(1)
        else:
            # The streamed TOK text may not render the <tool_call> tags
            # (special tokens decode to nothing); fall back to spotting
            # the JSON array itself.
            m = re.search(r"(\[\s*\{.*\}\s*\])", self.raw, re.S)
            if not m:
                return None
            text = m.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def led(self):
        if self.action:
            return self.action.get("color"), self.action.get("mode")
        call = self.call_json()
        if isinstance(call, list) and call:
            args = call[0].get("arguments", {})
            return args.get("color"), args.get("mode")
        return None, None


def _dur(ms):
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return "?"
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def render_agent(sess, backend_name):
    layout = Layout()
    layout.split_column(
        Layout(name="head", size=3),
        Layout(name="body"),
        Layout(name="foot", size=9),
    )

    col = {"booting": "grey50", "priming": "yellow", "ready": "green",
           "recording": "bold white", "transcribing": "cyan",
           "reading prompt": "yellow", "generating": "cyan",
           "acting": "magenta", "done": "green", "nothing heard": "red",
           "no match": "yellow", "error": "bold red"}.get(sess.status, "grey50")
    head = Text()
    head.append("  Whistle → Needle 2 ", style="bold white")
    head.append("· voice agent · on-device · ", style="grey58")
    head.append(backend_name, style="bold magenta")
    head.append("    ")
    head.append("●", style=col)
    head.append(f" {sess.status}", style=col)
    if not sess.think:
        head.append("   [fast: no reasoning]", style="yellow")
    layout["head"].update(Panel(head, border_style="grey30"))

    blocks = []

    if sess.priming:
        i, n, pct, eta = sess.priming
        bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
        p = Text()
        p.append("priming the tool schema (one time only)\n\n", style="bold yellow")
        p.append(bar + f"  {pct}%  ", style="yellow")
        p.append(f"{i}/{n} tokens · about {eta}s left", style="grey62")
        blocks.append(Align.center(p, vertical="middle"))
    else:
        if sess.voice and sess.text is not None:
            t = Text()
            t.append("“", style="grey50")
            t.append(sess.text or "(nothing)", style="bold white")
            t.append("”", style="grey50")
            title = "[grey58]heard[/]" + (f" [bold cyan]{LANG_NAMES.get(sess.lang, sess.lang)}[/]"
                                          if sess.lang else "")
            blocks.append(Panel(Align.center(t, vertical="middle"), title=title,
                                border_style="cyan", height=5))
            blocks.append(Text())
        elif sess.prompt:
            p = Text()
            p.append("› ", style="bold green")
            p.append(sess.prompt, style="bold white")
            blocks.append(p)
            blocks.append(Text())
        elif sess.status == "recording":
            blocks.append(Align.center(
                Text("listening...  release the button when you are done",
                     style="grey62"), vertical="middle"))
            blocks.append(Text())
        elif sess.status == "transcribing":
            blocks.append(Align.center(
                Text("transcribing on the ESP32-P4...", style="grey62"),
                vertical="middle"))
            blocks.append(Text())

        if sess.reading:
            i, n = sess.reading
            r = Text()
            r.append("needle 2 is reading your request  ", style="yellow")
            r.append("▪" * i + "▫" * max(0, n - i), style="yellow")
            r.append(f"  {i}/{n}", style="grey62")
            blocks.append(r)
            blocks.append(Text())

        think = sess.think_text()
        if think:
            blocks.append(Panel(Text(think, style="italic grey70"),
                                title="[grey58]reasoning[/]", border_style="grey30"))

        call = sess.call_json()
        if call is not None:
            if isinstance(call, list):
                blocks.append(Panel(Text(json.dumps(call, indent=2), style="bold cyan"),
                                    title="[grey58]tool call[/]", border_style="cyan"))
            else:
                blocks.append(Panel(Text(str(call), style="yellow"),
                                    title="[grey58]tool call (streaming)[/]",
                                    border_style="grey30"))

        if sess.error:
            blocks.append(Panel(Text(sess.error, style="bold red"),
                                title="error", border_style="red"))

        if not blocks:
            blocks = [Align.center(
                Text("hold BOOT and speak  ·  \"flash the light blue for two seconds\"\n"
                     "or restart with:  --send \"make the light red\"",
                     style="grey42"), vertical="middle")]

    for old, act in sess.history:
        if old is None:
            continue
        h = Text("  " + (old or "(nothing)"), style="grey42")
        if act and act.get("color"):
            h.append(f"   → {act['color']} {act.get('mode', '')}", style="grey35")
        blocks.append(h)
    layout["body"].update(Panel(Group(*blocks), border_style="grey30"))

    color, mode = sess.led()
    swatch = Text()
    if color:
        hexc = SWATCH.get(color, "#888888")
        bar = "█" * 22
        swatch.append(bar + "\n", style=hexc)
        swatch.append(bar + "\n", style="grey15" if mode == "flash" else hexc)
        swatch.append(bar + "\n", style=hexc)
        swatch.append(f"{color} · {mode or '-'}", style="bold " + hexc)
        if sess.action and sess.action.get("duration"):
            swatch.append(f" · {sess.action['duration']}s", style="grey62")
    else:
        swatch.append("\n\n  (no action)", style="grey35")

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="grey58", justify="right")
    stats.add_column(style="bold white", overflow="fold")

    # -- board section: always on, so "what is going on" needs no turn --
    board = Text(sess.status, style=col)
    if sess.priming:
        board.append(f" · priming {sess.priming[2]}%", style=col)
    if sess.mic:
        board.append(f" · mic {sess.mic}", style="grey58")
    if sess.bench:
        board.append(f" · bench {sess.bench.get('tps', '?')} tok/s", style="grey58")
    stats.add_row("board", board)

    if sess.st_info:
        stt = Text("✓ ", style="green")
        stt.append(f"whistle · depth {sess.st_info.get('depth', '?')}"
                   f" · vocab {sess.st_info.get('vocab', '?')}")
    else:
        stt = Text("… waiting for boot", style="grey42")
    stats.add_row("stt", stt)

    if sess.llm_info:
        llm = Text("✓ ", style="green")
        llm.append(f"needle2 · {sess.llm_info.get('layers', '?')} layers"
                   f" · window {sess.llm_info.get('window', '?')}")
        try:
            mb = float(sess.llm_info.get("psram_free", 0)) / 1e6
            llm.append(f" · psram {mb:.1f} MB free", style="grey58")
        except ValueError:
            pass
    else:
        llm = Text("… waiting for boot", style="grey42")
    stats.add_row("llm", llm)

    # -- turn section (a turn in progress replaces the idle line) --
    if sess.time or sess.prefill or sess.done:
        stats.add_row("", "")
        if sess.time:
            stats.add_row("heard", _dur(sess.time.get("audio_ms")))
            stats.add_row("stt", _dur(sess.time.get("total_ms")))
        if sess.prof:
            stats.add_row("phases", "  ".join(f"{k} {_dur(v)}" for k, v in sess.prof.items()
                                              if k in ("mel", "stem", "encoder",
                                                       "cross_kv", "decoder")))
        if sess.prefill:
            stats.add_row("prompt", f"{sess.prefill.get('tokens', '?')} tok  "
                                    f"{sess.prefill.get('tps', '?')} tok/s")
        if sess.done:
            stats.add_row("generated", f"{sess.done.get('tokens', '?')} tok  "
                                       f"{sess.done.get('tps', '?')} tok/s")
            ms = sess.done.get("ms")
            if ms:
                stats.add_row("llm", f"{float(ms) / 1000:.1f} s")
        if sess.conf is not None:
            filled = int(max(0.0, min(1.0, sess.conf)) * 20)
            c = "green" if sess.conf >= 0.5 else "yellow"
            bar = Text("█" * filled + "░" * (20 - filled), style=c)
            bar.append(f"  {sess.conf:.3f}", style="bold " + c)
            stats.add_row("confidence", bar)
    elif sess.last[0]:
        txt = sess.last[0]
        txt = txt[:44] + "…" if len(txt) > 44 else txt
        last = Text(txt, style="grey42")
        last.append(f"  {int(time.time() - sess.last[1])}s ago", style="grey35")
        stats.add_row("last", last)

    foot = Layout()
    foot.split_row(Layout(name="led", ratio=1), Layout(name="stat", ratio=2))
    foot["led"].update(Panel(swatch, title="[grey58]LED[/]", border_style="grey30"))
    foot["stat"].update(Panel(stats, title="[grey58]stats[/]", border_style="grey30"))
    layout["foot"].update(foot)
    return layout


# -------------------------------------------------- whistle_demo (STT only)


class WhistleSession:
    """State of one push-to-talk turn on p4/whistle_demo."""

    def __init__(self):
        self.status = "attaching"
        self.model_info = ""
        self.mic = None
        self.last = ("", 0.0)
        self.text = None
        self.action = None
        self.time = {}
        self.prof = {}
        self.speech = {}
        self.lang = None
        self.error = None
        self.history = []        # (transcript, action) of earlier turns

    def feed(self, line):
        if line and not line.startswith(("EVT =", "EVT  READY", "EVT  e.g")):
            self.last = (line, time.time())   # skip banner art
        if line.startswith("EVT ready"):
            if "model=needle2" in line:
                return          # needle_demo firmware on the wire, not us
            self.model_info = line[len("EVT ready "):]
            self.status, self.error = "ready", None
        elif line.startswith("EVT mic_channel="):
            self.mic = line.split("=")[1][:1]
        elif line.startswith("EVT selftest"):
            self.status = "self-test"
        elif line.startswith("EVT recording"):
            if self.text is not None:
                self.history = ([(self.text, self.action)] + self.history)[:4]
            self.status, self.text, self.action = "recording", None, None
            self.time, self.prof, self.speech, self.lang = {}, {}, {}, None
            self.error = None
        elif line.startswith("EVT released"):
            self.status = "transcribing"
        elif line.startswith("EVT speech"):
            self.speech = _kv(line)
        elif line.startswith(("EVT no_speech", "EVT too_short")):
            self.status = "nothing heard"
        elif line.startswith("EVT lang "):
            self.lang = line.split()[2]
        elif line.startswith("TXT"):
            self.text = line[4:] if len(line) > 4 else ""
        elif line.startswith("EVT time"):
            self.time = _kv(line)
        elif line.startswith("EVT prof"):
            parts = line.split()
            if len(parts) >= 4:
                self.prof[parts[2]] = parts[3]
        elif line.startswith("ACT "):
            self.action = _kv(line[4:]) if "=" in line else {}
            if self.status != "nothing heard":
                self.status = "done" if self.action else "no colour heard"
        elif line.startswith("ERR "):
            self.error = line[4:]
            self.status, self.action = "error", None
        elif line.startswith("EVT  READY"):
            if self.status in ("self-test", "error"):
                self.status, self.error = "ready", None


def render_whistle(sess, backend_name):
    layout = Layout()
    layout.split_column(
        Layout(name="head", size=3),
        Layout(name="body"),
        Layout(name="foot", size=9),
    )
    col = {"attaching": "grey50", "ready": "green", "self-test": "yellow",
           "recording": "bold white", "transcribing": "cyan",
           "done": "green", "nothing heard": "red",
           "no colour heard": "yellow", "error": "bold red"}.get(sess.status, "grey50")
    depth = _kv(sess.model_info).get("depth", "?")
    head = Text()
    head.append("  Whistle ", style="bold white")
    head.append(f"· speech-to-text · {depth}-layer decoder · on-device · ", style="grey58")
    head.append(backend_name, style="bold magenta")
    head.append("    ")
    head.append("●", style=col)
    head.append(f" {sess.status}", style=col)
    layout["head"].update(Panel(head, border_style="grey30"))

    blocks = []
    if sess.error is not None:
        blocks.append(Panel(Text(sess.error, style="red"), title="error", border_style="red"))
    hint = {"ready": "hold the button and speak  ·  \"flash the light blue\"",
            "self-test": "boot self-test: transcribing a built-in clip...",
            "recording": "listening...  release the button when you are done",
            "transcribing": "transcribing on the ESP32-P4..."}.get(sess.status)
    if sess.text is not None:
        t = Text()
        t.append("“", style="grey50")
        t.append(sess.text or "(nothing)", style="bold white")
        t.append("”", style="grey50")
        title = "[grey58]heard[/]" + (f" [bold cyan]{LANG_NAMES.get(sess.lang, sess.lang)}[/]"
                                      if sess.lang else "")
        blocks.append(Panel(Align.center(t, vertical="middle"), title=title,
                            border_style="cyan", height=5))
    elif hint:
        blocks.append(Align.center(Text(hint, style="grey62"), vertical="middle"))
    for old, act in sess.history:
        h = Text("  " + (old or "(nothing)"), style="grey42")
        if act and act.get("color"):
            h.append(f"   → {act['color']} {act.get('mode', '')}", style="grey35")
        blocks.append(h)
    layout["body"].update(Panel(Group(*blocks), border_style="grey30"))

    swatch = Text()
    a = sess.action or {}
    if a.get("color"):
        hexc = SWATCH.get(a["color"], "#888888")
        bar = "█" * 22
        swatch.append(bar + "\n", style=hexc)
        swatch.append(bar + "\n", style="grey15" if a.get("mode") == "flash" else hexc)
        swatch.append(bar + "\n", style=hexc)
        swatch.append(f"{a['color']} · {a.get('mode', '-')}", style="bold " + hexc)
        if a.get("duration"):
            swatch.append(f" · {a['duration']}s", style="grey62")
        if a.get("period"):
            swatch.append(f" · every {a['period']}s", style="grey62")
    elif sess.status == "nothing heard":
        swatch.append("\n  two red blinks", style="red")
    else:
        swatch.append("\n\n  (no action)", style="grey35")

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="grey58", justify="right")
    stats.add_column(style="bold white", overflow="fold")

    # -- board section: always on --
    board = Text(sess.status, style=col)
    if sess.mic:
        board.append(f" · mic {sess.mic}", style="grey58")
    stats.add_row("board", board)
    if sess.model_info:
        m = _kv(sess.model_info)
        try:
            mb = f" · psram {float(m.get('psram_free', 0)) / 1e6:.1f} MB free"
        except ValueError:
            mb = ""
        stats.add_row("model", f"whistle · depth {m.get('depth', '?')}"
                               f" · enc {m.get('enc_layers', '?')}{mb}")

    if sess.time:
        stats.add_row("speech", _dur(sess.time.get("audio_ms", 0)))
        stats.add_row("total", f"{_dur(sess.time.get('total_ms', 0))}  "
                               f"({sess.time.get('tokens', '?')} tokens)")
    if sess.prof:
        stats.add_row("phases", "  ".join(f"{k} {_dur(v)}" for k, v in sess.prof.items()
                                          if k in ("mel", "stem", "encoder", "cross_kv", "decoder")))
    if not sess.time and sess.last[0]:
        txt = sess.last[0]
        txt = txt[:44] + "…" if len(txt) > 44 else txt
        last = Text(txt, style="grey42")
        last.append(f"  {int(time.time() - sess.last[1])}s ago", style="grey35")
        stats.add_row("last", last)

    foot = Layout()
    foot.split_row(Layout(name="led", ratio=1), Layout(name="stat", ratio=2))
    foot["led"].update(Panel(swatch, title="[grey58]LED[/]", border_style="grey30"))
    foot["stat"].update(Panel(stats, title="[grey58]stats[/]", border_style="grey30"))
    layout["foot"].update(foot)
    return layout


# ------------------------------------------------------------------- mains


def agent_main(args):
    sess = AgentSession()
    backend = None
    try:
        backend = SerialBackend(args.serial)
        if args.reset or not ensure_board(backend, sess):
            pass    # board is (re)booting: the watch loop renders the boot
        else:
            sess.status = "ready"    # attached to a live, idle board
            if args.think is not None:
                backend.send(f"!think {args.think}")
        CONSOLE.clear()
        with Live(render_agent(sess, backend.name), console=CONSOLE,
                  refresh_per_second=8, screen=False) as live:
            sent = False
            ready_seen = False
            while True:
                try:
                    line = backend.q.get(timeout=0.25)
                except queue.Empty:
                    line = None
                if line is not None:
                    sess.feed(line)
                    if line.startswith("EVT  READY"):
                        ready_seen = True
                    if sent and line.startswith(("ACT ", "ERR ")):
                        live.update(render_agent(sess, backend.name))
                        break
                # Re-render on every pass (4 Hz), not only on new lines:
                # the "last ... ago" ticker ticks and the panel visibly lives.
                live.update(render_agent(sess, backend.name))
                # With --reset, hold the typed request until the boot (and
                # the one-time prime) has finished; the UART buffer would
                # hold it anyway, but this keeps the display truthful.
                if args.send and not sent and (not args.reset or ready_seen):
                    backend.send(args.send)
                    sess.new_turn(False, args.send)
                    sess.status = "reading prompt"
                    sent = True
                    live.update(render_agent(sess, backend.name))
    except KeyboardInterrupt:
        pass
    finally:
        if backend:
            backend.close()
    CONSOLE.print("\n[grey50]bye[/]")
    return 1 if sess.status == "error" else 0


def whistle_main(args):
    sess = WhistleSession()
    backend = None
    try:
        backend = SerialBackend(args.serial)
        if args.reset or not ensure_board(backend, sess):
            pass    # board is (re)booting: the watch loop renders the boot
        else:
            sess.status = "ready"    # attached to a live, idle board
        CONSOLE.clear()
        with Live(render_whistle(sess, backend.name), console=CONSOLE,
                  refresh_per_second=8, screen=False) as live:
            while True:
                try:
                    line = backend.q.get(timeout=0.25)
                except queue.Empty:
                    line = None
                if line is not None:
                    sess.feed(line)
                live.update(render_whistle(sess, backend.name))
    except KeyboardInterrupt:
        pass
    finally:
        if backend:
            backend.close()
    CONSOLE.print("\n[grey50]bye[/]")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serial", metavar="PORT", required=True)
    ap.add_argument("--reset", action="store_true",
                    help="pulse RTS to reboot the board cleanly first")
    ap.add_argument("--send", metavar="MSG",
                    help="send one typed request, exit when the turn ends")
    ap.add_argument("--think", type=int, choices=(0, 1), default=None,
                    help="set reasoning display off/on (persisted on the board)")
    ap.add_argument("--whistle", action="store_true",
                    help="follow p4/whistle_demo (push-to-talk STT only)")
    args = ap.parse_args()
    return whistle_main(args) if args.whistle else agent_main(args)


if __name__ == "__main__":
    sys.exit(main())
