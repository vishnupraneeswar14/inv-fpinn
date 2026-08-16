#!/usr/bin/env python3
"""paramseva_login: automated SSH login with TOTP-based 2FA.

Pure-stdlib TOTP (RFC 6238) + pexpect session. The SSH prompts are matched
in any order (password / username / verification code / host key), so it
works regardless of how the cluster orders them.

Credentials come from a .env file (default: ./.env) in dotenv format:
    HOST=login.cluster
    USERNAME=vishn
    PASSWORD=...
    TOTP_SECRET=JBSWY3DPEHPK3PXP      # base32, or a full otpauth:// URL
Lines starting with # are comments. Empty values fall back to the legacy
~/.paramseva_login file, and CLI flags override everything.

Key names: HOST, USERNAME (or USER), PASSWORD, TOTP_SECRET.

Usage:
    paramseva_login.py                       # log in using ./.env
    paramseva_login.py --env /path/.env --code
    paramseva_login.py --host ... --user ... # override .env
    paramseva_login.py --user 2              # log in as 2nd user block in .env

Multiple users: setup.py appends one block per user (blank line between
blocks). Pick interactively, or pass --user <1-based index>. --code uses
user 1 by default and prints the username next to the code.

Get the secret from Google Authenticator:
    App > Export accounts > verify PIN > scan the shown QR with zbarimg:
    zbarimg -q --raw qr.png   ->   otpauth://totp/...?secret=BASE32...
"""

import argparse
import base64
import hashlib
import hmac
import os
import re
import signal
import struct
import sys
import termios
import time
from pathlib import Path

import pexpect

CONFIG_PATH = Path.home() / ".paramseva_login"


def totp(secret_b32, window=30, digits=6, now=None):
    secret = secret_b32.strip().replace(" ", "").upper()
    secret = secret + ("=" * ((8 - len(secret) % 8) % 8))
    key = base64.b32decode(secret)
    counter = struct.pack(">Q", int(now if now is not None else time.time()) // window)
    digest = hmac.new(key, counter, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return f"{code:0{digits}d}"


def extract_secret(value):
    value = value.strip().lstrip("=")
    if "secret=" in value:
        m = re.search(r"(?:^|[?&])secret=([^&]+)", value)
        if m:
            return m.group(1)
    return value.split("&")[0].split("?")[0]


def load_config():
    cfg = {}
    if CONFIG_PATH.exists():
        for line in CONFIG_PATH.read_text().splitlines():
            for k, v in parse_dotenv_line(line):
                cfg[k] = v
    return cfg


def parse_dotenv_line(line):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return []
    k, _, v = line.partition("=")
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1]
    return [(k.strip().upper(), v)]


def load_dotenv(path=".env"):
    env = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            for k, v in parse_dotenv_line(line):
                env[k] = v
    return env


def load_dotenv_entries(path=".env"):
    """Parse .env into an ordered list of user blocks.

    Blocks are separated by blank lines; each block carries one user's
    HOST / USERNAME / PASSWORD / TOTP_SECRET. A legacy flat file (no
    blank lines) comes back as a single entry, so old setups keep working.
    """
    entries = []
    cur = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                if cur:
                    entries.append(cur)
                    cur = {}
                continue
            for k, v in parse_dotenv_line(line):
                cur[k] = v
    if cur:
        entries.append(cur)
    return entries


def pick_user(entries):
    print("multiple users in config, pick one:")
    for i, e in enumerate(entries, 1):
        name = e.get("USERNAME") or e.get("USER") or "?"
        host = e.get("HOST") or "?"
        print(f"  [{i}] {name} @ {host}")
    while True:
        try:
            raw = input(f"pick 1-{len(entries)} (Enter = 1): ").strip()
        except EOFError:
            sys.exit(f"multiple users found; pass --user <1-{len(entries)}>")
        if not raw:
            return 0
        if raw.isdigit() and 1 <= int(raw) <= len(entries):
            return int(raw) - 1
        print("invalid choice")


PROMPTS = [
    r"(?i)verification code",      # 0  code
    r"(?i)one[- ]time password",   # 1  code
    r"(?i)authentication code",    # 2  code
    r"(?i)security code",          # 3  code
    r"(?i)\botp\b",                # 4  code
    r"(?i)password",               # 5  password
    r"(?i)username",               # 6  username
    r"login as",                   # 7  username
    r"yes/no",                     # 8  host key accept
    r"Type the string above",      # 9  captcha
    r"(?i)last login",             # 10 done
    pexpect.EOF,                   # 11
    pexpect.TIMEOUT,               # 12
]
CODE_IDX = (0, 1, 2, 3, 4)
PASS_IDX = 5
USER_IDX = (6, 7)
CAPTCHA_IDX = 9
DONE_IDX = 10


def solve_captcha(text):
    for m in re.findall(r"\(\s*([^)]*)\)", text):
        chars = [c for c in re.split(r"[\s|]+", m) if c.isalnum()]
        if chars:
            return "".join(chars)
    return None


def safe_code(secret):
    try:
        return totp(extract_secret(secret))
    except Exception:
        sys.exit("TOTP_SECRET in the config is invalid: expected base32 chars (A-Z, 2-7)")


def sync_terminal(child):
    if not sys.stdin.isatty():
        return
    try:
        fd = child.child_fd if hasattr(child, "child_fd") else child.fd
        attrs = termios.tcgetattr(sys.stdin.fileno())
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        cols, rows = os.get_terminal_size()
        child.setwinsize(rows, cols)
    except (OSError, termios.error):
        pass

    def _winch(sig, frame):
        try:
            cols, rows = os.get_terminal_size()
            child.setwinsize(rows, cols)
        except OSError:
            pass

    signal.signal(signal.SIGWINCH, _winch)


def login(ssh_cmd, host, user, password, secret, timeout=30, attempts=5, backoff=15, debug=False):
    print(f"connecting as {user}@{host} ...", file=sys.stderr)
    seen_prompt = False
    for attempt in range(1, attempts + 1):
        child = pexpect.spawn(ssh_cmd,
                              ["-o", "PubkeyAuthentication=no",
                               "-o", "PreferredAuthentications=keyboard-interactive,password",
                               f"{user}@{host}"],
                              encoding="utf-8", timeout=timeout)
        sync_terminal(child)
        code_hits = 0
        captcha_hits = 0
        seen_prompt = False
        retry_needed = False
        try:
            while True:
                i = child.expect(PROMPTS)
                if debug:
                    print(f"[debug] attempt {attempt} matched idx {i}: "
                          f"{repr((child.before or '')[-80:])}", file=sys.stderr)
                if i in CODE_IDX:
                    seen_prompt = True
                    code_hits += 1
                    if code_hits > 3:
                        child.close(force=True)
                        sys.exit("verification code rejected (check TOTP_SECRET)")
                    child.sendline(safe_code(secret))
                elif i == PASS_IDX:
                    seen_prompt = True
                    child.sendline(password)
                elif i in USER_IDX:
                    seen_prompt = True
                    child.sendline(user)
                elif i == CAPTCHA_IDX:
                    seen_prompt = True
                    captcha_hits += 1
                    if captcha_hits > 3:
                        child.close(force=True)
                        sys.exit("captcha rejected 3 times (could not solve) - check the captcha format")
                    answer = solve_captcha(child.before or "")
                    print(f"captcha round {captcha_hits}: {answer!r}",
                          file=sys.stderr)
                    if not answer:
                        child.close(force=True)
                        sys.exit("could not parse captcha text")
                    child.sendline(answer)
                elif i == 8:
                    child.sendline("yes")
                elif i == DONE_IDX:
                    break
                else:
                    tail = (child.before or child.buffer or "").strip()
                    child.close(force=True)
                    if not seen_prompt:
                        print(f"attempt {attempt}/{attempts}: network refused connection, retrying in {backoff}s",
                              file=sys.stderr)
                        retry_needed = True
                        break
                    sys.exit(f"login failed: {tail[:400]}")
            if retry_needed:
                time.sleep(backoff)
                continue
            sync_terminal(child)
            child_fd = child.child_fd if hasattr(child, "child_fd") else child.fd
            a = list(termios.tcgetattr(child_fd))
            a[0] = termios.IGNBRK
            a[1] = 0
            a[2] = (a[2] & ~(termios.CSIZE | termios.PARENB)) | termios.CS8
            a[3] = 0
            a[6][termios.VMIN] = 1
            a[6][termios.VTIME] = 0
            termios.tcsetattr(child_fd, termios.TCSANOW, a)
            return child
        except pexpect.exceptions.EOF:
            child.close(force=True)
            if not seen_prompt:
                print(f"attempt {attempt}/{attempts}: connection dropped, retrying in {backoff}s",
                      file=sys.stderr)
                time.sleep(backoff)
                continue
            tail = (child.before or "").strip()
            sys.exit(f"login failed: {tail[:400]}")
    sys.exit("all connection attempts failed (network blocked or server down)")


def interact_session(child):
    """Raw-mode the controlling tty (/dev/tty), not stdin, so keystrokes
    pass through untouched even if the script was launched with redirected
    stdin. Ctrl-] exits, like pexpect's interact."""
    try:
        import select
        import tty as tty_mod
    except ImportError:
        child.interact()
        return
    try:
        fd = os.open("/dev/tty", os.O_RDWR)
    except OSError:
        child.interact()
        return
    saved = termios.tcgetattr(fd)
    tty_mod.setraw(fd)
    try:
        if child.buffer:
            buf = child.buffer
            if isinstance(buf, str):
                buf = buf.encode("utf-8")
            os.write(fd, bytes(buf))
    except OSError:
        pass
    try:
        while child.isalive():
            ready, _, _ = select.select([fd, child.child_fd], [], [], 0.2)
            if fd in ready:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                out = bytes(b for b in data if b != 0x1d)
                if out:
                    os.write(child.child_fd, out)
                if len(out) != len(data):
                    break
            if child.child_fd in ready:
                try:
                    data = os.read(child.child_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                os.write(fd, data)
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSANOW, saved)
        except termios.error:
            pass
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", help="SSH host")
    parser.add_argument("--user", help="user index (1-based) or SSH username")
    parser.add_argument("--password", help="SSH password")
    parser.add_argument("--secret", help="TOTP secret (base32 or otpauth:// URL)")
    parser.add_argument("--ssh-cmd", default=os.environ.get("PARAMSEVA_SSH_CMD", "ssh"),
                        help="command to spawn (default: ssh)")
    parser.add_argument("--env", default=".env",
                        help="dotenv file with credentials (default: ./.env)")
    parser.add_argument("--attempts", type=int, default=5,
                        help="connection retries on network failure (default: 5)")
    parser.add_argument("--backoff", type=int, default=5,
                        help="seconds between retries (default: 5)")
    parser.add_argument("--code", action="store_true", help="print current TOTP code and exit")
    parser.add_argument("--debug", action="store_true",
                        help="log every prompt matched during login")
    args = parser.parse_args()

    entries = load_dotenv_entries(args.env)
    if not entries:
        entries = [load_config()]

    if args.user is not None and args.user.isdigit():
        idx = int(args.user) - 1
        if not 0 <= idx < len(entries):
            sys.exit(f"--user {args.user}: only {len(entries)} user(s) in config")
    elif len(entries) == 1 or args.code:
        idx = 0
    else:
        idx = pick_user(entries)

    entry = entries[idx]

    def eget(key):
        return entry.get(key) or (entry.get("USER") if key == "USERNAME" else None)

    missing = [k for k in ("HOST", "USERNAME", "PASSWORD", "TOTP_SECRET") if not eget(k)]
    if missing:
        if not any(entry.values()):
            print(f"Missing credentials. Write them to {args.env} (chmod 600):")
            print("    HOST=login.cluster")
            print("    USERNAME=vishn")
            print("    PASSWORD=...")
            print("    TOTP_SECRET=...   # base32 or otpauth:// URL")
        else:
            name = eget("USERNAME") or "?"
            sys.exit(f"user {idx + 1} ({name}) incomplete: missing {', '.join(missing)}. Re-run setup.py.")
        sys.exit(1)

    if args.code:
        secret = args.secret or eget("TOTP_SECRET")
        if not secret:
            sys.exit("no TOTP_SECRET set")
        print(f"{eget('USERNAME')}: {safe_code(secret)}")
        return

    host = args.host or eget("HOST")
    user = args.user if (args.user is not None and not args.user.isdigit()) else eget("USERNAME")
    password = args.password or eget("PASSWORD")
    secret = args.secret or eget("TOTP_SECRET")

    child = login(args.ssh_cmd, host, user, password, secret,
                  attempts=args.attempts, backoff=args.backoff, debug=args.debug)
    interact_session(child)


if __name__ == "__main__":
    main()