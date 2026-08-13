#!/usr/bin/env python3
"""setup.py: onboard paramseva_login credentials for a teammate.

Prompts for host / username / password / TOTP secret and writes ./.env.

The TOTP secret accepts any of:
    base32 string          e.g. JBSWY3DPEHPK3PXP
    otpauth URL            otpauth://totp/...?secret=...
    QR image path          PNG/JPEG of Google Authenticator export
                           (decoded via zbarimg + decode_qr.py)

Usage:
    python3 setup.py            # interactive
    python3 setup.py --env net.env   # write elsewhere
"""

import argparse
import base64
import getpass
import sys
from pathlib import Path

import decode_qr
from paramseva_login import extract_secret


def decode_entries_from_image(path):
    text = decode_qr.zbarimg_decode(path)
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.split("QR-Code:", 1)[-1].strip()
        if not line.startswith("otpauth") and "data=" not in line:
            line = "otpauth-migration://offline?data=" + line
        entries += decode_qr.decode_url(line)
    return entries


def b32(raw_secret):
    return base64.b32encode(raw_secret).decode().rstrip("=")


def looks_valid_b32(secret):
    s = secret.replace(" ", "").upper()
    try:
        base64.b32decode(s + ("=" * ((8 - len(s) % 8) % 8)))
        return True
    except Exception:
        return False


def ask_host():
    raw = input(f"Cluster host [paramseva.iith.ac.in]: ").strip()
    return raw or "paramseva.iith.ac.in"


def ask_username():
    while True:
        raw = input("Username: ").strip()
        if raw:
            return raw
        print("username required")


def ask_password():
    while True:
        p1 = getpass.getpass("Password (hidden): ")
        if not p1:
            print("password required")
            continue
        p2 = getpass.getpass("Confirm password: ")
        if p1 == p2:
            return p1
        print("mismatch, retry")


def ask_secret():
    while True:
        raw = input(
            "TOTP secret (base32, otpauth URL, or path to QR image): ").strip()
        if not raw:
            print("empty input")
            continue

        path = Path(raw)
        if path.is_file():
            try:
                entries = decode_entries_from_image(path)
            except FileNotFoundError:
                print("zbarimg missing - install: sudo apt install zbar-tools")
                print("....or paste the otpauth-migration URL instead")
                continue
            except Exception as e:
                print(f"could not decode image: {e}")
                continue
            if not entries:
                print("no OTP entry found in image")
                continue
            if len(entries) == 1:
                e = entries[0]
                print(f"found: {e['name'] or '?'} ({e['issuer'] or '?'})")
                return b32(e["secret"])
            print("multiple entries - pick one:")
            for i, e in enumerate(entries, 1):
                print(f"  [{i}] {e['name'] or '?'} - {e['issuer'] or '?'}")
            while True:
                sel = input(f"pick 1-{len(entries)}: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(entries):
                    return b32(entries[int(sel) - 1]["secret"])
                print("invalid choice")
            continue

        secret = extract_secret(raw)
        if looks_valid_b32(secret):
            return secret
        if input(f"'{secret[:20]}...' does not look like valid base32. Write it anyway? (y/N): ").strip().lower() == "y":
            return secret
        print("re-enter the secret")


def write_env(path, host, user, password, secret):
    header = "# paramseva cluster credentials - keep this file private (never commit to git)\n"
    path.write_text(header
                    + f"HOST={host}\n"
                    + f"USERNAME={user}\n"
                    + f"PASSWORD={password}\n"
                    + f"TOTP_SECRET={secret}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=".env",
                        help="dotenv file to write (default: ./.env)")
    try:
        args = parser.parse_args()
        host = ask_host()
        user = ask_username()
        password = ask_password()
        secret = ask_secret()

        write_env(Path(args.env), host, user, password, secret)
        print(f"\nwrote {args.env}")
        print(f"  HOST={host}")
        print(f"  USERNAME={user}")
        print(f"  PASSWORD={'*' * len(password)}")
        print(f"  TOTP_SECRET={'*' * len(secret)}")
        print()
        print("verify codes match your phone:")
        print(f"  python3 paramseva_login.py --code --env {args.env}")
        print("then log in:")
        print(f"  python3 paramseva_login.py --env {args.env}")
    except KeyboardInterrupt:
        print("\naborted")
        sys.exit(130)


if __name__ == "__main__":
    main()