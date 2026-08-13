#!/usr/bin/env python3
"""decode_qr: extract TOTP secrets from Google Authenticator export QR content.

Google Authenticator's export QR codes contain
    otpauth-migration://offline?data=<url-encoded base64 protobuf>
instead of a plain otpauth:// URL. This script parses that protobuf
(pure stdlib, no protobuf dependency) and prints each account's name,
issuer and base32 secret.

Usage:
    decode_qr.py qr.png                          # decode a QR screenshot (runs zbarimg)
    decode_qr.py 'otpauth-migration://offline?data=CjwK...'   # or pass the URL directly
    zbarimg -q --raw qr.png | decode_qr.py                     # or pipe zbarimg output
    decode_qr.py --env .env qr.png               # write TOTP_SECRET into a .env file
"""

import argparse
import base64
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote

DEFAULTS = {"name": "", "issuer": "", "algorithm": 1, "digits": 1, "type": 2, "counter": 0}


def read_varint(buf, off):
    result = 0
    shift = 0
    while True:
        b = buf[off]
        off += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, off
        shift += 7


def parse_fields(data):
    out = []
    off = 0
    while off < len(data):
        tag, off = read_varint(data, off)
        fno, wt = tag >> 3, tag & 7
        if wt == 0:
            val, off = read_varint(data, off)
        elif wt == 2:
            length, off = read_varint(data, off)
            val = data[off:off + length]
            off += length
        elif wt == 5:
            val = data[off:off + 4]
            off += 4
        elif wt == 1:
            val = data[off:off + 8]
            off += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        out.append((fno, wt, val))
    return out


def decode_payload(data):
    otps = []
    for fno, wt, val in parse_fields(data):
        if fno == 1 and wt == 2:
            otp = dict(DEFAULTS)
            for sfno, swt, sval in parse_fields(val):
                if sfno == 1 and swt == 2:
                    otp["secret"] = sval
                elif sfno == 2 and swt == 2:
                    otp["name"] = sval.decode("utf-8", "replace")
                elif sfno == 3 and swt == 2:
                    otp["issuer"] = sval.decode("utf-8", "replace")
                elif sfno == 4 and swt == 0:
                    otp["algorithm"] = sval
                elif sfno == 5 and swt == 0:
                    otp["digits"] = sval
                elif sfno == 6 and swt == 0:
                    otp["type"] = sval
                elif sfno == 7 and swt == 0:
                    otp["counter"] = sval
            otps.append(otp)
    return otps


ALGORITHMS = {1: "SHA1", 2: "SHA256", 3: "SHA512", 4: "MD5"}
DIGITS = {1: "6", 2: "8"}
TYPES = {1: "hotp", 2: "totp"}


def otpauth_url(otp):
    scheme = TYPES.get(otp["type"], "totp")
    params = ["secret=" + base64.b32encode(otp["secret"]).decode().rstrip("=")]
    if otp["issuer"]:
        params.append("issuer=" + quote(otp["issuer"]))
    if otp["algorithm"] != 1:
        params.append("algorithm=" + ALGORITHMS.get(otp["algorithm"], "SHA1"))
    if otp["digits"] != 1:
        params.append("digits=" + DIGITS.get(otp["digits"], "6"))
    if otp["type"] == 1:
        params.append("counter=" + str(otp["counter"]))
    name = quote(otp["name"] or "account")
    return f"otpauth://{scheme}/{name}?" + "&".join(params)


def decode_url(url):
    m = re.search(r"[?&]data=([^&]+)", url.strip())
    if not m:
        raise ValueError("not an otpauth-migration URL / missing data= parameter")
    data_b64 = unquote(m.group(1)).replace(" ", "+")
    try:
        data = base64.b64decode(data_b64.replace("-", "+").replace("_", "/"))
    except Exception as e:
        raise ValueError(f"invalid base64 payload: {e}")
    return decode_payload(data)


def zbarimg_decode(image_path):
    try:
        proc = subprocess.run(["zbarimg", "-q", "--raw", str(image_path)],
                              capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        sys.exit("zbarimg not found - install it: sudo apt install zbar-tools")
    if proc.returncode != 0:
        raise ValueError(proc.stderr.strip() or f"zbarimg failed (exit {proc.returncode})")
    return proc.stdout


def update_env(path, secret):
    lines = []
    if path.exists():
        lines = path.read_text().splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("TOTP_SECRET"):
            lines[i] = f"TOTP_SECRET={secret}"
            replaced = True
    if not replaced:
        lines.append(f"TOTP_SECRET={secret}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="otpauth-migration URL or base64 data (else stdin)")
    parser.add_argument("--env", metavar="PATH", help="write TOTP_SECRET into a .env file (single entry)")
    args = parser.parse_args()

    raw = args.input if args.input else sys.stdin.read()
    inp = raw.strip()

    if inp and not inp.startswith("otpauth") and "data=" not in inp and Path(inp).is_file():
        source = zbarimg_decode(Path(inp))
    else:
        source = raw

    otps = []
    errors = []
    for line in source.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.split("QR-Code:", 1)[-1].strip()
        if not line.startswith("otpauth") and "data=" not in line:
            line = "otpauth-migration://offline?data=" + line
        try:
            otps += decode_url(line)
        except Exception as e:
            errors.append(str(e))
    if not otps:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit("no OTP entries decoded")

    for otp in otps:
        secret = base64.b32encode(otp["secret"]).decode().rstrip("=")
        print(f"name:    {otp['name']}")
        print(f"issuer:  {otp['issuer']}")
        print(f"secret:  {secret}")
        print(f"uri:     {otpauth_url(otp)}")
        print()

    if args.env:
        if len(otps) != 1:
            sys.exit("--env requires exactly one entry in the data")
        update_env(Path(args.env), base64.b32encode(otps[0]["secret"]).decode().rstrip("="))
        print(f"wrote TOTP_SECRET to {args.env}")


if __name__ == "__main__":
    main()