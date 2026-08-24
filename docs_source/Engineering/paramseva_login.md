# Paramseva Login

## 1. Summary

This work turns multistep cluster SSH login into a single command. Same machinery handles file transfers (`--scp-up` / `--scp-down`).

### Code

[`src/helper_scripts/paramseva/`](src/helper_scripts/paramseva):

- login: `paramseva_login.py`
- onboarding: `setup.py`
- QR decode: `decode_qr.py`
- credentials: `./.env` (Make sure to have it .gitignore)


```
setup.py (QR scan) → .env block → paramseva_login.py → raw shell
```

## 2. How manual login works


Without the script, every session requires these steps:

```
$ ssh username@paramseva.iith.ac.in
password: ********
Verification code: 123456                            # From Google Authenticator app in mobile
Captcha: (ab12cd34)                    
Last login: ...
$                                             
```
Challenges:
- Requires Google Authenticator in Phone out every time — the code refreshes every 30 seconds.
- Wrong code → prompts re-entering. Wrong Captcha → re-read the text, retype.

The script collapses this into one simple command: `python3 paramseva_login.py`.

## 3. File transfers (scp)

Same auth handling (password, captcha, verification code) for file transfers:

```
python3 paramseva_login.py --scp-down SRC DEST                  # cluster -> local
python3 paramseva_login.py --scp-up SRC DEST                    # local -> cluster
python3 paramseva_login.py --scp-down                           # interactive: asks direction + paths
```

- Transfer failure exits with scp's code (e.g. `scp failed (exit 3)`).

## 4. Setup (per user)

### Step 1: Run setup.py

```
python3 src/helper_scripts/paramseva/setup.py
```

This prompts for -> host (default `paramseva.iith.ac.in`), username, password (hidden, confirm-matched), TOTP secret.

### Step 2: Give the TOTP secret — three forms

1. **Base32 string**, e.g. `JBSWY3DPEHPK3PXP`
2. **otpauth URL**, e.g. `otpauth://totp/...?secret=...`
3. **QR image** — screenshot of the Google Authenticator QR code export (App > Export accounts > PIN). This QR code is decoded using `decode_qr.py`.


### .env

One block per user is **appended** once the setup is done:

```
HOST=paramseva.iith.ac.in
USERNAME=user1
PASSWORD=...
TOTP_SECRET=...

HOST=paramseva.iith.ac.in
USERNAME=user2
...
```


### Step 3: Verify and log in

```
python3 paramseva_login.py --code --user 1     # Prints Google Authenticator code for user 1
python3 paramseva_login.py                     # If only one user had been setup-> logs in directly. For more than one user, prompts to pick a number from 1-N (Default = 1)
python3 paramseva_login.py --user 2            # Directly logs into cluster acc of user 2
```

This setup allows single user to access multiple cluster accounts seamlessly