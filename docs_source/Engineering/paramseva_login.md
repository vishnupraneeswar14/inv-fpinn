# Paramseva Login

## 1. Summary

This setup converts multistep cluster SSH login, file transfers into a single command. This also aims to make accessing multiple cluster accounts by a single user seamless.

### Code

[`src/helper_scripts/paramseva/`](src/helper_scripts/paramseva):

- login: `paramseva_login.py`
- onboarding: `setup.py`
- QR decode: `decode_qr.py`
- credentials: `./.env` (Have it in .gitignore :) )


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
- Requires Google Authenticator in phone every time as the code refreshes periodically (30 seconds).
- If wrong code is entered → asks for code again
- If wrong captcha is entered → re-read the text, retype.

The script bundles all these into one simple command: `python paramseva_login.py`.

## 3. File transfers (scp)

This setup also takes care of auth handling (password, captcha, verification code) for cluster file transfers:

```
python paramseva_login.py --scp-down SRC DEST                  # cluster -> local
python paramseva_login.py --scp-up SRC DEST                    # local -> cluster
python paramseva_login.py --scp-down                           # interactive: asks direction + paths
```

- Transfer failure exits with scp's code (e.g. `scp failed (exit 3)`).

## 4. Setup (per user)

### Step 1: Run setup.py

```
python src/helper_scripts/paramseva/setup.py
```

This asks for
- host (default `paramseva.iith.ac.in`)
- username
- password (hidden, confirm-matched)
- TOTP secret

### Step 2: Three ways to give TOTP secret 

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
python paramseva_login.py --code --user 1     # Prints Google Authenticator code for user 1
python paramseva_login.py                     # If only one user had been setup-> logs in directly. For more than one user, asks to pick a number from 1-N (Default = 1)
python paramseva_login.py --user 2            # Directly logs into cluster acc of user 2
```