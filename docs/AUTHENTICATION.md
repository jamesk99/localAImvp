# Authentication Model for the RAG Flask App

This document explains **how authentication actually works** in the Flask RAG app:

- What the browser popup is doing.
- How environment variables and JSON files control which credentials are accepted.
- How to manage users (add/remove) without touching Python code.

It intentionally does **not** restate the app code; it explains behavior and workflow.

---

## 1. What the browser popup is doing

When someone visits the app (e.g. `http://192.168.1.23:8000/`):

1. The browser requests `/`.
2. The server replies with **401 Unauthorized** and a `WWW-Authenticate` header.
3. The browser shows the built-in **username/password popup**.
4. Whatever the user types gets sent back in an `Authorization` header.
5. The Flask app checks those credentials in `check_auth(...)`.
6. If they match something the server trusts → request is allowed.
   If not → 401 again, popup reappears.

So the popup is **just a way to send credentials**. The list of “allowed” usernames/passwords is **entirely defined on the server** (via env vars and/or a JSON file).

---

## 2. Two ways to define valid credentials

The app supports two layers:

1. **JSON file with many users** (recommended).
2. **Single username/password from environment variables** (fallback).

The **precedence** is:

1. If a valid JSON user file is loaded → **that controls auth**.
2. Otherwise → it falls back to a **single** `RAG_USER` / `RAG_PASS` pair.

There is **never** a mix; it’s either:

- JSON file (many users), or
- single env-based username/password.

---

## 3. JSON file with many users (recommended)

### 3.1 File format

The file must be a JSON object mapping usernames to passwords, like:

```json
{
  "alice": "alicePassword123",
  "bob": "bobSecret!",
  "dad": "SuperStrong_2025!"
}
```

- Keys → usernames.
- Values → plain-text passwords (what users type in the popup).

### 3.2 Where the app looks for the JSON file

The app decides which file to use like this:

1. If environment variable `RAG_USER_FILE` is set:
   - Use that path as the auth file.

2. Otherwise, if a file named `auth_users.json` exists in the same folder as `app.py` (`src/`):
   - Use `src/auth_users.json`.

3. If neither exists or cannot be loaded:
   - No JSON users are active (`AUTH_USERS = None`), and the app falls back to env-based auth.

### 3.3 Recommended simple setup

**Recommended**: use the default filename in `src/` so you don’t have to set extra env vars.

1. In `notebook/e6-phase0-dadproject/src/`, create this file:

   - `auth_users.json`

2. Put your allowed users in it:

   ```json
   {
     "alice": "alicePassword123",
     "bob": "bobSecret!",
     "dad": "SuperStrong_2025!"
   }
   ```

3. Start the app:

   ```powershell
   cd notebook\e6-phase0-dadproject\src
   python app.py
   ```

4. Now, in the browser popup, **each user** logs in with their own pair:

   - `alice` / `alicePassword123`
   - `bob` / `bobSecret!`
   - `dad` / `SuperStrong_2025!`

### 3.4 If you want a custom filename / location

If you prefer a different path or name (for example, `src/auth.json` or `C:\secrets\rag_auth.json`):

1. Create your JSON file at that path with the same `{ "username": "password" }` shape.
2. Before starting the app, set `RAG_USER_FILE` to that full path:

   ```powershell
   $env:RAG_USER_FILE = "C:\secrets\rag_auth.json"
   python app.py
   ```

3. The app will use that file instead of `auth_users.json`.

---

## 4. Env-based single credential (fallback path)

If **no** JSON auth file is successfully loaded, the app falls back to this behavior:

- It reads two environment variables:
  - `RAG_USER` – username (default: `raguser`).
  - `RAG_PASS` – password (default: `changeme`).

These are used as **one single allowed pair**. So:

- On the server, you do:

  ```powershell
  $env:RAG_USER = "myuser"
  $env:RAG_PASS = "myStrongPassword!"
  python app.py
  ```

- In the browser popup, anyone who wants in must type:
  - Username: `myuser`
  - Password: `myStrongPassword!`

This mode is mainly useful for quick dev/testing or when you truly only want a single shared credential.

---

## 5. How to add / remove users (JSON mode)

When you are in JSON-auth mode (either `auth_users.json` or a custom `RAG_USER_FILE` path), you manage users **by editing that JSON file and restarting the app**.

### 5.1 Add a new user

1. Open the JSON file in your editor.
2. Add a new key/value pair under the existing ones, for example:

   ```json
   {
     "alice": "alicePassword123",
     "bob": "bobSecret!",
     "dad": "SuperStrong_2025!",
     "newuser": "NewUser2025!"
   }
   ```

3. Save the file.
4. Restart the Flask app (`python app.py`).
5. Send `newuser` their username and password.

### 5.2 Remove a user

1. Delete that user’s line from the JSON file.
2. Save the file and restart the app.
3. Any further login attempts with that username will fail.

### 5.3 Change a user’s password

1. Edit the string value for that username.
2. Save + restart.
3. Tell the user their new password.

---

## 6. Which mode am I actually using?

At runtime, the app effectively does this:

1. **Try to load a JSON auth file**:
   - If `RAG_USER_FILE` is set → try that path.
   - Else if `auth_users.json` exists next to `app.py` → use that.

2. If JSON load succeeds:
   - All auth checks use the usernames/passwords from that JSON.

3. If JSON load fails or no file is configured:
   - Auth checks fall back to the single `RAG_USER` / `RAG_PASS` pair.

You can think of it as:

- JSON present and valid → **multi-user mode**.
- No JSON → **single shared credential mode**.

---

## 7. Security notes / expectations

- Passwords in the JSON file are currently stored in **plain text**. This is acceptable for a small, LAN-only internal tool, but:
  - Protect that file (e.g., put it somewhere only admins can read).
  - Do not commit it to a public repo.
- Basic Auth does not do fancy session handling; it relies on the browser caching credentials while the tab is open.
- For more advanced needs (per-user roles, password resets, stronger password storage), you would move to a more full-featured auth system (login page + database + hashed passwords). For now, this setup focuses on being simple and easy to operate.
