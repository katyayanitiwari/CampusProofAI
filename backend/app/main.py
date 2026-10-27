from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from datetime import date

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT
from app.ml.predict import predict_text
from app.ml.predict_financial_message import predict_financial_message
from app.schemas.models import SearchRequest, VerifyRequest
from app.services.eligibility_matcher import match_eligibility
from app.services.evidence_verifier import verify
from app.services.report_service import get_report, list_reports

AUTH_DB = ROOT / "artifacts" / "reports" / "campusproof_accounts.sqlite3"
AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
SESSION_SECRET = os.getenv("CAMPUSPROOF_SESSION_SECRET")
if not SESSION_SECRET:
    SESSION_SECRET = secrets.token_urlsafe(48)
    with (ROOT / ".env").open("a", encoding="utf-8") as env_file:
        env_file.write(f"\nCAMPUSPROOF_SESSION_SECRET={SESSION_SECRET}\n")
    os.environ["CAMPUSPROOF_SESSION_SECRET"] = SESSION_SECRET


def db_connect():
    conn = sqlite3.connect(AUTH_DB)
    conn.row_factory = sqlite3.Row
    return conn


with db_connect() as conn:
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
        salt BLOB NOT NULL,
        password_hash BLOB NOT NULL,
        created_at INTEGER NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        category TEXT NOT NULL,
        due_date TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        source_url TEXT NOT NULL DEFAULT '',
        completed INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS reminders_by_user_date ON reminders(user_id, due_date, completed)")


def password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


def session_signature(user_id: str, expires: str) -> str:
    return hmac.new(SESSION_SECRET.encode(), f"{user_id}.{expires}".encode(), hashlib.sha256).hexdigest()


def get_session_user(request: Request) -> str | None:
    try:
        user_id, expires, signature = request.cookies.get("campusproof_session", "").split(".", 2)
        if int(expires) < int(time.time()) or not hmac.compare_digest(signature, session_signature(user_id, expires)):
            return None
        return user_id
    except (ValueError, TypeError):
        return None


def set_session(response: Response, request: Request, user_id: int) -> None:
    uid = str(user_id)
    expires = str(int(time.time()) + 43200)
    token = f"{uid}.{expires}.{session_signature(uid, expires)}"
    secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").lower() == "https"
    response.set_cookie("campusproof_session", token, max_age=43200, httponly=True, secure=secure, samesite="lax", path="/")


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=10, max_length=256)


class ReminderCreate(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    category: str = Field(default="deadline", pattern="^(deadline|application|follow-up|document|other)$")
    due_date: str = Field(min_length=10, max_length=10, pattern="^\\d{4}-\\d{2}-\\d{2}$")
    notes: str = Field(default="", max_length=1000)
    source_url: str = Field(default="", max_length=500)


class ReminderUpdate(BaseModel):
    completed: bool

app = FastAPI(
    title="CampusProof AI",
    description="Student opportunity checks and evidence receipts.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

OPEN_AUTH_PATHS = {
    "/api/health", "/api/auth/status", "/api/auth/register",
    "/api/auth/login", "/api/auth/logout",
}


@app.middleware("http")
async def require_account(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.url.path not in OPEN_AUTH_PATHS:
        if not get_session_user(request):
            return JSONResponse({"detail": "Sign in to use CampusProof."}, status_code=401)
    return await call_next(request)

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/auth/status")
def auth_status(request: Request):
    with db_connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    user_id = get_session_user(request)
    username = None
    if user_id:
        with db_connect() as conn:
            row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            username = row["username"]
        else:
            user_id = None
    return {"authenticated": bool(user_id), "username": username, "setup_required": count == 0}


@app.post("/api/auth/register")
def register(credentials: Credentials, request: Request):
    username = credentials.username.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,40}", username):
        raise HTTPException(400, "Username must be 3–40 characters using letters, numbers, dot, dash, or underscore.")
    salt = secrets.token_bytes(16)
    digest = password_digest(credentials.password, salt)
    try:
        with db_connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users(username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, salt, digest, int(time.time())),
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "That username is already taken.") from exc
    response = JSONResponse({"created": True, "username": username})
    set_session(response, request, cursor.lastrowid)
    return response


@app.post("/api/auth/login")
def login(credentials: Credentials, request: Request):
    with db_connect() as conn:
        row = conn.execute("SELECT id, salt, password_hash FROM users WHERE username = ?", (credentials.username.strip(),)).fetchone()
    if not row or not hmac.compare_digest(password_digest(credentials.password, row["salt"]), row["password_hash"]):
        raise HTTPException(401, "Username or password is incorrect.")
    response = JSONResponse({"signed_in": True})
    set_session(response, request, row["id"])
    return response


@app.post("/api/auth/logout")
def logout():
    response = JSONResponse({"signed_out": True})
    response.delete_cookie("campusproof_session", path="/", httponly=True, samesite="lax")
    return response


def current_user_id(request: Request) -> int:
    user_id = get_session_user(request)
    if not user_id:
        raise HTTPException(401, "Sign in to use CampusProof.")
    return int(user_id)


@app.get("/api/reminders")
def list_reminders(request: Request):
    user_id = current_user_id(request)
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT id, title, category, due_date, notes, source_url, completed, created_at "
            "FROM reminders WHERE user_id = ? ORDER BY completed ASC, due_date ASC, id DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/reminders", status_code=201)
def create_reminder(payload: ReminderCreate, request: Request):
    user_id = current_user_id(request)
    try:
        due_date = date.fromisoformat(payload.due_date).isoformat()
    except ValueError as exc:
        raise HTTPException(422, "Enter a valid reminder date.") from exc
    source_url = payload.source_url.strip()
    if source_url:
        parsed = urlparse(source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(422, "Source link must start with http:// or https://.")
    with db_connect() as conn:
        cursor = conn.execute(
            "INSERT INTO reminders(user_id, title, category, due_date, notes, source_url, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, payload.title.strip(), payload.category, due_date, payload.notes.strip(), source_url, int(time.time())),
        )
        row = conn.execute(
            "SELECT id, title, category, due_date, notes, source_url, completed, created_at FROM reminders WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    return dict(row)


@app.patch("/api/reminders/{reminder_id}")
def update_reminder(reminder_id: int, payload: ReminderUpdate, request: Request):
    user_id = current_user_id(request)
    with db_connect() as conn:
        cursor = conn.execute(
            "UPDATE reminders SET completed = ? WHERE id = ? AND user_id = ?",
            (int(payload.completed), reminder_id, user_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(404, "Reminder not found.")
        row = conn.execute(
            "SELECT id, title, category, due_date, notes, source_url, completed, created_at FROM reminders WHERE id = ? AND user_id = ?",
            (reminder_id, user_id),
        ).fetchone()
    return dict(row)


@app.delete("/api/reminders/{reminder_id}", status_code=204)
def delete_reminder(reminder_id: int, request: Request):
    user_id = current_user_id(request)
    with db_connect() as conn:
        cursor = conn.execute("DELETE FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, user_id))
    if cursor.rowcount == 0:
        raise HTTPException(404, "Reminder not found.")
    return Response(status_code=204)


@app.get("/api/model/status")
def model_status():
    return predict_text("")


@app.post("/api/verify")
def verify_endpoint(req: VerifyRequest):
    try:
        result = verify(req.text, str(req.url) if req.url else None, req.source_text)
        result.model_signal = predict_financial_message(req.text)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/opportunities/search")
def search(req: SearchRequest):
    query = f"{req.query} {req.opportunity_type} {req.degree} {req.branch} scholarship internship India"
    return {
        "live": False,
        "results": [{
            "title": "Open web search results",
            "organisation": "Google Search",
            "url": "https://www.google.com/search?q=" + quote_plus(query),
            "snippet": "Search results open in a new browser tab. CampusProof does not retrieve or verify these pages.",
        }],
    }


@app.post("/api/eligibility/match-requirements")
def eligibility(payload: dict):
    return match_eligibility(str(payload.get("requirements", "")), payload.get("profile", {}))


@app.post("/api/eligibility/match")
def eligibility_match(payload: dict):
    return match_eligibility(str(payload.get("requirements", "")), payload.get("profile", {}))


@app.get("/api/reports")
def reports():
    if os.getenv("CAMPUSPROOF_PUBLIC_MODE") == "1":
        return []
    return list_reports()


@app.get("/api/reports/{report_id}")
def report(report_id: str):
    if os.getenv("CAMPUSPROOF_PUBLIC_MODE") == "1":
        raise HTTPException(404, "Report not found")
    result = get_report(report_id)
    if not result:
        raise HTTPException(404, "Report not found")
    return result


FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

