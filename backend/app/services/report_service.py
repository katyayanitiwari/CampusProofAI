import json,sqlite3
from pathlib import Path
from app.config import DB_PATH
def _connect():
 p=Path(DB_PATH);p.parent.mkdir(parents=True,exist_ok=True);c=sqlite3.connect(p);c.execute("CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY,created_at TEXT,status TEXT,payload TEXT)");return c
def save_report(d):
 c=_connect()
 try:c.execute("INSERT OR REPLACE INTO reports VALUES(?,?,?,?)",(d['report_id'],d['retrieved_at'],d['evidence_status'],json.dumps(d)));c.commit()
 finally:c.close()
def get_report(i):
 c=_connect()
 try:r=c.execute("SELECT payload FROM reports WHERE id=?",(i,)).fetchone();return json.loads(r[0]) if r else None
 finally:c.close()
def list_reports(limit=50):
 c=_connect()
 try:return [json.loads(r[0]) for r in c.execute("SELECT payload FROM reports ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()]
 finally:c.close()

