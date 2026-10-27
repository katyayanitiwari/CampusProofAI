import json,joblib
from functools import lru_cache
from app.config import MODEL_PATH,ROOT
@lru_cache(maxsize=1)
def load_model():return joblib.load(MODEL_PATH)
def predict_text(text):
 if not MODEL_PATH.exists():return {'available':False,'message':'No trained model found. Run python scripts/download_difraud_job_scams.py, then python -m app.ml.train_difraud.','domain':'Job-posting dataset only; not validated for scholarship fraud.'}
 model=load_model();meta_path=ROOT/'models/metadata.json';meta=json.loads(meta_path.read_text()) if meta_path.exists() else {};threshold=float(meta.get('threshold',0.0));score=float(model.decision_function([text])[0])
 return {'available':True,'signal':'higher_warning_signal' if score>=threshold else 'lower_warning_signal','decision_score':score,'threshold':threshold,'is_probability':False,'model':meta.get('model'),'domain':meta.get('domain','Job postings only; not validated for scholarship fraud.')}

