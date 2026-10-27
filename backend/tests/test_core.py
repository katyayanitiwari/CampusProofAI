import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from app.ml.preprocessing import prepare_csv
from app.services.eligibility_matcher import match_eligibility
def test_clean_labels_and_duplicates(tmp_path):
 p=tmp_path/'set.csv';pd.DataFrame({'title':['A','A','B'],'description':[None,'','text'],'fraudulent':[0,0,2]}).to_csv(p,index=False);d,s=prepare_csv(p);assert len(d)==1 and s['removed']['invalid_labels']==1
def test_eligibility_unknown():
 r=match_eligibility('Applicants must be in final year of engineering.',{'degree':'','branch':'','year_of_study':None});assert r['needs_confirmation']



