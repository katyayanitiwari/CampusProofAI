import pandas as pd
TEXT_COLUMNS=['title','company_profile','description','requirements','benefits']
ALIASES={'fraudulent':['fraudulent','label','target']}
def normalize_columns(df):
 df=df.copy();df.columns=[str(c).strip().lower().replace(' ','_') for c in df.columns]
 for target,aliases in ALIASES.items():
  if target not in df:
   for alias in aliases:
    if alias in df:df=df.rename(columns={alias:target});break
 return df
def prepare_csv(path):
 df=normalize_columns(pd.read_csv(path));removed={'duplicates':int(df.duplicated().sum())};df=df.drop_duplicates()
 if 'fraudulent' not in df:raise ValueError('Dataset needs a fraudulent label column (0 genuine, 1 fraudulent).')
 labels=pd.to_numeric(df['fraudulent'],errors='coerce');valid=labels.isin([0,1]);removed['invalid_labels']=int((~valid).sum());df=df[valid].copy();df['fraudulent']=labels[valid].astype(int)
 for c in TEXT_COLUMNS:
  if c not in df:df[c]=''
  df[c]=df[c].fillna('').astype(str)
 df['text']=df[TEXT_COLUMNS].agg(' '.join,axis=1).str.replace(r'\s+',' ',regex=True).str.strip()
 return df,{'rows':len(df),'removed':removed,'label_counts':df['fraudulent'].value_counts().sort_index().to_dict(),'columns':list(df.columns),'missing':df.isna().sum().to_dict()}

