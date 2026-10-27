import argparse
from pathlib import Path
from app.ml.preprocessing import prepare_csv
def main():
 p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--out',default='../data/processed/prepared.csv');a=p.parse_args();df,s=prepare_csv(a.csv);Path(a.out).parent.mkdir(parents=True,exist_ok=True);df.to_csv(a.out,index=False);print(s)
if __name__=='__main__':main()

