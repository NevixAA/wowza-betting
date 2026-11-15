import json, pandas as pd


p=pd.read_csv(r'C:\Users\nevo\dev\wtp2\baseline_main\output\predictions.csv',low_memory=False)
nf=p[p.model_type=='new_format']
empty={c for c in p.columns if p[c].isna().all()}
for m in ('newformat','btts','over15','over35','standard','ht_over05','ht_over15'):
    f=pd.read_csv(rf'C:\Users\nevo\dev\wtp2\baseline_main\models\feature_importances_{m}.csv')
    feats=set(f.iloc[:,0].astype(str))
    e=sorted(feats & empty)
    nfmiss=sorted(c for c in feats if c in nf.columns and nf[c].isna().mean()>0.9)
    print(m,'n_feats',len(feats),'fully-empty-on-board',len(e),e[:8],'| >90% missing on NF board',len(nfmiss),nfmiss[:8])


