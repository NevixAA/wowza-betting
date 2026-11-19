import pandas as pd
a=pd.read_csv(r'C:\Users\nevo\dev\wtp2\baseline_main\output\stake_audit.csv',low_memory=False)
a['ff']=pd.to_numeric(a.final_fraction,errors='coerce').fillna(0)
s=a[a.ff>0]
g=s.groupby(['date','fixture'])
multi=g.filter(lambda x: len(x)>1)
print('staked rows',len(s),'fixtures with >1 staked position',multi.groupby(['date','fixture']).ngroups,'rows',len(multi))
dirn=lambda r: 'UP' if str(r.side).upper() in ('OVER','YES','YES/OVER') else 'DOWN'
multi=multi.assign(d=multi.apply(dirn,axis=1))
opp=multi.groupby(['date','fixture']).d.nunique()
print('fixtures with opposite-direction staked positions',int((opp>1).sum()))
print(multi[multi.set_index(['date','fixture']).index.isin(opp[opp>1].index)][['date','fixture','market','side','ff','reason']].head(12).to_string())
print('dup identical rows (same date,fixture,market,side)',int(s.duplicated(['date','fixture','market','side']).sum()))
