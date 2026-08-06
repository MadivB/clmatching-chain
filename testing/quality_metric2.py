import glob,json,os,numpy as np
DUMPS='v030v3_out/dumps'; CLIP,GAIN,PEAK=60780.0,200.0,105
lab=json.load(open('quality_labels.json')); label={}
for r in lab['bad']: label[(r[0],r[1],r[2])]=1
for r in lab['good']: label[(r[0],r[1],r[2])]=0
def peaks(tr,frac=0.06,ms=8):
    thr=frac*tr.max(); pk=[]
    for t in range(2,len(tr)-2):
        if tr[t]>thr and tr[t]>=tr[t-1] and tr[t]>tr[t+1] and tr[t]>=tr[t-2]:
            if pk and t-pk[-1]<ms:
                if tr[t]>tr[pk[-1]]: pk[-1]=t
            else: pk.append(t)
    return pk
def feats(act,pred):
    sat=act>=0.98*CLIP; m=~sat
    a=act[m]; p=np.minimum(pred,CLIP)[m]
    alpha=float((p@a)/max(p@p,1e-9)); v=GAIN*np.maximum(a,100.0)
    chi2=float(((p-a)**2/v).mean()); chi2_scaled=float(((alpha*p-a)**2/v).mean())
    resid_scaled=float(np.abs(alpha*p-a).sum()/max(a.sum(),1))
    at=act.sum(0); pt=pred.sum(0); pk=peaks(at); W=4
    if pk:
        A=np.array([at[max(0,t-W):t+W+1].sum() for t in pk]); P=np.array([pt[max(0,t-W):t+W+1].sum() for t in pk])
        r=P/np.maximum(A,1.0); w=A/A.sum(); mr=np.median(r)
        ratio_mad=float(np.sum(w*np.abs(r-mr))); big=float(np.sum(w*(np.abs(r-mr)>0.3)))
    else: ratio_mad=big=0.0
    return dict(chi2=chi2,chi2_scaled=chi2_scaled,resid_scaled=resid_scaled,ratio_mad=ratio_mad,bigdev=big)
rows=[]
for fp in sorted(glob.glob(f'{DUMPS}/*.npz')):
    d=np.load(fp,allow_pickle=True); ft=os.path.basename(fp).split('_ev')[0]; ev=int(str(d['event']))
    tpc=np.asarray(d['tpc']); full=np.asarray(d['full_light'],float); base=np.asarray(d['base_image'],float)
    for c in np.unique(tpc):
        k=(ft,ev,int(c))
        if k not in label: continue
        F=feats(full[int(c)],base[int(c)]); F['label']=label[k]; F['key']=list(k); rows.append(F)
json.dump(rows,open('quality_metrics2.json','w'))
y=np.array([r['label'] for r in rows])
def auc(sc,y,hb=True):
    s=np.array(sc); s=s if hb else -s; pos=s[y==1];neg=s[y==0]
    o=np.argsort(np.concatenate([pos,neg])); rk=np.empty_like(o,float); rk[o]=np.arange(1,len(o)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
print(f"{len(rows)} TPCs ({y.sum()} BAD)")
for m in ['chi2','chi2_scaled','resid_scaled','ratio_mad','bigdev']:
    a=auc([r[m] for r in rows],y); gm=np.median([r[m] for r in rows if r['label']==0]); bm=np.median([r[m] for r in rows if r['label']==1])
    print(f"  {m:>13} AUC={a:.3f}  GOODmed={gm:.3f} BADmed={bm:.3f}")
X=np.array([[r['chi2_scaled'],r['ratio_mad'],r['bigdev']] for r in rows]); X=(X-X.mean(0))/(X.std(0)+1e-9)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    p=cross_val_predict(LogisticRegression(class_weight='balanced',max_iter=1000),X,y,cv=5,method='predict_proba')[:,1]
    print(f"  3-feat logistic CV AUC = {auc(p,y):.3f}")
except Exception as e: print("  sklearn:",e)
print("DONE")
