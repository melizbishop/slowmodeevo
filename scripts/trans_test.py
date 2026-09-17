import numpy as np, pandas as pd, os
CACHE=os.path.expanduser("~/mqcache")
z=np.load(CACHE+"/bout_transitions.npz")
sps=sorted(k[3:] for k in z.files if k.startswith("T__"))
lab=pd.read_csv(os.path.expanduser("~/mnt/slowmodeevo/moseq_syllable_labels.csv")).set_index("moseq_cluster")
EPS=1e-12; MIN=300; B=200; rng=np.random.default_rng(0)

def jsd(p,q):
    m=0.5*(p+q)
    def kl(a,b):
        mask=a>0; return float((a[mask]*np.log2(a[mask]/np.maximum(b[mask],EPS))).sum())
    return 0.5*kl(p,m)+0.5*kl(q,m)

rows=[]; lift_rows=[]
for sp in sps:
    T=z["T__"+sp]                       # (arm, half, S, S)
    T1, T2 = T[0].sum(0), T[1].sum(0)
    n1, n2 = T1.sum(1), T2.sum(1)
    use=(n1>=MIN)&(n2>=MIN)
    obs_list=[]; null_list=[]; w_list=[]; sig=0
    for s in np.nonzero(use)[0]:
        p1, p2 = T1[s]/n1[s], T2[s]/n2[s]
        pool=(T1[s]+T2[s]); pool=pool/pool.sum()
        o=jsd(p1,p2)
        k1,k2=int(round(n1[s])),int(round(n2[s]))
        nulls=[jsd(rng.multinomial(k1,pool)/k1, rng.multinomial(k2,pool)/k2) for _ in range(B)]
        nulls=np.array(nulls)
        obs_list.append(o); null_list.append(nulls.mean()); w_list.append(n2[s])
        sig += int(o > np.percentile(nulls,99))
    if not obs_list: continue
    w=np.array(w_list); w=w/w.sum()
    obs=np.array(obs_list); nul=np.array(null_list)
    rows.append(dict(species=sp, n_rows=len(obs), JSD_obs=float((w*obs).sum()),
                     JSD_null=float((w*nul).sum()),
                     excess=float((w*(obs-nul)).sum()),
                     ratio=float((w*obs).sum()/max((w*nul).sum(),EPS)),
                     frac_rows_p01=sig/len(obs)))
    # composition-controlled: lift = log2 P(next|cur,arm) - log2 P(next|arm)
    m1, m2 = T1.sum(0)/T1.sum(), T2.sum(0)/T2.sum()
    L1=np.log2((T1[use]/n1[use,None]+EPS)/(m1+EPS)); L2=np.log2((T2[use]/n2[use,None]+EPS)/(m2+EPS))
    keep=(T1[use]>0)&(T2[use]>0)
    lift_rows.append(dict(species=sp, n_cells=int(keep.sum()),
                          r_lift=float(np.corrcoef(L1[keep],L2[keep])[0,1])))
t=pd.DataFrame(rows).set_index("species")
print("Is the bout-to-bout transition kernel the same in both arms?")
print("(JSD of P(next|current) between arms, vs a count-matched permutation null)\n")
print(t.round(4).to_string())
print("\nComposition-controlled: correlation of transition LIFT between arms")
print("(lift = log2 P(next|cur,arm) / P(next|arm); removes availability differences)\n")
print(pd.DataFrame(lift_rows).set_index("species").round(3).to_string())
