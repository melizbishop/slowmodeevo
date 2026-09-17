import sys, os
sys.path.insert(0, os.path.expanduser("~/mnt/slowmodeevo"))
import json, os, numpy as np, pandas as pd, pickle, sys, time
R=os.path.expanduser("~/mnt/slowmodeevo/outputs/multispecies_slow_modes/global_clustering_modes/global_outputs/XY_EvenSampled_SlowModes/")
CACHE=os.path.expanduser("~/mqcache")
idx=json.load(open(CACHE+"/index.json"))
rec=pd.DataFrame(idx).T.rename_axis("recording_id").reset_index(); rec["n"]=rec["n"].astype(int)
md=pickle.load(open(R+"projection_result.pkl","rb"))["metadata"]
md=md if isinstance(md,list) else list(md.values())
meta={m["individual_id"].split("__subject__")[-1]: m for m in md}
man=pd.read_csv(R+"global_kmeans_state_manifest.csv"); man["subject"]=man.source_individual_id.astype(str)
statefile=dict(zip(man.subject, man.state_file)); d_embed=int(man.d_embed.iloc[0])
VIEW="with_low_occupancy_cutoff"
CHI={sp: np.load(f"{R}saved_slow_mode_arrays/{VIEW}/{sp}/chi_global_aligned.npy")
     for sp in sorted(os.listdir(f"{R}saved_slow_mode_arrays/{VIEW}"))}
NAMED=set(pd.read_csv(os.path.expanduser("~/mnt/slowmodeevo/moseq_syllable_labels.csv")).moseq_cluster)
S=50; n_arm=2
# T[species][arm][half] : bout-to-bout transition counts, soft-weighted
T={sp: np.zeros((n_arm,2,S,S)) for sp in CHI}
OCC={sp: np.zeros((n_arm,2,S)) for sp in CHI}
t0=time.time(); nsub=0
for h_i,(subj, g) in enumerate(rec.groupby("subject")):
    if subj not in meta or subj not in statefile: continue
    g=g.sort_values("recording_id"); nf=int(meta[subj]["n_frames"]); k=len(g)
    if nf % k: continue
    seg=nf//k; sp=g.species.iloc[0]
    if sp not in CHI: continue
    st=np.load(R+"states/"+os.path.basename(statefile[subj])).astype(np.int64)
    if len(st)!=nf-(d_embed-1): continue
    track=np.full(nf,-1,dtype=np.int16)
    for j,(_,r) in enumerate(g.iterrows()):
        s=np.load(f"{CACHE}/seq/{r.recording_id}.npy"); track[j*seg:j*seg+len(s)]=s[:seg]
    half = h_i % 2                                     # split by individual for the noise floor
    for j in range(k):                                  # never cross a recording boundary
        lo, hi = j*seg, min((j+1)*seg - (d_embed-1), len(st))
        if hi-lo < 3: continue
        syl=track[lo:hi].astype(np.int64); cl=st[lo:hi]
        ok=(syl>=0)&(cl>=0)&(cl<CHI[sp].shape[0])
        syl=np.where(ok,syl,-1)
        chg=np.nonzero(np.diff(syl)!=0)[0]              # bout boundaries
        if len(chg)<2: continue
        a_s, b_s = syl[chg], syl[chg+1]                 # from -> to
        w = CHI[sp][np.clip(cl[chg+1],0,CHI[sp].shape[0]-1)]
        good=(a_s>=0)&(b_s>=0)&(a_s<S)&(b_s<S)&ok[chg]&ok[chg+1]
        a_s,b_s,w=a_s[good],b_s[good],w[good]
        for arm in range(n_arm):
            np.add.at(T[sp][arm,half], (a_s,b_s), w[:,arm])
            np.add.at(OCC[sp][arm,half], a_s, w[:,arm])
    nsub+=1
    if time.time()-t0>150: print("TIMEOUT at subject",nsub); break
np.savez(CACHE+"/bout_transitions.npz", **{f"T__{sp}":T[sp] for sp in T},
         **{f"OCC__{sp}":OCC[sp] for sp in OCC})
print(f"subjects={nsub} elapsed={time.time()-t0:.0f}s")
for sp in CHI:
    tot=T[sp].sum((1,2,3))
    print(f"  {sp:26s} bout transitions: arm1={tot[0]:,.0f} arm2={tot[1]:,.0f}")
