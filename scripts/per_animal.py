import sys, os, json, numpy as np, pandas as pd, pickle, time
sys.path.insert(0, os.path.expanduser("~/mnt/slowmodeevo"))
R=os.path.expanduser("~/mnt/slowmodeevo/outputs/multispecies_slow_modes/global_clustering_modes/global_outputs/XY_EvenSampled_SlowModes/")
CACHE=os.path.expanduser("~/mqcache")
idx=json.load(open(CACHE+"/index.json")); rec=pd.DataFrame(idx).T.rename_axis("recording_id").reset_index()
rec["n"]=rec["n"].astype(int)
md=pickle.load(open(R+"projection_result.pkl","rb"))["metadata"]
md=md if isinstance(md,list) else list(md.values())
meta={m["individual_id"].split("__subject__")[-1]:m for m in md}
man=pd.read_csv(R+"global_kmeans_state_manifest.csv"); man["subject"]=man.source_individual_id.astype(str)
statefile=dict(zip(man.subject,man.state_file)); d_embed=int(man.d_embed.iloc[0])
V="without_low_occupancy_cutoff"
CHI={sp: np.load(f"{R}saved_slow_mode_arrays/{V}/{sp}/chi_global_aligned.npy")
     for sp in sorted(os.listdir(f"{R}saved_slow_mode_arrays/{V}"))}
S=50
occ={}; sylarm={}; spec={}
for subj,g in rec.groupby("subject"):
    if subj not in meta or subj not in statefile: continue
    g=g.sort_values("recording_id"); nf=int(meta[subj]["n_frames"]); k=len(g)
    if nf%k: continue
    seg=nf//k; sp=g.species.iloc[0]
    if sp not in CHI: continue
    st=np.load(R+"states/"+os.path.basename(statefile[subj])).astype(np.int64)
    if len(st)!=nf-(d_embed-1): continue
    track=np.full(nf,-1,dtype=np.int16)
    for j,(_,r) in enumerate(g.iterrows()):
        s=np.load(f"{CACHE}/seq/{r.recording_id}.npy"); track[j*seg:j*seg+len(s)]=s[:seg]
    fi=np.arange(len(st)); crosses=((fi%seg)+d_embed-1)>=seg
    syl=track[fi]
    keep=(~crosses)&(syl>=0)&(syl<S)&(st>=0)&(st<1000)
    occ[subj]=np.bincount(st[(st>=0)&(st<1000)],minlength=1000).astype(float)
    m=CHI[sp][st[keep]]; sk=syl[keep].astype(np.int64)
    A=np.zeros((2,S))
    for a in range(2): A[a]=np.bincount(sk,weights=m[:,a],minlength=S)
    sylarm[subj]=A; spec[subj]=sp
subs=sorted(occ)
np.savez(CACHE+"/per_animal.npz", subjects=np.array(subs),
         species=np.array([spec[s] for s in subs]),
         occ=np.stack([occ[s] for s in subs]),
         sylarm=np.stack([sylarm[s] for s in subs]))
print("animals:", len(subs))
print(pd.Series([spec[s] for s in subs]).value_counts().to_string())
