import json, os, numpy as np, pandas as pd, pickle, sys, time
sys.path.insert(0, os.path.expanduser("~/mnt/slowmodeevo"))
R=os.path.expanduser("~/mnt/slowmodeevo/outputs/multispecies_slow_modes/global_clustering_modes/global_outputs/XY_EvenSampled_SlowModes/")
CACHE=os.path.expanduser("~/mqcache")
idx=json.load(open(CACHE+"/index.json"))
rec=pd.DataFrame(idx).T.rename_axis("recording_id").reset_index()
rec["n"]=rec["n"].astype(int)
md=pickle.load(open(R+"projection_result.pkl","rb"))["metadata"]
md=md if isinstance(md,list) else list(md.values())
meta={m["individual_id"].split("__subject__")[-1]: m for m in md}
man=pd.read_csv(R+"global_kmeans_state_manifest.csv")
man["subject"]=man.source_individual_id.astype(str)
statefile=dict(zip(man.subject, man.state_file)); d_embed=int(man.d_embed.iloc[0])
VIEW=sys.argv[1] if len(sys.argv)>1 else "with_low_occupancy_cutoff"
CHI={sp: np.load(f"{R}saved_slow_mode_arrays/{VIEW}/{sp}/chi_global_aligned.npy")
     for sp in sorted(os.listdir(f"{R}saved_slow_mode_arrays/{VIEW}")) }
n_arm=next(iter(CHI.values())).shape[1]
NSYL=100
acc={sp: np.zeros((n_arm, NSYL)) for sp in CHI}
stats=dict(subjects=0, frames=0, dropped_boundary=0, dropped_trunc=0, dropped_missing=0, bad=[])
t0=time.time()
for subj, g in rec.groupby("subject"):
    if subj not in meta or subj not in statefile: stats["bad"].append(("nometa",subj)); continue
    g=g.sort_values("recording_id")
    nf=int(meta[subj]["n_frames"]); k=len(g)
    if nf % k: stats["bad"].append(("nondiv",subj)); continue
    seg=nf//k
    if not set(g.n.unique()) <= {seg, 215995}: stats["bad"].append(("seglen",subj)); continue
    sp=g.species.iloc[0]
    if sp not in CHI: stats["bad"].append(("nospecies",subj)); continue
    st=np.load(R+"states/"+os.path.basename(statefile[subj])).astype(np.int64)
    if len(st) != nf-(d_embed-1): stats["bad"].append(("statelen",subj)); continue
    # concatenated syllable track, -1 where truncated/missing
    track=np.full(nf, -1, dtype=np.int16)
    for j,(_,r) in enumerate(g.iterrows()):
        s=np.load(f"{CACHE}/seq/{r.recording_id}.npy")
        track[j*seg : j*seg+len(s)] = s[:seg]
        stats["dropped_trunc"] += seg-min(len(s),seg)
    fi=np.arange(len(st))
    seg_of=fi//seg
    crosses=((fi % seg) + d_embed-1) >= seg          # embedding window spans a boundary
    syl=track[fi]
    keep=(~crosses) & (syl>=0) & (st>=0) & (st<CHI[sp].shape[0])
    stats["dropped_boundary"]+=int(crosses.sum()); stats["dropped_missing"]+=int(((syl<0)&~crosses).sum())
    s_ok=syl[keep].astype(np.int64); c_ok=st[keep]
    m=CHI[sp][c_ok]
    for a in range(n_arm):
        acc[sp][a] += np.bincount(s_ok, weights=m[:,a], minlength=NSYL)
    stats["subjects"]+=1; stats["frames"]+=int(keep.sum())
rows=[{"species":sp,"aligned_arm":a+1,"moseq_cluster":s,"soft_frames_mine":acc[sp][a,s]}
      for sp in acc for a in range(n_arm) for s in range(NSYL) if acc[sp][a,s]>0]
out=pd.DataFrame(rows); out.to_csv(f"{CACHE}/soft_frames_reproduced_{VIEW}.csv", index=False)
print(f"view={VIEW} subjects={stats['subjects']} frames={stats['frames']:,} "
      f"boundary={stats['dropped_boundary']:,} trunc={stats['dropped_trunc']:,} "
      f"missing={stats['dropped_missing']:,} elapsed={time.time()-t0:.0f}s")
if stats["bad"]: print("skipped:", pd.Series([b[0] for b in stats['bad']]).value_counts().to_dict())
ref=pd.read_csv(R+"matched_arm_syllable_probs.csv")[["species","aligned_arm","moseq_cluster","soft_frames"]]
cmp=ref.merge(out, on=["species","aligned_arm","moseq_cluster"], how="outer")
cmp["ratio"]=cmp.soft_frames_mine/cmp.soft_frames
print(f"\nrows ref={len(ref)} mine={len(out)} merged={len(cmp)} | both present={cmp.dropna().shape[0]}")
print("ratio (mine/reference) summary:"); print(cmp.ratio.describe().round(4).to_string())
tot=cmp.groupby("species")[["soft_frames","soft_frames_mine"]].sum()
tot["ratio"]=tot.soft_frames_mine/tot.soft_frames
print("\nper-species totals:"); print(tot.round(1).to_string())
