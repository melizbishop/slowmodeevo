import numpy as np, os, json, time, sys
SRC=os.path.expanduser("~/mnt/moseq_dataverse/multispecies_moseq_timeseries.csv")
OUT=os.path.expanduser("~/mqcache"); os.makedirs(OUT+"/seq", exist_ok=True)
budget=float(sys.argv[1]) if len(sys.argv)>1 else 140.0
t0=time.time()
idx_path=OUT+"/index.json"
index=json.load(open(idx_path)) if os.path.exists(idx_path) else {}
with open(SRC) as f:
    header=f.readline().rstrip("\n").split(",")
    tcols=[i for i,h in enumerate(header) if h.startswith("t_")]
    first_t, last_t = tcols[0], tcols[-1]
    n_t = len(tcols)
    n_trail = len(header)-1-last_t
    i_rec = header.index("recording_id")
    assert i_rec < first_t, "unexpected layout"
    tail_names = header[last_t+1:]
    done=0
    for line in f:
        line=line.rstrip("\n")
        pos=-1
        for _ in range(first_t): pos=line.index(",",pos+1)
        start=pos+1
        end=len(line)
        for _ in range(n_trail): end=line.rindex(",",0,end)
        head=line[:start-1].split(",")
        rec=head[i_rec]
        if rec in index: continue
        tail=line[end+1:].split(",")
        body=line[start:end]
        arr=np.fromstring(body, sep=",", dtype=np.float32)
        if len(arr)!=body.count(",")+1 or len(arr)!=n_t:
            print("PARSE FAIL", rec, len(arr), body.count(",")+1, n_t); break
        # real length: trailing NaN/sentinel trimmed
        finite=np.isfinite(arr)
        L=int(np.max(np.nonzero(finite)[0]))+1 if finite.any() else 0
        seq=np.where(np.isfinite(arr[:L]), arr[:L], -1).astype(np.int16)
        np.save(f"{OUT}/seq/{rec}.npy", seq)
        t=dict(zip(tail_names, tail))
        index[rec]={"subject":t["subject"],"species":t["species"].replace(" ","_"),
                    "n":int(L),"n_missing":int((seq<0).sum())}
        done+=1
        if time.time()-t0>budget: break
json.dump(index, open(idx_path,"w"))
print(f"cached_now={done} total={len(index)}/740 elapsed={time.time()-t0:.0f}s")
if index:
    k=next(iter(index)); print("example:", k, index[k])
