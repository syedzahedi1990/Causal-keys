s=open('pilot_flag.py').read()
s=s.replace("""for core in R:
    d = prep(tok, core, ARMFIT)
    a3, zb, zf, _ = a3_and_flag_runs(d)
    recs.append((a3, zb, zf))""","""for i, core in enumerate(R):
    d = prep(tok, core, ARMFIT)
    a3, zb, zf, _ = a3_and_flag_runs(d)
    recs.append((a3, zb, zf))
    if i % 5 == 0:
        log(f"rank story {i} T={d['T']}")""")
s=s.replace("torch.set_num_threads(4)","torch.set_num_threads(int(__import__('os').environ.get('NT','3')))")
s=s.replace("""        rows.append(dict(iB=d["iB"], iS=d["iS"], iX=d["iX"], iI=d["iI"], r=r))""","""        rows.append(dict(iB=d["iB"], iS=d["iS"], iX=d["iX"], iI=d["iI"], iD=d["iD"], r=r))
        if len(rows) % 5 == 0:
            log(f"eval {arm} story {len(rows)}")""")
open('pilot_flag.py','w').write(s)
