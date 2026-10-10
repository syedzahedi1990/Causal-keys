s=open('pilot_flag.py').read()
s=s.replace("""g = torch.Generator().manual_seed(0)
RAND =""","""if OUT:
    json.dump({"model": MODEL, "Hs": Hs}, open(OUT, "w"))
    torch.save({l: Delta[l] for l in Ls}, OUT.replace(".json", "_delta.pt"))
    torch.save(flags, OUT.replace(".json", "_flags.pt"))
g = torch.Generator().manual_seed(0)
RAND =""")
s=s.replace("""if OUT:
    json.dump({"model": MODEL, "Hs": Hs, "res": res""","""if OUT:
    json.dump({"model": MODEL, "Hs": Hs, "res": res""")
s=s.replace("""    torch.save({l: Delta[l] for l in Ls}, OUT.replace(".json", "_delta.pt"))
log("done")""","""log("done")""")
open('pilot_flag.py','w').write(s)
