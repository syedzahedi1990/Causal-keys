import json
import sys

import numpy as np

J = json.load(open(sys.argv[1]))
for arm, rows in J["res"].items():
    names = list(rows[0]["r"])
    r = lambda n, k: np.array([x["r"][n][x[k]] for x in rows])  # noqa: E731
    lse = lambda n: np.array([np.log(np.exp(x["r"][n]).sum()) for x in rows])  # noqa: E731
    am = lambda n: np.array([int(np.argmax(x["r"][n])) for x in rows])  # noqa: E731
    c = lambda n: r(n, "iS") - r(n, "iX")  # noqa: E731
    idK = 0.5 * ((c("KS") - c("none")) - (c("KX") - c("none")))
    iB, iX, iS, iI = (np.array([x[k] for x in rows]) for k in ("iB", "iX", "iS", "iI"))
    iD = np.array([x.get("iD", -1) for x in rows])
    mass = np.mean(np.exp(lse("none")))
    ment = np.mean([(a in (b, i, dd)) for a, b, i, dd in zip(am("none"), iB, iI, iD)])
    print(f"{arm}: n={len(rows)} cand mass {mass:.2f}; clean argmax mentioned {ment:.2f} (B {np.mean(am('none') == iB):.2f}, S {np.mean(am('none') == iS):.2f}, X {np.mean(am('none') == iX):.2f})")
    print(f"   ID_K {idK.mean():+.2f} (sd {idK.std():.2f}, se {idK.std() / np.sqrt(len(idK)):.2f}); dlp(B) under K_S {np.mean(r('KS', 'iB') - r('none', 'iB')):+.2f}; argmax B under K_S {np.mean(am('KS') == iB):.2f}")
    for n in names:
        lX = r(n, "iX") - lse(n) - (r("none", "iX") - lse("none"))
        print(f"   {n:9s} dl_X(renorm) {lX.mean():+6.2f} (se {lX.std() / np.sqrt(len(lX)):.2f})  argmax X {np.mean(am(n) == iX):.2f} B {np.mean(am(n) == iB):.2f}")
