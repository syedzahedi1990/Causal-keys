"""Per-model data of the stage-9 scorer: the stories (and l_b) files, the passage files and the edit file of one model,
as written by experiments/stage9_stories.py, stage9_passages.py and stage9_edits.py (layouts: docs/stage9/SPEC.md
sections 11.4-11.6), with the per-item arrays every statistic is built from.

Stories (SPEC 9.4-9.5): an item is competent in format f (C9(f)) when its ``none`` row answers B and its clean S run
answers S; flip(f, row) = the share of C9(f) whose generated answer is S, stale = B; the E-argmax version reads the
argmax of the row's E six-vector (canonical word order) instead of the generated answer. One index set per file: every
item of the file in item order (an item absent from a format is outside that format's subset).
Passages (SPEC 9.3-9.4): per split, competent in f when who(none) = B and who(S_run) = S under the answer rule
(containment primary: the builder's who_contain, whose normaliser drops possessives and Unicode punctuation for every
entity alike; strict, the stage-8a rule, the sensitivity); flip = who(row) = S; the argmax version reads the argmax of
the decision-token log-probabilities lp. Clusters: the article.
BF16 floor (SPEC 14, CONTROLS): the none row (edits: self) against the clean B run of the same item in another batch
shape (stories: "clean_B" of the capture pass; passages: "clean_B" of the capture batch; edits: the CAA batch's self row,
the B prompt unclamped); none against none2 (self against self2) is an in-batch determinism check (the same ids and
tables in one forward: 0 expected, also in BF16).
Edits (SPEC 9.7): per core and row Z, hit(Z) = the mean over t in (S, X) of 1[answer of Z|t = t] (stale: = B), so a
rate over cores is the rate over (core, t) and both targets move together; ID(R) = 1/2 [(d_S(R|S) - d_S(R|X)) +
(d_X(R|X) - d_X(R|S))] with d_w(row) = E_w(row) - E_w(self) (nats) or the logit margin m_w = L_w - L_B taken relative to
self; r_K(Z|t) = stats[l][Z|t].proj_k (CAA: E1's). Clusters: (base, source).
"""
from __future__ import annotations

import functools

import numpy as np


def _argmax(v) -> int:
    """The index of the first maximum (numpy.argmax's rule) of a short list, without numpy's per-call overhead."""
    return max(range(len(v)), key=v.__getitem__)

from .stats import NAN, boot_of

STORY_FORMATS = ("NONE", "BEFORE", "POST-NULL", "POST", "AFTER", "LETTER")
EDIT_FORMATS = ("NONE", "AFTER", "LETTER", "POST")
PASSAGE_FORMATS = ("NOM", "OPTB", "OPTA")
LOCATIONS = ("box", "basket", "shelf", "drawer", "cabinet", "closet")   # ckeys.story.LOCATIONS (asserted in the tests)
TARGETS = ("S", "X")


def story_onsets(nL: int) -> list[int]:
    return list(range(2, nL - 1, 2))


def passage_onsets(nL: int) -> list[int]:
    return [round(x * nL) for x in (0.15, 0.3, 0.45, 0.6)]


def nearest_onset(onsets, x: float) -> int:
    """The grid onset nearest to x (ties: the lower one); equals round(x) when round(x) is on the grid."""
    return min(onsets, key=lambda l: (abs(l - x), l))


# --------------------------------------------------------------------------- stories (and the l_b file)
class StoryRun:
    """One stories eval file (population N9) or l_b file (population C) of one model."""

    def __init__(self, J: dict, label: str = ""):
        self.J, self.label = J, label
        self.P = J.get("provenance") or {}
        self.rows = list(J.get("rows") or [])
        self.L = int(self.P.get("n_layers") or 0)
        self.onsets = sorted(int(r.split("@")[1]) for r in self.rows if r.startswith("late@"))
        self.kv_onsets = sorted(int(r.split("@")[1]) for r in self.rows if r.startswith("Klate@"))
        recs = J.get("items") or []
        self.by: dict = {}
        meta: dict = {}
        for r in recs:
            self.by.setdefault(r["fmt"], {})[r["key"]] = r
            meta.setdefault(r["key"], r)
        self.keys = sorted(meta, key=lambda k: (meta[k].get("idx", 0), k))
        self.meta = [meta[k] for k in self.keys]
        self.n = len(self.keys)
        self.fmts = [f for f in STORY_FORMATS if f in self.by]
        self.clusters = [tuple(m["cluster"]) for m in self.meta]
        self.B = boot_of(self.clusters) if self.n else None
        self.track = {t: np.array([m["track"][t] for m in self.meta], dtype=object) for t in ("S", "B", "X")}

    def has(self, fmt) -> bool:
        return fmt in self.by

    def _rec(self, fmt, i):
        return self.by.get(fmt, {}).get(self.keys[i])

    def _run(self, fmt, i, row):
        r = self._rec(fmt, i)
        if r is None:
            return None
        return r["clean_S"] if row == "clean_S" else r["rows"].get(row)

    @functools.lru_cache(maxsize=None)
    def present(self, fmt) -> np.ndarray:
        return np.array([self._rec(fmt, i) is not None for i in range(self.n)], float)

    @functools.lru_cache(maxsize=None)
    def ans(self, fmt, row, how="gen") -> np.ndarray:
        """The answer word per item ("" if absent): the generated answer, or the E-argmax word (how="E")."""
        out = []
        for i in range(self.n):
            d = self._run(fmt, i, row)
            if d is None:
                out.append("")
            elif how == "E":
                out.append(self.meta[i]["words"][_argmax(d["E"])])
            else:
                out.append(d.get("ans", ""))
        return np.array(out, dtype=object)

    @functools.lru_cache(maxsize=None)
    def is_(self, fmt, row, who, how="gen") -> np.ndarray:
        return (self.ans(fmt, row, how) == self.track[who]).astype(float)

    @functools.lru_cache(maxsize=None)
    def comp(self, fmt) -> np.ndarray:
        """C9(fmt): present, none answers B, the clean S run answers S."""
        return self.present(fmt) * self.is_(fmt, "none", "B") * self.is_(fmt, "clean_S", "S")

    def mask(self, fmt, sub="comp") -> np.ndarray:
        return self.comp(fmt) if sub == "comp" else self.present(fmt)

    @functools.lru_cache(maxsize=None)
    def absp(self, fmt, row, who="B") -> np.ndarray:
        """exp(E_who) of the row per item (absolute probability, not renormalised); NaN if absent."""
        out = []
        for i in range(self.n):
            d = self._run(fmt, i, row)
            out.append(NAN if d is None else float(np.exp(d["E"][self.meta[i]["words"].index(self.track[who][i])])))
        return np.array(out, float)

    @functools.lru_cache(maxsize=None)
    def sixp(self, fmt, row, who="B") -> np.ndarray:
        """The six-way p(who) = softmax(E) of the row per item; NaN if absent."""
        out = []
        for i in range(self.n):
            d = self._run(fmt, i, row)
            if d is None:
                out.append(NAN)
                continue
            e = np.asarray(d["E"], float)
            p = np.exp(e - e.max())
            out.append(float(p[self.meta[i]["words"].index(self.track[who][i])] / p.sum()))
        return np.array(out, float)

    @functools.lru_cache(maxsize=None)
    def e_floor(self, fmt) -> tuple[float, float, int]:
        """The BF16 floor: (mean |E(none) - E(clean_B)| over the items and the six entries, the share of items whose
        E-argmax words differ, the number of items). clean_B is the B prompt's own pass in the capture batch (B, S, X),
        another batch shape than the scoring batch; it has no generated answer, so the answers compared are the
        E-argmax words. (NaN, NaN, 0) when the file has no clean_B."""
        d, a = [], []
        for i in range(self.n):
            r = self._rec(fmt, i)
            x, y = (None, None) if r is None else (r["rows"].get("none"), r.get("clean_B"))
            if x is None or y is None:
                continue
            d.append(float(np.mean(np.abs(np.asarray(x["E"]) - np.asarray(y["E"])))))
            a.append(_argmax(x["E"]) != _argmax(y["E"]))
        return (float(np.mean(d)) if d else NAN, float(np.mean(a)) if a else NAN, len(d))

    @functools.lru_cache(maxsize=None)
    def determinism(self, fmt) -> tuple[float, float, int]:
        """In-batch determinism: (max |E(none) - E(none2)| over the items and entries, the share of items whose
        generated answers differ, the number of items); 0 and 0 expected (the same ids and tables in one forward)."""
        d, a = [], []
        for i in range(self.n):
            x, y = self._run(fmt, i, "none"), self._run(fmt, i, "none2")
            if x is None or y is None:
                continue
            d.append(float(np.max(np.abs(np.asarray(x["E"]) - np.asarray(y["E"])))))
            a.append(x.get("ans") != y.get("ans"))
        return (float(np.max(d)) if d else NAN, float(np.mean(a)) if a else NAN, len(d))

    def letter_conflicts(self, fmt="LETTER") -> tuple[int, int, int, dict]:
        """SPEC 9.2 / 14 (a sensitivity count): (rows whose answer is a parsed letter while the word parse of the same
        text names another candidate word, i.e. ``word`` is neither "other" nor the letter's word; rows with a letter
        answer; rows in all; {row: count} of the rows with a conflict), over every item's rows and clean S run of
        ``fmt``. (0, 0, 0, {}) when no row stores ``word``."""
        c = nl = nr = 0
        per: dict = {}
        for r in self.by.get(fmt, {}).values():
            for name, d in list(r["rows"].items()) + [("clean_S", r["clean_S"])]:
                if "word" not in d:
                    continue
                nr += 1
                if d.get("how") != "letter":
                    continue
                nl += 1
                if d["word"] not in ("other", d.get("ans")):
                    c += 1
                    per[name] = per.get(name, 0) + 1
        return c, nl, nr, per

    def agreement(self, fmt) -> tuple[int, int]:
        """(rows whose generated answer equals the E-argmax word, rows whose generated answer names a candidate)."""
        num = den = 0
        for i in range(self.n):
            r = self._rec(fmt, i)
            if r is None:
                continue
            for d in list(r["rows"].values()) + [r["clean_S"]]:
                if d.get("ans", "other") == "other":
                    continue
                den += 1
                num += self.meta[i]["words"][_argmax(d["E"])] == d["ans"]
        return num, den

    def b_before_s(self) -> np.ndarray:
        """1 where B is listed before S in the item's order (the listing shows words[order[k]] as option k)."""
        out = []
        for m in self.meta:
            W, o = m["words"], list(m["order"])
            out.append(o.index(W.index(m["track"]["B"])) < o.index(W.index(m["track"]["S"])))
        return np.array(out, float)

    def lex(self) -> np.ndarray:
        return np.array([m.get("lex") for m in self.meta])


def late_flips_all(run: StoryRun, fmt: str = "AFTER") -> dict:
    """{onset: share of every item of ``fmt`` (no competence filter) whose late@onset answer is S} (the l_b rule)."""
    recs = [r for r in run.by.get(fmt, {}).values()]
    if not recs:
        return {}
    return {l: sum(r["rows"][f"late@{l}"]["ans"] == r["track"]["S"] for r in recs) / len(recs) for l in run.onsets}


def lb_of(flips: dict, thr: float = 0.2):
    """l_b = the first onset (ascending) whose late AFTER flip is <= thr, else None (SPEC 9.6)."""
    return next((l for l in sorted(flips) if flips[l] <= thr), None)


# --------------------------------------------------------------------------- passages
ENT = ("B", "S", "X", "Z", "D")


class PassageSplit:
    """One split (E or R) of a passage file: items in the order of the file, the formats run."""

    def __init__(self, fm: dict):
        self.by = {f: {r["id"]: r for r in recs} for f, recs in fm.items()}
        ids, arts = [], {}
        for f, recs in fm.items():
            for r in recs:
                if r["id"] not in arts:
                    ids.append(r["id"])
                    arts[r["id"]] = r["art"]
        self.ids = ids
        self.n = len(ids)
        self.clusters = [arts[i] for i in ids]
        self.B = boot_of(self.clusters) if self.n else None
        self.fmts = list(fm)

    def has(self, fmt) -> bool:
        return fmt in self.by

    def _run(self, fmt, i, row):
        r = self.by.get(fmt, {}).get(self.ids[i])
        if r is None:
            return None
        return r["S_run"] if row == "S_run" else r["rows"].get(row)

    @functools.lru_cache(maxsize=None)
    def present(self, fmt) -> np.ndarray:
        return np.array([self.ids[i] in self.by.get(fmt, {}) for i in range(self.n)], float)

    @functools.lru_cache(maxsize=None)
    def who(self, fmt, row, rule="contain") -> np.ndarray:
        """The entity per item: who_contain (rule "contain"), who_strict ("strict") or the argmax of lp ("lp")."""
        out = []
        for i in range(self.n):
            d = self._run(fmt, i, row)
            if d is None:
                out.append("")
            elif rule == "lp":
                lp = d.get("lp") or {}
                out.append(max(lp, key=lp.get) if lp else "")
            else:
                out.append(d.get("who_contain" if rule == "contain" else "who_strict", ""))
        return np.array(out, dtype=object)

    @functools.lru_cache(maxsize=None)
    def is_(self, fmt, row, Y, rule="contain") -> np.ndarray:
        return (self.who(fmt, row, rule) == Y).astype(float)

    @functools.lru_cache(maxsize=None)
    def comp(self, fmt, rule="contain") -> np.ndarray:
        """Competent: who(none) = B and who(S_run) = S under the rule (the argmax rule uses containment's)."""
        r = "contain" if rule == "lp" else rule
        return self.present(fmt) * self.is_(fmt, "none", "B", r) * self.is_(fmt, "S_run", "S", r)

    def mask(self, fmt, sub="comp", rule="contain") -> np.ndarray:
        return self.comp(fmt, rule) if sub == "comp" else self.present(fmt)

    def floor(self, fmt) -> tuple[float, float, int]:
        """The BF16 floor: (mean |lp(none) - lp(clean_B)| over the items and the decision tokens, the share of items
        whose decision-token argmax differs, the number of items). clean_B is the B prompt's decision log-probabilities
        from the capture batch (B, S, X, Z + w), another batch shape than the scoring batch. (NaN, NaN, 0) without it."""
        d, a = [], []
        for i in range(self.n):
            r = self.by.get(fmt, {}).get(self.ids[i])
            x, y = (None, None) if r is None else (r["rows"].get("none"), r.get("clean_B"))
            if x is None or y is None or not x.get("lp"):
                continue
            lx, ly = x["lp"], y["lp"]
            d.append(float(np.mean([abs(lx[Y] - ly[Y]) for Y in lx])))
            a.append(max(lx, key=lx.get) != max(ly, key=ly.get))
        return (float(np.mean(d)) if d else NAN, float(np.mean(a)) if a else NAN, len(d))

    def determinism(self, fmt) -> tuple[float, float, int]:
        """In-batch determinism: (max |lp(none) - lp(none2)|, the share of items whose containment answers differ, the
        number of items); 0 and 0 expected."""
        d, a = [], []
        for i in range(self.n):
            x, y = self._run(fmt, i, "none"), self._run(fmt, i, "none2")
            if x is None or y is None:
                continue
            d.append(float(max(abs(x["lp"][Y] - y["lp"][Y]) for Y in x["lp"])) if x.get("lp") else 0.0)
            a.append(x.get("who_contain") != y.get("who_contain"))
        return (float(np.max(d)) if d else NAN, float(np.mean(a)) if a else NAN, len(d))

    def agreement(self, fmt) -> tuple[int, int]:
        num = den = 0
        for i in range(self.n):
            r = self.by.get(fmt, {}).get(self.ids[i])
            if r is None:
                continue
            for d in r["rows"].values():
                if d.get("who_contain", "other") == "other" or not d.get("lp"):
                    continue
                den += 1
                num += max(d["lp"], key=d["lp"].get) == d["who_contain"]
        return num, den


class PassageRun:
    """A passage file of one model (NOM, OPTB, OPTA on E and R; or the LETA file)."""

    def __init__(self, J: dict, label: str = ""):
        self.J, self.label = J, label
        self.P = J.get("provenance") or {}
        self.rows = list(J.get("rows") or [])
        self.L = int(self.P.get("n_layers") or 0)
        self.onsets = sorted(int(r.split("@")[1]) for r in self.rows if r.startswith("late@"))
        self.splits = {name: PassageSplit(fm) for name, fm in (J.get("splits") or {}).items()}

    def split(self, name):
        return self.splits.get(name)


# --------------------------------------------------------------------------- edits
class EditRun:
    """The edit file of one model: E9 cores x formats x blocks (table rows and the CAA batch)."""

    def __init__(self, J: dict, label: str = ""):
        self.J, self.label = J, label
        self.P = J.get("provenance") or {}
        self.cores = sorted(J.get("cores") or [], key=lambda c: c["index"])
        self.n = len(self.cores)
        self.clusters = [tuple(c["cluster"]) for c in self.cores]
        self.B = boot_of(self.clusters) if self.n else None
        self.blocks = [int(x) for x in (self.P.get("blocks") or [])]
        self.formats = list(self.P.get("formats") or [])
        self.word = {"S": [c["core"]["source"] for c in self.cores], "X": [c["X"] for c in self.cores],
                     "B": [c["core"]["base"] for c in self.cores]}
        self.L = int(self.P.get("n_layers") or 0)

    def cell(self, i, fmt, l):
        return self.cores[i]["cells"].get(f"{fmt}@{l}")

    @functools.lru_cache(maxsize=None)
    def has_cell(self, fmt, l) -> bool:
        """The cell exists in every core with its self and self2 rows and the CAA batch's self row (what every statistic
        of the cell reads besides its own rows); a malformed cell is a MISMATCH of the grid check and NOT EVALUABLE here."""
        if self.n == 0:
            return False
        for i in range(self.n):
            c = self.cell(i, fmt, l)
            if c is None or "self" not in c or "self2" not in c or "self" not in (c.get("caa") or {}):
                return False
        return True

    @functools.lru_cache(maxsize=None)
    def has_row(self, fmt, l, Z) -> bool:
        """The cell exists in every core and holds Z|S and Z|X in every core (a row missing from one core makes the
        row NOT EVALUABLE, never a KeyError in a later section)."""
        if not self.has_cell(fmt, l):
            return False
        for i in range(self.n):
            c = self.cell(i, fmt, l)
            rows = (c.get("caa") or {}) if Z == "CAA" else c
            if f"{Z}|S" not in rows or f"{Z}|X" not in rows:
                return False
        return True

    def _row(self, i, fmt, l, name):
        c = self.cell(i, fmt, l)
        if c is None:
            raise KeyError(f"cell {fmt}@{l} missing for core {self.cores[i]['index']}")
        if name.startswith("CAA|"):
            return c["caa"][name]
        return c[name]

    def _self(self, i, fmt, l, Z):
        c = self.cell(i, fmt, l)
        return c["caa"]["self"] if Z == "CAA" else c["self"]

    @functools.lru_cache(maxsize=None)
    def hit(self, fmt, l, Z, who="t", how="gen") -> np.ndarray:
        """Per core: the mean over t in (S, X) of 1[answer of Z|t = t] (who "t"), or of 1[answer = B] (who "B");
        how "E": the E-argmax location instead of the generated answer."""
        out = []
        for i in range(self.n):
            v = 0.0
            for t in TARGETS:
                d = self._row(i, fmt, l, f"{Z}|{t}")
                a = LOCATIONS[_argmax(d["E"])] if how == "E" else d.get("ans", "")
                v += a == self.word[t if who == "t" else "B"][i]
            out.append(v / len(TARGETS))
        return np.array(out, float)

    @functools.lru_cache(maxsize=None)
    def unit_hit(self, fmt, l, Z, t) -> np.ndarray:
        """Per core: 1[answer of Z|t = t]."""
        return np.array([float(self._row(i, fmt, l, f"{Z}|{t}").get("ans", "") == self.word[t][i]) for i in range(self.n)])

    @functools.lru_cache(maxsize=None)
    def self_hit(self, fmt, l, who="B") -> np.ndarray:
        return np.array([float(self._self(i, fmt, l, "nat").get("ans", "") == self.word[who][i]) for i in range(self.n)])

    def floor(self, fmt, l) -> tuple[float, float, float, int]:
        """The BF16 floor of a cell: (mean |E(self) - E(clean B)| over the cores and six entries, the share of cores
        whose E-argmax differs, the share whose generated answers differ, the number of cores). The clean B run is the
        CAA batch's self row: the B prompt unclamped (its CAA vector is zero) in a batch of 3, against the table batch's
        self row (B's own tables at p from the natural pass)."""
        d, a, g = [], [], []
        for i in range(self.n):
            c = self.cell(i, fmt, l)
            if c is None or "self" not in c or "self" not in (c.get("caa") or {}):
                continue
            x, y = c["self"], c["caa"]["self"]
            d.append(float(np.mean(np.abs(np.asarray(x["E"]) - np.asarray(y["E"])))))
            a.append(_argmax(x["E"]) != _argmax(y["E"]))
            g.append(x.get("ans") != y.get("ans"))
        f = lambda v: float(np.mean(v)) if v else NAN  # noqa: E731
        return f(d), f(a), f(g), len(d)

    def determinism(self, fmt, l) -> tuple[float, float, int]:
        """In-batch determinism of a cell: (max |E(self) - E(self2)|, the share of cores whose generated answers differ,
        the number of cores); 0 and 0 expected."""
        d, a = [], []
        for i in range(self.n):
            c = self.cell(i, fmt, l)
            if c is None or "self" not in c or "self2" not in c:
                continue
            d.append(float(np.max(np.abs(np.asarray(c["self"]["E"]) - np.asarray(c["self2"]["E"])))))
            a.append(c["self"].get("ans") != c["self2"].get("ans"))
        return (float(np.max(d)) if d else NAN, float(np.mean(a)) if a else NAN, len(d))

    def letter_conflicts(self, l, fmt="LETTER") -> tuple[int, int, int, dict]:
        """As StoryRun.letter_conflicts, over every core's table rows and CAA rows of the cell ``fmt``@``l``; {kind: count}
        with the row names' kinds (the target dropped)."""
        c = nl = nr = 0
        per: dict = {}
        for i in range(self.n):
            cell = self.cell(i, fmt, l)
            if cell is None:
                continue
            for name, d in [(k, v) for k, v in cell.items() if k != "caa"] + list((cell.get("caa") or {}).items()):
                if "word" not in d:
                    continue
                nr += 1
                if d.get("how") != "letter":
                    continue
                nl += 1
                if d["word"] not in ("other", d.get("ans")):
                    c += 1
                    k = name.split("|")[0]
                    per[k] = per.get(k, 0) + 1
        return c, nl, nr, per

    def agreement(self, fmt, l) -> tuple[int, int]:
        """(rows whose generated answer equals the E-argmax location, rows whose answer names a location) in a cell."""
        num = den = 0
        for i in range(self.n):
            c = self.cell(i, fmt, l)
            if c is None:
                continue
            for d in [v for k, v in c.items() if k != "caa"] + list(c.get("caa", {}).values()):
                if d.get("ans", "other") == "other":
                    continue
                den += 1
                num += LOCATIONS[_argmax(d["E"])] == d["ans"]
        return num, den

    @functools.lru_cache(maxsize=None)
    def ident(self, fmt, l, Z, sg="E") -> np.ndarray:
        """ID(Z) per core in nats (sg "E") or in logit margins (sg "logit"), relative to the batch's self row."""
        out = []
        for i in range(self.n):
            c = self.cores[i]
            iS, iX, iB = c["iS"], c["iX"], c["iB"]
            s = self._self(i, fmt, l, Z)

            def d(row, w):
                r = self._row(i, fmt, l, row)
                if sg == "E":
                    return r["E"][w] - s["E"][w]
                return (r["L"][w] - r["L"][iB]) - (s["L"][w] - s["L"][iB])
            out.append(0.5 * ((d(f"{Z}|S", iS) - d(f"{Z}|X", iS)) + (d(f"{Z}|X", iX) - d(f"{Z}|S", iX))))
        return np.array(out, float)

    @functools.lru_cache(maxsize=None)
    def rk_unit(self, l, Z, t) -> np.ndarray:
        """r_K of Z|t at block l per core: stats[l][Z|t].proj_k (CAA: E1's)."""
        Z = "E1" if Z == "CAA" else Z
        out = []
        for c in self.cores:
            s = (c.get("stats") or {}).get(str(l), {}).get(f"{Z}|{t}")
            out.append(NAN if s is None or s.get("proj_k") is None else float(s["proj_k"]))
        return np.array(out, float)

    def rk(self, l, Z) -> np.ndarray:
        return 0.5 * (self.rk_unit(l, Z, "S") + self.rk_unit(l, Z, "X"))

    def kinds(self, l) -> list[str]:
        """The table-row kinds of block l in batch order (without the target), then CAA."""
        if not self.n:
            return []
        c = next((self.cell(0, f, l) for f in self.formats if self.cell(0, f, l) is not None), None)
        if c is None:
            return []
        ks = []
        for name in c:
            if "|" in name:
                k = name.split("|")[0]
                if k not in ks:
                    ks.append(k)
        return ks + (["CAA"] if "caa" in c else [])
