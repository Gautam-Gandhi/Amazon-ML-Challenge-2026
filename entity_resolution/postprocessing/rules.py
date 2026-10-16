"""Rules applied to the final test probabilities. Each one targets a pair type found by label-free count matching
(accepted pairs per type on test vs true pairs per type in validation) or by word-role statistics.

  france_word_roles     unseen countries: accept a one-word swap at the same house number when the swapped-in word has a
                        noise role in that country's own statistics (replaces a word, over-represented in records)
  us_legal_form_siblings US: demote same-name pairs where the S1 entity has no legal form, the record has one and the
                        house number differs (sibling companies the test contains far more often than training)
  french_acronyms       unseen countries: accept acronym records (initials of the S1 name, same house number, unique)
  french_descriptor_swaps unseen countries: demote descriptor-for-descriptor swaps at the same house number
                        ("Club" -> "Ecole": sibling organisations; vocabulary learned from the country's own records)
  french_acronym_candidates unseen countries: add acronym pairs that pruning lost (same number + street, unique S1)
"""
import numpy as np
import polars as pl

from .. import config as C
from ..utils import LOG


def assigned(pred):
    """Each record's best S1 entity."""
    return pred.filter(pl.col("p") == pl.col("p").max().over("r")).unique(subset=["r"], keep="first")


def _first_number(col):
    return col.str.split(" ").list.first().fill_null("")


# ---------------------------------------------------------------------------------------------- France: word roles
def _role_pairs(a, s1, r):
    """Assigned pairs whose names differ by one extra record word (+ <= 1 missing S1 word), with role statistics."""
    c = s1["country"].to_numpy()
    si, ri = a["s1"].to_numpy(), a["r"].to_numpy()
    A = s1["name_core"].gather(si).str.split(" ").list.unique()
    B = r["name_core"].gather(ri).str.split(" ").list.unique()
    h1 = s1["addr_nums"].gather(si).str.split(" ").list.first()
    h2 = r["addr_nums"].gather(ri).str.split(" ").list.first()
    a = a.with_columns(B.list.set_difference(A).alias("extra"), A.list.set_difference(B).alias("miss"),
                       pl.Series("country", c[si]), ((h1 == h2) & (h1 != "")).alias("hn_eq"))
    d = a.filter((pl.col("extra").list.len() == 1) & (pl.col("miss").list.len() <= 1)).with_columns(
        pl.col("extra").list.first().alias("w"), (pl.col("miss").list.len() == 1).alias("swap"))
    role = d.group_by(["country", "w"]).agg(pl.len().alias("wn"), pl.col("swap").mean().alias("role_swap"))
    ns = []
    for cc in np.unique(c):
        ta = s1.filter(pl.col("country") == cc)["name_core"].str.split(" ").list.unique().explode().value_counts()
        rc = r.filter(pl.col("country") == cc)
        tb = rc["name_core"].str.split(" ").list.unique().explode().value_counts()
        n1 = int((c == cc).sum())
        t = tb.rename({tb.columns[0]: "w", "count": "nr"}).join(
            ta.rename({ta.columns[0]: "w", "count": "ns1"}), on="w", how="left").fill_null(0)
        ns.append(t.with_columns(pl.lit(cc).alias("country"),
                                 ((pl.col("nr") + 1) / (pl.col("ns1") + 1)).log().sub(np.log(rc.height / n1)).alias("nsc"))
                  .select("country", "w", "nsc"))
    return d.join(role, on=["country", "w"]).join(pl.concat(ns), on=["country", "w"], how="left")


def france_word_roles(ws, pred):
    R = C.ROLE_RULE
    s1 = pl.read_parquet(ws.table("test", "s1"), columns=["eid", "country", "name_core", "addr_nums"])
    r = pl.read_parquet(ws.table("test", "r"), columns=["eid", "country", "name_core", "addr_nums"])
    d = _role_pairs(assigned(pred), s1, r)
    rule = d.filter(pl.col("swap") & pl.col("hn_eq") & (pl.col("role_swap") >= R["min_role"]) & (pl.col("wn") >= R["min_count"])
                    & (pl.col("nsc") > R["min_over"]) & ~pl.col("country").is_in(list(C.TRAINING_COUNTRIES)))
    flip = rule.filter((pl.col("p") <= R["p_set"]) & (pl.col("p") > R["p_floor"]))
    LOG.info("France word-role rule: %s pairs accepted", f"{flip.height:,}")
    return pred.join(flip.select("s1", "r", pl.lit(R["p_set"], pl.Float32).alias("pn")), on=["s1", "r"], how="left") \
               .with_columns(pl.max_horizontal("p", pl.col("pn").fill_null(0)).alias("p")).drop("pn")


# ---------------------------------------------------------------------------------------------- US legal-form siblings
def us_legal_form_siblings(ws, pred):
    R = C.SIBLING_RULE
    s1 = pl.read_parquet(ws.table("test", "s1"), columns=["country", "name_core", "addr_nums", "name_sfx"])
    r = pl.read_parquet(ws.table("test", "r"), columns=["name_core", "addr_nums", "name_sfx"])
    si, ri = pred["s1"].to_numpy(), pred["r"].to_numpy()
    d = pl.DataFrame({
        "ctry": s1["country"].to_numpy()[si],
        "same_name": (s1["name_core"].gather(si) == r["name_core"].gather(ri)).to_numpy(),
        "f1": s1["name_sfx"].gather(si).fill_null(""), "f2": r["name_sfx"].gather(ri).fill_null(""),
        "h1": _first_number(s1["addr_nums"].gather(si)), "h2": _first_number(r["addr_nums"].gather(ri))})
    d = d.with_columns((pl.col("h1").str.extract(r"(\d+)").cast(pl.Int64, strict=False)
                        - pl.col("h2").str.extract(r"(\d+)").cast(pl.Int64, strict=False)).abs().alias("dnum"))
    m = (pl.col("ctry").is_in(list(R["countries"])) & pl.col("same_name") & (pl.col("f1") == "") & (pl.col("f2") != "")
         & (pl.col("h1") != "") & (pl.col("h2") != "") & (pl.col("h1") != pl.col("h2")))
    d = d.with_columns(m.alias("m"))
    mask, dnum = d["m"].to_numpy(), d["dnum"].fill_null(10**9).to_numpy()
    p = pred["p"].to_numpy().astype(np.float32)
    hit = mask & (p <= np.where(dnum <= 12, R["t_near"], R["t_far"]))
    LOG.info("US legal-form sibling rule: %s pairs demoted", f"{int(hit.sum()):,}")
    return pred.with_columns(pl.Series("p", np.where(hit, p * R["demote"], p).astype(np.float32)))


# ---------------------------------------------------------------------------------------------- French acronyms
STOP = {"la", "le", "les", "de", "du", "des", "d", "l", "et", "and", "of", "the", "a", "en", "au", "aux"}


def initials(core, drop_stop):
    toks = [t for t in core.split() if t]
    if drop_stop:
        toks = [t for t in toks if t not in STOP]
    return "".join(t[0] for t in toks)


def french_acronyms(ws, pred):
    R = C.ACRONYM_RULE
    s1 = pl.read_parquet(ws.table("test", "s1"), columns=["country", "name_core", "addr_nums"])
    r = pl.read_parquet(ws.table("test", "r"), columns=["name_core", "addr_nums"])
    rc = r["name_core"].to_numpy()
    short = np.array([(2 <= len(x) <= 5 and " " not in x and x.isalpha()) for x in rc])
    t = pred.with_row_index("i").filter(pl.Series(short[pred["r"].to_numpy()]))
    t = t.with_columns(pl.col("p").rank("ordinal", descending=True).over("r").alias("rk"))
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    s1c = s1["name_core"].to_numpy()
    rn = rc[ri]
    acro = np.array([(initials(s1c[s], False) == x) or (initials(s1c[s], True) == x) for s, x in zip(si, rn)])
    t = t.with_columns(pl.Series("acro", acro))
    t = t.with_columns(pl.col("acro").sum().over("r").alias("n_acro"))
    t = t.filter(pl.col("acro"))
    si, ri = t["s1"].to_numpy(), t["r"].to_numpy()
    h1 = _first_number(s1["addr_nums"].gather(si))
    h2 = _first_number(r["addr_nums"].gather(ri))
    t = t.with_columns(pl.Series("country", s1["country"].to_numpy()[si]), h1.alias("h1"), h2.alias("h2"))
    f = t.filter(~pl.col("country").is_in(list(C.TRAINING_COUNTRIES)) & (pl.col("h1") != "") & (pl.col("h1") == pl.col("h2"))
                 & (pl.col("n_acro") == 1) & (pl.col("rk") == 1) & (pl.col("p") > R["p_floor"]) & (pl.col("p") <= C.UNSEEN_THRESHOLD))
    idx = f["i"].to_numpy()
    LOG.info("French acronym rule: %s pairs accepted", f"{len(idx):,}")
    p = pred["p"].to_numpy().copy()
    p[idx] = np.maximum(p[idx], R["p_set"])
    return pred.with_columns(pl.Series("p", p.astype(np.float32)))


# ---------------------------------------------------------------------------------------------- French descriptor swaps
def _descriptor_vocab(s1, r, country, swapped_in):
    """Descriptor words of a country: in >= min_s1 S1 names, not over-represented in records, and swapped in at the
    same house number >= min_swaps times."""
    R = C.DESCRIPTOR_RULE
    s = s1.filter(pl.col("country") == country)
    rr = r.filter(pl.col("country") == country)
    cnt = lambda d: d.select(pl.col("name_core").str.split(" ").list.unique().alias("w")).explode("w").group_by("w").len()
    a = cnt(s).rename({"len": "n_s1"})
    b = cnt(rr).rename({"len": "n_r"})
    m = a.join(b, on="w", how="left").fill_null(0).with_columns(((pl.col("n_r") / rr.height) / (pl.col("n_s1") / s.height)).alias("over"))
    sw = swapped_in.value_counts().rename({swapped_in.name: "w", "count": "n_swap"})
    m = m.join(sw, on="w", how="left").fill_null(0)
    v = m.filter((pl.col("n_s1") >= R["min_s1"]) & (pl.col("over") < R["max_over"]) & (pl.col("n_swap") >= R["min_swaps"])
                 & (pl.col("w").str.len_chars() >= 3) & ~pl.col("w").is_in(list(R["exclude"])))
    return set(v["w"].to_list())


def french_descriptor_swaps(ws, pred):
    R = C.DESCRIPTOR_RULE
    cols = ["country", "name_core", "addr_nums"]
    s1 = pl.read_parquet(ws.table("test", "s1"), columns=cols)
    r = pl.read_parquet(ws.table("test", "r"), columns=cols)
    si, ri = pred["s1"].to_numpy(), pred["r"].to_numpy()
    A = s1["name_core"].gather(si).str.split(" ").list.unique()
    B = r["name_core"].gather(ri).str.split(" ").list.unique()
    t = pred.with_row_index("i").with_columns(
        pl.Series("country", s1["country"].to_numpy()[si]),
        A.list.set_difference(B).alias("miss"), B.list.set_difference(A).alias("extra"),
        s1["addr_nums"].gather(si).str.split(" ").list.first().fill_null("").alias("h1"),
        r["addr_nums"].gather(ri).str.split(" ").list.first().fill_null("").alias("h2"))
    t = t.filter(~pl.col("country").is_in(list(C.TRAINING_COUNTRIES)) & (pl.col("miss").list.len() == 1)
                 & (pl.col("extra").list.len() == 1) & (pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")))
    t = t.with_columns(pl.col("miss").list.first().alias("a"), pl.col("extra").list.first().alias("b"))
    idx = []
    for c in sorted(t["country"].unique().to_list()):
        tc = t.filter(pl.col("country") == c)
        vocab = _descriptor_vocab(s1, r, c, tc["b"].rename("b"))
        hit = tc.filter(pl.col("a").is_in(list(vocab)) & pl.col("b").is_in(list(vocab)))
        LOG.info("French descriptor-swap rule (%s): %d descriptor words, %s pairs demoted", c, len(vocab), f"{hit.height:,}")
        idx.append(hit["i"].to_numpy())
    idx = np.concatenate(idx) if idx else np.array([], dtype=np.int64)
    p = pred["p"].to_numpy().copy()
    p[idx] = p[idx] * R["demote"]
    return pred.with_columns(pl.Series("p", p.astype(np.float32)))


# ---------------------------------------------------------------------------------------------- French acronym candidates
def _street(df):
    return df.with_columns(
        pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("h"),
        pl.col("addr_clean").str.split(" ").list.eval(
            pl.element().filter(~pl.element().str.contains(r"\d") & (pl.element().str.len_chars() > 1)
                                & ~pl.element().str.starts_with("fr"))).alias("w"))


def french_acronym_candidates(ws, pred):
    """Acronym records whose initials match exactly one S1 entity at the same house number and street are added as
    candidates (they are near-certain matches that pruning lost), unless the record is already accepted elsewhere."""
    cols = ["country", "name_core", "addr_clean", "addr_nums"]
    ctry = pl.read_parquet(ws.table("test", "s1"), columns=["country"])["country"]
    unseen = sorted(set(ctry.unique().to_list()) - set(C.TRAINING_COUNTRIES))
    s1 = pl.read_parquet(ws.table("test", "s1"), columns=cols).with_row_index("s1").with_columns(pl.col("s1").cast(pl.Int32))
    r = pl.read_parquet(ws.table("test", "r"), columns=cols).with_row_index("r").with_columns(pl.col("r").cast(pl.Int32))
    s1 = _street(s1.filter(pl.col("country").is_in(unseen)))
    r = _street(r.filter(pl.col("country").is_in(unseen)))
    s1 = s1.with_columns(pl.col("name_core").map_elements(lambda c: initials(c, False), return_dtype=pl.Utf8).alias("i1"),
                         pl.col("name_core").map_elements(lambda c: initials(c, True), return_dtype=pl.Utf8).alias("i2"))
    ra = r.filter(pl.col("name_core").str.contains(r"^[a-z]{2,5}$") & (pl.col("h") != ""))
    pa = pl.concat([ra.join(s1.select("s1", "country", pl.col(i).alias("name_core"), "h", pl.col("w").alias("w1")),
                            on=["country", "name_core", "h"]) for i in ("i1", "i2")]).unique(subset=["s1", "r"])
    pa = pa.with_columns((pl.col("w").list.set_intersection("w1").list.len()
                          / pl.col("w").list.set_union("w1").list.len().clip(1, None)).alias("j")).filter(pl.col("j") >= 0.5)
    pa = pa.filter(pl.len().over("r") == 1).select("s1", "r", "country")
    new = pa.join(pred.select("s1", "r"), on=["s1", "r"], how="anti")
    busy = pred.filter(pl.col("p") > C.UNSEEN_THRESHOLD).select("r").unique()
    add = new.join(busy, on="r", how="anti")
    LOG.info("French acronym candidates: %s pairs added", f"{add.height:,}")
    return pl.concat([pred, add.select(pl.col("s1").cast(pred["s1"].dtype), pl.col("r").cast(pred["r"].dtype),
                                       pl.lit(C.ACRONYM_CANDIDATES_P, dtype=pl.Float32).alias("p"))])
