"""Paraphrased queries (no shared content words with the target issue): rank of the target."""
import sys
from repo_fix_agent.schema import load_demos
from repo_fix_agent.pipeline.retrieval import RetrievalIndex
Q = {
 "seed-001-pagination-offset": "Browsing results: screen two never shows the first ten entries, they are just gone",
 "seed-012-querystring-plus-decoding": "Typing a query containing a plus symbol loses that symbol on the server side",
 "seed-030-union-find-size-root": "Group merging reports every component as having a single member",
 "seed-052-tz-replace-vs-astimezone": "Per-day registration totals are wrong for users outside Greenwich time",
 "seed-059-init-bypasses-property-setter": "Heating controller loaded from a stored preset ignores temperature limits",
 "seed-066-lambda-late-binding": "Validation messages on the registration page point at the wrong input box",
 "seed-086-tax-on-pre-discount-subtotal": "Promo code reduces the price, but the levy is still computed on the original amount",
 "seed-092-overnight-booking-overlap": "Two reservations for the same room are allowed when one spans 11pm to 2am",
 "seed-099-discount-allocation-rounding": "Returning the final product of an order with a promo fails with an amount error",
 "seed-037-flatten-string-recursion": "Gathering labels hits maximum depth exceeded",
}
train = load_demos("runs/e2e/splits/train.jsonl")
specs = [None, "hash"] + sys.argv[1:]
for spec in specs:
    idx = RetrievalIndex.build(train, embedder=spec, alpha=0.7)
    ranks = []
    for iid, q in Q.items():
        hits = idx.query(q, k=len(idx))
        ranks.append(next(i for i, h in enumerate(hits) if h.entry["instance_id"] == iid) + 1)
    top1 = sum(r == 1 for r in ranks); top3 = sum(r <= 3 for r in ranks)
    mrr = sum(1 / r for r in ranks) / len(ranks)
    print(f"{spec or 'tf-idf':32} top1 {top1}/10  top3 {top3}/10  MRR {mrr:.2f}  ranks {ranks}")
