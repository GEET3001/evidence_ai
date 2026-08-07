"""Calibrate the corpus-coverage thresholds in config.py against real claims.

MIN_SIMILARITY thresholds a per-query min-max-normalized score, which forces
the top hit of every query toward 1.0 and so cannot detect an off-corpus
claim. The replacement gate uses raw, pre-normalization similarity — but a raw
cosine floor is only meaningful if it is measured, because its useful range
depends entirely on the embedding model and the corpus.

So this script measures it. Two claim sets are scored against the live index:
claims the corpus genuinely covers, and claims it does not, split into
progressively harder categories (other scientific domains, outright nonsense,
and near-miss claims that read as biomedical but sit outside a mental-health
corpus). Thresholds are then chosen from where the two distributions actually
separate, and the separation quality is reported honestly — including the
overlap, if the distributions overlap.

Usage:
    python -m app.pipeline.calibrate_coverage
    python -m app.pipeline.calibrate_coverage --out ../eval/results/similarity_calibration.md
"""

from __future__ import annotations

import app._thread_limits  # noqa: F401  (must precede the faiss/torch imports below)

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from app.config import BASE_DIR, settings
from app.pipeline.retrieval import RetrievalIndex

# Claims the corpus is genuinely about: the five seeded contested topics plus
# the broader mental-health themes the MeSH distribution shows it covers
# (depression, anxiety, CBT, digital/telehealth, exercise, sleep, screen time).
IN_DOMAIN_CLAIMS = [
    "Mindfulness-based interventions significantly reduce ADHD symptoms.",
    "Cannabis use increases the risk of psychosis and schizophrenia.",
    "SSRIs are more effective than placebo for mild-to-moderate depression.",
    "Social media use causes increased depression and anxiety in adolescents.",
    "Single-session psychological debriefing after trauma prevents PTSD.",
    "Cognitive behavioural therapy reduces symptoms of major depressive disorder.",
    "Smartphone-delivered mental health interventions reduce anxiety symptoms.",
    "Physical exercise improves depressive symptoms in adults.",
    "Telemedicine is as effective as in-person therapy for common mental disorders.",
    "Mindfulness meditation reduces symptoms of generalized anxiety disorder.",
    "Antidepressants increase the risk of suicidal ideation in young people.",
    "Screen time is associated with poor sleep quality in adolescents.",
    "Internet-delivered cognitive behavioural therapy is effective for insomnia.",
    "Prenatal cannabis exposure affects neurodevelopmental outcomes in offspring.",
    "Antipsychotic medication reduces relapse rates in schizophrenia.",
    "Parental involvement improves outcomes of ADHD interventions in children.",
    "Problematic social media use is associated with body image concerns.",
    "Psychological therapy is effective for post-traumatic stress disorder.",
    "Depression screening in primary care improves patient outcomes.",
    "Mindfulness training reduces occupational stress in healthcare workers.",
]

# Off-corpus claims, hardest category last. "near_miss" is the real test: these
# are legitimate biomedical claims a mental-health corpus still cannot answer,
# and they are where an embedding-similarity gate is most likely to fail.
OFF_CORPUS_CLAIMS: dict[str, list[str]] = {
    "other_domain": [
        "Quantum entanglement enables faster-than-light communication.",
        "The CRISPR-Cas9 system can be used to edit plant genomes.",
        "Reinforcement learning improves robotic grasping accuracy.",
        "Atlantic hurricane frequency has increased due to ocean warming.",
        "Sourdough fermentation reduces the glycaemic index of bread.",
        "Lithium-ion battery degradation accelerates at high charge rates.",
        "Roman concrete achieves self-healing durability through lime clasts.",
    ],
    "nonsense": [
        "Eating chocolate cake improves stock market forecasting.",
        "Wearing purple socks increases the boiling point of water.",
        "Listening to jazz makes bicycles more aerodynamic.",
        "Counting clouds reduces the tensile strength of steel.",
        "Alphabetising a bookshelf raises local property taxes.",
        "Left-handed pigeons navigate using prime numbers.",
    ],
    "near_miss": [
        "Statins reduce the risk of cardiovascular mortality in older adults.",
        "Metformin improves glycaemic control in type 2 diabetes.",
        "Early mobilisation shortens ICU length of stay after cardiac surgery.",
        "Vitamin D supplementation reduces the incidence of bone fractures.",
        "Proton pump inhibitors increase the risk of chronic kidney disease.",
        "Physiotherapy improves recovery after anterior cruciate ligament reconstruction.",
        "Antibiotic prophylaxis reduces surgical site infection rates.",
    ],
}


@dataclass
class Measurement:
    claim: str
    category: str  # "in_domain" or an off-corpus subcategory
    max_cosine: float
    mean_topk_cosine: float
    top_bm25: float


def measure(index: RetrievalIndex, claim: str, category: str) -> Measurement:
    cosine_scores, bm25_scores = index.raw_scores(claim)
    signals = index.coverage_signals(cosine_scores, bm25_scores)
    return Measurement(
        claim=claim,
        category=category,
        max_cosine=signals.max_cosine,
        mean_topk_cosine=signals.mean_topk_cosine,
        top_bm25=signals.top_bm25,
    )


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": ordered[len(ordered) // 4],
        "median": statistics.median(ordered),
        "p75": ordered[(3 * len(ordered)) // 4],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _histogram(
    in_domain: list[float], off_corpus: list[float], width: int = 46, bins: int = 14
) -> list[str]:
    """Overlaid text histogram. '#' = in-domain, 'o' = off-corpus, 'X' = both.

    Text rather than an image on purpose: it stays readable inside the markdown
    report, diffs cleanly in git, and needs no plotting dependency.
    """
    everything = in_domain + off_corpus
    lo, hi = min(everything), max(everything)
    span = hi - lo or 1e-9
    edges = [lo + span * i / bins for i in range(bins + 1)]

    def counts(values: list[float]) -> list[int]:
        out = [0] * bins
        for v in values:
            idx = min(int((v - lo) / span * bins), bins - 1)
            out[idx] += 1
        return out

    in_counts, off_counts = counts(in_domain), counts(off_corpus)
    peak = max(max(in_counts), max(off_counts)) or 1
    lines = []
    for i in range(bins):
        n_in, n_off = in_counts[i], off_counts[i]
        bar_in = round(n_in / peak * width)
        bar_off = round(n_off / peak * width)
        overlap = min(bar_in, bar_off)
        bar = "X" * overlap + "#" * (bar_in - overlap) + "o" * (bar_off - overlap)
        lines.append(f"  {edges[i]:6.3f}-{edges[i + 1]:6.3f} | {bar:<{width}} {n_in:>2}# {n_off:>2}o")
    return lines


@dataclass
class ThresholdChoice:
    signal: str
    threshold: float
    separated: bool
    min_in_domain: float
    max_off_corpus: float
    in_domain_rejected: int
    off_corpus_caught: int
    off_corpus_total: int


# Fraction of the in-domain spread held back as slack when a signal's
# distributions overlap, so a threshold never sits exactly on the lowest
# observed in-domain value.
_OVERLAP_SAFETY_MARGIN = 0.10

# Where in the gap to sit when the distributions *are* separated. 0.5 would be
# the midpoint; this deliberately sits nearer the off-corpus side. Both edges of
# the gap are sample minima/maxima — the least stable statistics there are — and
# a half-sample holdout showed a midpoint threshold rejecting 40% of unseen
# in-domain claims, because a smaller sample's observed minimum lands higher
# than the true one. Biasing downward trades a little detection for robustness
# on the error that actually breaks working functionality.
_GAP_POSITION = 0.25


def choose_threshold(
    signal: str, in_domain: list[float], off_corpus: list[float]
) -> ThresholdChoice:
    """Pick a floor for one signal.

    The two error types are not symmetric. Letting an off-corpus claim through
    returns a wrong verdict — bad, and the bug this gate exists to fix. But
    rejecting an in-domain claim breaks functionality that currently works, for
    a user asking a question the corpus *can* answer. So the policy is: never
    reject an observed in-domain claim, and catch as much off-corpus material
    as that constraint allows.

    Cleanly separated distributions give both for free, and the midpoint of the
    gap is the maximum-margin choice. When they overlap, no threshold does both,
    so the floor is placed below the lowest in-domain observation with slack —
    a safety net that catches only extreme cases. Reporting that a signal
    contributes little is the point; tuning it up to look effective would just
    move the failure onto real users.
    """
    min_in, max_off = min(in_domain), max(off_corpus)
    separated = min_in > max_off

    if separated:
        threshold = max_off + _GAP_POSITION * (min_in - max_off)
    else:
        spread = max(in_domain) - min_in
        threshold = min_in - _OVERLAP_SAFETY_MARGIN * spread

    return ThresholdChoice(
        signal=signal,
        threshold=threshold,
        separated=separated,
        min_in_domain=min_in,
        max_off_corpus=max_off,
        in_domain_rejected=sum(1 for v in in_domain if v < threshold),
        off_corpus_caught=sum(1 for v in off_corpus if v < threshold),
        off_corpus_total=len(off_corpus),
    )


def holdout_check(measurements: list[Measurement]) -> tuple[int, int, int, int]:
    """Calibrate on half the claims, score the other half.

    Thresholds picked and evaluated on the same claims report their own
    training-set fit, which always flatters. Splitting every category in half by
    alternating index keeps both halves category-balanced and gives a first
    honest read on whether the thresholds generalise at all. With 40 claims this
    is indicative, not conclusive.
    """
    calib, holdout = [], []
    by_category: dict[str, list[Measurement]] = {}
    for m in measurements:
        by_category.setdefault(m.category, []).append(m)
    for group in by_category.values():
        for i, m in enumerate(group):
            (calib if i % 2 == 0 else holdout).append(m)

    def split(ms: list[Measurement]) -> tuple[list[Measurement], list[Measurement]]:
        return (
            [m for m in ms if m.category == "in_domain"],
            [m for m in ms if m.category != "in_domain"],
        )

    calib_in, calib_off = split(calib)
    hold_in, hold_off = split(holdout)

    max_c = choose_threshold(
        "max_cosine", [m.max_cosine for m in calib_in], [m.max_cosine for m in calib_off]
    )
    mean_c = choose_threshold(
        "mean_topk_cosine",
        [m.mean_topk_cosine for m in calib_in],
        [m.mean_topk_cosine for m in calib_off],
    )

    def gated(m: Measurement) -> bool:
        return m.max_cosine < max_c.threshold or m.mean_topk_cosine < mean_c.threshold

    return (
        sum(1 for m in hold_off if gated(m)),
        len(hold_off),
        sum(1 for m in hold_in if gated(m)),
        len(hold_in),
    )


def build_report(
    measurements: list[Measurement],
    max_choice: ThresholdChoice,
    mean_choice: ThresholdChoice,
) -> str:
    in_domain = [m for m in measurements if m.category == "in_domain"]
    off_corpus = [m for m in measurements if m.category != "in_domain"]

    def gated_out(m: Measurement) -> bool:
        return (
            m.max_cosine < max_choice.threshold
            or m.mean_topk_cosine < mean_choice.threshold
        )

    in_rejected = [m for m in in_domain if gated_out(m)]
    off_caught = [m for m in off_corpus if gated_out(m)]

    lines: list[str] = []
    add = lines.append

    add("# Corpus-coverage threshold calibration")
    add("")
    add(
        "Regenerate with `python -m app.pipeline.calibrate_coverage` from `backend/`. "
        "These thresholds are specific to this corpus and to "
        f"`{settings.EMBEDDING_MODEL}` — rebuilding the index or changing the "
        "embedding model invalidates them."
    )
    add("")
    add("## Why this exists")
    add("")
    add(
        "`MIN_SIMILARITY` is compared against a per-query min-max-normalized "
        "score. Normalization sets the top-ranked passage of *every* query to "
        "1.0, so the threshold answers \"is this the best of what we found?\" "
        "and never \"is any of this actually relevant?\". An off-corpus claim "
        "therefore sails through it and receives a confident verdict. The gate "
        "calibrated here uses raw pre-normalization cosine instead, which is "
        "comparable across queries — but its useful range is a property of the "
        "embedding model and corpus, so it has to be measured rather than guessed."
    )
    add("")
    add("## Claim sets")
    add("")
    add(f"- **in_domain** — {len(in_domain)} claims on topics the corpus covers.")
    for name, claims in OFF_CORPUS_CLAIMS.items():
        add(f"- **{name}** — {len(claims)} off-corpus claims.")
    add("")
    add(
        "`near_miss` is the category that matters. Those are real biomedical "
        "claims that a mental-health corpus still cannot answer, and they sit "
        "closest to the decision boundary."
    )
    add("")

    add("## Distributions")
    add("")
    for signal, getter in (
        ("max_cosine", lambda m: m.max_cosine),
        ("mean_topk_cosine", lambda m: m.mean_topk_cosine),
        ("top_bm25", lambda m: m.top_bm25),
    ):
        add(f"### {signal}")
        add("")
        add("| Set | n | min | p25 | median | p75 | max | mean |")
        add("|---|---|---|---|---|---|---|---|")
        groups = [("in_domain", in_domain)] + [
            (name, [m for m in measurements if m.category == name])
            for name in OFF_CORPUS_CLAIMS
        ]
        groups.append(("ALL off-corpus", off_corpus))
        for name, group in groups:
            s = _summary([getter(m) for m in group])
            add(
                f"| {name} | {s['n']:.0f} | {s['min']:.3f} | {s['p25']:.3f} | "
                f"{s['median']:.3f} | {s['p75']:.3f} | {s['max']:.3f} | {s['mean']:.3f} |"
            )
        add("")
        add("```")
        add("  in-domain = #    off-corpus = o    overlap = X")
        lines.extend(
            _histogram([getter(m) for m in in_domain], [getter(m) for m in off_corpus])
        )
        add("```")
        add("")

    add("## Chosen thresholds")
    add("")
    add("| Signal | Threshold | Distributions | Lowest in-domain | Highest off-corpus |")
    add("|---|---|---|---|---|")
    for choice in (max_choice, mean_choice):
        state = "cleanly separated" if choice.separated else "**OVERLAP**"
        add(
            f"| `{choice.signal}` | **{choice.threshold:.3f}** | {state} | "
            f"{choice.min_in_domain:.3f} | {choice.max_off_corpus:.3f} |"
        )
    add("")
    for choice in (max_choice, mean_choice):
        if choice.separated:
            add(
                f"- `{choice.signal}` separates cleanly "
                f"({choice.max_off_corpus:.3f} off-corpus max vs "
                f"{choice.min_in_domain:.3f} in-domain min), so the threshold sits "
                f"{_GAP_POSITION:.0%} of the way up the gap — deliberately below "
                "the midpoint, see the holdout note below."
            )
        else:
            add(
                f"- `{choice.signal}` **does not separate the two sets**: the "
                f"highest off-corpus value ({choice.max_off_corpus:.3f}) is above "
                f"the lowest in-domain one ({choice.min_in_domain:.3f}). No "
                "threshold can catch every off-corpus claim without also "
                "rejecting valid ones. It is therefore set below the lowest "
                "in-domain observation as a safety net only, catching "
                f"{choice.off_corpus_caught}/{choice.off_corpus_total} off-corpus "
                "claims on its own rather than being tuned up at the cost of "
                "false rejections."
            )
    add("")
    add(
        "The two error types are not symmetric, which is what drives that policy. "
        "Letting an off-corpus claim through produces a wrong verdict — the bug "
        "being fixed. Rejecting an in-domain claim breaks a question the corpus "
        "can actually answer. The second is worse, so thresholds are chosen to "
        "never reject an observed in-domain claim."
    )
    add("")
    add(
        "`top_bm25` is measured and carried through the pipeline but is **not** "
        "part of the gate. It is an unbounded, corpus-frequency-dependent score "
        "rather than a bounded one, so it has no stable cross-query scale to put "
        "a fixed floor on. Its distribution is reported above because near-zero "
        "lexical overlap is a useful independent diagnostic when a gate decision "
        "looks wrong."
    )
    add("")

    add("## Separation quality (combined gate)")
    add("")
    add(
        "The gate rejects a claim when `max_cosine` is below its floor **or** "
        "`mean_topk_cosine` is below its floor."
    )
    add("")
    add(f"- Off-corpus claims correctly caught: **{len(off_caught)}/{len(off_corpus)}**")
    add(f"- In-domain claims wrongly rejected: **{len(in_rejected)}/{len(in_domain)}**")
    add("")
    caught_h, total_off_h, rejected_h, total_in_h = holdout_check(measurements)
    add(
        f"**Those two numbers are training-set fit** — the thresholds were chosen "
        f"on these same claims, so they are an upper bound, not an estimate of "
        f"real-world behaviour. Recalibrating on half the claims and scoring the "
        f"held-out half gives: **{caught_h}/{total_off_h}** off-corpus caught, "
        f"**{rejected_h}/{total_in_h}** in-domain wrongly rejected. With 40 claims "
        f"total this is indicative only."
    )
    add("")
    margin = mean_choice.min_in_domain - mean_choice.max_off_corpus
    if mean_choice.separated and margin < 0.02:
        add(
            f"**The separating margin is thin: {margin:.3f}.** The two "
            "distributions do not overlap, but they very nearly touch, so this "
            "gate should be expected to misclassify claims that sit between the "
            "sets sampled here. It is a working guard, not a solved problem."
        )
        add("")
    add("| Off-corpus category | Caught | Total |")
    add("|---|---|---|")
    for name in OFF_CORPUS_CLAIMS:
        group = [m for m in measurements if m.category == name]
        add(f"| {name} | {sum(1 for m in group if gated_out(m))} | {len(group)} |")
    add("")
    if in_rejected:
        add("**In-domain claims wrongly rejected — these are the real cost:**")
        add("")
        for m in in_rejected:
            add(
                f"- *{m.claim}* — max_cosine {m.max_cosine:.3f}, "
                f"mean_topk {m.mean_topk_cosine:.3f}"
            )
        add("")
    missed = [m for m in off_corpus if not gated_out(m)]
    if missed:
        add("**Off-corpus claims that still get through:**")
        add("")
        for m in missed:
            add(
                f"- [{m.category}] *{m.claim}* — max_cosine {m.max_cosine:.3f}, "
                f"mean_topk {m.mean_topk_cosine:.3f}"
            )
        add("")

    add("## Per-claim measurements")
    add("")
    add("| Category | Claim | max_cosine | mean_topk_cosine | top_bm25 | Gated out |")
    add("|---|---|---|---|---|---|")
    for m in measurements:
        add(
            f"| {m.category} | {m.claim} | {m.max_cosine:.3f} | "
            f"{m.mean_topk_cosine:.3f} | {m.top_bm25:.2f} | "
            f"{'yes' if gated_out(m) else 'no'} |"
        )
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "eval" / "results" / "similarity_calibration.md",
        help="Where to write the calibration report.",
    )
    args = parser.parse_args()

    print("Loading retrieval index...")
    index = RetrievalIndex()
    print(f"  {len(index.passages)} passages\n")

    measurements: list[Measurement] = []
    print(f"Scoring {len(IN_DOMAIN_CLAIMS)} in-domain claims...")
    for claim in IN_DOMAIN_CLAIMS:
        measurements.append(measure(index, claim, "in_domain"))
    for category, claims in OFF_CORPUS_CLAIMS.items():
        print(f"Scoring {len(claims)} {category} claims...")
        for claim in claims:
            measurements.append(measure(index, claim, category))

    in_domain = [m for m in measurements if m.category == "in_domain"]
    off_corpus = [m for m in measurements if m.category != "in_domain"]

    max_choice = choose_threshold(
        "max_cosine",
        [m.max_cosine for m in in_domain],
        [m.max_cosine for m in off_corpus],
    )
    mean_choice = choose_threshold(
        "mean_topk_cosine",
        [m.mean_topk_cosine for m in in_domain],
        [m.mean_topk_cosine for m in off_corpus],
    )

    report = build_report(measurements, max_choice, mean_choice)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    print("\n" + "=" * 70)
    for choice in (max_choice, mean_choice):
        state = "separated" if choice.separated else "OVERLAP — no clean split"
        print(
            f"{choice.signal:>18}: threshold {choice.threshold:.4f}  ({state}); "
            f"in-domain rejected {choice.in_domain_rejected}/{len(in_domain)}, "
            f"off-corpus caught {choice.off_corpus_caught}/{choice.off_corpus_total}"
        )
    print("=" * 70)
    print("\nSet these in config.py:")
    print(f"    MIN_MAX_COSINE: float = {max_choice.threshold:.3f}")
    print(f"    MIN_MEAN_TOPK_COSINE: float = {mean_choice.threshold:.3f}")
    print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
