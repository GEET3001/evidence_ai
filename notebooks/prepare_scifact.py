"""Prepare SciFact as training data for stance classification fine-tuning.

Local-only data prep (no GPU) — run before any Colab fine-tuning step, from
the backend venv which already has `datasets`/`huggingface_hub`:

    backend\\venv\\Scripts\\python.exe notebooks\\prepare_scifact.py

WHY THIS EXISTS: stance.py currently uses zero-shot NLI (bart-large-mnli),
which measurably under-detects CONTRADICT on the real corpus (see the
pipeline verification writeup) — general-domain MNLI models aren't
calibrated for hedged/statistical scientific skepticism language. stance.py
is already label-name-mapped (SUPPORT/CONTRADICT/NEUTRAL by name, not index),
so a fine-tuned checkpoint at STANCE_MODEL_PATH drops in with no code changes.

VERIFIED LIVE, NOT ASSUMED (both findings below only became apparent by
actually loading the data — see the printed schema dump each run):

1. `allenai/scifact` uses a legacy dataset LOADING SCRIPT, which `datasets`
   5.x has fully removed support for ("Dataset scripts are no longer
   supported"). It has no `--trust-remote-code`-style escape hatch; the fix
   is to pull the auto-converted parquet files directly from the
   `refs/convert/parquet` branch via `huggingface_hub.hf_hub_download`,
   bypassing the removed script/config-name mechanism entirely (that branch
   collapses named configs to a single 'default' builder, which is why we
   fetch exact file paths rather than using `load_dataset(..., "claims")`).

2. The claims parquet is row-per-(claim, cited_doc), not row-per-claim:
   claim ids repeat (a claim can have separate SUPPORT evidence from one
   paper and separate CONTRADICT evidence from another, or simply cite a
   paper it wasn't annotated against). `evidence_sentences` holds sentence
   INDICES into that doc's `abstract` array (already sentence-split), not
   text — the passage is reconstructed by indexing the corpus, not present
   inline. `evidence_label` is one of '', 'SUPPORT', 'CONTRADICT' (verified
   live — not 'SUPPORTS'/'REFUTES', though the mapping below tolerates both
   in case a future revision changes this). The official `test` split has
   ZERO evidence annotations of any kind (not even `cited_doc_ids`) — it's
   the held-out leaderboard set with hidden labels, not usable for building
   triples. `validation` is the closest thing to a labeled held-out set.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "allenai/scifact"
REVISION = "refs/convert/parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"
SPLITS = ["train", "validation", "test"]

# Tolerate both label vocabularies even though live inspection confirmed
# SUPPORT/CONTRADICT (not SUPPORTS/REFUTES) in the current parquet revision.
_LABEL_MAP = {
    "SUPPORT": "SUPPORT",
    "SUPPORTS": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "REFUTES": "CONTRADICT",
}

CONTRADICT_WARNING_THRESHOLD = 0.15


def _download_parquet(path_in_repo: str) -> str:
    return hf_hub_download(
        repo_id=REPO_ID, filename=path_in_repo, repo_type="dataset", revision=REVISION
    )


def load_corpus() -> dict[int, dict]:
    path = _download_parquet("corpus/train/0000.parquet")
    corpus = load_dataset("parquet", data_files=path, split="train")
    return {row["doc_id"]: row for row in corpus}


def load_claims_split(split: str):
    path = _download_parquet(f"claims/{split}/0000.parquet")
    return load_dataset("parquet", data_files=path, split="train")


def print_schema(claims_by_split: dict, corpus: dict[int, dict]) -> None:
    print("=" * 70)
    print("SciFact schema (verified live, per split)")
    print("=" * 70)
    any_split = next(iter(claims_by_split.values()))
    print("claims features:", any_split.features)
    print("claims sample row:", dict(any_split[0]))
    print()
    first_doc = next(iter(corpus.values()))
    print("corpus features:", {k: type(v).__name__ for k, v in first_doc.items()})
    print("corpus sample row (abstract truncated):", {
        **first_doc,
        "abstract": list(first_doc["abstract"][:2]) + ["..."],
    })
    print()
    for split, ds in claims_by_split.items():
        n_labeled = sum(1 for r in ds if r["evidence_label"])
        n_noinfo = sum(1 for r in ds if not r["evidence_label"] and len(r["cited_doc_ids"]) > 0)
        n_empty = len(ds) - n_labeled - n_noinfo
        print(
            f"  {split}: {len(ds)} rows, {len(set(ds['id']))} unique claim ids, "
            f"labeled={n_labeled} noinfo(cited,no-label)={n_noinfo} no-citation={n_empty}"
        )
    print("=" * 70)


def build_triples(claims, corpus: dict[int, dict], split: str) -> tuple[list[dict], int]:
    """Row-wise: SUPPORT/CONTRADICT rows use the annotated rationale sentences
    as the passage; NOINFO rows (empty evidence_label but real citations) get
    one NEUTRAL triple per cited doc, using that doc's full abstract — there's
    no sentence-level annotation to key on for an unevidenced citation."""
    triples = []
    skipped_missing_doc = 0

    for row in claims:
        claim_id = row["id"]
        claim_text = row["claim"]
        raw_label = row["evidence_label"]

        if raw_label:
            label = _LABEL_MAP.get(raw_label)
            if label is None:
                continue  # unrecognized label — skip rather than guess
            doc = corpus.get(int(row["evidence_doc_id"]))
            if doc is None:
                skipped_missing_doc += 1
                continue
            indices = sorted(int(i) for i in row["evidence_sentences"])
            passage = " ".join(doc["abstract"][i] for i in indices if 0 <= i < len(doc["abstract"]))
            if not passage:
                continue
            triples.append(
                {
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "passage": passage,
                    "label": label,
                    "doc_id": int(row["evidence_doc_id"]),
                    "source": "scifact",
                    "split": split,
                }
            )
        else:
            for raw_doc_id in row["cited_doc_ids"]:
                doc = corpus.get(int(raw_doc_id))
                if doc is None:
                    skipped_missing_doc += 1
                    continue
                passage = " ".join(doc["abstract"])
                if not passage:
                    continue
                triples.append(
                    {
                        "claim_id": claim_id,
                        "claim": claim_text,
                        "passage": passage,
                        "label": "NEUTRAL",
                        "doc_id": int(raw_doc_id),
                        "source": "scifact",
                        "split": split,
                    }
                )

    return triples, skipped_missing_doc


def report_health_datasets() -> None:
    """Investigated live against the HF Hub (see this module's docstring for
    the SciFact-side version of the same class of problem). Neither dataset
    ends up contributing usable CONTRADICT triples, for two different reasons
    — reported here rather than silently skipped."""
    print("\n" + "=" * 70)
    print("--include-health investigation")
    print("=" * 70)
    print(
        "HealthVer (dwadden/healthver_entailment): BLOCKED. Script-based dataset "
        "with no `refs/convert/parquet` fallback available on the HF Hub (verified "
        "via `list_repo_refs` — `converts` is empty) — unlike allenai/scifact, "
        "there is no alternate route to load it under datasets>=5.0. Not included."
    )
    print(
        "PUBHEALTH (ImperialCollegeLondon/health_fact, also bigbio/pubhealth): "
        "LOADABLE but DECLINED — different label scheme, not forced. Its labels "
        "are true/false/mixture/unproven, a claim-VERACITY judgment made against "
        "a full fact-check article (politics/news domain, checked live: PolitiFact-"
        "style pieces, not scientific abstracts), not a claim-vs-PASSAGE stance "
        "label like SciFact's SUPPORT/CONTRADICT/NOINFO. Even bigbio's "
        "'pubhealth_bigbio_pairs' reformatting keeps the same veracity labels over "
        "the same full-article text — it standardizes the *shape*, not the "
        "*meaning*. 'mixture' in particular has no defensible 1:1 mapping to "
        "SUPPORT/CONTRADICT/NEUTRAL (it means 'partially true and partially "
        "false', not 'unrelated to the claim'). Forcing this mapping would inject "
        "mislabeled examples and a severe domain/length mismatch (full articles "
        "vs. paper-abstract sentences) into fine-tuning data. Not included."
    )
    print("=" * 70)


def report_distribution(all_triples: list[dict]) -> None:
    counts = Counter(t["label"] for t in all_triples)
    total = sum(counts.values())
    print("\n" + "=" * 70)
    print("Class distribution (train + validation combined)")
    print("=" * 70)
    for label in ("SUPPORT", "CONTRADICT", "NEUTRAL"):
        n = counts.get(label, 0)
        pct = n / total if total else 0
        print(f"  {label:<12} {n:>5}  ({pct:.1%})")
    print(f"  {'TOTAL':<12} {total:>5}")

    contradict_share = counts.get("CONTRADICT", 0) / total if total else 0
    if contradict_share < CONTRADICT_WARNING_THRESHOLD:
        print(
            f"\nCONTRADICT is {contradict_share:.1%} of examples, below the "
            f"{CONTRADICT_WARNING_THRESHOLD:.0%} flag threshold — this is the "
            f"known SciFact imbalance (CONTRADICT is consistently the rarest "
            f"class in the source annotations, not an artifact of this pairing "
            f"logic). Recommendation: use class-weighted loss (e.g. weight "
            f"CONTRADICT's cross-entropy term by roughly NEUTRAL_count/"
            f"CONTRADICT_count) rather than oversampling. Class weighting costs "
            f"nothing extra at train time and doesn't duplicate the same small "
            f"set of CONTRADICT passages, which oversampling would — repeating "
            f"the same ~100-200 CONTRADICT examples many times over risks the "
            f"model memorizing their surface phrasing rather than learning the "
            f"general contradiction signal, and inflates apparent epoch-over-"
            f"epoch progress without proportionate generalization. Class "
            f"weighting is the safer default; consider oversampling only as a "
            f"secondary experiment if weighting alone underperforms."
        )
    print("=" * 70)


def print_examples(all_triples: list[dict], rng: random.Random) -> None:
    print("\n" + "=" * 70)
    print("Example rows per class")
    print("=" * 70)
    by_label: dict[str, list[dict]] = {}
    for t in all_triples:
        by_label.setdefault(t["label"], []).append(t)
    for label in ("SUPPORT", "CONTRADICT", "NEUTRAL"):
        examples = by_label.get(label, [])
        print(f"\n--- {label} ({len(examples)} available) ---")
        for t in rng.sample(examples, min(3, len(examples))):
            print(f"  claim:   {t['claim']}")
            print(f"  passage: {t['passage'][:200]}")
            print()


def write_jsonl(triples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SciFact triples for stance fine-tuning.")
    parser.add_argument(
        "--include-health",
        action="store_true",
        help="Investigate HealthVer/PUBHEALTH as additional CONTRADICT sources "
        "and report why each is or isn't included (see report_health_datasets).",
    )
    parser.add_argument("--seed", type=int, default=42, help="For the example-row sample only.")
    args = parser.parse_args()

    print("Downloading SciFact (auto-converted parquet, refs/convert/parquet)...")
    corpus = load_corpus()
    claims_by_split = {split: load_claims_split(split) for split in SPLITS}
    print_schema(claims_by_split, corpus)

    all_triples: list[dict] = []
    for split in SPLITS:
        triples, skipped = build_triples(claims_by_split[split], corpus, split)
        write_jsonl(triples, OUTPUT_DIR / f"{split}.jsonl")
        print(f"\n{split}: wrote {len(triples)} triples to {OUTPUT_DIR / f'{split}.jsonl'}")
        if skipped:
            print(f"  ({skipped} rows skipped — cited doc not found in corpus)")
        if split == "test" and not triples:
            print(
                "  test.jsonl is EMPTY and expected to be: the official SciFact "
                "test split carries no evidence annotations at all (verified "
                "live — 0/300 rows have a label or even a citation; it's the "
                "held-out leaderboard set with hidden gold labels). Use "
                "validation.jsonl as the held-out eval set for fine-tuning "
                "instead of test.jsonl."
            )
        all_triples.extend(t for t in triples if t["split"] != "test")

    report_distribution(all_triples)
    print_examples(all_triples, random.Random(args.seed))

    if args.include_health:
        report_health_datasets()


if __name__ == "__main__":
    main()
