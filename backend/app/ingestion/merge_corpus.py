"""Merge all collection sources into the final data/corpus.json.

Combines:
  - data/corpus.json          (PMC + PsyArXiv)
  - data/pubmed_papers.json   (stratified PubMed + contested-topic retrieval)
  - data/manual_papers.json   (optional, if present)

Deduplication is transitive (union-find) on exact DOI, then exact PMID, then
fuzzy title match >= FUZZY_TITLE_MATCH_THRESHOLD.

Duplicates are merged rather than dropped: the most complete record becomes the
base, gaps are backfilled from the others, mesh_terms and all_source_databases
are unioned, and is_preprint is False if any copy is peer reviewed.

Usage:
    python -m app.ingestion.merge_corpus
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from app.config import settings

FUZZY_TITLE_MATCH_THRESHOLD = 0.90

INPUT_FILES = [
    settings.corpus_path,
    settings.data_dir / "pubmed_papers.json",
    settings.data_dir / "manual_papers.json",
]


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _doi_key(paper: dict) -> str | None:
    doi = paper.get("doi")
    return doi.strip().lower().removeprefix("https://doi.org/") if doi else None


def _pmid_key(paper: dict) -> str | None:
    pmid = paper.get("pmid")
    return pmid.strip() if pmid else None


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _completeness_score(paper: dict) -> float:
    """Higher = more complete metadata. Used to pick the base record on merge."""
    score = 0.0
    score += len(paper.get("abstract") or "") / 100  # long abstracts weighted lightly
    score += len(paper.get("authors") or []) * 2
    score += 5 if paper.get("doi") else 0
    score += 5 if paper.get("pmid") else 0
    score += 3 if paper.get("journal") else 0
    score += 3 if paper.get("publication_tier") else 0
    score += 3 if paper.get("publication_type") else 0
    score += len(paper.get("mesh_terms") or []) * 0.5
    score += 5 if paper.get("openalex_checked") else 0
    return score


def _merge_group(papers: list[dict]) -> dict:
    if len(papers) == 1:
        base = dict(papers[0])
    else:
        base = dict(max(papers, key=_completeness_score))
        for other in papers:
            if other is base:
                continue
            for field_name in (
                "doi", "pmid", "journal", "publication_type", "publication_tier",
                "contested_topic", "source_url",
            ):
                if not base.get(field_name) and other.get(field_name):
                    base[field_name] = other[field_name]
            if len(other.get("abstract") or "") > len(base.get("abstract") or ""):
                base["abstract"] = other["abstract"]
            if len(other.get("authors") or []) > len(base.get("authors") or []):
                base["authors"] = other["authors"]
            base["mesh_terms"] = sorted(
                set(base.get("mesh_terms") or []) | set(other.get("mesh_terms") or [])
            )
            if other.get("openalex_checked") and not base.get("openalex_checked"):
                for field_name in (
                    "openalex_checked", "is_retracted", "cited_by_count", "is_open_access",
                    "open_access_url", "concepts", "referenced_works_count",
                    "author_institutions",
                ):
                    base[field_name] = other[field_name]
            # A paper found via ANY peer-reviewed source is peer-reviewed, even if
            # another copy came from a preprint server.
            if not other.get("is_preprint", True):
                base["is_preprint"] = False
            if other.get("retrieved_at") and (
                not base.get("retrieved_at") or other["retrieved_at"] < base["retrieved_at"]
            ):
                base["retrieved_at"] = other["retrieved_at"]

    # Fold in each member's OWN prior all_source_databases (not just its singular
    # source_database) so re-running the merge on an already-merged corpus doesn't
    # discard provenance recorded by an earlier merge pass.
    all_sources: set[str] = set()
    for p in papers:
        all_sources.update(p.get("all_source_databases") or [])
        if p.get("source_database"):
            all_sources.add(p["source_database"])
    base["all_source_databases"] = sorted(all_sources)
    base.setdefault("openalex_checked", False)
    base.setdefault("is_retracted", False)
    return base


@dataclass
class MergeReport:
    input_counts: dict[str, int] = field(default_factory=dict)
    total_raw: int = 0
    total_merged: int = 0
    groups_with_multiple_sources: int = 0
    doi_merges: int = 0
    pmid_merges: int = 0
    title_merges: list[tuple[str, str, float]] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=dict)


def load_sources() -> tuple[list[dict], MergeReport]:
    report = MergeReport()
    all_papers: list[dict] = []
    for path in INPUT_FILES:
        if not path.exists():
            print(f"  (skipping {path.name} — not found)")
            continue
        with open(path, encoding="utf-8") as f:
            papers = json.load(f)
        report.input_counts[path.name] = len(papers)
        all_papers.extend(papers)
    report.total_raw = len(all_papers)
    return all_papers, report


def merge(all_papers: list[dict], report: MergeReport) -> list[dict]:
    n = len(all_papers)
    uf = UnionFind(n)

    by_doi: dict[str, list[int]] = {}
    by_pmid: dict[str, list[int]] = {}
    for i, p in enumerate(all_papers):
        doi_key = _doi_key(p)
        if doi_key:
            by_doi.setdefault(doi_key, []).append(i)
        pmid_key = _pmid_key(p)
        if pmid_key:
            by_pmid.setdefault(pmid_key, []).append(i)

    for indices in by_doi.values():
        for i in indices[1:]:
            if uf.find(i) != uf.find(indices[0]):
                report.doi_merges += 1
            uf.union(indices[0], i)
    for indices in by_pmid.values():
        for i in indices[1:]:
            if uf.find(i) != uf.find(indices[0]):
                report.pmid_merges += 1
            uf.union(indices[0], i)

    normalized_titles = [_normalize_title(p.get("title", "")) for p in all_papers]
    for i in range(n):
        if not normalized_titles[i]:
            continue
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j) or not normalized_titles[j]:
                continue
            ratio = SequenceMatcher(None, normalized_titles[i], normalized_titles[j]).ratio()
            if ratio >= FUZZY_TITLE_MATCH_THRESHOLD:
                report.title_merges.append(
                    (all_papers[i].get("paper_id", "?"), all_papers[j].get("paper_id", "?"), ratio)
                )
                uf.union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, p in enumerate(all_papers):
        groups.setdefault(uf.find(i), []).append(p)

    merged_papers = []
    for group in groups.values():
        if len(group) > 1:
            report.groups_with_multiple_sources += 1
        merged_papers.append(_merge_group(group))

    report.total_merged = len(merged_papers)
    for p in merged_papers:
        for src in p.get("all_source_databases", []):
            report.source_counts[src] = report.source_counts.get(src, 0) + 1

    return merged_papers


def print_report(report: MergeReport) -> None:
    print("\n" + "=" * 70)
    print("Corpus merge report")
    print("=" * 70)
    print("Inputs:")
    for name, count in report.input_counts.items():
        print(f"  {name:<25} {count}")
    print(f"\nTotal raw records:        {report.total_raw}")
    print(f"Merged into:              {report.total_merged} unique papers")
    print(f"  via DOI match:          {report.doi_merges}")
    print(f"  via PMID match:         {report.pmid_merges}")
    print(f"  via fuzzy title match:  {len(report.title_merges)}")
    if report.title_merges:
        print("  fuzzy title merges (review these — the riskiest match type):")
        for a, b, ratio in report.title_merges:
            print(f"    {ratio:.0%}  {a}  <->  {b}")
    print(f"Papers found in >1 source: {report.groups_with_multiple_sources}")
    print("\nFinal per-source-database counts (a merged paper counts toward every "
          "source it was found in):")
    for src, count in sorted(report.source_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<12} {count}")
    print("=" * 70)


def main() -> None:
    print("Loading sources...")
    all_papers, report = load_sources()
    print(f"Loaded {report.total_raw} raw records.")

    merged_papers = merge(all_papers, report)

    with open(settings.corpus_path, "w", encoding="utf-8") as f:
        json.dump(merged_papers, f, indent=2, ensure_ascii=False)

    print_report(report)
    print(f"\nWrote {len(merged_papers)} merged papers to {settings.corpus_path}")


if __name__ == "__main__":
    main()
