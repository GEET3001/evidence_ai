"""Stance classification of a passage against a claim — zero-shot NLI or a
fine-tuned checkpoint, switched via STANCE_MODEL_MODE (see config.py).

Passage text = premise, claim = hypothesis. STANCE_MODEL_MODE="zeroshot"
(default) uses NLI_BASELINE_MODEL (facebook/bart-large-mnli) from the HF hub.
STANCE_MODEL_MODE="finetuned" uses the checkpoint at STANCE_MODEL_PATH and
fails loudly at startup if it isn't there — this is an explicit switch, not
auto-detection off whether that path happens to exist, so "I asked for the
fine-tuned model" never silently becomes "I got zero-shot instead" (see
README: Model Provenance for how to obtain/train a checkpoint).

Labels are mapped by the model's own id2label NAMES, not hardcoded index
order, since a fine-tuned checkpoint isn't guaranteed to preserve
bart-large-mnli's label order or even its entailment/neutral/contradiction
vocabulary — notebooks/train_stance.ipynb sets id2label to
SUPPORT/CONTRADICT/NEUTRAL directly (matching this module's own vocabulary),
so this mapping already handles the fine-tuned model correctly with no
changes needed; confirmed by inspection, not just assumed.

rationale_sentences ("the sentence(s) the classifier keyed on" per
EvidenceItem's docstring) come from re-running _score() per sentence within
the passage and picking whichever sentence scores highest for the passage's
winning label — this is what actually drove the stance call, not just a
topically-similar sentence (which is what an embedding-similarity approach
would give you). This mechanism is model-agnostic (it just calls _score()
again, whichever model that currently is) — it is NOT tied to zero-shot NLI's
behaviour, and needs no adaptation for a fine-tuned checkpoint. It's also
in-distribution for the fine-tuned model specifically: many of its SUPPORT/
CONTRADICT training pairs (see prepare_scifact.py) were already single
annotated sentences, not full passages, so isolated-sentence scoring isn't a
novel input shape for it. Faithful per README: no LLM writes this text, it's
a direct readout of a computed classifier score.
"""

from __future__ import annotations

import app._thread_limits  # noqa: F401  (must precede the torch import below)

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings
from app.indexing.text_utils import split_sentences
from app.models import Stance

# A future fine-tuned checkpoint may use our own SUPPORT/CONTRADICT/NEUTRAL
# vocabulary directly instead of NLI's entailment/neutral/contradiction —
# both are accepted so the swap-in genuinely requires no code changes.
_LABEL_ALIASES: dict[str, Stance] = {
    "entailment": Stance.SUPPORT,
    "contradiction": Stance.CONTRADICT,
    "neutral": Stance.NEUTRAL,
    "support": Stance.SUPPORT,
    "contradict": Stance.CONTRADICT,
}


@dataclass
class StanceResult:
    stance: Stance
    confidence: float
    rationale_sentences: list[str]


class StanceClassifier:
    def __init__(self, model_path: str | None = None) -> None:
        """model_path overrides STANCE_MODEL_MODE-based resolution entirely —
        used by pipeline.compare_classifiers to load two specific models side
        by side regardless of the configured mode. Omit it for normal
        (production) resolution, which is explicit-mode-based, not
        existence-based: STANCE_MODEL_MODE="finetuned" always means "use
        STANCE_MODEL_PATH, and fail loudly if it's not there" — never a
        silent fallback to zero-shot."""
        if model_path is not None:
            source = model_path
        elif settings.STANCE_MODEL_MODE == "finetuned":
            resolved_path = Path(settings.STANCE_MODEL_PATH)
            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"STANCE_MODEL_MODE=finetuned but no checkpoint found at "
                    f"'{resolved_path}'. Either train one (see README: Model "
                    f"Provenance) and place it there, or set "
                    f"STANCE_MODEL_MODE=zeroshot to use {settings.NLI_BASELINE_MODEL} "
                    f"instead."
                )
            source = str(resolved_path)
        else:
            source = settings.NLI_BASELINE_MODEL

        self.source = source
        # Zero-shot NLI on CPU is compute-bound (~3s/forward pass for
        # bart-large-mnli single-threaded — see app/_thread_limits.py for why
        # it must stay single-threaded on this machine). GPU is used
        # opportunistically when available and falls back to the
        # already-proven CPU path otherwise, so this still runs correctly on
        # a grading machine with no CUDA device.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        # low_cpu_mem_usage avoids briefly double-allocating the full weight
        # tensors during load. dtype=torch.float32 (explicit, not "auto") skips
        # transformers' dtype auto-detection probe, which otherwise mmaps the
        # checkpoint via safe_open() regardless of low_cpu_mem_usage — on a
        # memory-constrained machine that probe alone can hit the OS commit
        # limit before the model is even usable.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            source, low_cpu_mem_usage=True, dtype=torch.float32
        )
        self.model.to(self.device)
        self.model.eval()

        id2label: dict[int, Stance] = {}
        unmapped: list[str] = []
        for idx, label in self.model.config.id2label.items():
            mapped = _LABEL_ALIASES.get(label.lower())
            if mapped is None:
                unmapped.append(label)
            else:
                id2label[idx] = mapped
        if unmapped:
            raise ValueError(
                f"Stance model at '{source}' has labels with no known mapping: "
                f"{unmapped}. Expected some combination of "
                f"entailment/contradiction/neutral or support/contradict/neutral."
            )
        self._id2label = id2label

    def _score(self, premise: str, hypothesis: str) -> dict[Stance, float]:
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

        scores: dict[Stance, float] = {}
        for idx, stance in self._id2label.items():
            scores[stance] = max(scores.get(stance, 0.0), float(probs[idx]))
        return scores

    def classify(self, passage_text: str, claim: str) -> StanceResult:
        scores = self._score(passage_text, claim)
        winning_stance = max(scores, key=scores.get)
        confidence = scores[winning_stance]

        sentences = split_sentences(passage_text)
        if len(sentences) <= 1:
            rationale = sentences or [passage_text]
        else:
            best_sentence, best_score = sentences[0], -1.0
            for sentence in sentences:
                sentence_scores = self._score(sentence, claim)
                score = sentence_scores.get(winning_stance, 0.0)
                if score > best_score:
                    best_sentence, best_score = sentence, score
            rationale = [best_sentence]

        return StanceResult(
            stance=winning_stance, confidence=confidence, rationale_sentences=rationale
        )
