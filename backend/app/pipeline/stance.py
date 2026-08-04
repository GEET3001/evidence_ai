"""Stance classification of a passage against a claim.

The passage is the premise and the claim is the hypothesis. Runs either
zero-shot NLI or a fine-tuned checkpoint, selected by STANCE_MODEL_MODE.
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

# Matched by name rather than index, so a fine-tuned checkpoint can use either
# NLI's vocabulary or ours without having to preserve bart-large-mnli's ordering.
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
        """Load the stance model.

        `model_path` bypasses STANCE_MODEL_MODE, so compare_classifiers can hold
        both models open at once regardless of the configured mode.
        """
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        # An explicit dtype skips transformers' auto-detection probe, which mmaps
        # the checkpoint and can exhaust the commit limit on a 16GB machine.
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
        """Classify the passage and report the sentence that drove the call.

        The rationale sentence is picked by re-scoring each sentence alone and
        keeping the one scoring highest for the winning stance, so it reads out
        the classifier itself rather than a separate similarity heuristic.
        """
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
