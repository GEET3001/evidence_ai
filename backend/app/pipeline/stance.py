"""Zero-shot NLI-based stance classification of a passage against a claim.

Passage text = premise, claim = hypothesis. Uses STANCE_MODEL_PATH if it
exists locally (a future fine-tuned checkpoint, produced in Colab per the
README) — else NLI_BASELINE_MODEL from the HF hub — so a fine-tuned drop-in
needs no code changes here. Labels are mapped by the model's own id2label
NAMES, not hardcoded index order, since a future checkpoint isn't guaranteed
to preserve bart-large-mnli's label order or even its
entailment/neutral/contradiction vocabulary.

rationale_sentences ("the sentence(s) the classifier keyed on" per
EvidenceItem's docstring) come from re-running NLI per sentence within the
passage and picking whichever sentence scores highest for the passage's
winning label — this is what actually drove the stance call, not just a
topically-similar sentence (which is what an embedding-similarity approach
would give you). Faithful per README: no LLM writes this text, it's a direct
readout of a computed classifier score.
"""

from __future__ import annotations

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
    def __init__(self) -> None:
        model_path = Path(settings.STANCE_MODEL_PATH)
        source = str(model_path) if model_path.exists() else settings.NLI_BASELINE_MODEL

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
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
