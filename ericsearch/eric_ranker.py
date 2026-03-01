from dataclasses  import dataclass, field
import re

from sentence_transformers import CrossEncoder

from ericsearch.utils import split_sentences,  es_get_device, es_get_logger
from ericsearch.utils import EricDocument, RankerResult


@dataclass
class RankerCallArgs:
    bs: int = 32
    limit: int = 1


# Internal dataclasses.
@dataclass
class Sentence:
    text: str
    doc_idx: int
    paragraph_idx: int
    score: float = 0


@dataclass
class Paragraph:
    text: str
    doc_idx: int
    paragraph_idx: int
    doc_score: float
    paragraph_crossencoder_score: float = 0
    sentences: list[Sentence] = field(default_factory=list)
    best_sentence: Sentence | None = None
    aggregated_score: float = 0


class EricRanker:
    ignore_original_score: bool

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.ignore_original_score = False

        self.device = es_get_device()

        self.model = CrossEncoder(model_name, device=str(self.device))

        self.logger = es_get_logger()

    @staticmethod
    def get_blocks(
        eric_searchset_result: list[EricDocument],
    ) -> tuple[list[Paragraph], list[Sentence]]:
        # Each input doc gets one block. So output blocks align to input docs.

        kept_paragraphs = []
        kept_sentences = []

        overall_p_id = 0

        for doc_idx, doc_search_result in enumerate(eric_searchset_result):
            text = doc_search_result.text
            doc_score = doc_search_result.score

            text = text.replace("\r\n", "\n").replace("\r", "\n")
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            for paragraph_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.split()) > 5:
                    kept_paragraphs.append(
                        Paragraph(
                            text=paragraph,
                            doc_idx=doc_idx,
                            paragraph_idx=paragraph_idx,
                            doc_score=doc_score,
                        )
                    )

                    sentences = split_sentences(paragraph)
                    for sentence in sentences:
                        if len(sentence.split()) > 5:
                            kept_sentences.append(
                                Sentence(
                                    text=sentence,
                                    doc_idx=doc_idx,
                                    paragraph_idx=paragraph_idx,
                                )
                            )
                            kept_paragraphs[-1].sentences.append(kept_sentences[-1])
                overall_p_id += 1

        return kept_paragraphs, kept_sentences

    def __call__(
        self,
        text: str,
        docs: list[EricDocument],
        args: RankerCallArgs = RankerCallArgs(),
    ) -> list[RankerResult]:
        if not docs:
            return []

        paragraphs, sentences = self.get_blocks(docs)

        self.logger.info("EricRanker(): scoring paragraphs...")
        paragraph_scores = self.model.predict(
            [(text, p.text) for p in paragraphs], batch_size=args.bs
        )
        for paragraph, score in zip(paragraphs, paragraph_scores):
            paragraph.paragraph_crossencoder_score = score

        self.logger.info("EricRanker(): scoring sentences...")
        sent_scores = self.model.predict(
            [(text, s.text) for s in sentences], batch_size=args.bs
        )
        for sentence, score in zip(sentences, sent_scores):
            sentence.score = score

        # Find best sentence for each paragraph
        # and compute each paragraph's aggregated score.
        i = 0
        for paragraph in paragraphs:
            i += 1
            if not paragraph.sentences:
                paragraph.best_sentence = None
                paragraph.aggregated_score = (
                    (0.6 * paragraph.doc_score) * (not self.ignore_original_score)
                    + 0.3 * paragraph.paragraph_crossencoder_score
                    # + 0.1 * best_sentence.score   # no sentence to use
                )
                continue

            paragraph.best_sentence = max(paragraph.sentences, key=lambda s: s.score)
            paragraph.aggregated_score = (
                (0.6 * paragraph.doc_score) * (not self.ignore_original_score)
                + 0.3 * paragraph.paragraph_crossencoder_score
                + 0.1 * paragraph.best_sentence.score
            )
        # Rank the paragraphs and return results.
        paragraphs = sorted(paragraphs, key=lambda p: p.aggregated_score, reverse=True)
        return [
            RankerResult(
                text=paragraph.text,
                score=paragraph.aggregated_score.item(),
                best_sentence=(
                    paragraph.best_sentence.text if paragraph.best_sentence else ""
                ),
                metadata=docs[paragraph.doc_idx].metadata,
            )
            for paragraph in paragraphs[: args.limit]
        ]
