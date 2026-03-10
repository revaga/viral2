"""
Viral title evaluation metrics.

Ensures every model version (e.g. Gemma-2b vs Llama 3.2 3B, different LoRA runs)
is compared evenly. English-only; assumes references and predictions are already
from your viral pipeline (quantile-based viral definition lives upstream).

Research interpretation:
- ROUGE-L F1: structural match / hook structure
- Perplexity (base vs tuned) on refs: style transfer (tuned finds viral refs less surprising)
- Max cosine: style proximity to real viral examples
- Mean abs compound: emotional volume (high = loud, near zero = boring)
- Props (pos/neg): sentiment bias (rage-bait vs positivity)
- cosine_max_sim_prop_above_95: mode collapse / memorization risk
- Diversity (TTR / Self-BLEU): variety vs one-trick
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np  # type: ignore[import-untyped]

_DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
_MODE_COLLAPSE_THRESHOLD = 0.95


def _safe_mean_std(arr: np.ndarray | list[float]) -> tuple[float, float]:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a)) if len(a) > 1 else 0.0


def _filter_pairs(references: list[str], predictions: list[str]) -> tuple[list[str], list[str]]:
    refs, preds = [], []
    for r, p in zip(references, predictions):
        r, p = (s.strip() if s else "" for s in (r, p))
        if r and p:
            refs.append(r)
            preds.append(p)
    return refs, preds


def _compute_rouge(references: list[str], predictions: list[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer  # type: ignore[import-untyped]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    out: dict[str, float] = {}
    for name in ["rouge1", "rouge2", "rougeL"]:
        precs, recs, f1s = [], [], []
        for ref, pred in zip(references, predictions):
            s = scorer.score(ref, pred)[name]
            precs.append(s.precision)
            recs.append(s.recall)
            f1s.append(s.fmeasure)
        for key, vals in [("p", precs), ("r", recs), ("f1", f1s)]:
            m, s = _safe_mean_std(vals)
            out[f"{name}_{key}_mean"] = m
            out[f"{name}_{key}_std"] = s
    return out


def _compute_cosine(predictions: list[str], corpus_titles: list[str], embedding_model: Any) -> dict[str, float]:
    if not corpus_titles or not predictions:
        nan = float("nan")
        return {
            "cosine_mean_corpus_mean": nan,
            "cosine_mean_corpus_std": nan,
            "cosine_max_sim_mean": nan,
            "cosine_max_sim_std": nan,
            "cosine_max_sim_prop_above_95": nan,
        }

    pred_emb = embedding_model.encode(predictions, convert_to_numpy=True)
    corpus_emb = embedding_model.encode(corpus_titles, convert_to_numpy=True)
    pred_emb = pred_emb / np.linalg.norm(pred_emb, axis=1, keepdims=True)
    corpus_emb = corpus_emb / np.linalg.norm(corpus_emb, axis=1, keepdims=True)

    mean_corpus = np.mean(corpus_emb, axis=0)
    mean_corpus = mean_corpus / np.linalg.norm(mean_corpus)
    mean_to_corpus = np.dot(pred_emb, mean_corpus)

    sim_matrix = np.dot(pred_emb, corpus_emb.T)
    max_sim = np.max(sim_matrix, axis=1)
    prop_above_95 = float(np.mean(max_sim >= _MODE_COLLAPSE_THRESHOLD))

    m1, s1 = _safe_mean_std(mean_to_corpus.tolist())
    m2, s2 = _safe_mean_std(max_sim.tolist())
    return {
        "cosine_mean_corpus_mean": m1,
        "cosine_mean_corpus_std": s1,
        "cosine_max_sim_mean": m2,
        "cosine_max_sim_std": s2,
        "cosine_max_sim_prop_above_95": prop_above_95,
    }


def _perplexity_for_texts(
    texts: list[str], model: Any, tokenizer: Any, device: Any, max_length: int = 128
) -> list[float]:
    import torch  # type: ignore[import-untyped]

    ppls: list[float] = []
    for text in texts:
        if not text.strip():
            ppls.append(float("nan"))
            continue
        inputs = tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        with torch.no_grad():
            out = model(**inputs)
        loss = out.loss
        if loss is not None and math.isfinite(loss.item()):
            ppls.append(math.exp(loss.item()))
        else:
            ppls.append(float("nan"))
    return ppls


def _compute_perplexity(
    references: list[str],
    predictions: list[str],
    base_model: Any | None,
    base_tokenizer: Any | None,
    ft_model: Any | None,
    ft_tokenizer: Any | None,
    device: Any,
) -> dict[str, float]:
    nan = float("nan")
    out: dict[str, float] = {
        "perplexity_base_pred_mean": nan,
        "perplexity_base_pred_std": nan,
        "perplexity_base_ref_mean": nan,
        "perplexity_base_ref_std": nan,
        "perplexity_self_pred_mean": nan,
        "perplexity_self_pred_std": nan,
        "perplexity_self_ref_mean": nan,
        "perplexity_self_ref_std": nan,
    }
    if base_model is not None and base_tokenizer is not None:
        ppl_pred = _perplexity_for_texts(predictions, base_model, base_tokenizer, device)
        ppl_ref = _perplexity_for_texts(references, base_model, base_tokenizer, device)
        m1, s1 = _safe_mean_std(ppl_pred)
        m2, s2 = _safe_mean_std(ppl_ref)
        out["perplexity_base_pred_mean"] = m1
        out["perplexity_base_pred_std"] = s1
        out["perplexity_base_ref_mean"] = m2
        out["perplexity_base_ref_std"] = s2
    if ft_model is not None and ft_tokenizer is not None:
        ppl_pred = _perplexity_for_texts(predictions, ft_model, ft_tokenizer, device)
        ppl_ref = _perplexity_for_texts(references, ft_model, ft_tokenizer, device)
        m1, s1 = _safe_mean_std(ppl_pred)
        m2, s2 = _safe_mean_std(ppl_ref)
        out["perplexity_self_pred_mean"] = m1
        out["perplexity_self_pred_std"] = s1
        out["perplexity_self_ref_mean"] = m2
        out["perplexity_self_ref_std"] = s2
    return out


def _compute_sentiment(predictions: list[str]) -> dict[str, float]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore[import-untyped]
    except ImportError:
        from vaderSentiment import SentimentIntensityAnalyzer  # type: ignore[import-untyped]

    analyzer = SentimentIntensityAnalyzer()
    compounds: list[float] = []
    abs_compounds: list[float] = []
    pos_count = neg_count = neu_count = 0
    for p in predictions:
        if not p.strip():
            compounds.append(float("nan"))
            abs_compounds.append(float("nan"))
            continue
        d = analyzer.polarity_scores(p.strip())
        c = d["compound"]
        compounds.append(c)
        abs_compounds.append(abs(c))
        if c > 0.05:
            pos_count += 1
        elif c < -0.05:
            neg_count += 1
        else:
            neu_count += 1
    total = len(predictions) or 1
    m1, s1 = _safe_mean_std(compounds)
    m2, s2 = _safe_mean_std(abs_compounds)
    return {
        "sentiment_compound_mean": m1,
        "sentiment_compound_std": s1,
        "sentiment_compound_abs_mean": m2,
        "sentiment_compound_abs_std": s2,
        "sentiment_prop_positive": pos_count / total,
        "sentiment_prop_negative": neg_count / total,
        "sentiment_prop_neutral": neu_count / total,
    }


def _compute_ttr(predictions: list[str]) -> float:
    all_tokens: list[str] = []
    for p in predictions:
        if p and p.strip():
            all_tokens.extend(re.findall(r"\b\w+\b", p.strip().lower()))
    if not all_tokens:
        return float("nan")
    return len(set(all_tokens)) / len(all_tokens)


def _compute_self_bleu(predictions: list[str], n: int = 4) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore[import-untyped]
    except ImportError:
        return float("nan")
    preds = [p.strip() for p in predictions if p and p.strip()]
    if len(preds) < 2:
        return float("nan")
    smoothing = SmoothingFunction()
    scores: list[float] = []
    weights = tuple(1.0 / n for _ in range(n))
    for i, hyp in enumerate(preds):
        refs = [preds[j].split() for j in range(len(preds)) if j != i]
        hyp_tok = hyp.split()
        if not hyp_tok or not any(refs):
            continue
        try:
            sc = sentence_bleu(refs, hyp_tok, weights=weights, smoothing_function=smoothing.method1)
        except Exception:
            sc = 0.0
        scores.append(sc)
    return float(np.mean(scores)) if scores else float("nan")


def _compute_diversity(predictions: list[str]) -> dict[str, float]:
    return {
        "diversity_ttr": _compute_ttr(predictions),
        "diversity_self_bleu": _compute_self_bleu(predictions),
    }


class Evaluator:
    """
    Evaluator for viral title generation. Loads embedding model and (optionally)
    base model once for reuse across evaluate() calls.
    """

    def __init__(
        self,
        embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
        base_model_name: str | None = None,
        device: str | None = None,
    ):
        self.embedding_model_name = embedding_model_name
        self.base_model_name = base_model_name
        self._embedding_model = None
        self._base_model = None
        self._base_tokenizer = None
        if device is None:
            try:
                import torch  # type: ignore[import-untyped]

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self._embedding_model.to(self._device)
        return self._embedding_model

    def _get_base_model_and_tokenizer(self):
        if self._base_model is not None:
            return self._base_model, self._base_tokenizer
        if self.base_model_name is None:
            return None, None
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

        self._base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self._base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        self._base_model.to(self._device)
        self._base_model.eval()
        return self._base_model, self._base_tokenizer

    def evaluate(
        self,
        references: list[str],
        predictions: list[str],
        corpus_titles: list[str] | None = None,
        finetuned_model: Any = None,
        finetuned_tokenizer: Any = None,
    ) -> dict[str, float]:
        refs, preds = _filter_pairs(references, predictions)
        if not refs:
            return self._empty_metrics()

        if corpus_titles is None:
            corpus_titles = list(refs)
        else:
            corpus_titles = [t.strip() for t in corpus_titles if t and t.strip()]

        results: dict[str, float] = {}
        results.update(_compute_rouge(refs, preds))
        results.update(_compute_cosine(preds, corpus_titles, self.embedding_model))

        base_model, base_tok = self._get_base_model_and_tokenizer()
        results.update(
            _compute_perplexity(
                refs,
                preds,
                base_model,
                base_tok,
                finetuned_model,
                finetuned_tokenizer,
                self._device,
            )
        )
        results.update(_compute_sentiment(preds))
        results.update(_compute_diversity(preds))
        return results

    def _empty_metrics(self) -> dict[str, float]:
        nan = float("nan")
        rouge_keys: list[str] = []
        for name in ["rouge1", "rouge2", "rougeL"]:
            for m in ["p", "r", "f1"]:
                rouge_keys.extend([f"{name}_{m}_mean", f"{name}_{m}_std"])
        return dict.fromkeys(rouge_keys, nan) | {
            "cosine_mean_corpus_mean": nan,
            "cosine_mean_corpus_std": nan,
            "cosine_max_sim_mean": nan,
            "cosine_max_sim_std": nan,
            "cosine_max_sim_prop_above_95": nan,
            "perplexity_base_pred_mean": nan,
            "perplexity_base_pred_std": nan,
            "perplexity_base_ref_mean": nan,
            "perplexity_base_ref_std": nan,
            "perplexity_self_pred_mean": nan,
            "perplexity_self_pred_std": nan,
            "perplexity_self_ref_mean": nan,
            "perplexity_self_ref_std": nan,
            "sentiment_compound_mean": nan,
            "sentiment_compound_std": nan,
            "sentiment_compound_abs_mean": nan,
            "sentiment_compound_abs_std": nan,
            "sentiment_prop_positive": nan,
            "sentiment_prop_negative": nan,
            "sentiment_prop_neutral": nan,
            "diversity_ttr": nan,
            "diversity_self_bleu": nan,
        }


def compute_metrics(
    references: list[str],
    predictions: list[str],
    corpus_titles: list[str] | None = None,
    embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
    base_model_name: str | None = None,
    finetuned_model: Any = None,
    finetuned_tokenizer: Any = None,
    device: str | None = None,
    **_: Any,
) -> dict[str, float]:
    evaluator = Evaluator(
        embedding_model_name=embedding_model_name,
        base_model_name=base_model_name,
        device=device,
    )
    return evaluator.evaluate(
        references=references,
        predictions=predictions,
        corpus_titles=corpus_titles,
        finetuned_model=finetuned_model,
        finetuned_tokenizer=finetuned_tokenizer,
    )

