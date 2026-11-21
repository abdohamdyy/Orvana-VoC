"""Classifier backends. Each exposes .name and .classify(segments) -> [(Category, confidence)]."""


def get_backend(kind: str, **kw):
    if kind == "llm":
        from .llm_ollama import OllamaBackend

        return OllamaBackend(**kw)
    if kind == "tfidf":
        from .tfidf import TfidfBackend

        return TfidfBackend.load()
    if kind == "ft":
        from .finetuned import FinetunedBackend

        return FinetunedBackend.load()
    if kind == "cascade":
        from .llm_ollama import OllamaBackend

        return make_cascade(OllamaBackend(**kw))
    raise ValueError(f"unknown backend {kind!r} (llm | tfidf | ft | cascade)")


def make_cascade(llm):
    """Fine-tuned model + LLM fallback, configured by <model_dir>/routing.json (written by
    scripts/routing_analysis.py from the suite-dev results; defaults: confidence threshold only)."""
    import json

    from ..config import settings
    from .cascade import CascadeBackend
    from .finetuned import FinetunedBackend

    cfg_path = settings.model_dir / "routing.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    second = retriever = None
    if cfg.get("disagree"):
        from .tfidf import TfidfBackend
        second = TfidfBackend.load()
    if cfg.get("min_similarity") is not None:
        from ..retriever import load_retriever
        retriever = load_retriever()
    return CascadeBackend(FinetunedBackend.load(), llm, threshold=cfg.get("threshold", settings.cascade_threshold),
                          second_opinion=second, retriever=retriever, min_similarity=cfg.get("min_similarity"))
