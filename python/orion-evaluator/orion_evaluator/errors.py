"""Library-specific exceptions for the orion-evaluator package."""


class EvaluatorError(Exception):
    """Base exception for all orion-evaluator errors."""


class ModelLoadError(EvaluatorError):
    """Raised when loading a model fails."""
