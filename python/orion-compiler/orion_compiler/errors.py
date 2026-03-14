"""Library-specific exceptions for the orion-compiler package."""


class CompilerError(Exception):
    """Base exception for all orion-compiler errors."""


class CompilationError(CompilerError):
    """Raised when compilation fails."""


class ValidationError(CompilerError):
    """Raised when parameter or input validation fails."""
