"""Types for caller-supplied personal-information detection."""

from collections.abc import Callable

PIIDetector = Callable[[str], bool]
