"""Shared utility functions."""


def mask_email(email: str) -> str:
    """Mask email for safe logging: 'user@example.com' -> 'u***@example.com'."""
    parts = email.split("@", 1)
    if len(parts) != 2 or not parts[0]:
        return "***"
    return f"{parts[0][0]}***@{parts[1]}"
