import os


def redact(text: str) -> str:
    """Do not persist deployment credentials embedded in upstream command errors."""
    values = {
        value for name, value in os.environ.items()
        if value and any(token in name.upper() for token in ["PASSWORD", "API_KEY", "ACCESS_TOKEN"])
    }
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, "<redacted>")
    return text
