"""Single source of truth for allowed labels. Scorers (Step 4) import these."""

CATEGORIES = ["database", "network", "authentication", "filesystem", "memory", "application", "other"]
SEVERITIES = ["low", "medium", "high", "critical"]