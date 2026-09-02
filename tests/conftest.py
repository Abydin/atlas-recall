import os
import textwrap

import pytest

from atlas_recall.config import Config

SAMPLE_NOTES = {
    "rule-descale-weekly.md": """---
name: rule-descale-weekly
description: descale the espresso machine weekly regardless of visible buildup
priority: hard-rule
---

Run a full descale cycle weekly, regardless of visible buildup. Skipping
it lets scale coat the pump and heating element, which is expensive to
replace.
""",
    "reference-bike-maintenance-log.md": """---
name: reference-bike-maintenance-log
description: where the bike maintenance log lives and how it's updated
priority: normal
---

The maintenance log gets an entry after every ride over 20 miles. See
[[gear-inventory]] for the full list of active gear.
""",
    "gear-inventory.md": """---
name: gear-inventory
description: master list of gear and where it's stored
priority: normal
---

Inventory of active gear and where it lives.
""",
    "rule-check-tire-pressure.md": """---
name: rule-check-tire-pressure
description: check tire pressure before every long ride
priority: high
---

Always check tire pressure before a long ride -- never trust the previous
reading. See [[rule-descale-weekly]] for another hard rule in the same
spirit.
""",
    "loose-ride-notes.md": """Just a plain note with no frontmatter at all,
to make sure the parser doesn't choke on one.
""",
}


@pytest.fixture
def notes_dir(tmp_path):
    d = tmp_path / "notes"
    d.mkdir()
    for name, content in SAMPLE_NOTES.items():
        (d / name).write_text(textwrap.dedent(content), encoding="utf-8")
    return str(d)


@pytest.fixture
def cfg(notes_dir, tmp_path):
    return Config(
        notes_dir=notes_dir,
        chroma_dir=str(tmp_path / "chroma"),
        dense_enabled=False,
    )
