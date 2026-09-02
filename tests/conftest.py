import os
import textwrap

import pytest

from atlas_recall.config import Config

SAMPLE_NOTES = {
    "feedback-no-em-dashes.md": """---
name: feedback-no-em-dashes
description: never use em dashes in outward-facing text
priority: hard-rule
---

Use a middot or a comma instead of an em dash in anything an outside party
can see: resumes, PR descriptions, commit messages, public docs.
""",
    "reference-portfolio-site.md": """---
name: reference-portfolio-site
description: where the portfolio site lives and how it deploys
priority: normal
---

The portfolio site deploys from main automatically. See
[[project-registry]] for the full list of active projects.
""",
    "project-registry.md": """---
name: project-registry
description: master project map
priority: normal
---

Registry of active projects and where their code lives.
""",
    "feedback-verify-before-claiming.md": """---
name: feedback-verify-before-claiming
description: run the test before asserting it passes
priority: high
---

Evidence before assertions. Run the command, read the output, then claim
something works -- never the other way around. See
[[feedback-no-em-dashes]] for another hard rule in the same spirit.
""",
    "idea-mrr-portfolio.md": """Just a plain note with no frontmatter at all,
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
