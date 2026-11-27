---
title: Check the Mermaid diagrams in Firefox
kind: maintenance
status: ready
priority: 4
effort: XS
blocked_by: []
raised_by: [Mermaid diagrams (#126)]
---

# Check the Mermaid diagrams in Firefox

The documentation draws its diagrams with sphinxcontrib-mermaid: the introduction's CAT-lifecycle
flowchart and the class trees. They were checked in Chromium. No one has checked Firefox, because
Playwright's Firefox would not launch when they were added.

Open the built site in Firefox, in light and dark themes. Check that each diagram renders at its
natural size and that its text is readable. `_static/mermaid.css` restores the natural size and
turns off pydata's dark-mode invert filter. Fix what differs, or delete this card if nothing does.
