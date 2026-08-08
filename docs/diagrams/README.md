# Diagrams

Mermaid source plus the rendered PNGs that the report and README embed.

| Source | Rendered | Shows |
|---|---|---|
| `architecture.mmd` | `architecture.png` | The offline/online split: what is built once versus what runs per request. |
| `verdict-decision-tree.mmd` | `verdict-decision-tree.png` | Every branch through verdict aggregation, including all three routes to `INSUFFICIENT_EVIDENCE`. |

## Regenerating

```bash
cd docs/diagrams
export PUPPETEER_EXECUTABLE_PATH="/c/Program Files/Google/Chrome/Application/chrome.exe"
npx -y @mermaid-js/mermaid-cli -i architecture.mmd -o architecture.png -w 2000 -b white
npx -y @mermaid-js/mermaid-cli -i verdict-decision-tree.mmd -o verdict-decision-tree.png -w 1250 -b white
```

`mermaid-cli` drives a headless browser. Pointing `PUPPETEER_EXECUTABLE_PATH` at
an installed Chrome avoids downloading a second copy of Chromium. The `-b white`
matters: the default transparent background renders as black when the PNG is
pasted into a dark-themed document.

Keep the source files free of leading `%%` comment blocks. A comment block above
the `flowchart` declaration makes the parser read the diagram type as part of
the comment and the render fails.

## Keeping them true

Both diagrams are transcriptions of code, not sketches of intent, so they go
stale silently when the code moves:

- `architecture.mmd` hardcodes the corpus size (183 papers / 446 passages), the
  fusion weights, and both coverage floors.
- `verdict-decision-tree.mmd` hardcodes every threshold in
  `backend/app/pipeline/verdict.py` and `config.py` — 0.911, 0.905, 0.45, the
  source floor of 3, the 0.10 tie margin, and the four certainty adjustments.

Changing a constant in `config.py`, rebuilding the index, or adding a branch to
`aggregate()` means re-rendering these.
