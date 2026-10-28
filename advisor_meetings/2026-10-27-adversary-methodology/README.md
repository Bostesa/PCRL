# Advisor meeting: adversary before methodology

Read `output/adversary_methodology.pdf` first. Slides 1–2 are the three-to-five-minute explanation. Slides 3–5 are the appendix. `output/adversary_methodology.pptx` is editable and contains speaker notes. The same notes are readable in `SPEAKER_NOTES.md`. `output/advisor_brief.pdf` is a one-page brief, with editable Markdown and LaTeX sources alongside it.

`ACCESS_EVIDENCE.csv`/`.json` are the evidence table prepared before slide drafting. They distinguish what each paper states, what inspected code implements, what our evaluation does, and the recommended target. `SOURCE_MAP.md` supplies paper sections, pinned code and unresolved facts. Unknown acquisition pathways are not filled in with hypothetical evidence.

The admitted empirical manuscript is `papers/combined_empirical_v2/` at 9e6d01f28 (16 October 2026), not the later ACS method-development study. The two original manuscripts, the four component experiments, the corrected adversary patch v2 and the unavailable-version limitation are identified in the source map.

## Recommendation

Make the release-only single recipient primary. State the labelled auxiliary-release budget explicitly as a benchmark grant. The particular service instantiation remains illustrative. Treat a perfectly linked two-recipient coalition and clean-auxiliary access as separate extensions. A recipient that also holds raw inputs, public weights or permitted repeated queries is a different plausible actor, not an illegitimate one. The current evidence is development-only, conditional on the fitted models/readers. No competitive new algorithm is established.

## What was checked

Two role-separated AI assistants checked implementation/access and motivation/primary sources independently. The motivation reviewer asked whether the actor still matters if every method resists. The implementation reviewer compared the table with actual runners rather than the preparation access lattice. Their corrections were incorporated: frozen rather than trained backbone, row_id linkage rather than conflating role keys with linkage, all useful-head surfaces, the label-only comparator exception and the correct interpretation of the 0.02 AUC contrast target. The source reviewer corrected an attribution and distinguished the access abstraction from the actual persistent noise construction. These checks are not independent human adjudication.

Every slide and the brief was rendered and visually inspected. The presentation package/integrity, slide count, native tables and import checks passed. LibreOffice rendered the PDF, so it is not a claim of opening the deck in Microsoft PowerPoint. Fonts in both PDFs are embedded. All 21 source-handoff hashes match. No historical scientific file was changed.

No training, reader fit, new scientific statistic, real-person array read, data-role opening, cloud resource, paid API request, message or publication occurred. The material is for discussion, not a new preregistration or result.

## Local rebuild

The presentation uses the installed `@oai/artifact-tool` JavaScript runtime. Font choice is Arial in the editable deck. The local PDF renderer uses its Liberation Sans equivalent. `build_slides.mjs` contains presentation authoring only, with no scientific-data loader or evaluator path. `deck_content.json` is the editable content source. The finalizer deliberately refuses existing output/receipt paths. To reproduce, copy this folder to a scratch directory, remove only that scratch copy's `output/adversary_methodology.pptx`, and create a new empty `.build` directory.

Set these paths to the installed equivalents on the machine:

```sh
export SKILL_DIR=/Users/nathansamson/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11814/skills/presentations
export RUNTIME_NODE_MODULES=/Users/nathansamson/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules
export RUNTIME_PYTHON=/Users/nathansamson/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
ln -s "$RUNTIME_NODE_MODULES" node_modules
/Users/nathansamson/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node build_slides.mjs
/Users/nathansamson/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice --headless --convert-to pdf --outdir output output/adversary_methodology.pptx
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=.build ADVISOR_BRIEF.tex
cp .build/ADVISOR_BRIEF.pdf output/advisor_brief.pdf
```

The originals and all other worktrees remain untouched. This branch is local only. The package includes no data arrays, model weights, credentials or new attack implementation.
