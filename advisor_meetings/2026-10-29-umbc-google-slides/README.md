# Dr. Yus meeting: UMBC slides and brief

Open the editable [Google Slides deck](https://docs.google.com/presentation/d/1R1DXAxTXviP6HBjsFsOfqxcppBcKJHdte2e9LxsgBPI/edit). The [one-page brief](https://drive.google.com/file/d/1HQF0eXz4O65cDXYAx2l-5ATvQfcJw-tM/view) is in the same private Drive folder. No sharing permissions were changed.

## Visual revision

The previous revision replaced the comparison table with paragraphs. This revision restores a visual explanation in the actual UMBC template:

1. A compact editable table compares prior adversaries with our proposed recipient.
2. An editable diagram shows the service output, labelled examples and extra inference.
3. An editable methodology diagram separates fitting, model selection and scoring. Its dashed arrow shows the validation feedback used for defense tuning.
4. The existing useful-head figure compares recovery with task usefulness.
5. The existing output-format and pooling figure shows the separate release surfaces.

Slides 1–2 are the main discussion. Slides 3–5 are the appendix. This keeps Dr. Yus's requested order: agree on the adversary, then discuss the methodology. There is no established successful new algorithm to present. Paper structure comes afterward.

The UMBC banner, logo, corner artwork, master, layouts and title typography remain. Body paragraphs were replaced with native tables, shapes and arrows. The paper plots are images exported from unchanged source PDFs. Their underlying plot objects are not editable in Slides. Both source PDFs and faithful PNG renders are in `assets/`.

The [Wikipedia writing guide](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) informed the wording changes. The slides remove bold-label paragraph lists, vague abstractions and inflated claims. Detailed qualifications and sources remain in speaker notes. The guide is not an AI detector, and its advice does not replace checking the scientific claims.

## Evidence and limits

The unchanged package in `../2026-10-27-adversary-methodology/` supplies the full access table and source map. The plot sources are `papers/combined_empirical_v2/figures/fig_useful_head.pdf` and `fig_formats_pairs.pdf`, with captions and qualifications preserved in notes. The first plot's displayed bars are marginal 90% intervals. The second shows three-seed means without intervals and uses axes starting near 0.45.

The service example is illustrative. The benchmark grants sensitive labels paired with matching releases, whose practical availability remains unknown. The linked-release extension grants perfect matching. These are assumptions for the meeting, not documented deployments. The figures retain the failed usefulness qualification and reused-data limitations. No calculations, scientific records or paper sources changed.

## Files

- `output/adversary_methodology_umbc.pdf`: final visually inspected Google Slides export.
- `output/adversary_methodology_umbc.pptx`: editable backup with five slides and five speaker-note pages.
- `output/advisor_brief.pdf`: unchanged one-page brief from the preceding revision.
- `CONTENT.json`, `SLIDE_PLAN.json`, `SPEAKER_NOTES.md`: slide content, template mapping and detailed notes.
- `assets/`: unchanged paper figure PDFs and rendered PNGs.
- `VALIDATION.json`, `MANIFEST.json`: readback, preservation checks and deliverable hashes.

The native deck is the editing destination. Export snapshots in this repository do not automatically follow later edits in Drive.

## Verification

All five final Google-rendered slides were visually inspected. Native titles, table cells, diagram labels and speaker notes match the content record. The template's one master and eleven layouts remain. The PowerPoint archive passes its CRC check. All 21 historical handoff hashes match. No experiments, reader fits, data-role access, new scientific statistics or evaluator calls occurred.

The connector's streamed export did not materialize locally. A bounded raw export was used and rendered with the Slides skill helper. This changes the export route, not the deck's content or permissions.
