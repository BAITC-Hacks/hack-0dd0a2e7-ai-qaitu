# QAITU: Liquid Glass interface

## Baseline and scope

Based on main at `5cab6c1`. QAITU is an analytical Streamlit workspace with
document upload, four result tabs, a function ownership matrix, source evidence,
and Markdown/JSON export. Preserve these workflows, the emerald brand accent,
Russian labels, source escaping, and explicit consent for external AI review.
The existing interface has a useful structure but flat surfaces, a weak empty
state, and little distinction between controls and evidence.

## Reference interpretation

- [Agatha Richards: Liquid Glass in Figma](https://www.youtube.com/watch?v=w_ZV24jeRmA): rounded translucent material, fine reflective edges, background visible through the control.
- [bentomotion: Liquid Glass in After Effects](https://www.youtube.com/watch?v=J9MORPelLZk): dimensional inset highlights, soft shadow, and smooth material transitions.

Use a web approximation with CSS backdrop blur, layered highlights, and shadows.
Do not describe this as Apple's native rendering or exact optical refraction.

## Direction

A restrained glass workspace is preferred over a cosmetic color-only refresh
(too little connection to the references) or a highly refractive full-screen
effect (poor reading conditions for dense evidence). Design variance 3/10,
motion 3/10, density 5/10. Use Apple-inspired materials, existing Streamlit
controls, and the platform font stack. No additional UI framework or remote assets.

Glass belongs on the sidebar, navigation, controls, and overview surfaces.
Matrix cells and quoted evidence remain high-opacity, with semantic color and
existing text markers. Preserve green branding and distinct blue/red/amber
matrix meanings. Use 24px panel, 16px control, and pill navigation radii.

The welcome screen explains the upload > compare > inspect workflow with a
clear hierarchy. It includes an actionable control example and clearly identifies
synthetic data. After analysis, six native anchor links remain pinned above the
report: overview, matrix, conclusion, structure, sources, and export. All sections
are visible in one report, and navigation preserves filters and analysis state.

The completed workspace expands to 1840px. The matrix has a wider function
column, ten rows per page and a 480px scroll region. Evidence appears in balanced
before/after columns, stacking when the available area is narrower than 860px.
Source metadata is secondary, body text uses a readable sans-serif face, and
repeated quotations expand on demand. Informational notices use the neutral
green palette while warning/error and matrix status colors retain their meaning.

## Implementation and verification

Separate local presentation helpers and CSS from analytical code. Keep consent
and error-handling behavior. All document-derived HTML stays escaped. Do not
transmit documents to any new service.

Acceptance: desktop and narrow mobile views have no page overflow; the matrix
scrolls within its named region; keyboard focus remains visible; native light
and dark themes remain readable; reduced motion, increased contrast and reduced
transparency have useful fallbacks. Check initial state, missing uploads, demo,
all six anchors, pagination, filters, citations, and both exports in addition to the existing
unittest suite. Record before/after screenshots and a reproducible change patch.
