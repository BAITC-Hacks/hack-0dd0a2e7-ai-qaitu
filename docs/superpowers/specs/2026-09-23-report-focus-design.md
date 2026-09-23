# QAITU: focused report workspace

The primary scenario is to understand the comparison, choose a question to
review, and inspect its evidence. Use 1920×1080 as the desktop acceptance viewport,
not a fixed canvas. Keep the existing dark/green identity and native light theme.

## Composition

- Compact title and pinned section links. Summary remains before the matrix.
- One row of restrained indicators; a visible synthetic-data label for the demo.
- Review questions precede the detailed matrix. Remove repeated count paragraphs,
  repeated structural-change lists, permanent legends and repeated caveats.
- Keep limitations, reading guidance, metadata and source context available on
  demand. Processing failures and uncertain ownership stay visible.
- Use full department names by default. The before/after column sets remain
  aligned; remove decorative cell hatching and soften empty-cell markers.
- Source text uses regular weight; status and metadata are secondary. One clear
  type scale: 28px title, 22px section, 17px evidence heading, 15–16px body.

## Expanded matrix

A labeled expand button opens a native Streamlit dialog on the current page.
It uses nearly the viewport width/height, keeps the current filters and selected
row highlighted, and provides bounded pagination for large filtered results.
Close via the visible close control, Escape, or outside click. Returning retains
the main report, its page, selection and filter state. No analysis/API call is
triggered by expansion. Document values stay HTML-escaped in both views.

## Material and motion

Flat reading surfaces, subtle separators, green accents and no background glow.
Keep glass subdued on navigation only. No content entrance sequence or animated
numbers. Button feedback lasts 100–140ms; dialog opacity/transform entrance is
160ms, with reduced-motion disabling movement. Native focus handling is verified
in the browser, including Escape and return to the trigger.

## Acceptance

Run the existing suite and add behavior checks for expanded filtered content,
pagination, state retention and escaping. Check 1920×1080, 1440×900 and 390×844,
both themes, dialog scrolling and close/reopen, keyboard focus, and reduced
motion. Backend analysis and complete Markdown/JSON exports remain unchanged.
