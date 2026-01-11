# Citation Extraction Update - Section Structure Fix

## Summary

Updated the USC Dash App to properly extract both inline citations and multi-line citation blocks from section content and move them to the "Editorial Notes & Annotations" section.

## Problem

Previously, the app only recognized editorial notes when they appeared as standalone section headers (e.g., "EDITORIAL NOTES", "AMENDMENTS", etc.). Two types of citations were being incorrectly displayed as part of the legislative content:

### 1. Inline Citations (within text):

- `(R.S. §27; Feb. 14, 1899, ch. 154, 30 Stat. 836.)`
- `(Pub. L. 93–554, title I, ch. III, Dec. 27, 1974, 88 Stat. 1777.)`
- `R.S. §27 derived from acts Feb. 28, 1871, ch. 99, §19, 16 Stat. 440...`

### 2. Multi-line Citation Blocks (at end of sections):

```
(Aug. 14, 1935, ch. 531, title I, §4, 49 Stat. 622; Aug. 28, 1950, ch. 809, title III, pt. 6, §361(c), (d),
64 Stat. 558; 1953 Reorg. Plan No. 1, §§5, 8, eff. Apr. 11, 1953, 18 F.R. 2053, 67 Stat. 631; Pub. L.
86–778, title VI, §601(e), Sept. 13, 1960, 74 Stat. 991; Pub. L. 90–248, title II, §245, Jan. 2, 1968, 81
Stat. 918; Pub. L. 96–88, title V, §509(b), Oct. 17, 1979, 93 Stat. 695.)
```

## Solution

Added two complementary functions to handle both types of citations:

### 1. `is_citation_line(line)` - Detects multi-line citation blocks

Identifies standalone lines that contain citation patterns:

- Date patterns: `Aug. 14, 1935` or `Feb. 25, 1944`
- Statute references: `49 Stat. 622` or `64 Stat. 558`
- Public Law references: `Pub. L. 86-778`
- Chapter references: `ch. 531` or `ch. 63`
- Federal Register: `18 F.R. 2053`
- Reorganization Plans: `Reorg. Plan No. 1`

When a line contains 2+ citation patterns or starts with a statute reference, it's treated as a citation line and moved to editorial notes.

### 2. `extract_inline_citations(text)` - Extracts inline citations

Identifies and extracts citations embedded within text:

- Parenthetical citations containing legal references
- Derivation statements like "R.S. §27 derived from..."

## Implementation Details

### New Function: `is_citation_line(line)`

Located in: `usc_dash_app.py` (around line 2791)

**Parameters:**

- `line` (str): A single line of text to check

**Returns:**

- `bool`: True if the line is a citation line, False otherwise

**Detection Logic:**

```python
citation_patterns = [
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.\s+\d+,\s+\d{4}',  # Date
    r'\d+\s+Stat\.\s+\d+',  # Statute
    r'Pub\.\s*L\.\s+\d+[–-]\d+',  # Public Law
    r'ch\.\s+\d+',  # Chapter
    r'\d+\s+F\.R\.\s+\d+',  # Federal Register
    r'Reorg\.\s+Plan\s+No\.',  # Reorganization Plan
]

# If line contains 2+ patterns OR starts with a statute reference, it's a citation line
```

### New Function: `extract_inline_citations(text)`

Located in: `usc_dash_app.py` (around line 2830)

**Parameters:**

- `text` (str): The section content to process

**Returns:**

- `cleaned_text` (str): Main legislative text with citations removed
- `citations` (list): List of extracted citation strings

**Patterns Matched:**

1. **Parenthetical Citations:**

   ```regex
   \([^)]*?(?:R\.S\.|Pub\.\s*L\.|Stat\.|ch\.\s*\d+|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d+,\s*\d{4})[^)]*?\)
   ```

2. **Derivation Statements:**
   ```regex
   R\.S\.\s*§\s*\d+\s+derived from.*?\.(?=\s+[A-Z§]|$)
   ```

### Integration Points

The citation extraction is applied at three key locations:

1. **During Line-by-Line Parsing** (line ~2933):

   ```python
   # Check if this line is a citation line (multi-line citation blocks)
   if section_started and is_citation_line(line_clean):
       in_editorial_section = True
       if first_editorial_idx is None:
           first_editorial_idx = i
       editorial_content.append(line_clean)
       continue
   ```

2. **Section Intro Text** (line ~2976):

   ```python
   if section_intro_text:
       intro_combined = ' '.join(section_intro_text)
       cleaned_intro, extracted_citations = extract_inline_citations(intro_combined)
       section_intro_text = [cleaned_intro] if cleaned_intro else []
       if extracted_citations:
           editorial_content.extend(extracted_citations)
   ```

3. **Subsection Content** (line ~3020):

   ```python
   # Stop collecting if we hit a citation line
   if is_citation_line(next_line):
       break

   # Extract inline citations from subsection content
   cleaned_content, subsection_citations = extract_inline_citations(content_text)
   content_map[path] = cleaned_content
   if subsection_citations:
       editorial_content.extend(subsection_citations)
   ```

## Testing

Created and ran comprehensive tests to verify:

- ✓ Multi-line citation blocks are correctly identified
- ✓ Regular legislative text is NOT identified as citations
- ✓ Subsection markers like "(a)" are NOT identified as citations
- ✓ Section titles are NOT identified as citations
- ✓ Continuation lines (starting with years/stats) are identified
- ✓ Parenthetical citations are correctly extracted
- ✓ Derivation statements are correctly extracted
- ✓ Main legislative text is preserved without citations
- ✓ Multiple citations in the same section are all extracted
- ✓ No text is lost during extraction

All 10+ test cases passed successfully.

## Result

Section content is now properly organized:

**Main Section Content:**

- Clean, readable legislative text
- Subsection hierarchy clearly visible
- No distracting citation clutter
- Citations don't interrupt the flow

**Editorial Notes & Annotations Section:**

- All inline citations collected
- All multi-line citation blocks collected
- Parenthetical legal references
- Derivation statements
- Traditional editorial headers (AMENDMENTS, etc.)

## Example Output

### Before:

```
§4. Benefits

The Secretary shall provide benefits to eligible individuals.
(Aug. 14, 1935, ch. 531, title I, §4, 49 Stat. 622; Aug. 28, 1950, ch. 809,
title III, pt. 6, §361(c), (d), 64 Stat. 558; 1953 Reorg. Plan No. 1, §§5, 8,
eff. Apr. 11, 1953, 18 F.R. 2053, 67 Stat. 631; Pub. L. 86–778, title VI,
§601(e), Sept. 13, 1960, 74 Stat. 991; Pub. L. 90–248, title II, §245,
Jan. 2, 1968, 81 Stat. 918; Pub. L. 96–88, title V, §509(b), Oct. 17, 1979,
93 Stat. 695.)
```

### After:

**Main Content:**

```
§4. Benefits

The Secretary shall provide benefits to eligible individuals.
```

**📋 Editorial Notes & Annotations:**

```
(Aug. 14, 1935, ch. 531, title I, §4, 49 Stat. 622; Aug. 28, 1950, ch. 809,
title III, pt. 6, §361(c), (d),
64 Stat. 558; 1953 Reorg. Plan No. 1, §§5, 8, eff. Apr. 11, 1953, 18 F.R.
2053, 67 Stat. 631; Pub. L.
86–778, title VI, §601(e), Sept. 13, 1960, 74 Stat. 991; Pub. L. 90–248,
title II, §245, Jan. 2, 1968, 81
Stat. 918; Pub. L. 96–88, title V, §509(b), Oct. 17, 1979, 93 Stat. 695.)
```

## Files Modified

- `usc_dash_app.py`:
  - Added `is_citation_line()` function (line ~2791)
  - Added `extract_inline_citations()` function (line ~2830)
  - Integrated citation detection into section parsing (line ~2933)
  - Added citation stopping condition for subsection content (line ~3015)

## No Breaking Changes

- Existing functionality preserved
- All search and navigation features unchanged
- Display format enhanced, not altered
- Compatible with existing USC database and index

## Performance Impact

- Minimal: Citation detection adds simple regex pattern matching per line
- No impact on database or indexing operations
- Parsing happens only when viewing sections (same as before)

## Next Steps (Optional Enhancements)

1. Consider applying this extraction logic during database building (in `build_search_db.py`)
2. Add formatting to distinguish between different types of citations in the editorial section
3. Make citations clickable if they reference other sections
4. Add collapsible section for very long citation blocks

---

**Date:** October 19, 2025  
**Impact:** Significantly improves readability of US Code sections by properly organizing content and separating legislative text from citations  
**Tests:** 10+ test cases verified successfully
