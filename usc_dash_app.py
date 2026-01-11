import dash
from dash import dcc, html, Input, Output, State, callback, ALL, no_update
import dash_bootstrap_components as dbc
import re
import os
from datetime import datetime
import glob
import json
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import zipfile
import io
import base64

# Import SQLite search database for fast content search
try:
    import search_database
    SEARCH_DB_AVAILABLE = True
    print("✓ SQLite search database module loaded")
except ImportError:
    SEARCH_DB_AVAILABLE = False
    print("⚠️  SQLite search database not available - using fallback search")

# Import Ollama AI integration for testing
try:
    import ollama_integration
    AI_AVAILABLE = True
    print("✓ Ollama AI integration module loaded")
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  Ollama AI integration not available - AI features disabled")

# Import the search functions from the existing script
def load_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except IOError:
        print(f"Error: Unable to read file at {file_path}")
        return None

# Global variable to store the preloaded index
USC_INDEX = None

def load_usc_index():
    """Load the pre-built USC sections index"""
    global USC_INDEX
    if USC_INDEX is not None:
        return USC_INDEX
    
    try:
        with open('usc_sections_index.json', 'r', encoding='utf-8') as f:
            USC_INDEX = json.load(f)
        print(f"✓ Loaded USC index with {USC_INDEX['metadata']['total_sections']} sections from {USC_INDEX['metadata']['total_files']} files")
        return USC_INDEX
    except FileNotFoundError:
        print("Warning: usc_sections_index.json not found. Run 'python build_usc_index.py' to create it.")
        print("Falling back to on-demand parsing (slower)...")
        return None
    except Exception as e:
        print(f"Error loading USC index: {e}")
        return None

@lru_cache(maxsize=128)
def get_document_sections(filename):
    """Get sections from pre-built index or fall back to parsing"""
    # Try to use the pre-built index first
    index = load_usc_index()
    if index and filename in index['files']:
        file_data = index['files'][filename]
        sections = {}
        
        # Convert index format to the expected format
        for section_num, section_info in file_data['sections'].items():
            sections[section_num] = {
                'title': section_info['title'],
                'subsections': section_info['subsections']
            }
        
        return sections
    
    # Fallback to original parsing method if index is not available
    print(f"Index not available for {filename}, using on-demand parsing...")
    file_path = f"txt/{filename}"
    document = load_document(file_path)
    if not document:
        return {}
    
    sections = {}
    
    # Find all sections (§number.) - avoid matching citations in parentheses like "(R.S. §3073.)"
    all_matches = list(re.finditer(r'(?:^|\n\s*)§(\d+[A-Za-z]?)\.\s*([^\n]+)', document, re.MULTILINE))
    
    # Filter out parenthetical citations
    section_matches = []
    for match in all_matches:
        # Get some context before the match to check for parentheses
        start_pos = max(0, match.start() - 50)
        context_before = document[start_pos:match.start()]
        
        # Skip if this section reference is inside parentheses
        open_parens = context_before.count('(')
        close_parens = context_before.count(')')
        
        if open_parens > close_parens:
            continue
            
        # Also skip if the immediate character before § is '('
        if match.start() > 0 and document[match.start()-1] == '(':
            continue
            
        section_matches.append(match)
    
    for match in section_matches:
        section_num = match.group(1)
        section_title = match.group(2).strip()
        
        # Skip sections that are omitted, repealed, or transferred
        title_lower = section_title.lower()
        if title_lower.startswith('omitted') or title_lower.startswith('repealed') or title_lower.startswith('transferred'):
            continue
        
        # Extract the full section content
        section_content = extract_full_section(document, section_num)
        
        # Find subsections within this section
        subsections = {}
        if section_content and not section_content.startswith('Section'):
            # Look for subsections like (a), (1), (A), etc.
            subsection_matches = re.finditer(r'\(([a-zA-Z0-9]+)\)', section_content)
            for sub_match in subsection_matches:
                sub_key = sub_match.group(1)
                # Get a snippet of the subsection content
                start_pos = sub_match.end()
                snippet = section_content[start_pos:start_pos+100].strip()
                # Clean up the snippet
                snippet = re.split(r'\([a-zA-Z0-9]+\)', snippet)[0].strip()
                if snippet:
                    subsections[sub_key] = snippet[:80] + "..." if len(snippet) > 80 else snippet
        
        sections[section_num] = {
            'title': section_title[:100] + "..." if len(section_title) > 100 else section_title,
            'subsections': subsections
        }
    
    return sections

def find_usc_section(document, input_string):
    section_identifier, search_sequence = parse_input(input_string)
    section_number = section_identifier.strip('§.')
    
    # Find the section
    section_content = extract_full_section(document, section_number)
    
    if section_content.startswith(f"Section {section_number} not found"):
        return section_content
    
    if not search_sequence:
        return section_content
    
    # If subsections are specified, search within the section content
    current_text = section_content
    for i, subsection in enumerate(search_sequence):
        subsection_pattern = re.compile(rf"\({subsection}\).*?(?=\({get_next_sibling(subsection)}\)|\({get_next_parent(search_sequence, i)}\)|(?:^|\n\s*)§\d+[A-Za-z]?\.|$)", re.DOTALL | re.MULTILINE)
        subsection_match = subsection_pattern.search(current_text)
        
        if not subsection_match:
            return f"Subsection {subsection} not found in Section {section_number}"
        
        current_text = subsection_match.group()
    
    return current_text.strip()

def parse_input(input_string):
    parts = input_string.split(maxsplit=1)
    section_number = parts[0]
    section_identifier = f"§{section_number}."
    
    if len(parts) > 1:
        search_sequence = re.findall(r"\(([^)]+)\)", parts[1])
    else:
        search_sequence = []
    
    return section_identifier, search_sequence

def get_next_sibling(subsection):
    if subsection.isdigit():
        return str(int(subsection) + 1)
    elif subsection.isalpha():
        return chr(ord(subsection) + 1)
    return ""

def get_next_parent(search_sequence, current_index):
    if current_index > 0:
        return search_sequence[current_index - 1]
    return ""

def extract_full_section(document, section_number):
    # Create the section identifier pattern
    section_start = rf"§{section_number}\."
    
    # Find the start of the section
    section_start_match = re.search(section_start, document)
    if not section_start_match:
        return f"Section {section_number} not found"
    
    start_index = section_start_match.start()
    
    # Find the start of the next section
    # Use word boundary and line start to avoid matching citations like "(R.S. §3073.)"
    next_section_match = re.search(r"(?:^|\n\s*)§\d+[A-Za-z]?\.", document[start_index + 1:], re.MULTILINE)
    
    if next_section_match:
        end_index = start_index + 1 + next_section_match.start()
        section_content = document[start_index:end_index].strip()
    else:
        # If no next section is found, extract until the end of the document
        section_content = document[start_index:].strip()
    
    return section_content

# Get available USC titles
def get_available_titles():
    # Try to use the pre-built index first for faster loading
    index = load_usc_index()
    if index:
        titles = []
        title_info = []
        
        for filename, file_data in index['files'].items():
            title_data = file_data['title_info']
            title_info.append({
                'label': title_data['title_label'],
                'value': filename,
                'title_num': title_data['title_num']
            })
        
        # Sort by title number
        title_info.sort(key=lambda x: (len(x['title_num']), x['title_num']))
        
        # Return only label and value for Dash dropdown
        for item in title_info:
            titles.append({
                'label': item['label'],
                'value': item['value']
            })
        
        return titles
    
    # Fallback to file system scanning if index is not available
    print("Index not available, scanning file system for titles...")
    txt_files = glob.glob("txt/usc*.txt")
    titles = []
    title_info = []  # Store title info separately for sorting
    
    for file in txt_files:
        filename = os.path.basename(file)
        # Extract title number from filename - handle different formats
        # Format 1: "usc09@118-105.txt" 
        # Format 2: "usc42_ch40to81_Secs3271to6892@118-105.txt"
        match = re.match(r'usc(\d+[A-Z]?)(?:_.*)?@', filename)
        if match:
            title_num = match.group(1)
            
            # Create a readable label for complex titles
            if '_ch' in filename:
                # Extract chapter info for Title 42 files
                ch_match = re.search(r'_ch(\d+)to(\d+)', filename)
                if ch_match:
                    label = f'Title {title_num} (Chapters {ch_match.group(1)}-{ch_match.group(2)})'
                else:
                    label = f'Title {title_num} (Partial)'
            else:
                label = f'Title {title_num}'
            
            title_info.append({
                'label': label,
                'value': filename,
                'title_num': title_num
            })
    
    # Sort by title number
    title_info.sort(key=lambda x: (len(x['title_num']), x['title_num']))
    
    # Return only label and value for Dash dropdown
    for item in title_info:
        titles.append({
            'label': item['label'],
            'value': item['value']
        })
    
    return titles

# Helper: find a filename for a given title number
def extract_title_num_from_filename(filename: str) -> str:
    m = re.match(r'usc(\d+[A-Z]?)(?:_.*)?@', filename)
    return m.group(1) if m else ''

def find_filename_for_title_num(title_num: str, current_filename: str = None) -> str:
    # Prefer staying in the same file if title matches
    if current_filename and extract_title_num_from_filename(current_filename) == title_num:
        return current_filename
    # Otherwise pick the first available for that title (prefer non-partial if exists)
    candidates = [t['value'] for t in available_titles if t['label'].startswith(f'Title {title_num}')]
    if not candidates:
        return current_filename or (available_titles[0]['value'] if available_titles else None)
    # Prefer candidates without '(' (i.e., full title files), else first
    fulls = [c for c in candidates if '(' not in c]
    return fulls[0] if fulls else candidates[0]

# Helper: linkify references in text
ref_pattern = re.compile(
    r"(§\s*(?P<sec1>\d+[A-Za-z]?))|(section\s+(?P<sec2>\d+[A-Za-z]?))(?:\s+of\s+title\s+(?P<title>\d+[A-Z]?))?(?:\s+of\s+this\s+title)?",
    re.IGNORECASE
)

def linkify_text(text: str, current_filename: str):
    elements = []
    last = 0
    for m in ref_pattern.finditer(text):
        start, end = m.start(), m.end()
        if start > last:
            elements.append(text[last:start])
        section_num = m.group('sec1') or m.group('sec2')
        title_num = m.group('title') or extract_title_num_from_filename(current_filename)
        target_file = find_filename_for_title_num(title_num, current_filename)
        label = m.group(0)
        elements.append(
            html.A(
                label,
                id={
                    'type': 'usc-ref',
                    'title': title_num,
                    'section': section_num,
                    'file': target_file,
                    'index': len(elements)  # Add index to make IDs unique
                },
                n_clicks=0,
                style={
                    'textDecoration': 'underline',
                    'cursor': 'pointer',
                    'color': '#0d6efd',
                    'fontWeight': '500'
                }
            )
        )
        last = end
    if last < len(text):
        elements.append(text[last:])
    return elements if elements else [text]

def search_single_file(args):
    """Search a single file for a set of terms - designed to run in parallel"""
    file_path, search_terms = args
    results = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            filename = os.path.basename(file_path)
            
            content_lower = content.lower()
            # Quick check: if any term is not in the file, we can skip it.
            if not all(term in content_lower for term in search_terms):
                return []

            # Get title info once for efficiency
            title_match = re.match(r'usc(\d+[A-Z]?)(?:_.*)?@', filename)
            title_num = title_match.group(1) if title_match else "Unknown"
            if '_ch' in filename:
                ch_match = re.search(r'_ch(\d+)to(\d+)', filename)
                title_label = f'Title {title_num} (Ch. {ch_match.group(1)}-{ch_match.group(2)})' if ch_match else f'Title {title_num} (Partial)'
            else:
                title_label = f'Title {title_num}'

            section_pattern = re.compile(r'(?:^|\n\s*)§(\d+[A-Za-z]?)\.\s*([^\n]+)', re.MULTILINE)
            all_matches = list(section_pattern.finditer(content))
            
            # Filter out parenthetical citations
            all_sections = []
            for match in all_matches:
                # Get some context before the match to check for parentheses
                start_pos = max(0, match.start() - 50)
                context_before = content[start_pos:match.start()]
                
                # Skip if this section reference is inside parentheses
                open_parens = context_before.count('(')
                close_parens = context_before.count(')')
                
                if open_parens > close_parens:
                    continue
                    
                # Also skip if the immediate character before § is '('
                if match.start() > 0 and content[match.start()-1] == '(':
                    continue
                    
                all_sections.append(match)
            
            # Define editorial patterns to exclude
            editorial_patterns = [
                r'\bEDITORIAL NOTES\b',
                r'\bAMENDMENTS\b',
                r'\bDERIVATION\b',
                r'\bSTATUTORY NOTES AND RELATED SUBSIDIARIES\b',
                r'\bREFERENCES IN TEXT\b',
                r'\bEFFECTIVE DATE\b',
                r'\bSHORT TITLE\b',
                r'\bHISTORICAL AND REVISION NOTES\b',
                r'\bPRIOR PROVISIONS\b',
                r'\bCODIFICATION\b',
                r'\bREPEALS\b'
            ]
            editorial_regex = re.compile('|'.join(editorial_patterns), re.IGNORECASE)
            
            # Iterate through each section in the document
            for i, section_match in enumerate(all_sections):
                section_start_pos = section_match.start()
                next_section_idx = i + 1
                section_end_pos = all_sections[next_section_idx].start() if next_section_idx < len(all_sections) else len(content)
                
                # Extract the full section content
                full_section_content = content[section_start_pos:section_end_pos]
                section_lower = full_section_content.lower()

                # Check if all search terms are present in this section
                if all(term in section_lower for term in search_terms):
                    section_num = section_match.group(1)
                    section_title = section_match.group(2).strip()
                    
                    # Skip sections that are omitted, repealed, or transferred
                    title_lower = section_title.lower()
                    if title_lower.startswith('omitted') or title_lower.startswith('repealed') or title_lower.startswith('transferred'):
                        continue
                    
                    section_key = (filename, section_num)
                    
                    # Clean the section content by removing editorial material
                    cleaned_content = full_section_content
                    
                    # Find the first occurrence of editorial patterns
                    editorial_match = editorial_regex.search(full_section_content)
                    if editorial_match:
                        # Keep content only up to the editorial section
                        cleaned_content = full_section_content[:editorial_match.start()].strip()
                    
                    # If the cleaned content is very long, we can optionally truncate it
                    # but let's keep it full for now as requested
                    context = cleaned_content.strip()
                    
                    # Remove excessive whitespace but preserve structure
                    context = re.sub(r'\n\s*\n\s*\n+', '\n\n', context)  # Replace 3+ newlines with 2
                    
                    # Find the first occurrence of any search term for highlighting
                    first_term_pos = -1
                    for term in search_terms:
                        pos = context.lower().find(term)
                        if pos != -1 and (first_term_pos == -1 or pos < first_term_pos):
                            first_term_pos = pos
                    
                    # Highlight all search terms in the full content
                    highlight_pattern = '|'.join(re.escape(term) for term in search_terms)
                    highlighted_context = re.sub(
                        f'({highlight_pattern})', 
                        r'**\1**', 
                        context, 
                        flags=re.IGNORECASE
                    )
                    
                    # Count total occurrences of all terms
                    occurrence_count = sum(section_lower.count(term) for term in search_terms)

                    results[section_key] = {
                        'title_num': title_num,
                        'title_label': title_label,
                        'section_num': section_num,
                        'section_title': section_title,
                        'context': highlighted_context,
                        'filename': filename,
                        'occurrence_count': occurrence_count,
                        'position': section_start_pos
                    }

    except Exception as e:
        print(f"Error searching {file_path}: {e}")
    
    return list(results.values())

def word_search_sequential(search_terms, files_to_search, max_results=10000):
    """Fallback sequential search for environments that don't support multiprocessing"""
    all_results = []
    section_keys_seen = set()
    
    for file_path in files_to_search:
        if len(all_results) >= max_results:
            break
            
        file_results = search_single_file((file_path, search_terms))
        
        # Add unique results
        for result in file_results:
            section_key = (result['filename'], result['section_num'])
            if section_key not in section_keys_seen:
                section_keys_seen.add(section_key)
                all_results.append(result)
                
                if len(all_results) >= max_results:
                    break
    
    return all_results

def estimate_tokens(text):
    """
    Rough estimation of token count (approximately 4 characters per token).
    This is a conservative estimate for splitting purposes.
    """
    return len(text) // 4

def parse_section_hierarchy(section_content):
    """
    Dynamically parse the actual hierarchy structure of a section.
    
    Uses the sequence and type of subsections to infer hierarchy.
    Common USC pattern: first type seen = level 0, each new type = nested level
    
    Enforces strict ordering:
    - Each level must start at the beginning (1, a, i, A)
    - Siblings must be sequential (a, b, c... no skipping)
    - Only valid structural markers are recognized
    
    Args:
        section_content: Full text content of the section
    
    Returns:
        dict: Hierarchy information with structure:
            {
                'subsection_key': {
                    'level': int (0 = top-level parent, 1+ = nested child),
                    'parent': str or None (parent subsection key),
                    'position': int (order of appearance)
                }
            }
    """
    hierarchy = {}
    lines = section_content.split('\n')
    
    # Editorial/citation patterns to exclude from structural parsing
    # Note: Month pattern must be specific to citation format (e.g., "Feb. 14, 1899")
    # NOT generic month mentions (e.g., "the first Monday in February" or "January 1, 1968")
    editorial_citation_patterns = [
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.\s+\d{1,2},?\s*\d{4}',  # Citation dates: Feb. 14, 1899 (requires period after month)
        r'\bPub\. L\.',
        r'\d+ Stat\.',
        r'\bch\.\s*\d+',
        r'\bEDITORIAL NOTES\b',
        r'\bAMENDMENTS\b',
        r'\bDERIVATION\b',
        r'\bSTATUTORY NOTES\b',
        r'\bREFERENCES IN TEXT\b',
        r'\bEFFECTIVE DATE\b',
        r'\bSHORT TITLE\b',
        r'\bHISTORICAL AND REVISION NOTES\b',
        r'\bPRIOR PROVISIONS\b',
        r'\bCODIFICATION\b',
        r'\bREPEALS\b'
    ]
    editorial_regex = re.compile('|'.join(editorial_citation_patterns), re.IGNORECASE)
    
    # Track subsections in order
    subsections = []  # [(key, type), ...]
    
    # Citation patterns - specific phrases that indicate a reference, not structure
    citation_patterns = [
        r'subsection\s+\(',
        r'paragraph\s+\(',
        r'section\s+\d*\s*\(',
        r'clause\s+\(',
        r'described\s+in\s+\(',
        r'referred\s+to\s+in\s+\(',
        r'pursuant\s+to\s+\(',
        r'under\s+\(',
        r'subsec\.\s+\(',
        r'subparagraph\s+\(',
        r'set\s+out\s+in\s+\(',
    ]
    citation_regex = re.compile('|'.join(citation_patterns), re.IGNORECASE)
    
    # First pass: collect all subsections strictly at line starts, handling consecutive markers like (1)(A)(i)
    # This avoids missing parents in lines that include multiple markers consecutively
    for line in lines:
        line_clean = line.strip()
        
        # Skip editorial/citation lines
        if editorial_regex.search(line_clean):
            continue
        
        # Extract ALL consecutive markers at the start of the line
        remaining_line = line_clean
        consecutive_markers = []
        while True:
            marker_match = re.match(r'^\(([a-zA-Z0-9]+)\)', remaining_line)
            if not marker_match:
                break
            subsection_key = marker_match.group(1)
            subsection_type = _get_subsection_type(subsection_key)
            if subsection_type == 'other':
                break
            consecutive_markers.append((subsection_key, subsection_type))
            remaining_line = remaining_line[marker_match.end():]
            # If next char isn't another marker, stop
            if not remaining_line.startswith('('):
                break
        
        # Add all consecutive markers (if any)
        if consecutive_markers:
            subsections.extend(consecutive_markers)
    
    if not subsections:
        return {}
    
    # Second pass: determine hierarchy based on context and enforce sequential ordering
    # Use a stack-based approach to track nesting
    level_stack = []  # [(subsection_key, subsection_type), ...]
    expected_next = {}  # level -> expected next key for that level
    
    for position, (subsection_key, subsection_type) in enumerate(subsections):
        # Determine the level based on context
        if position == 0:
            # First subsection is always at level 0
            # Must start with valid sequence start (1, a, A, i)
            if not _is_valid_sequence_start(subsection_key, subsection_type):
                # Invalid first subsection - skip it
                continue
            
            level = 0
            parent = None
            expected_next[level] = _get_next_in_sequence(subsection_key, subsection_type)
        else:
            # Look at the previous subsection to determine relationship
            prev_key, prev_type = subsections[position - 1]
            
            if subsection_type == prev_type:
                # Same type as previous = sibling at same level
                if level_stack:
                    level = len(level_stack) - 1
                    # Parent is the same as previous subsection's parent
                    if level > 0:
                        parent = level_stack[level - 1][0]
                    else:
                        parent = None
                else:
                    level = 0
                    parent = None
                
                # Enforce sequential ordering for siblings
                if level in expected_next:
                    if subsection_key != expected_next[level]:
                        # Out of order - skip this subsection
                        continue
                    # Update expected next
                    expected_next[level] = _get_next_in_sequence(subsection_key, subsection_type)
                else:
                    # First of this type at this level - should start sequence
                    if not _is_valid_sequence_start(subsection_key, subsection_type):
                        continue
                    expected_next[level] = _get_next_in_sequence(subsection_key, subsection_type)
            else:
                # Different type from previous
                # Check if this type was seen before at a higher level (moving back up)
                found_level = None
                
                # Search through the stack to see if this type appeared before
                # This handles cases like: (a) > (1)(2)(3) > (b) where (b) should be sibling of (a)
                for i in range(len(level_stack)):
                    if level_stack[i][1] == subsection_type:
                        # Found the same type at a higher level
                        found_level = i
                        break
                
                # Determine if this is truly returning to a previous level or starting a new child sequence
                current_stack_depth = len(level_stack)
                
                if found_level is not None and found_level < current_stack_depth - 1:
                    # The type was found at least 2 levels up (not the immediate parent)
                    # This is clearly returning to that level as a sibling
                    level = found_level
                    level_stack = level_stack[:level]
                    if level > 0:
                        parent = level_stack[level - 1][0]
                    else:
                        parent = None
                    
                    # Enforce sequential ordering
                    if level in expected_next:
                        if subsection_key != expected_next[level]:
                            continue
                        expected_next[level] = _get_next_in_sequence(subsection_key, subsection_type)
                    else:
                        expected_next[level] = _get_next_in_sequence(subsection_key, subsection_type)
                    
                    # Clear expected_next for deeper levels
                    for deeper_level in list(expected_next.keys()):
                        if deeper_level > level:
                            del expected_next[deeper_level]
                            
                elif found_level is not None and found_level == current_stack_depth - 1:
                    # The type was found at the immediate parent level
                    # This COULD be a new child sequence if it starts with proper key (1/a/A/i)
                    # Otherwise, it might be a citation
                    if _is_valid_sequence_start(subsection_key, subsection_type):
                        # Valid start - this is a new child level
                        level = len(level_stack)
                        parent = level_stack[-1][0] if level_stack else None
                        expected_next[level] = _get_next_in_sequence(subsection_key, subsection_type)
                    else:
                        # Invalid start - return to the sibling level
                        level = found_level
                        level_stack = level_stack[:level]
                        if level > 0:
                            parent = level_stack[level - 1][0]
                        else:
                            parent = None
                        
                        # Enforce sequential ordering
                        if level in expected_next:
                            if subsection_key != expected_next[level]:
                                continue
                            expected_next[level] = _get_next_in_sequence(subsection_key, subsection_type)
                        
                        # Clear expected_next for deeper levels
                        for deeper_level in list(expected_next.keys()):
                            if deeper_level > level:
                                del expected_next[deeper_level]
                else:
                    # Type not found in stack - this is a new type, new child level
                    # MUST validate that it starts at the beginning of the sequence
                    if not _is_valid_sequence_start(subsection_key, subsection_type):
                        # Invalid sequence start - likely a reference or citation, not structure
                        # Skip this subsection (don't add to hierarchy)
                        continue
                    
                    level = len(level_stack)
                    if level > 0:
                        parent = level_stack[-1][0]
                    else:
                        parent = None
                    expected_next[level] = _get_next_in_sequence(subsection_key, subsection_type)
        
        # Update hierarchy
        hierarchy[subsection_key] = {
            'level': level,
            'parent': parent,
            'position': position
        }
        
        # Update stack
        # If same type as current level, replace it; otherwise, extend
        if len(level_stack) > level:
            level_stack[level] = (subsection_key, subsection_type)
            level_stack = level_stack[:level + 1]
        else:
            level_stack.append((subsection_key, subsection_type))
    
    return hierarchy


def parse_hierarchy_with_paths(section_content):
    """
    Robust parser that builds a path-based hierarchy and maps each start-of-line
    structural block to its final (deepest) node path.
    
    Returns:
        tuple(dict, list):
            - hierarchy_by_path: {
                path: { 'level': int, 'parent': str|None, 'position': int,
                        'marker': str }
              }
            - blocks: [ { 'path': str, 'line_index': int, 'first_content': str } ]
    """
    lines = section_content.split('\n')
    # Editorial/citation patterns to exclude from structural parsing
    # Note: Month pattern must be specific to citation format (e.g., "Feb. 14, 1899")
    # NOT generic month mentions (e.g., "the first Monday in February" or "January 1, 1968")
    editorial_citation_patterns = [
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.\s+\d{1,2},?\s*\d{4}',  # Citation dates: Feb. 14, 1899 (requires period after month)
        r'\bPub\. L\.',
        r'\d+ Stat\.',
        r'\bch\.\s*\d+',
        r'\bEDITORIAL NOTES\b',
        r'\bAMENDMENTS\b',
        r'\bDERIVATION\b',
        r'\bSTATUTORY NOTES\b',
        r'\bREFERENCES IN TEXT\b',
        r'\bEFFECTIVE DATE\b',
        r'\bSHORT TITLE\b',
        r'\bHISTORICAL AND REVISION NOTES\b',
        r'\bPRIOR PROVISIONS\b',
        r'\bCODIFICATION\b',
        r'\bREPEALS\b'
    ]
    editorial_regex = re.compile('|'.join(editorial_citation_patterns), re.IGNORECASE)

    # 1) Extract structural blocks at the start of lines
    blocks_raw = []  # [{ 'markers': [keys...], 'line_index': i, 'first_content': str }]
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        if editorial_regex.search(line_clean):
            # editorial lines are not structural blocks
            continue
        remaining = line_clean
        markers = []
        while True:
            m = re.match(r'^\(([a-zA-Z0-9]+)\)', remaining)
            if not m:
                break
            key = m.group(1)
            t = _get_subsection_type(key)
            if t == 'other':
                break
            markers.append((key, t))
            remaining = remaining[m.end():]
            if not remaining.startswith('('):
                break
        if markers:
            blocks_raw.append({
                'markers': markers,
                'line_index': i,
                'first_content': remaining.lstrip()
            })

    if not blocks_raw:
        return {}, []

    # 2) Build hierarchy with path-based keys while walking blocks sequentially
    hierarchy = {}
    blocks_with_paths = []
    level_stack = []  # [(key, type, path)]
    expected_next = {}  # level -> next key str for that level
    position = 0

    for block in blocks_raw:
        last_path_for_block = None
        for idx, (sub_key, sub_type) in enumerate(block['markers']):
            if position == 0 and idx == 0:
                # First node in entire sequence must be valid start
                if not _is_valid_sequence_start(sub_key, sub_type):
                    continue
                level = 0
                parent_path = None
                path = sub_key
                expected_next[level] = _get_next_in_sequence(sub_key, sub_type)
            else:
                # Determine relation vs previous token using stack and expected_next
                prev_type = level_stack[-1][1] if level_stack else None
                if prev_type == sub_type:
                    # Sibling of current level
                    level = len(level_stack) - 1 if level_stack else 0
                    # Enforce sequential ordering at this level
                    if level in expected_next and expected_next[level] is not None:
                        if sub_key != expected_next[level]:
                            # out of order; skip
                            continue
                    # Parent path for this level
                    parent_path = level_stack[level - 1][2] if level > 0 else None
                    path = f"{parent_path}.{sub_key}" if parent_path else sub_key
                    expected_next[level] = _get_next_in_sequence(sub_key, sub_type)
                    # clear deeper expectations
                    for l in list(expected_next.keys()):
                        if l > level:
                            del expected_next[l]
                else:
                    # Different type: either going deeper or moving up to an ancestor that had this type
                    found_level = None
                    for i in range(len(level_stack)):
                        if level_stack[i][1] == sub_type:
                            found_level = i
                            break
                    current_depth = len(level_stack)
                    if found_level is not None and found_level < current_depth - 1:
                        # Jumping back to an ancestor level with same type -> sibling there
                        level = found_level
                        level_stack = level_stack[:level]
                        parent_path = level_stack[level - 1][2] if level > 0 else None
                        # Enforce sequential ordering
                        if level in expected_next and expected_next[level] is not None:
                            if sub_key != expected_next[level]:
                                continue
                        path = f"{parent_path}.{sub_key}" if parent_path else sub_key
                        expected_next[level] = _get_next_in_sequence(sub_key, sub_type)
                        # Clear deeper expectations
                        for l in list(expected_next.keys()):
                            if l > level:
                                del expected_next[l]
                    elif found_level is not None and found_level == current_depth - 1:
                        # Same type as immediate parent level: this could be start of a new child sequence
                        if _is_valid_sequence_start(sub_key, sub_type):
                            level = len(level_stack)
                            parent_path = level_stack[-1][2] if level_stack else None
                            path = f"{parent_path}.{sub_key}" if parent_path else sub_key
                            expected_next[level] = _get_next_in_sequence(sub_key, sub_type)
                        else:
                            # Treat as sibling at that level
                            level = found_level
                            level_stack = level_stack[:level]
                            parent_path = level_stack[level - 1][2] if level > 0 else None
                            if level in expected_next and expected_next[level] is not None:
                                if sub_key != expected_next[level]:
                                    continue
                            path = f"{parent_path}.{sub_key}" if parent_path else sub_key
                            expected_next[level] = _get_next_in_sequence(sub_key, sub_type)
                            for l in list(expected_next.keys()):
                                if l > level:
                                    del expected_next[l]
                    else:
                        # New deeper child type, must start at sequence start
                        if not _is_valid_sequence_start(sub_key, sub_type):
                            continue
                        level = len(level_stack)
                        parent_path = level_stack[-1][2] if level > 0 else None
                        path = f"{parent_path}.{sub_key}" if parent_path else sub_key
                        expected_next[level] = _get_next_in_sequence(sub_key, sub_type)

            # Record/update node in hierarchy
            if path not in hierarchy:
                hierarchy[path] = {
                    'level': level,
                    'parent': parent_path,
                    'position': position,
                    'marker': sub_key
                }
            position += 1

            # Update stack for this level
            if len(level_stack) > level:
                level_stack[level] = (sub_key, sub_type, path)
                level_stack = level_stack[:level + 1]
            else:
                level_stack.append((sub_key, sub_type, path))

            last_path_for_block = path

        if last_path_for_block is not None:
            blocks_with_paths.append({
                'path': last_path_for_block,
                'line_index': block['line_index'],
                'first_content': block['first_content']
            })

    return hierarchy, blocks_with_paths

def build_hierarchy_tree(section_hierarchy, content_map):
    """
    Build a tree structure from the flat hierarchy dictionary.
    
    Args:
        section_hierarchy: Dict from parse_section_hierarchy with level/parent info
        content_map: Dict mapping subsection keys to their content text
    
    Returns:
        List of root nodes, where each node is:
        {
            'key': subsection key (e.g., 'a', '1', 'i'),
            'content': text content for this subsection,
            'level': nesting level,
            'children': list of child nodes (same structure)
        }
    """
    # Build a map of parent -> children
    children_map = {}
    for key, info in section_hierarchy.items():
        parent = info.get('parent')
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append({
            'id': key,  # path or raw key
            'marker': info.get('marker', key),
            'content': content_map.get(key, ''),
            'level': info['level'],
            'position': info['position']
        })
    
    # Sort children by position
    for parent in children_map:
        children_map[parent].sort(key=lambda x: x['position'])
    
    # Recursively build tree
    def build_node_tree(node):
        """Add children to a node recursively"""
        node_key = node['id']
        if node_key in children_map:
            node['children'] = []
            for child in children_map[node_key]:
                child_with_tree = build_node_tree(child)
                node['children'].append(child_with_tree)
        else:
            node['children'] = []
        return node
    
    # Get root nodes (those with parent=None)
    root_nodes = children_map.get(None, [])
    
    # Build full tree for each root
    tree = [build_node_tree(node) for node in root_nodes]
    
    return tree


def render_hierarchy_tree(tree, section_id_prefix="section"):
    """
    Render a hierarchical tree structure as collapsible Dash components.
    
    Args:
        tree: List of tree nodes from build_hierarchy_tree
        section_id_prefix: Prefix for unique IDs for collapse components
    
    Returns:
        List of Dash components representing the tree
    """
    def render_node(node, parent_path=""):
        """Recursively render a tree node and its children"""
        node_id = node['id']
        marker = node.get('marker', node_id)
        content = node['content']
        level = node['level']
        children = node.get('children', [])
        
        # Create unique ID for this node's collapse
        node_path = f"{parent_path}_{node_id}" if parent_path else node_id
        collapse_id = f"{section_id_prefix}_{node_path}"
        button_id = f"{section_id_prefix}_btn_{node_path}"
        
        # Determine indentation based on level
        indent_rem = level * 1.5
        
        # Create the main content display
        has_children = len(children) > 0
        
        # Subsection marker styling
        marker_style = {
            'font-weight': '600',
            'color': '#0d6efd',
            'font-size': '0.95rem'
        }
        
        # Content text styling
        content_style = {
            'color': '#212529',
            'font-size': '0.9rem',
            'margin-left': '0.5rem',
            'display': 'inline'
        }
        
        # Container styling with indentation
        container_style = {
            'margin-left': f'{indent_rem}rem',
            'margin-bottom': '0.5rem',
            'padding': '0.4rem 0.6rem',
            'border-left': '2px solid #e0e0e0' if level > 0 else 'none',
            'padding-left': '0.8rem' if level > 0 else '0'
        }
        
        # Visual indicator for parent items (items with children)
        if has_children:
            indicator = html.Span("▸ ", style={
                'margin-right': '0.3rem',
                'color': '#6c757d',
                'font-size': '0.7rem'
            })
        else:
            indicator = html.Span(style={'display': 'inline-block', 'width': '1rem'})
        
        # Main node content
        node_content = html.Div([
            indicator,
            html.Span(f"({marker})", style=marker_style),
            html.Span(content, style=content_style) if content else html.Span()
        ])
        
        # Recursively render children
        result = [html.Div([node_content], style=container_style)]
        
        if has_children:
            for child in children:
                result.extend(render_node(child, node_path))
        
        return result
    
    # Render all root nodes
    all_components = []
    for root in tree:
        all_components.extend(render_node(root))
    
    return all_components


def prune_tree_to_path(tree, target_path):
    """
    Return only the subtree matching the provided path. If not found, return the original tree.
    """
    if not target_path:
        return tree

    def find_subtree(nodes):
        for node in nodes:
            if node.get('id') == target_path:
                return [node]
            found = find_subtree(node.get('children', []))
            if found:
                return found
        return []

    pruned = find_subtree(tree)
    return pruned if pruned else tree


def _get_subsection_type(key):
    """Get the type of a subsection key"""
    # Explicit list of valid roman numerals to avoid ambiguity
    # Single 'i', 'v', 'x' are roman, but 'c', 'd', 'l', 'm' are lowercase letters
    roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                      'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
                      'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii', 'xxviii', 'xxix', 'xxx']
    
    if key.lower() in roman_numerals:
        return 'roman'
    elif re.match(r'^[a-z]$', key):
        return 'lower'
    elif re.match(r'^[A-Z]$', key):
        return 'upper'
    elif re.match(r'^\d+$', key):
        return 'number'
    else:
        return 'other'

def _is_valid_sequence_start(key, subsection_type):
    """
    Check if this key is a valid starting point for a new sequence.
    New nested levels must start at the beginning of their sequence:
    - Numbers must start with '1'
    - Lowercase letters must start with 'a'
    - Uppercase letters must start with 'A'
    - Roman numerals must start with 'i'
    """
    if subsection_type == 'number':
        return key == '1'
    elif subsection_type == 'lower':
        return key == 'a'
    elif subsection_type == 'upper':
        return key == 'A'
    elif subsection_type == 'roman':
        return key.lower() == 'i'
    else:
        return True  # Unknown types allowed

def _get_next_in_sequence(key, subsection_type):
    """
    Get the next expected key in a sequence for enforcing sibling ordering.
    
    Args:
        key: Current subsection key (e.g., 'a', '1', 'i', 'A')
        subsection_type: Type of the subsection ('lower', 'upper', 'number', 'roman')
    
    Returns:
        str: The next expected key in the sequence, or None if no next key
    """
    if subsection_type == 'number':
        return str(int(key) + 1)
    elif subsection_type == 'lower':
        if key == 'z':
            return None  # End of sequence
        return chr(ord(key) + 1)
    elif subsection_type == 'upper':
        if key == 'Z':
            return None  # End of sequence
        return chr(ord(key) + 1)
    elif subsection_type == 'roman':
        # Roman numeral sequence: i, ii, iii, iv, v, vi, vii, viii, ix, x, etc.
        roman_sequence = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 
                         'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
                         'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii', 'xxviii', 'xxix', 'xxx']
        try:
            idx = roman_sequence.index(key.lower())
            if idx + 1 < len(roman_sequence):
                return roman_sequence[idx + 1]
            else:
                return None  # End of our known sequence
        except ValueError:
            return None  # Unknown roman numeral
    else:
        return None  # Unknown type

def get_parent_section_hierarchy(section_num):
    """
    Extract the parent section number from a section identifier.
    
    For example:
    - "1234" -> "1234" (it is the parent)
    - "1234a" -> "1234a" (section with letter suffix)
    
    This identifies the top-level section number.
    
    Args:
        section_num: Section number string (e.g., "1234", "1234a")
    
    Returns:
        str: Parent section number
    """
    # Extract just the numeric and letter part
    match = re.match(r'^(\d+[A-Za-z]?)', section_num)
    if match:
        return match.group(1)
    return section_num

def find_common_parent_subsection(results, filename):
    """
    Find the smallest common parent subsection that contains all search results.
    
    For example, if results are in (a)(1) and (a)(2), return (a).
    If results are in (a) and (b), return the section itself.
    
    Args:
        results: List of search result dictionaries
        filename: The USC file being searched
    
    Returns:
        dict: {
            'scope_type': 'section' or 'subsection',
            'scope_identifier': section number or subsection key,
            'parent_key': parent subsection if scope is subsection,
            'all_content': full content to download
        }
    """
    if not results:
        return None
    
    # All results should be from the same section
    section_num = results[0].get('section_num', '')
    
    # Load the full section content to analyze hierarchy
    file_path = f"txt/{filename}"
    document = load_document(file_path)
    if not document:
        return None
    
    section_content = extract_full_section(document, section_num)
    section_hierarchy = parse_section_hierarchy(section_content)
    
    # Extract subsection keys from results (if they have subsections mentioned)
    # This is tricky - we need to parse the context to find which subsections match
    result_subsections = set()
    for result in results:
        context = result.get('context', '')
        # Find subsection markers in the context
        subsection_matches = re.findall(r'\(([a-zA-Z0-9]+)\)', context[:200])  # Check first 200 chars
        result_subsections.update(subsection_matches)
    
    if not result_subsections:
        # No subsections identified, return full section
        return {
            'scope_type': 'section',
            'scope_identifier': section_num,
            'parent_key': None,
            'all_content': section_content
        }
    
    # Find the common parent of all result subsections
    common_parent = None
    all_have_same_parent = True
    
    for subsection_key in result_subsections:
        if subsection_key in section_hierarchy:
            parent = section_hierarchy[subsection_key]['parent']
            if common_parent is None:
                common_parent = parent
            elif common_parent != parent:
                all_have_same_parent = False
                break
    
    if all_have_same_parent and common_parent:
        # All results share the same parent subsection
        # Extract content for just that parent subsection
        parent_pattern = rf'\({common_parent}\).*?(?=\([a-zA-Z0-9]+\)(?!.*\({common_parent}\))|(?:^|\n\s*)§\d+[A-Za-z]?\.|$)'
        parent_match = re.search(parent_pattern, section_content, re.DOTALL | re.MULTILINE)
        
        if parent_match:
            return {
                'scope_type': 'subsection',
                'scope_identifier': common_parent,
                'parent_key': section_hierarchy[common_parent].get('parent') if common_parent in section_hierarchy else None,
                'all_content': parent_match.group()
            }
    
    # Default to full section if no common parent found
    return {
        'scope_type': 'section',
        'scope_identifier': section_num,
        'parent_key': None,
        'all_content': section_content
    }

def group_results_by_parent_section(results):
    """
    Group search results by their parent section.
    
    This helps narrow down downloads to only include content within
    the same parent section hierarchy.
    
    Args:
        results: List of search result dictionaries
    
    Returns:
        dict: Dictionary mapping parent section numbers to lists of results
    """
    grouped = {}
    
    for result in results:
        section_num = result.get('section_num', '')
        parent = get_parent_section_hierarchy(section_num)
        
        # Create a composite key: title_num + parent_section
        title_num = result.get('title_num', '')
        key = f"{title_num}§{parent}"
        
        if key not in grouped:
            grouped[key] = {
                'parent_section': parent,
                'title_num': title_num,
                'title_label': result.get('title_label', ''),
                'results': []
            }
        
        grouped[key]['results'].append(result)
    
    return grouped

def filter_content_by_keyword_location(section_content, search_terms, section_num):
    """
    Filter section content to only include subsections where keywords are found.
    
    Args:
        section_content: Full text content of the section
        search_terms: List or tuple of search terms to find
        section_num: Section number for reference
    
    Returns:
        str: Filtered content containing only subsections with keywords
    """
    if not section_content or not search_terms:
        return section_content
    
    # Parse the hierarchy
    hierarchy = parse_section_hierarchy(section_content)
    lines = section_content.split('\n')
    
    # Convert search_terms to list if needed
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    
    # Find which subsections contain keywords
    subsections_with_keywords = set()
    current_subsection = None
    current_subsection_content = []
    subsection_contents = {}  # subsection_key -> content lines
    intro_content = []  # Content before first subsection
    section_header = []  # Section title and header
    in_section_header = True
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Detect section header (first few lines before subsections)
        if line_clean.startswith('§'):
            section_header.append(line)
            in_section_header = True
            continue
        
        # Check for subsection start
        subsection_match = re.match(r'^\(([a-zA-Z0-9]+)\)', line_clean)
        if subsection_match and subsection_match.group(1) in hierarchy:
            # Save previous subsection content
            if current_subsection is not None:
                subsection_contents[current_subsection] = current_subsection_content
            elif in_section_header:
                intro_content = current_subsection_content
                in_section_header = False
            
            # Start new subsection
            current_subsection = subsection_match.group(1)
            current_subsection_content = [line]
        else:
            # Add line to current subsection or intro
            current_subsection_content.append(line)
    
    # Save last subsection
    if current_subsection is not None:
        subsection_contents[current_subsection] = current_subsection_content
    elif in_section_header:
        intro_content = current_subsection_content
    
    # Check which subsections contain keywords
    for subsection_key, content_lines in subsection_contents.items():
        content_text = '\n'.join(content_lines).lower()
        for term in search_terms:
            if str(term).lower() in content_text:
                subsections_with_keywords.add(subsection_key)
                # Also include parent subsections for context
                if subsection_key in hierarchy:
                    parent = hierarchy[subsection_key].get('parent')
                    while parent:
                        subsections_with_keywords.add(parent)
                        parent = hierarchy[parent].get('parent') if parent in hierarchy else None
                break
    
    # Also check intro content
    intro_has_keyword = False
    if intro_content:
        intro_text = '\n'.join(intro_content).lower()
        for term in search_terms:
            if str(term).lower() in intro_text:
                intro_has_keyword = True
                break
    
    # If no subsections found, return original content
    if not subsections_with_keywords and not intro_has_keyword:
        return section_content
    
    # Build filtered content
    filtered_lines = []
    filtered_lines.extend(section_header)
    
    # Add intro if it has keywords or if we're including subsections
    if intro_content and (intro_has_keyword or subsections_with_keywords):
        filtered_lines.extend(intro_content)
    
    # Add only subsections with keywords (maintaining hierarchy)
    # Sort by position to maintain order
    sorted_subsections = sorted(
        [(key, hierarchy[key]['position']) for key in subsections_with_keywords if key in hierarchy],
        key=lambda x: x[1]
    )
    
    for subsection_key, _ in sorted_subsections:
        if subsection_key in subsection_contents:
            filtered_lines.extend(subsection_contents[subsection_key])
    
    return '\n'.join(filtered_lines)

def generate_word_search_txt_chunks(results, search_term, max_tokens=100000):
    """
    Generate text file chunks containing word search results, split at approximately 100k tokens.
    
    Args:
        results: List of search result dictionaries
        search_term: The search term used
        max_tokens: Maximum tokens per chunk (default 100,000)
    
    Returns:
        list: List of text content chunks for download
    """
    if not results:
        return [f"No results found for search term: '{search_term}'"]
    
    # Create header template
    def create_header(chunk_num, total_chunks, total_sections, parent_sections_info=""):
        header = []
        header.append("=" * 80)
        header.append(f"US CODE SEARCH RESULTS")
        header.append(f"Search Term: '{search_term}'")
        header.append(f"Total Sections Found: {total_sections}")
        if parent_sections_info:
            header.append(f"Organized by Parent Section Hierarchy")
            header.append(parent_sections_info)
        header.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if total_chunks > 1:
            header.append(f"Chunk: {chunk_num} of {total_chunks}")
        header.append("=" * 80)
        header.append("")
        return header
    
    # Group results by parent section hierarchy
    grouped_by_parent = group_results_by_parent_section(results)
    
    # Create a summary of parent sections with scope information
    parent_sections_list = []
    parent_sections_list.append("DOWNLOAD SCOPE: Key Results Only - Content filtered to subsections containing keywords")
    parent_sections_list.append("")
    
    for key, group_data in grouped_by_parent.items():
        parent = group_data['parent_section']
        title_label = group_data['title_label']
        count = len(group_data['results'])
        parent_sections_list.append(f"  • {title_label} §{parent}: {count} results")
        parent_sections_list.append(f"    (only subsections with keyword matches)")
    
    parent_sections_info = "\n".join(parent_sections_list)
    
    # Sort parent section groups for consistent output
    sorted_parent_keys = sorted(grouped_by_parent.keys())
    
    # Prepare all content sections organized by parent section
    all_sections = []
    for parent_key in sorted_parent_keys:
        group_data = grouped_by_parent[parent_key]
        parent_section = group_data['parent_section']
        title_label = group_data['title_label']
        parent_results = group_data['results']
        
        # Sort sections within each parent by section number
        parent_results.sort(key=lambda x: (
            int(re.search(r'\d+', x['section_num']).group()) if re.search(r'\d+', x['section_num']) else 0,
            x['section_num']
        ))
        
        # Add parent section header
        parent_header = [
            f"\n{'=' * 60}",
            f"PARENT SECTION: {title_label} §{parent_section}",
            f"Results in this hierarchy: {len(parent_results)}",
            f"{'=' * 60}",
            ""
        ]
        all_sections.append(('parent_header', parent_header))
        
        # Add each section within this parent
        for result in parent_results:
            section_content = result['context']
            # Remove ** highlighting markers
            section_content = re.sub(r'\*\*(.*?)\*\*', r'\1', section_content)
            
            # Filter content to only subsections containing the keywords
            search_terms_for_filter = result.get('search_terms', [])
            if not search_terms_for_filter:
                # Try to extract search terms from the search_term string
                if '&' in search_term:
                    search_terms_for_filter = [t.strip() for t in search_term.split('&')]
                else:
                    search_terms_for_filter = [search_term]
            
            filtered_content = filter_content_by_keyword_location(
                section_content, 
                search_terms_for_filter,
                result['section_num']
            )
            
            section_data = [
                f"\n{'-' * 40}",
                f"Section: §{result['section_num']}",
                f"Title: {result['section_title']}",
                f"Occurrences: {result.get('occurrence_count', 1)}",
                f"Note: Showing only subsections containing keywords",
                f"{'-' * 40}",
                "",
                filtered_content,
                ""
            ]
            all_sections.append(('section', section_data))
    
    # Create chunks
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_num = 1
    
    # Add header to first chunk
    header = create_header(1, 1, len(results), parent_sections_info)  # Will update total_chunks later
    current_chunk.extend(header)
    current_tokens += estimate_tokens('\n'.join(header))
    
    for section_type, section_data in all_sections:
        section_text = '\n'.join(section_data)
        section_tokens = estimate_tokens(section_text)
        
        # Check if adding this section would exceed the limit
        if current_tokens + section_tokens > max_tokens and current_chunk:
            # Finalize current chunk
            current_chunk.extend([
                "\n" + "=" * 80,
                "End of Chunk",
                "=" * 80
            ])
            chunks.append('\n'.join(current_chunk))
            
            # Start new chunk
            chunk_num += 1
            current_chunk = []
            current_tokens = 0
            
            # Add header to new chunk
            header = create_header(chunk_num, 1, len(results), parent_sections_info)  # Will update total_chunks later
            current_chunk.extend(header)
            current_tokens += estimate_tokens('\n'.join(header))
        
        # Add section to current chunk
        current_chunk.extend(section_data)
        current_tokens += section_tokens
    
    # Add final chunk if there's content
    if current_chunk:
        current_chunk.extend([
            "\n" + "=" * 80,
            "End of Search Results",
            "=" * 80
        ])
        chunks.append('\n'.join(current_chunk))
    
    # Update headers with correct total chunk count
    total_chunks = len(chunks)
    if total_chunks > 1:
        updated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            # Replace the header in each chunk
            lines = chunk.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith("Chunk: ") and " of " in line:
                    new_lines.append(f"Chunk: {i} of {total_chunks}")
                else:
                    new_lines.append(line)
            updated_chunks.append('\n'.join(new_lines))
        return updated_chunks
    
    return chunks

def generate_word_search_txt(results, search_term):
    """
    Generate a text file containing all word search results.
    This is a wrapper that returns the first chunk for backward compatibility.
    
    Args:
        results: List of search result dictionaries
        search_term: The search term used
    
    Returns:
        str: Formatted text content for download
    """
    chunks = generate_word_search_txt_chunks(results, search_term)
    return chunks[0] if chunks else f"No results found for search term: '{search_term}'"

def create_chunks_zip(chunks, search_term):
    """
    Create a ZIP file containing all chunks.
    
    Args:
        chunks: List of chunk text content
        search_term: The search term used
    
    Returns:
        bytes: ZIP file content
    """
    # Create a BytesIO object to hold the ZIP file
    zip_buffer = io.BytesIO()
    
    # Create filename base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_search_term = re.sub(r'[^\w\s-]', '', search_term).strip()
    safe_search_term = re.sub(r'[-\s]+', '_', safe_search_term)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, chunk in enumerate(chunks, 1):
            filename = f"usc_search_{safe_search_term}_chunk{i}_{timestamp}.txt"
            zip_file.writestr(filename, chunk)
        
        # Add a README file
        readme_content = f"""US Code Search Results - {search_term}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Chunks: {len(chunks)}

This ZIP file contains {len(chunks)} text files with KEY RESULTS for the term "{search_term}".
Each file contains approximately 100,000 tokens for optimal readability and processing.

IMPORTANT - KEY RESULTS SCOPE:
These downloads contain ONLY the subsections where your keywords were found,
not entire sections. This significantly reduces file size while preserving all
relevant content. Parent subsections are included for context when needed.

Files included:
"""
        for i in range(1, len(chunks) + 1):
            readme_content += f"- usc_search_{safe_search_term}_chunk{i}_{timestamp}.txt\n"
        
        readme_content += f"""
Usage:
- Extract all files to access individual chunks
- Each chunk is a complete, standalone text file
- Chunks are organized by USC Title and Section
- Only subsections containing keywords are included
- Search term occurrences are preserved in context

For questions or issues, refer to the original USC Search Dashboard.
"""
        
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def word_search_across_documents(search_terms, title_filter=None, max_results=20000):
    """
    Search for a set of words/phrases across USC documents.
    
    Uses SQLite database for lightning-fast search when available,
    falls back to file scanning if SQLite is not available.
    """
    if not search_terms:
        return []
    
    # Try SQLite search first (much faster)
    if SEARCH_DB_AVAILABLE:
        try:
            # Convert search_terms to a single string if needed
            if isinstance(search_terms, (list, tuple)):
                search_text = " ".join(str(term) for term in search_terms)
            else:
                search_text = str(search_terms)
            
            # Use SQLite search
            start_time = datetime.now()
            results = search_database.simple_search(
                search_text=search_text, 
                title_filter=title_filter if title_filter != "all" else None,
                max_results=max_results
            )
            
            end_time = datetime.now()
            search_duration = (end_time - start_time).total_seconds()
            print(f"🚀 SQLite search completed in {search_duration:.3f}s - found {len(results)} results")
            
            return results
            
        except Exception as e:
            print(f"SQLite search failed, falling back to file search: {e}")
            # Fall through to file-based search
    
    # Fallback to original file-based search
    print("⚠️  Using slower file-based search...")
    search_term = search_terms # Keep variable name for now, but it's a tuple
    
    # Get list of files to search
    if title_filter and title_filter != "all":
        files_to_search = [f"txt/{title_filter}"]
    else:
        files_to_search = glob.glob("txt/usc*.txt")
    
    # Sort files by size (smaller first) for faster initial results
    try:
        files_to_search.sort(key=lambda x: os.path.getsize(x))
    except:
        pass  # If file size check fails, continue with original order
    
    all_results = []
    section_keys_seen = set()
    
    # Try parallel processing first, fall back to sequential if it fails
    try:
        # Check if we're in an environment that supports multiprocessing
        if hasattr(os, 'fork') or os.name == 'nt':  # Unix-like or Windows
            # Determine number of workers (cap at 6 for PythonAnywhere compatibility)
            num_workers = min(mp.cpu_count(), 6, len(files_to_search))
            
            # Only use multiprocessing if we have multiple files and workers
            if len(files_to_search) > 1 and num_workers > 1:
                # Prepare arguments for parallel processing
                search_args = [(file_path, search_terms) for file_path in files_to_search]
                
                # Use ProcessPoolExecutor for parallel file processing
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all file search tasks
                    future_to_file = {executor.submit(search_single_file, args): args[0] for args in search_args}
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_file):
                        try:
                            file_results = future.result()
                            
                            # Add unique results (deduplicate across files)
                            for result in file_results:
                                section_key = (result['filename'], result['section_num'])
                                if section_key not in section_keys_seen:
                                    section_keys_seen.add(section_key)
                                    all_results.append(result)
                                    
                                    # Stop early if we have enough results
                                    if len(all_results) >= max_results:
                                        break
                            
                            # Break if we have enough results
                            if len(all_results) >= max_results:
                                break
                                
                        except Exception as e:
                            file_path = future_to_file[future]
                            print(f"Error processing {file_path}: {e}")
            else:
                # Single file or single worker - use sequential
                all_results = word_search_sequential(search_terms, files_to_search, max_results)
        else:
            # Environment doesn't support multiprocessing
            all_results = word_search_sequential(search_terms, files_to_search, max_results)
            
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        # Fallback to sequential processing
        all_results = word_search_sequential(search_terms, files_to_search, max_results)
    
    # Sort results by title and section number
    all_results.sort(key=lambda x: (
        int(re.search(r'\d+', x['title_num']).group()) if re.search(r'\d+', x['title_num']) else 0,
        int(re.search(r'\d+', x['section_num']).group()) if re.search(r'\d+', x['section_num']) else 0
    ))
    
    return all_results[:max_results]

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "USC Search Dashboard"

# Pre-load the USC index for faster performance
print("Starting USC Search Dashboard...")
load_usc_index()

# Get available titles
available_titles = get_available_titles()

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("US Code Search Dashboard", className="text-center mb-4"),
            html.P("Search through United States Code sections and subsections with auto-complete", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Main content
    dbc.Row([
        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Search Controls")),
                dbc.CardBody([
                    # Title selection
                    html.Label("Select USC Title:", className="fw-bold"),
                    dcc.Dropdown(
                        id='title-dropdown',
                        options=available_titles,
                        value=available_titles[0]['value'] if available_titles else None,
                        placeholder="Select a USC Title...",
                        className="mb-3"
                    ),
                    
                    # Section selection with auto-complete
                    html.Label("Select Section:", className="fw-bold"),
                    dcc.Dropdown(
                        id='section-dropdown',
                        placeholder="Choose a section...",
                        className="mb-3",
                        disabled=True
                    ),
                    
                    # Subsection selection
                    html.Label("Select Subsection (Optional):", className="fw-bold"),
                    dcc.Dropdown(
                        id='subsection-dropdown',
                        placeholder="Choose a subsection...",
                        className="mb-3",
                        disabled=True,
                        multi=True
                    ),
                    
                    # Manual search input (alternative)
                    html.Hr(),
                    html.Label("Or Enter Manually:", className="fw-bold"),
                    html.P("Examples: '12', '12 (a)', '12 (a)(1)'", className="text-muted small"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='search-input',
                            placeholder="Enter section number...",
                            type="text"
                        ),
                        dbc.Button(
                            "Search",
                            id='search-button',
                            color="primary",
                            n_clicks=0
                        )
                    ], className="mb-3"),
                    
                    # Navigation buttons
                    html.Hr(),
                    html.Label("Navigation:", className="fw-bold"),
                    dbc.ButtonGroup([
                        dbc.Button(
                            [html.I(className="fas fa-arrow-left me-1"), "Back"],
                            id='nav-back-button',
                            color="secondary",
                            size="sm",
                            disabled=True
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-arrow-right me-1"), "Forward"],
                            id='nav-forward-button',
                            color="secondary",
                            size="sm",
                            disabled=True
                        )
                    ], className="mb-3 d-grid"),
                    
                    # Quick search examples
                    html.Label("Quick Examples:", className="fw-bold"),
                    html.Div([
                        dbc.Button("§1", id="example-1", size="sm", color="outline-secondary", className="me-2 mb-2"),
                        dbc.Button("§2", id="example-2", size="sm", color="outline-secondary", className="me-2 mb-2"),
                        dbc.Button("§3 (a)", id="example-3", size="sm", color="outline-secondary", className="me-2 mb-2"),
                    ]),
                    
                    # Hotkey help
                    html.Hr(),
                    html.Label("⌨️ Keyboard Shortcuts:", className="fw-bold text-primary"),
                    html.Div([
                        html.P([
                            html.Kbd("↑↓", className="me-2"),
                            html.Span("Navigate titles", className="small text-muted")
                        ], className="mb-1"),
                        html.P([
                            html.Kbd("←→", className="me-2"), 
                            html.Span("Navigate sections", className="small text-muted")
                        ], className="mb-1"),
                        html.P([
                            html.Small("Searches automatically!", className="text-success fst-italic")
                        ], className="mb-1"),
                        html.P([
                            html.Small(id="hotkey-status", children="Press arrow keys to test...", className="text-muted")
                        ], className="mb-0")
                    ], className="p-2 bg-light rounded")
                ])
            ]),
            
            # Word Search Card - Advanced Search
            dbc.Card([
                dbc.CardHeader([
                    html.H4("🔍 Advanced Search", className="mb-0 d-inline"),
                    dbc.Button(
                        [html.I(className="fas fa-book-open me-1"), "Guide"],
                        id="search-syntax-help-btn",
                        color="outline-info",
                        size="sm",
                        className="ms-2",
                    )
                ]),
                dbc.CardBody([
                    html.Label("Search for words/phrases:", className="fw-bold"),
                    html.P([
                        "Supports ",
                        html.Span("phrases", className="text-primary"),
                        ", ",
                        html.Span("OR", className="text-success"),
                        ", ",
                        html.Span("NOT", className="text-danger"),
                        ", ",
                        html.Span("wildcards", className="text-warning"),
                        " & ",
                        html.Span("proximity", className="text-info")
                    ], className="text-muted small"),
                    
                    # Word search input with loading indicator
                    dbc.InputGroup([
                        dbc.Input(
                            id='word-search-input',
                            placeholder='e.g., "federal agency" OR regulation*',
                            type="text"
                        ),
                        dbc.Button(
                            html.Span("Search", id="word-search-button-text"),
                            id='word-search-button',
                            color="success",
                            n_clicks=0
                        )
                    ], className="mb-2"),
                    
                    # Parsed query display (shows how query was interpreted)
                    html.Div(id="parsed-query-display", className="mb-3"),
                    
                    # Search scope selection
                    html.Label("Search Scope:", className="fw-bold"),
                    dcc.RadioItems(
                        id='search-scope',
                        options=[
                            {'label': ' All USC Titles', 'value': 'all'},
                            {'label': ' Current Title Only', 'value': 'current'}
                        ],
                        value='all',
                        className="mb-3",
                        labelStyle={'display': 'block', 'marginBottom': '5px'}
                    ),
                    
                    # Search options
                    dbc.Checklist(
                        options=[
                            {"label": "Case sensitive", "value": "case_sensitive"},
                            {"label": "Whole words only", "value": "whole_words"}
                        ],
                        value=[],
                        id="search-options",
                        className="mb-3"
                    )
                ])
            ], className="mt-3"),
            
            # Advanced Search Help Modal
            dbc.Modal([
                dbc.ModalHeader([
                    dbc.ModalTitle([
                        html.I(className="fas fa-search me-2 text-primary"),
                        "Advanced Search Guide"
                    ])
                ], close_button=True),
                dbc.ModalBody([
                    # Quick Reference Section
                    html.H5([
                        html.I(className="fas fa-bolt me-2 text-warning"),
                        "Quick Reference"
                    ], className="border-bottom pb-2 mb-3"),
                    
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Syntax", style={"width": "30%"}),
                                html.Th("Description"),
                                html.Th("Example", style={"width": "35%"})
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(html.Code('"exact phrase"', className="bg-light px-2 py-1 rounded")),
                                html.Td("Match exact phrase"),
                                html.Td(dbc.Button('"federal agency"', id="help-example-phrase", size="sm", color="outline-primary", className="py-0"))
                            ]),
                            html.Tr([
                                html.Td(html.Code('word1 OR word2', className="bg-light px-2 py-1 rounded")),
                                html.Td("Match either term"),
                                html.Td(dbc.Button('highway OR interstate', id="help-example-or", size="sm", color="outline-primary", className="py-0"))
                            ]),
                            html.Tr([
                                html.Td(html.Code('-word', className="bg-light px-2 py-1 rounded")),
                                html.Td("Exclude term from results"),
                                html.Td(dbc.Button('criminal -misdemeanor', id="help-example-not", size="sm", color="outline-primary", className="py-0"))
                            ]),
                            html.Tr([
                                html.Td(html.Code('word*', className="bg-light px-2 py-1 rounded")),
                                html.Td("Wildcard prefix matching"),
                                html.Td(dbc.Button('regulat*', id="help-example-wildcard", size="sm", color="outline-primary", className="py-0"))
                            ]),
                            html.Tr([
                                html.Td(html.Code('word1 NEAR word2', className="bg-light px-2 py-1 rounded")),
                                html.Td("Terms within 10 words"),
                                html.Td(dbc.Button('tax NEAR fraud', id="help-example-near", size="sm", color="outline-primary", className="py-0"))
                            ]),
                            html.Tr([
                                html.Td(html.Code('NEAR/N', className="bg-light px-2 py-1 rounded")),
                                html.Td("Terms within N words"),
                                html.Td(dbc.Button('due NEAR/3 process', id="help-example-near-n", size="sm", color="outline-primary", className="py-0"))
                            ]),
                            html.Tr([
                                html.Td(html.Code('title:N', className="bg-light px-2 py-1 rounded")),
                                html.Td("Filter by USC title"),
                                html.Td(dbc.Button('fraud title:18', id="help-example-title", size="sm", color="outline-primary", className="py-0"))
                            ]),
                        ])
                    ], bordered=True, hover=True, size="sm", className="mb-4"),
                    
                    # How It Works Section
                    html.H5([
                        html.I(className="fas fa-cogs me-2 text-info"),
                        "How It Works"
                    ], className="border-bottom pb-2 mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6([html.I(className="fas fa-check-circle text-success me-2"), "Default Behavior"], className="mb-2"),
                                    html.P("Multiple words without operators require ALL terms to match.", className="small mb-2"),
                                    html.Code("contract breach → finds both words", className="d-block bg-light p-2 rounded small")
                                ])
                            ], className="h-100")
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6([html.I(className="fas fa-quote-left text-primary me-2"), "Phrase Search"], className="mb-2"),
                                    html.P("Wrap in quotes for exact phrase matching.", className="small mb-2"),
                                    html.Code('"breach of contract" → exact phrase', className="d-block bg-light p-2 rounded small")
                                ])
                            ], className="h-100")
                        ], md=6, className="mb-3"),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6([html.I(className="fas fa-code-branch text-warning me-2"), "OR Logic"], className="mb-2"),
                                    html.P("Use OR (case-insensitive) between alternatives.", className="small mb-2"),
                                    html.Code("highway OR freeway OR interstate", className="d-block bg-light p-2 rounded small")
                                ])
                            ], className="h-100")
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6([html.I(className="fas fa-ban text-danger me-2"), "Exclusion"], className="mb-2"),
                                    html.P("Prefix with - or use NOT to exclude terms.", className="small mb-2"),
                                    html.Code("tax -income  OR  tax NOT income", className="d-block bg-light p-2 rounded small")
                                ])
                            ], className="h-100")
                        ], md=6, className="mb-3"),
                    ]),
                    
                    # Pro Tips Section
                    html.H5([
                        html.I(className="fas fa-lightbulb me-2 text-warning"),
                        "Pro Tips"
                    ], className="border-bottom pb-2 mb-3"),
                    
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Strong("Combine operators: ", className="text-primary"),
                            html.Code('"federal agency" OR "government agency" -state', className="ms-2")
                        ], className="py-2"),
                        dbc.ListGroupItem([
                            html.Strong("Use wildcards for variations: ", className="text-primary"),
                            html.Code('regulat* → regulate, regulation, regulatory, regulator...', className="ms-2")
                        ], className="py-2"),
                        dbc.ListGroupItem([
                            html.Strong("Narrow with title filter: ", className="text-primary"),
                            html.Code('"due process" title:42 → searches only Title 42', className="ms-2")
                        ], className="py-2"),
                        dbc.ListGroupItem([
                            html.Strong("Find related concepts: ", className="text-primary"),
                            html.Code('environment* NEAR/5 protect* → environmental protection, etc.', className="ms-2")
                        ], className="py-2"),
                    ], flush=True, className="mb-4"),
                    
                    # Common USC Titles Reference
                    html.H5([
                        html.I(className="fas fa-bookmark me-2 text-secondary"),
                        "Common USC Title Codes"
                    ], className="border-bottom pb-2 mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Ul([
                                html.Li([html.Strong("title:5"), " - Government Organization"]),
                                html.Li([html.Strong("title:11"), " - Bankruptcy"]),
                                html.Li([html.Strong("title:15"), " - Commerce & Trade"]),
                                html.Li([html.Strong("title:17"), " - Copyrights"]),
                            ], className="small mb-0")
                        ], md=4),
                        dbc.Col([
                            html.Ul([
                                html.Li([html.Strong("title:18"), " - Crimes"]),
                                html.Li([html.Strong("title:26"), " - Tax Code (IRC)"]),
                                html.Li([html.Strong("title:28"), " - Judiciary"]),
                                html.Li([html.Strong("title:29"), " - Labor"]),
                            ], className="small mb-0")
                        ], md=4),
                        dbc.Col([
                            html.Ul([
                                html.Li([html.Strong("title:35"), " - Patents"]),
                                html.Li([html.Strong("title:42"), " - Public Health"]),
                                html.Li([html.Strong("title:47"), " - Telecommunications"]),
                                html.Li([html.Strong("title:49"), " - Transportation"]),
                            ], className="small mb-0")
                        ], md=4),
                    ])
                ]),
                dbc.ModalFooter([
                    html.Small("Click any example to try it!", className="text-muted me-auto"),
                    dbc.Button("Close", id="close-search-help-modal", color="secondary")
                ])
            ], id="search-help-modal", size="lg", scrollable=True)
        ], width=4),
        
        # Results area
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Search Results", className="mb-0"),
                    html.Small(id="search-info", className="text-muted")
                ]),
                dbc.CardBody([
                    # Simple tabs to organize output
                    dbc.Tabs([
                        dbc.Tab(label="📄 Search Results", tab_id="results-tab"),
                        dbc.Tab(label="📋 Section Structure", tab_id="structure-tab"),
                        dbc.Tab(label="🔗 References", tab_id="references-tab"),
                        dbc.Tab(label="🔍 Word Search", tab_id="word-search-tab"),
                        dbc.Tab(label="🤖 AI Assistant", tab_id="ai-assistant-tab") if AI_AVAILABLE else None,
                    ], id="result-tabs", active_tab="structure-tab", className="mb-3"),
                    
                    # Content area for tabs
                    dcc.Loading(
                        id="loading",
                        children=[html.Div(id="tab-content")],
                        type="default"
                    )
                ])
            ])
        ], width=8)
    ]),
    
    # Search history
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Recent Searches")),
                dbc.CardBody([
                    html.Div(id="search-history")
                ])
            ], className="mt-4")
        ])
    ]),
    
    # Hidden store for cross-reference clicks
    dcc.Store(id='ref-click'),
    
    # Navigation history store
    dcc.Store(id='nav-history', data={'history': [], 'current_index': -1}),
    
    # Store for search results data
    dcc.Store(id='search-results-store'),
    
    # Store for word search results
    dcc.Store(id='word-search-results-store'),
    
    # Store for word search loading state
    dcc.Store(id='word-search-loading-store', data=False),
    
    # Store for word search chunks
    dcc.Store(id='word-search-chunks-store'),
    
    # Store for hotkey navigation state
    dcc.Store(id='hotkey-navigation-store', data={'current_title_index': 0, 'current_section_index': 0}),
    
    # Store for AI assistant responses (only if AI available)
    dcc.Store(id='ai-response-store') if AI_AVAILABLE else html.Div(),
    
    # Store for AI conversation history (only if AI available)
    dcc.Store(id='ai-conversation-store', data={'messages': [], 'section_content': '', 'section_title': ''}) if AI_AVAILABLE else html.Div(),
    
    # Download component for word search results
    dcc.Download(id="download-word-search-file"),
    
    # Hidden buttons for hotkey simulation
    html.Div([
        dbc.Button(id='hotkey-up', n_clicks=0, style={'display': 'none'}),
        dbc.Button(id='hotkey-down', n_clicks=0, style={'display': 'none'}),
        dbc.Button(id='hotkey-left', n_clicks=0, style={'display': 'none'}),
        dbc.Button(id='hotkey-right', n_clicks=0, style={'display': 'none'}),
    ])
], fluid=True)

# Store for search history
search_history = []

# Helper function to add word search to navigation history
def _add_word_search_to_nav_history(nav_data, word_search_data, search_query, search_scope, current_title):
    """Add a word search to navigation history with complete state for instant restoration."""
    import datetime
    
    if nav_data is None:
        nav_data = {'history': [], 'current_index': -1}
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    # Create navigation entry with full state
    current_search = {
        'type': 'word',
        'search_data': None,  # No section search data
        'word_search_data': word_search_data,
        'active_tab': 'word-search-tab',
        'form_state': {
            'title': current_title or '',
            'search_input': '',
            'word_search_input': search_query,
            'search_scope': search_scope
        },
        'timestamp': timestamp
    }
    
    nav_history = nav_data.get('history', [])
    current_index = nav_data.get('current_index', -1)
    
    # Remove any items after current index (if user navigated back then did new search)
    if current_index < len(nav_history) - 1:
        nav_history = nav_history[:current_index + 1]
    
    nav_history.append(current_search)
    current_index = len(nav_history) - 1
    
    # Keep only last 20 items
    if len(nav_history) > 20:
        nav_history = nav_history[-20:]
        current_index = len(nav_history) - 1
    
    return {
        'history': nav_history,
        'current_index': current_index
    }

# Callback to populate sections when title is selected
@app.callback(
    [Output('section-dropdown', 'options'),
     Output('section-dropdown', 'disabled')],
    [Input('title-dropdown', 'value')]
)
def update_sections(selected_title):
    if not selected_title:
        return [], True
    
    sections = get_document_sections(selected_title)
    section_options = []
    
    for section_num, section_data in sections.items():
        section_options.append({
            'label': f"§{section_num} - {section_data['title']}",
            'value': section_num
        })
    
    return section_options, False

# Callback to populate subsections when section is selected
@app.callback(
    [Output('subsection-dropdown', 'options'),
     Output('subsection-dropdown', 'disabled'),
     Output('subsection-dropdown', 'value')],
    [Input('section-dropdown', 'value'),
     Input('title-dropdown', 'value')]
)
def update_subsections(selected_section, selected_title):
    if not selected_section or not selected_title:
        return [], True, None
    
    sections = get_document_sections(selected_title)
    if selected_section not in sections:
        return [], True, None
    
    subsections = sections[selected_section]['subsections']
    subsection_options = []
    
    for sub_key, sub_desc in subsections.items():
        subsection_options.append({
            'label': f"({sub_key}) {sub_desc}",
            'value': sub_key
        })
    
    return subsection_options, False, None

# Handle cross-reference clicks
@app.callback(
    [Output('ref-click', 'data'),
     Output('title-dropdown', 'value')],
    [Input({'type': 'usc-ref', 'title': ALL, 'section': ALL, 'file': ALL, 'index': ALL}, 'n_clicks')],
    [State('title-dropdown', 'value')],
    prevent_initial_call=True
)
def on_ref_click(n_clicks_list, current_title_value):
    ctx = dash.callback_context
    print(f"DEBUG: Cross-ref callback triggered: {ctx.triggered}")
    print(f"DEBUG: n_clicks_list: {n_clicks_list}")
    
    if not ctx.triggered:
        return no_update, no_update
    
    # Check if any clicks occurred
    if not n_clicks_list or not any(n_clicks_list):
        print("DEBUG: No clicks detected")
        return no_update, no_update
    
    # Find which button was actually clicked by checking n_clicks > 0
    for i, clicks in enumerate(n_clicks_list):
        if clicks and clicks > 0:
            # Get the input from the context to find the component ID
            inputs = ctx.inputs_list[0]  # Get the input list for our Input pattern
            if i < len(inputs):
                triggered_id = inputs[i]['id']
                print(f"DEBUG: Found clicked reference at index {i}: {triggered_id}")
                
                target_title_num = triggered_id.get('title')
                target_section = triggered_id.get('section')
                target_file = triggered_id.get('file')
                
                print(f"DEBUG: Target - title: {target_title_num}, section: {target_section}, file: {target_file}")
                
                if not target_section:
                    continue
                
                data = {
                    'filename': target_file,
                    'title_num': target_title_num,
                    'section': target_section,
                    'timestamp': datetime.now().isoformat()  # Force update
                }
                
                print(f"DEBUG: Returning data: {data}")
                # Update title dropdown to the file containing the target
                return data, target_file
    
    print("DEBUG: No valid clicked reference found")
    return no_update, no_update

# Callback to handle tab switching
@app.callback(
    Output('tab-content', 'children'),
    [Input('result-tabs', 'active_tab'),
     Input('search-results-store', 'data'),
     Input('word-search-results-store', 'data')]
)
def render_tab_content(active_tab, search_data, word_search_data):
    try:
        # Handle None or empty active_tab
        if not active_tab:
            active_tab = "structure-tab"
        
        if not search_data:
            # Initial state - show welcome message
            welcome_msg = html.Div([
                html.H5("👋 Welcome to USC Search!", className="text-primary mb-3"),
                html.P("Select a title and section from the sidebar to begin exploring the US Code."),
                html.P("You can also manually enter a section number in the search box.", className="text-muted")
            ], className="text-center mt-5")
            return welcome_msg
        
        if active_tab == "results-tab":
            result = search_data.get('formatted_result')
            if result is None:
                return html.P("No results available.", className="text-muted")
            return result
        
        elif active_tab == "structure-tab":
            structure = search_data.get('structure_content')
            if structure is None:
                return html.P("No structure information available.", className="text-muted")
            return structure
        
        elif active_tab == "references-tab":
            references = search_data.get('references_content')
            if references is None:
                return html.P("No references found.", className="text-muted")
            return references
        
        elif active_tab == "word-search-tab":
            if word_search_data is None or not word_search_data.get('results'):
                return html.Div([
                    html.H5("💡 Word Search", className="text-success mb-3"),
                    html.P("Use the Word Search section in the sidebar to search for words or phrases across all USC documents."),
                    html.P("Examples: 'highway', 'federal agency', 'criminal procedure'", className="text-muted")
                ], className="text-center mt-5")
            
            # Check if we have chunk data to show chunk download options
            chunks_data = word_search_data.get('chunks_data')
            if chunks_data and chunks_data.get('chunks'):
                # Show chunk download interface
                chunks = chunks_data['chunks']
                search_term = chunks_data['search_term']
                total_chunks = chunks_data['total_chunks']
                
                chunk_buttons = []
                for i, chunk in enumerate(chunks, 1):
                    chunk_buttons.append(
                        dbc.Button(
                            [html.I(className="fas fa-download me-2"), f"Download Chunk {i}"],
                            id={
                                'type': 'chunk-download',
                                'chunk_index': i - 1,
                                'index': i
                            },
                            color="outline-primary",
                            size="sm",
                            className="me-2 mb-2"
                        )
                    )
                
                return html.Div([
                    html.H5(f"📦 Chunked Download: '{search_term}'", className="text-primary mb-3"),
                    html.P(f"Results split into {total_chunks} chunks of ~100k tokens each", className="text-muted"),
                    
                    # Download all chunks button
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("Download Options", className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Button(
                                [html.I(className="fas fa-file-archive me-2"), f"Download All {total_chunks} Chunks (ZIP)"],
                                id={
                                    'type': 'download-all-chunks',
                                    'search_term': search_term,
                                    'index': 0
                                },
                                color="success",
                                size="sm",
                                className="mb-3 w-100"
                            ),
                            html.P([
                                html.I(className="fas fa-info-circle me-2"),
                                f"Downloads a ZIP file containing all {total_chunks} chunks plus a README file."
                            ], className="text-muted small mb-0")
                        ])
                    ], className="mb-4"),
                    
                    # Individual chunk downloads
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("Download Individual Chunks", className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Div(chunk_buttons),
                            html.Hr(),
                            html.P([
                                html.I(className="fas fa-info-circle me-2"),
                                "Each chunk contains approximately 100,000 tokens for optimal readability and processing."
                            ], className="text-muted small mb-0")
                        ])
                    ], className="mb-4"),
                    
                    # Show preview of first chunk
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("Preview: Chunk 1", className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Pre(
                                chunks[0][:1000] + "..." if len(chunks[0]) > 1000 else chunks[0],
                                className="bg-light p-3 rounded",
                                style={'max-height': '300px', 'overflow-y': 'auto', 'font-size': '0.8rem'}
                            )
                        ])
                    ])
                ])
            
            results = word_search_data.get('results', [])
            search_term = word_search_data.get('search_term', '')
            total_results = len(results)
            
            if total_results == 0:
                return html.Div([
                    dbc.Alert([
                        html.H5("No Results Found", className="alert-heading"),
                        html.P(f"No sections found containing '{search_term}'"),
                        html.P("Try different keywords or check spelling.", className="mb-0")
                    ], color="info")
                ])
            
            # Group results by title for better organization
            results_by_title = {}
            for result in results:
                title_label = result['title_label']
                if title_label not in results_by_title:
                    results_by_title[title_label] = []
                results_by_title[title_label].append(result)
            
            # Create result components
            result_components = []
            
            # Estimate if we need multiple chunks
            estimated_tokens = sum(estimate_tokens(result['context']) for result in results)
            needs_chunking = estimated_tokens > 100000
            
            result_components.append(
                html.Div([
                    html.H5(f"Found '{search_term}' in {total_results} sections", className="text-success mb-3"),
                    html.P(f"Results from {len(results_by_title)} titles", className="text-muted"),
                    html.P(f"Estimated size: ~{estimated_tokens:,} tokens", className="text-muted small"),
                    
                    # Download options
                    html.Div([
                        dbc.Button(
                            [html.I(className="fas fa-download me-2"), f"Download Key Results ({total_results} sections)"],
                            id="download-word-search-results",
                            color="success",
                            size="sm",
                            className="me-2"
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-file-archive me-2"), "Download in Chunks"],
                            id="download-word-search-chunks",
                            color="outline-success",
                            size="sm",
                            disabled=not needs_chunking
                        ) if needs_chunking else html.Div()
                    ], className="mt-2"),
                    
                    # Explain Key Results scope
                    html.P([
                        html.I(className="fas fa-filter me-2"),
                        html.Strong("Key Results: "),
                        "Downloads only the subsections containing your search keywords, reducing file size significantly."
                    ], className="text-muted small mt-2"),
                    
                    # Show chunking info if needed
                    dbc.Alert([
                        html.I(className="fas fa-info-circle me-2"),
                        f"Large result set detected (~{estimated_tokens:,} tokens). ",
                        "Consider downloading in chunks for better readability and processing."
                    ], color="info", className="mt-2") if needs_chunking else html.Div()
                ])
            )
            
            for title_label, title_results in results_by_title.items():
                # Create expandable accordion for each title
                accordion_items = []
                
                for result in title_results:
                    # Create clickable result card
                    context_lines = result['context'].split('\n')
                    
                    # For word search display, show first few paragraphs or up to 800 characters
                    context_preview = result['context'][:800]
                    if len(result['context']) > 800:
                        # Find a good break point (end of sentence or paragraph)
                        break_point = context_preview.rfind('.')
                        if break_point > 400:  # If we found a sentence break after reasonable content
                            context_preview = context_preview[:break_point + 1] + '...'
                        else:
                            context_preview = context_preview + '...'
                    
                    # Convert **term** to highlighted span
                    highlighted_context = re.sub(
                        r'\*\*(.*?)\*\*', 
                        r'<mark>\1</mark>', 
                        context_preview
                    )
                    
                    # Format occurrence count
                    occurrence_text = ""
                    if result.get('occurrence_count', 1) > 1:
                        occurrence_text = f" ({result['occurrence_count']} occurrences)"
                    
                    section_content = html.Div([
                        html.P([
                            html.Span("Context: ", className="fw-bold text-secondary"),
                            html.Span(
                                dcc.Markdown(highlighted_context, dangerously_allow_html=True),
                                className="text-muted"
                            )
                        ]),
                        dbc.Button(
                            [html.I(className="fas fa-external-link-alt me-1"), "View Full Section"],
                            id={
                                'type': 'word-search-ref',
                                'title': result['title_num'],
                                'section': result['section_num'],
                                'file': result['filename'],
                                'index': len(result_components) + len(accordion_items)
                            },
                            size="sm",
                            color="outline-primary",
                            n_clicks=0,
                            className="mt-2"
                        )
                    ])
                    
                    accordion_items.append(
                        dbc.AccordionItem(
                            section_content,
                            title=f"§{result['section_num']} - {result['section_title'][:60]}{'...' if len(result['section_title']) > 60 else ''}{occurrence_text}",
                            item_id=f"section-{result['filename']}-{result['section_num']}"
                        )
                    )
                
                # Create the main title accordion (collapsed by default)
                title_accordion = dbc.Accordion([
                    dbc.AccordionItem(
                        dbc.Accordion(accordion_items, flush=True, always_open=True),
                        title=f"{title_label} ({len(title_results)} sections)",
                        item_id=f"title-{title_label.replace(' ', '-').replace('(', '').replace(')', '')}",
                    )
                ], start_collapsed=True, className="mb-4")
                
                result_components.append(title_accordion)
            
            return html.Div(result_components)
        
        elif active_tab == "ai-assistant-tab" and AI_AVAILABLE:
            if not search_data or not search_data.get('result_text'):
                return html.Div([
                    html.H5("🤖 AI Assistant", className="text-primary mb-3"),
                    html.P("Navigate to a section first, then return here to ask questions about it."),
                    dbc.Alert([
                        html.H6("How to use:", className="alert-heading"),
                        html.P("1. Search for and select a USC section"),
                        html.P("2. Come back to this AI Assistant tab"),
                        html.P("3. Ask questions about the section content", className="mb-0")
                    ], color="info")
                ], className="text-center mt-3")
            
            section_content = search_data.get('result_text', '')
            section_title = search_data.get('search_query', '')
            
            return html.Div([
                html.H5("🤖 AI Assistant", className="text-primary mb-3"),
                html.P(f"Ask questions about: §{section_title}", className="text-muted"),
                
                # Simple explanation button
                dbc.Card([
                    dbc.CardBody([
                        dbc.Button(
                            [html.I(className="fas fa-lightbulb me-2"), "Get Simple Explanation"],
                            id="ai-simple-btn",
                            color="primary",
                            size="lg",
                            n_clicks=0,
                            className="w-100"
                        ),
                        html.P("Click for a clear, concise explanation in plain English", 
                               className="text-muted mt-2 mb-0 small text-center")
                    ])
                ], className="mb-3"),

                # Question input area
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("💬 Custom Question", className="mb-0 text-primary")
                    ]),
                    dbc.CardBody([
                        html.Label("Or ask your own question:", className="fw-bold mb-2"),
                        dbc.InputGroup([
                            dbc.Textarea(
                                id="ai-question-input",
                                placeholder="e.g., How does this section interact with other laws? What are the implementation requirements?",
                                rows=3,
                                className="mb-2"
                            ),
                        ]),
                        dbc.Button(
                            [html.I(className="fas fa-robot me-2"), "Ask Custom Question"],
                            id="ai-ask-button",
                            color="primary",
                            n_clicks=0,
                            className="mt-2"
                        )
                    ])
                ], className="mb-3"),
                
                # AI Response area
                html.Div(id="ai-response-area", children=[
                    dbc.Alert(
                        "Enter a question above and click 'Ask AI' to get help understanding this section.",
                        color="light"
                    )
                ]),
                
                # Follow-up question area (initially hidden)
                html.Div(id="ai-followup-area", style={'display': 'none'}, children=[
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("💬 Follow-up Question", className="mb-0 text-success")
                        ]),
                        dbc.CardBody([
                            html.Label("Ask a follow-up question:", className="fw-bold mb-2"),
                            dbc.InputGroup([
                                dbc.Textarea(
                                    id="ai-followup-input",
                                    placeholder="e.g., Can you clarify that point? What are the exceptions?",
                                    rows=2,
                                    className="mb-2"
                                ),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        [html.I(className="fas fa-robot me-2"), "Ask Follow-up"],
                                        id="ai-followup-button",
                                        color="success",
                                        n_clicks=0,
                                        className="w-100"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Button(
                                        [html.I(className="fas fa-plus me-2"), "New Conversation"],
                                        id="ai-new-conversation-button",
                                        color="outline-secondary",
                                        n_clicks=0,
                                        className="w-100"
                                    )
                                ], width=6)
                            ])
                        ])
                    ], className="mb-3")
                ]),
                
                # Performance Statistics
                html.Div(id="ai-performance-stats", className="mt-3")
            ])
        
        return html.P("Invalid tab selected.", className="text-warning")
        
    except Exception as e:
        # Catch any errors and display them
        return html.Div([
            dbc.Alert(f"Tab rendering error: {str(e)}", color="danger"),
            html.P(f"Active tab: {active_tab}", className="small text-muted"),
            html.P(f"Search data type: {type(search_data)}", className="small text-muted"),
            html.P(f"Search data keys: {list(search_data.keys()) if isinstance(search_data, dict) else 'Not a dict'}", className="small text-muted")
        ])

# Main search callback (updated to store data for tabs)
@app.callback(
    [Output('search-results-store', 'data'),
     Output('search-info', 'children'),
     Output('search-history', 'children'),
     Output('nav-history', 'data', allow_duplicate=True)],
    [Input('search-button', 'n_clicks'),
     Input('example-1', 'n_clicks'),
     Input('example-2', 'n_clicks'),
     Input('example-3', 'n_clicks'),
     Input('section-dropdown', 'value'),
     Input('subsection-dropdown', 'value'),
     Input('ref-click', 'data')],
    [State('title-dropdown', 'value'),
     State('search-input', 'value'),
     State('nav-history', 'data')],
    prevent_initial_call=True
)
def perform_search(search_clicks, ex1_clicks, ex2_clicks, ex3_clicks, selected_section, selected_subsections, ref_click_data, selected_title, search_query, nav_data):
    ctx = dash.callback_context
    
    print(f"DEBUG: Main search callback triggered: {ctx.triggered}")
    print(f"DEBUG: ref_click_data: {ref_click_data}")
    
    if not ctx.triggered:
        empty_data = {
            'formatted_result': html.P("Select a title and section to begin, or use manual search.", className="text-muted"),
            'structure_content': html.P("No structure available.", className="text-muted"),
            'references_content': html.P("No references available.", className="text-muted"),
            'result_text': "",
            'search_query': "",
            'title': ""
        }
        return empty_data, "", "", nav_data or {'history': [], 'current_index': -1}
    
    # Determine which input triggered the search
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"DEBUG: Button ID: {button_id}")
    
    # Build search query from ref click, dropdowns, or manual input
    override_file = None
    
    if button_id == 'ref-click' and ref_click_data:
        # Cross-reference navigation
        override_file = ref_click_data.get('filename')
        search_query = ref_click_data.get('section')
    elif button_id in ['section-dropdown', 'subsection-dropdown'] and selected_section:
        if selected_subsections:
            subsection_str = " ".join([f"({sub})" for sub in selected_subsections])
            search_query = f"{selected_section} {subsection_str}"
        else:
            search_query = selected_section
    elif button_id == 'example-1':
        search_query = "1"
    elif button_id == 'example-2':
        search_query = "2"
    elif button_id == 'example-3':
        search_query = "3 (a)"
    
    if (not selected_title and not override_file) or not search_query:
        no_query_data = {
            'formatted_result': dbc.Alert("Please select a title and enter a search query.", color="warning"),
            'structure_content': html.P("No structure available.", className="text-muted"),
            'references_content': html.P("No references available.", className="text-muted"),
            'result_text': "",
            'search_query': search_query or "",
            'title': selected_title or ""
        }
        return no_query_data, "", "", nav_data or {'history': [], 'current_index': -1}
    
    # Load the document
    file_path = f"txt/{override_file or selected_title}"
    document = load_document(file_path)
    
    if not document:
        error_data = {
            'formatted_result': dbc.Alert(f"Error: Could not load {override_file or selected_title}", color="danger"),
            'structure_content': html.P("No structure available - file could not be loaded.", className="text-muted"),
            'references_content': html.P("No references available - file could not be loaded.", className="text-muted"),
            'result_text': f"Error: Could not load {override_file or selected_title}",
            'search_query': search_query or "",
            'title': selected_title or ""
        }
        return error_data, "", "", nav_data or {'history': [], 'current_index': -1}
    
    # Perform search
    try:
        result = find_usc_section(document, search_query.strip())
        
        # Add to search history
        timestamp = datetime.now().strftime("%H:%M:%S")
        title_value = override_file or selected_title
        search_history.insert(0, {
            'time': timestamp,
            'title': title_value,
            'query': search_query,
            'success': not result.startswith('Section') or not result.endswith('not found')
        })
        
        # Keep only last 10 searches
        if len(search_history) > 10:
            search_history.pop()
        
        # Format result with linkified references
        def format_line(line):
            line = line.strip()
            if not line:
                return None
            children = linkify_text(line, override_file or selected_title)
            if line.startswith('§'):
                return html.H5(children, className="text-primary mt-3")
            elif re.match(r'^\([a-zA-Z0-9]+\)', line):
                return html.H6(children, className="text-secondary mt-2")
            else:
                return html.P(children, className="mb-2")
        
        if result.startswith('Section') and result.endswith('not found'):
            formatted_result = dbc.Alert(result, color="warning")
            structure_content = dbc.Alert("No structure available - section not found.", color="info")
            references_content = dbc.Alert("No references available - section not found.", color="info")
        elif result.startswith('Subsection') and 'not found' in result:
            formatted_result = dbc.Alert(result, color="warning")
            structure_content = dbc.Alert("No structure available - subsection not found.", color="info")
            references_content = dbc.Alert("No references available - subsection not found.", color="info")
        else:
            # Always build structure from the full section content for reliability,
            # but format/display lines from the result content
            lines = result.split('\n')  # for formatted result output
            formatted_lines = []
            structure_items = []
            references = []

            # Determine section and scope from the search query
            section_identifier, search_sequence = parse_input(search_query.strip())
            section_num_for_structure = section_identifier.strip('§.')
            full_section_for_structure = extract_full_section(document, section_num_for_structure)
            section_for_structure = (
                full_section_for_structure if full_section_for_structure and not full_section_for_structure.startswith('Section')
                else result
            )
            struct_lines = section_for_structure.split('\n')
            
            # Helper function to check if a line is a PDF artifact
            def is_pdf_artifact(line):
                """Check if a line is a PDF conversion artifact (footnote numbers, page markers, etc.)"""
                stripped = line.strip()
                # Filter out standalone digits (1-3 digits) that are likely footnote/page numbers
                if stripped.isdigit() and len(stripped) <= 3:
                    return True
                # Filter out very short non-meaningful strings
                if len(stripped) <= 2 and not stripped.isalpha():
                    return True
                return False
            
            def is_citation_line(line):
                """
                Check if a line appears to be a citation/reference line that should go in editorial notes.
                
                Citation lines typically contain:
                - Date patterns: Aug. 14, 1935 or Feb. 25, 1944
                - Statute references: 49 Stat. 622 or 64 Stat. 558
                - Public Law references: Pub. L. 86-778
                - Chapter references: ch. 531 or ch. 63
                - Multiple semicolons separating citations
                """
                line = line.strip()
                if not line:
                    return False
                
                # Check for common citation patterns
                citation_patterns = [
                    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.\s+\d+,\s+\d{4}',  # Date: Aug. 14, 1935
                    r'\d+\s+Stat\.\s+\d+',  # Statute: 49 Stat. 622
                    r'Pub\.\s*L\.\s+\d+[–-]\d+',  # Public Law: Pub. L. 86-778
                    r'ch\.\s+\d+',  # Chapter: ch. 531
                    r'\d+\s+F\.R\.\s+\d+',  # Federal Register: 18 F.R. 2053
                    r'Reorg\.\s+Plan\s+No\.',  # Reorganization Plan
                ]
                
                # If the line contains multiple citation elements, it's likely a citation line
                pattern_count = sum(1 for pattern in citation_patterns if re.search(pattern, line, re.IGNORECASE))
                
                # Citation lines typically have 2+ citation patterns or start with a date/statute
                if pattern_count >= 2:
                    return True
                
                # Also check if line starts with a date or statute reference (continuation of previous citation)
                starts_with_citation = re.match(r'^\d+\s+Stat\.|^\d{4};|^Stat\.\s+\d+', line, re.IGNORECASE)
                if starts_with_citation:
                    return True
                
                return False
            
            def extract_inline_citations(text):
                """
                Extract inline citations and references that should be moved to editorial notes.
                Returns: (cleaned_text, list_of_citations)
                
                Patterns matched:
                - (R.S. §27; Feb. 14, 1899, ch. 154, 30 Stat. 836.)
                - R.S. §27 derived from acts...
                - (Pub. L. 93–554, title I, ch. III, Dec. 27, 1974, 88 Stat. 1777.)
                """
                citations = []
                
                # Pattern 1: Parenthetical citations with legal references
                # Matches: (R.S. §...), (Pub. L. ...), etc.
                parenthetical_pattern = r'\([^)]*?(?:R\.S\.|Pub\.\s*L\.|Stat\.|ch\.\s*\d+|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d+,\s*\d{4})[^)]*?\)'
                
                for match in re.finditer(parenthetical_pattern, text, re.IGNORECASE):
                    citation_text = match.group(0)
                    citations.append(citation_text)
                
                # Remove the citations from the text
                cleaned_text = re.sub(parenthetical_pattern, '', text, flags=re.IGNORECASE)
                
                # Pattern 2: "R.S. §... derived from..." statements (not in parentheses)
                # This typically appears after the parenthetical citation
                # Match until we find a sentence-ending period (period followed by space and capital or end of string)
                derivation_pattern = r'R\.S\.\s*§\s*\d+\s+derived from.*?\.(?=\s+[A-Z§]|$)'
                
                for match in re.finditer(derivation_pattern, cleaned_text, re.IGNORECASE):
                    citation_text = match.group(0).strip()
                    citations.append(citation_text)
                
                # Remove the derivation statements
                cleaned_text = re.sub(derivation_pattern, '', cleaned_text, flags=re.IGNORECASE)
                
                # Clean up extra whitespace
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                cleaned_text = re.sub(r'\s+([.,;])', r'\1', cleaned_text)  # Fix spacing before punctuation
                cleaned_text = cleaned_text.strip()
                
                return cleaned_text, citations
            
            # Extract structure information with proper hierarchy tracking
            # Parse the full section to build a robust path-based hierarchy
            section_hierarchy_paths, structural_blocks = parse_hierarchy_with_paths(section_for_structure)
            
            # Build content map: subsection key -> text content
            content_map = {}
            section_title = None
            section_intro_text = []  # Collect text between section title and first subsection
            has_subsections = False
            section_started = False
            editorial_content = []  # Collect editorial material
            in_editorial_section = False
            
            # Define editorial header patterns (only anchored headers trigger editorial collection)
            editorial_header_patterns = [
                r'^\s*EDITORIAL NOTES\s*:?\s*$',
                r'^\s*AMENDMENTS\s*:?\s*$',
                r'^\s*DERIVATION\s*:?\s*$',
                r'^\s*STATUTORY NOTES AND RELATED SUBSIDIARIES\s*:?\s*$',
                r'^\s*REFERENCES IN TEXT\s*:?\s*$',
                r'^\s*EFFECTIVE DATE\s*:?\s*$',
                r'^\s*SHORT TITLE\s*:?\s*$',
                r'^\s*HISTORICAL AND REVISION NOTES\s*:?\s*$',
                r'^\s*PRIOR PROVISIONS\s*:?\s*$',
                r'^\s*CODIFICATION\s*:?\s*$',
                r'^\s*REPEALS\s*:?\s*$'
            ]
            editorial_header_regex = re.compile('|'.join(editorial_header_patterns), re.IGNORECASE)
            
            # Track key line indices to robustly compute the pre-subsection intro
            title_line_idx = None
            first_structural_idx = None
            first_editorial_idx = None

            # First pass: format the result content for the Results tab
            for line in lines:
                comp = format_line(line)
                if comp is not None:
                    formatted_lines.append(comp)

            # Second pass: analyze the full section for structure/editorial
            for i, line in enumerate(struct_lines):
                line_clean = line.strip()

                # Check if we're entering an editorial section (only on anchored headers)
                if editorial_header_regex.search(line_clean):
                    in_editorial_section = True
                    if first_editorial_idx is None:
                        first_editorial_idx = i
                    editorial_content.append(line_clean)
                    continue

                # If we're in editorial section, continue collecting until a new section title appears
                if in_editorial_section:
                    if line_clean.startswith('§') and not editorial_header_regex.search(line_clean):
                        in_editorial_section = False
                        # fall through to process as a section title
                    else:
                        editorial_content.append(line_clean)
                        continue
                
                # Check if this line is a citation line (multi-line citation blocks)
                if section_started and is_citation_line(line_clean):
                    in_editorial_section = True
                    if first_editorial_idx is None:
                        first_editorial_idx = i
                    editorial_content.append(line_clean)
                    continue

                if line_clean.startswith('§') and section_title is None:
                    section_title = line_clean
                    section_intro_text = []
                    section_started = True
                    has_subsections = False
                    title_line_idx = i
                    continue

                if re.match(r'^\([a-zA-Z0-9]+\)', line_clean):
                    has_subsections = True
                    if first_structural_idx is None:
                        first_structural_idx = i
                    continue

                if section_started and not has_subsections and line_clean and not is_pdf_artifact(line_clean):
                    section_intro_text.append(line_clean)

            # After scanning lines, recompute intro more robustly: from title to first subsection/editorial header
            if title_line_idx is not None:
                end_idx_candidates = []
                if first_structural_idx is not None:
                    end_idx_candidates.append(first_structural_idx)
                if first_editorial_idx is not None:
                    end_idx_candidates.append(first_editorial_idx)
                end_idx = min(end_idx_candidates) if end_idx_candidates else len(lines)
                # Build full intro lines slice, ignoring trivial artifacts
                intro_slice = []
                for j in range(title_line_idx + 1, end_idx):
                    lc = struct_lines[j].strip()
                    if not lc or is_pdf_artifact(lc):
                        continue
                    intro_slice.append(lc)
                if intro_slice:
                    section_intro_text = intro_slice
            
            # Extract inline citations from intro text and move them to editorial content
            if section_intro_text:
                intro_combined = ' '.join(section_intro_text)
                cleaned_intro, extracted_citations = extract_inline_citations(intro_combined)
                
                # Update the intro text with cleaned version
                if cleaned_intro:
                    section_intro_text = [cleaned_intro]
                else:
                    section_intro_text = []
                
                # Add extracted citations to editorial content
                if extracted_citations:
                    editorial_content.extend(extracted_citations)
            
            # Build content map keyed by path using structural_blocks
            content_map = {}
            if structural_blocks:
                # Build a fast lookup of block line indices to path
                block_by_line = {b['line_index']: b for b in structural_blocks}
                block_lines = sorted(block_by_line.keys())
                for idx, line_idx in enumerate(block_lines):
                    path = block_by_line[line_idx]['path']
                    # Start content with any trailing text on the block line
                    content_lines = []
                    first_trailer = block_by_line[line_idx]['first_content']
                    if first_trailer:
                        content_lines.append(first_trailer)
                    # Collect subsequent lines until next structural block or section/editorial header or citation line
                    stop_line = block_lines[idx + 1] if idx + 1 < len(block_lines) else len(struct_lines)
                    for j in range(line_idx + 1, stop_line):
                        next_line = struct_lines[j].strip()
                        if not next_line:
                            continue
                        if editorial_header_regex.search(next_line):
                            break
                        if next_line.startswith('§'):
                            break
                        # Stop collecting if we hit a citation line
                        if is_citation_line(next_line):
                            break
                        content_lines.append(next_line)
                    content_text = ' '.join(content_lines).strip()
                    
                    # Extract inline citations from subsection content
                    cleaned_content, subsection_citations = extract_inline_citations(content_text)
                    content_map[path] = cleaned_content
                    
                    # Add any extracted citations to editorial content
                    if subsection_citations:
                        editorial_content.extend(subsection_citations)

            # Build the hierarchical tree structure
            if section_hierarchy_paths:
                tree = build_hierarchy_tree(section_hierarchy_paths, content_map)

                # If a specific subsection path was requested, prune the tree to that subtree
                if search_sequence:
                    # Try to resolve the exact path in a case-insensitive manner for letters/roman
                    seq_norm = [k.lower() for k in search_sequence]
                    resolved_path = None
                    for path_key in section_hierarchy_paths.keys():
                        tokens = path_key.split('.')
                        if len(tokens) != len(seq_norm):
                            continue
                        ok = True
                        for i_tok, tok in enumerate(tokens):
                            t_norm = tok.lower()
                            if t_norm != seq_norm[i_tok]:
                                ok = False
                                break
                        if ok:
                            resolved_path = path_key
                            break
                    if not resolved_path:
                        resolved_path = '.'.join(search_sequence)
                    tree = prune_tree_to_path(tree, resolved_path)

                tree_components = render_hierarchy_tree(tree, section_id_prefix=f"struct_{timestamp}")
                
                # Add section title at the top
                if section_title:
                    structure_items.append(html.H6(section_title, className="text-primary mb-3"))
                
                # Add intro text (always render if present)
                if section_intro_text:
                    intro_content = ' '.join(section_intro_text).strip()
                    if intro_content:
                        structure_items.append(
                            html.P(intro_content, className="mb-3", style={'font-weight': 'normal', 'background': 'none'})
                        )
                
                # Add the tree
                structure_items.extend(tree_components)
            else:
                # No hierarchy found - still show title and any intro text
                if section_title:
                    structure_items.append(html.H6(section_title, className="text-primary mb-3"))
                if section_intro_text:
                    intro_content = ' '.join(section_intro_text).strip()
                    if intro_content:
                        structure_items.append(
                            html.P(intro_content, className="mb-3", style={'font-weight': 'normal', 'background': 'none'})
                        )
            
            # Add editorial content if found
            if editorial_content:
                editorial_text = ' '.join(editorial_content).strip()
                if editorial_text:
                    # Limit display length for editorial content since it can be very long
                    if len(editorial_text) > 500:
                        editorial_display = editorial_text[:500] + "... [Editorial content truncated]"
                    else:
                        editorial_display = editorial_text
                    
                    structure_items.append(
                        html.Div([
                            html.Hr(className="mt-4 mb-3"),
                            html.H6("📋 Editorial Notes & Annotations", className="text-warning mb-2"),
                            html.P(editorial_display, className="text-muted small", style={'font-style': 'italic'})
                        ])
                    )
                
                # Extract references from the collected editorial text
                ref_matches = re.findall(r'section\s+\d+[A-Za-z]?|§\s*\d+[A-Za-z]?', editorial_text, re.IGNORECASE)
                for ref in ref_matches:
                    if ref not in references:
                        references.append(ref)
            
            formatted_result = html.Div(formatted_lines)
            
            # Create structure content
            if structure_items:
                # Simple structure display with just a title
                structure_content = html.Div([
                    html.H6("Section Structure:", className="text-primary mb-3"),
                    html.Div(structure_items)
                ])
            else:
                structure_content = dbc.Alert("No section content found to display structure.", color="info")
            
            # Create references content
            if references:
                references_content = html.Div([
                    html.H6("Legal References Found:", className="text-primary mb-3"),
                    html.Ul([
                        html.Li(ref, className="mb-1") for ref in references[:10]  # Limit to 10
                    ])
                ])
            else:
                references_content = dbc.Alert("No legal references found in this content.", color="info")
        
        # Create search info
        title_name = next((t['label'] for t in available_titles if t['value'] == (override_file or selected_title)), (override_file or selected_title))
        search_info = f"Searched {title_name} for '{search_query}'"
        
        # Create history display
        history_items = []
        for item in search_history:
            color = "success" if item['success'] else "warning"
            history_items.append(
                dbc.Badge(
                    f"{item['time']} - {item['query']} in {item['title'][:10]}...",
                    color=color,
                    className="me-2 mb-1"
                )
            )
        
        # Prepare data for tab content rendering (do this first so we can store in nav history)
        search_data = {
            'formatted_result': formatted_result,
            'structure_content': structure_content,
            'references_content': references_content,
            'result_text': result,
            'search_query': search_query,
            'title': title_name
        }
        
        # Update navigation history with complete state
        if nav_data is None:
            nav_data = {'history': [], 'current_index': -1}
        
        # Add current search to navigation with full state for instant restoration
        current_search = {
            'type': 'section',
            'search_data': search_data,
            'word_search_data': None,
            'active_tab': 'structure-tab',
            'form_state': {
                'title': title_value,
                'search_input': search_query,
                'word_search_input': '',
                'search_scope': 'all'
            },
            'timestamp': timestamp
        }
        
        nav_history = nav_data.get('history', [])
        current_index = nav_data.get('current_index', -1)
        
        # Remove any items after current index (if user navigated back)
        if current_index < len(nav_history) - 1:
            nav_history = nav_history[:current_index + 1]
        
        nav_history.append(current_search)
        current_index = len(nav_history) - 1
        
        # Keep only last 20 items
        if len(nav_history) > 20:
            nav_history = nav_history[-20:]
            current_index = len(nav_history) - 1
        
        updated_nav_data = {
            'history': nav_history,
            'current_index': current_index
        }
        
        return search_data, search_info, html.Div(history_items), updated_nav_data
        
    except Exception as e:
        error_alert = dbc.Alert(f"Error: {str(e)}", color="danger")
        error_data = {
            'formatted_result': error_alert,
            'structure_content': dbc.Alert("Error occurred - no structure available.", color="warning"),
            'references_content': dbc.Alert("Error occurred - no references available.", color="warning"),
            'result_text': f"Error: {str(e)}",
            'search_query': search_query or "unknown",
            'title': "Error"
        }
        return error_data, "", "", nav_data or {'history': [], 'current_index': -1}

# Separate callback to handle section dropdown updates from ref-click and examples
@app.callback(
    Output('section-dropdown', 'value', allow_duplicate=True),
    [Input('ref-click', 'data'),
     Input('example-1', 'n_clicks'),
     Input('example-2', 'n_clicks'),
     Input('example-3', 'n_clicks')],
    prevent_initial_call=True
)
def update_section_dropdown_from_refs(ref_click_data, ex1_clicks, ex2_clicks, ex3_clicks):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'ref-click' and ref_click_data:
        # Cross-reference navigation - update section dropdown to match clicked section
        return ref_click_data.get('section')
    elif button_id == 'example-1':
        return "1"
    elif button_id == 'example-2':
        return "2"
    elif button_id == 'example-3':
        return "3"
    
    return no_update

# Toggle search help modal
@app.callback(
    Output('search-help-modal', 'is_open'),
    [Input('search-syntax-help-btn', 'n_clicks'),
     Input('close-search-help-modal', 'n_clicks'),
     Input('help-example-phrase', 'n_clicks'),
     Input('help-example-or', 'n_clicks'),
     Input('help-example-not', 'n_clicks'),
     Input('help-example-wildcard', 'n_clicks'),
     Input('help-example-near', 'n_clicks'),
     Input('help-example-near-n', 'n_clicks'),
     Input('help-example-title', 'n_clicks')],
    [State('search-help-modal', 'is_open')],
    prevent_initial_call=True
)
def toggle_search_help_modal(open_clicks, close_clicks, ex1, ex2, ex3, ex4, ex5, ex6, ex7, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Open modal when help button clicked
    if trigger_id == 'search-syntax-help-btn':
        return True
    # Close modal when close button or any example clicked
    elif trigger_id in ['close-search-help-modal', 'help-example-phrase', 'help-example-or', 
                        'help-example-not', 'help-example-wildcard', 'help-example-near',
                        'help-example-near-n', 'help-example-title']:
        return False
    
    return is_open

# Populate search input from help modal examples
@app.callback(
    Output('word-search-input', 'value'),
    [Input('help-example-phrase', 'n_clicks'),
     Input('help-example-or', 'n_clicks'),
     Input('help-example-not', 'n_clicks'),
     Input('help-example-wildcard', 'n_clicks'),
     Input('help-example-near', 'n_clicks'),
     Input('help-example-near-n', 'n_clicks'),
     Input('help-example-title', 'n_clicks')],
    prevent_initial_call=True
)
def populate_search_from_help(ex1, ex2, ex3, ex4, ex5, ex6, ex7):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    examples = {
        'help-example-phrase': '"federal agency"',
        'help-example-or': 'highway OR interstate',
        'help-example-not': 'criminal -misdemeanor',
        'help-example-wildcard': 'regulat*',
        'help-example-near': 'tax NEAR fraud',
        'help-example-near-n': 'due NEAR/3 process',
        'help-example-title': 'fraud title:18'
    }
    
    return examples.get(trigger_id, no_update)

# Word search callback - now with advanced search syntax and navigation history
@app.callback(
    [Output('word-search-results-store', 'data'),
     Output('result-tabs', 'active_tab', allow_duplicate=True),
     Output('word-search-loading-store', 'data'),
     Output('parsed-query-display', 'children'),
     Output('nav-history', 'data', allow_duplicate=True)],
    [Input('word-search-button', 'n_clicks')],
    [State('word-search-input', 'value'),
     State('search-scope', 'value'),
     State('search-options', 'value'),
     State('title-dropdown', 'value'),
     State('nav-history', 'data')],
    prevent_initial_call=True
)
def perform_word_search(search_clicks, search_term, search_scope, search_options, current_title, nav_data):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return no_update, no_update, False, no_update, no_update
    
    search_query = search_term.strip() if search_term else ''
    
    if not search_query or len(search_query) < 2:
        return {
            'results': [],
            'search_term': search_query,
            'error': 'Please enter search terms with 2 or more characters.'
        }, "word-search-tab", False, html.Small("Enter a search query...", className="text-muted"), no_update

    try:
        # Determine search scope - title filter from UI
        title_filter = None
        if search_scope == 'current' and current_title:
            # Extract title number from filename
            title_match = re.match(r'usc(\d+[A-Z]?)(?:_.*)?@', current_title)
            if title_match:
                title_filter = title_match.group(1)
        
        # Use advanced search if SQLite database is available
        if SEARCH_DB_AVAILABLE:
            results, parse_info = search_database.advanced_search(
                search_query, 
                title_filter=title_filter,
                max_results=20000
            )
            
            # Build parsed query display
            parsed_display = []
            if parse_info.get('parsed_explanation'):
                parsed_display.append(
                    html.Div([
                        html.I(className="fas fa-check-circle text-success me-1"),
                        html.Small(parse_info['parsed_explanation'], className="text-muted")
                    ], className="mt-1")
                )
            if parse_info.get('title_filter'):
                parsed_display.append(
                    html.Div([
                        html.I(className="fas fa-filter text-info me-1"),
                        html.Small(f"Filtering: Title {parse_info['title_filter']}", className="text-info")
                    ])
                )
            if parse_info.get('error'):
                parsed_display.append(
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle text-warning me-1"),
                        html.Small(f"Parse issue: {parse_info['error']}", className="text-warning")
                    ])
                )
            
            search_data = {
                'results': results,
                'search_term': search_query,
                'search_scope': search_scope,
                'total_results': len(results),
                'parse_info': parse_info,
                'chunks_data': None
            }
            
            # Add to navigation history
            updated_nav_data = _add_word_search_to_nav_history(
                nav_data, search_data, search_query, search_scope, current_title
            )
            
            return search_data, "word-search-tab", False, html.Div(parsed_display), updated_nav_data
        
        else:
            # Fallback to legacy search for compatibility
            # Convert advanced syntax to simple terms (best effort)
            if '&' in search_query:
                search_terms = tuple(sorted(t.strip().lower() for t in search_query.split('&') if len(t.strip()) >= 2))
            else:
                # Remove advanced operators for fallback
                simple_query = re.sub(r'\bOR\b|\bNOT\b|\bNEAR\b|"|\*|title:\d+', ' ', search_query, flags=re.IGNORECASE)
                search_terms = tuple(t.strip().lower() for t in simple_query.split() if len(t.strip()) >= 2)
            
            results = word_search_across_documents(
                search_terms, 
                title_filter=title_filter if title_filter else "all",
                max_results=20000
            )
            
            search_data = {
                'results': results,
                'search_term': search_query,
                'search_scope': search_scope,
                'total_results': len(results),
                'chunks_data': None
            }
            
            fallback_msg = html.Div([
                html.I(className="fas fa-info-circle text-warning me-1"),
                html.Small("Advanced syntax unavailable - using simple search", className="text-warning")
            ])
            
            # Add to navigation history
            updated_nav_data = _add_word_search_to_nav_history(
                nav_data, search_data, search_query, search_scope, current_title
            )
            
            return search_data, "word-search-tab", False, fallback_msg, updated_nav_data
        
    except Exception as e:
        error_data = {
            'results': [],
            'search_term': search_query,
            'error': f'Search error: {str(e)}'
        }
        error_display = html.Div([
            html.I(className="fas fa-exclamation-circle text-danger me-1"),
            html.Small(f"Error: {str(e)}", className="text-danger")
        ])
        return error_data, "word-search-tab", False, error_display, no_update

# Immediate loading state callback - triggers when search button is clicked
@app.callback(
    Output('word-search-loading-store', 'data', allow_duplicate=True),
    [Input('word-search-button', 'n_clicks')],
    [State('word-search-input', 'value')],
    prevent_initial_call=True
)
def set_loading_state(search_clicks, search_term):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    
    # Only set loading if there's actually a search term
    search_query = search_term.strip() if search_term else ''
    if search_query and len(search_query) >= 2:
        return True
    return False

# Simplified loading state callback (no spinner)
@app.callback(
    [Output('word-search-button-text', 'children'),
     Output('word-search-button', 'disabled')],
    [Input('word-search-loading-store', 'data')],
    prevent_initial_call=True
)
def update_loading_display(is_loading):
    if is_loading:
        return "Searching...", True
    else:
        return "Search", False

# Handle word search reference clicks
@app.callback(
    [Output('ref-click', 'data', allow_duplicate=True),
     Output('title-dropdown', 'value', allow_duplicate=True),
     Output('result-tabs', 'active_tab', allow_duplicate=True)],
    [Input({'type': 'word-search-ref', 'title': ALL, 'section': ALL, 'file': ALL, 'index': ALL}, 'n_clicks')],
    [State('title-dropdown', 'value')],
    prevent_initial_call=True
)
def on_word_search_ref_click(n_clicks_list, current_title_value):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    # Check if any clicks occurred
    if not n_clicks_list or not any(n_clicks_list):
        return no_update, no_update, no_update
    
    # Find which button was actually clicked
    for i, clicks in enumerate(n_clicks_list):
        if clicks and clicks > 0:
            inputs = ctx.inputs_list[0]
            if i < len(inputs):
                triggered_id = inputs[i]['id']
                
                target_title_num = triggered_id.get('title')
                target_section = triggered_id.get('section')
                target_file = triggered_id.get('file')
                
                if not target_section:
                    continue
                
                data = {
                    'filename': target_file,
                    'title_num': target_title_num,
                    'section': target_section,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Navigate to the section and switch to results tab
                return data, target_file, "results-tab"
    
    return no_update, no_update, no_update

# Allow Enter key to trigger search
app.clientside_callback(
    """
    function(n_submit, n_clicks) {
        return n_submit + n_clicks;
    }
    """,
    Output('search-button', 'n_clicks'),
    [Input('search-input', 'n_submit')],
    [State('search-button', 'n_clicks')]
)

# Allow Enter key to trigger word search
app.clientside_callback(
    """
    function(n_submit, n_clicks) {
        return n_submit + n_clicks;
    }
    """,
    Output('word-search-button', 'n_clicks'),
    [Input('word-search-input', 'n_submit')],
    [State('word-search-button', 'n_clicks')]
)

# Allow Enter key to trigger follow-up question
app.clientside_callback(
    """
    function(n_submit, n_clicks) {
        return n_submit + n_clicks;
    }
    """,
    Output('ai-followup-button', 'n_clicks'),
    [Input('ai-followup-input', 'n_submit')],
    [State('ai-followup-button', 'n_clicks')]
)

# Keyboard event handler - triggers hidden buttons and tab switching
app.clientside_callback(
    """
    function() {
        if (!window.hotkeySetup) {
            window.hotkeySetup = true;
            
            // Tab IDs in order (1-5)
            const tabIds = ['results-tab', 'structure-tab', 'references-tab', 'word-search-tab', 'ai-assistant-tab'];
            
            document.addEventListener('keydown', function(event) {
                // Don't interfere if user is typing
                if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                    return;
                }
                
                let buttonId = null;
                let keyName = '';
                
                // Handle number keys 1-5 for tab switching
                if (event.key >= '1' && event.key <= '5') {
                    const tabIndex = parseInt(event.key) - 1;
                    const tabId = tabIds[tabIndex];
                    
                    // Find and click the tab
                    const tabsContainer = document.getElementById('result-tabs');
                    if (tabsContainer) {
                        const tabLinks = tabsContainer.querySelectorAll('.nav-link');
                        if (tabLinks[tabIndex]) {
                            tabLinks[tabIndex].click();
                            
                            // Update status
                            const status = document.getElementById('hotkey-status');
                            if (status) {
                                const tabNames = ['Results', 'Structure', 'References', 'Word Search', 'AI'];
                                status.textContent = 'Tab: ' + tabNames[tabIndex];
                                setTimeout(() => {
                                    status.textContent = 'Ready for hotkeys...';
                                }, 1000);
                            }
                        }
                    }
                    return;
                }
                
                switch(event.key) {
                    case 'ArrowUp':
                        event.preventDefault();
                        buttonId = 'hotkey-up';
                        keyName = '↑';
                        break;
                    case 'ArrowDown':
                        event.preventDefault();
                        buttonId = 'hotkey-down';
                        keyName = '↓';
                        break;
                    case 'ArrowLeft':
                        event.preventDefault();
                        buttonId = 'hotkey-left';
                        keyName = '←';
                        break;
                    case 'ArrowRight':
                        event.preventDefault();
                        buttonId = 'hotkey-right';
                        keyName = '→';
                        break;
                }
                
                if (buttonId) {
                    console.log('Hotkey pressed:', keyName, 'triggering:', buttonId);
                    const button = document.getElementById(buttonId);
                    if (button) {
                        button.click();
                        
                        // Update status temporarily
                        const status = document.getElementById('hotkey-status');
                        if (status) {
                            status.textContent = 'Last key: ' + keyName;
                            setTimeout(() => {
                                status.textContent = 'Ready for hotkeys...';
                            }, 1000);
                        }
                    } else {
                        console.log('Button not found:', buttonId);
                    }
                }
            });
        }
        
        return "Hotkeys ready!";
    }
    """,
    Output('hotkey-status', 'children'),
    [Input('hotkey-up', 'id')]  # Just trigger once on page load
)

# Navigation button state callback
@app.callback(
    [Output('nav-back-button', 'disabled'),
     Output('nav-forward-button', 'disabled')],
    [Input('nav-history', 'data')],
    prevent_initial_call=True
)
def update_nav_buttons(nav_data):
    if not nav_data:
        return True, True
    
    history = nav_data.get('history', [])
    current_index = nav_data.get('current_index', -1)
    
    back_disabled = current_index <= 0
    forward_disabled = current_index >= len(history) - 1
    
    return back_disabled, forward_disabled

# Navigation action callback - handles back/forward button clicks
# Now restores complete state including search results, active tab, and form inputs
@app.callback(
    [Output('nav-history', 'data', allow_duplicate=True),
     Output('search-results-store', 'data', allow_duplicate=True),
     Output('word-search-results-store', 'data', allow_duplicate=True),
     Output('result-tabs', 'active_tab', allow_duplicate=True),
     Output('title-dropdown', 'value', allow_duplicate=True),
     Output('search-input', 'value', allow_duplicate=True),
     Output('word-search-input', 'value', allow_duplicate=True),
     Output('search-scope', 'value', allow_duplicate=True)],
    [Input('nav-back-button', 'n_clicks'),
     Input('nav-forward-button', 'n_clicks')],
    [State('nav-history', 'data')],
    prevent_initial_call=True
)
def navigate_history(back_clicks, forward_clicks, nav_data):
    ctx = dash.callback_context
    if not ctx.triggered or not nav_data:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    history = nav_data.get('history', [])
    current_index = nav_data.get('current_index', -1)
    
    if not history:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    # Update the current index based on button clicked
    if button_id == 'nav-back-button' and current_index > 0:
        current_index -= 1
    elif button_id == 'nav-forward-button' and current_index < len(history) - 1:
        current_index += 1
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
    # Get the state to navigate to
    current_state = history[current_index]
    
    # Update navigation data
    updated_nav_data = {
        'history': history,
        'current_index': current_index
    }
    
    # Determine search type and restore appropriate state
    search_type = current_state.get('type', 'section')  # Default to section for backward compatibility
    
    # Get form state (with fallbacks for backward compatibility)
    form_state = current_state.get('form_state', {})
    title_value = form_state.get('title') or current_state.get('title', '')
    search_input_value = form_state.get('search_input') or current_state.get('query', '')
    word_search_input_value = form_state.get('word_search_input', '')
    search_scope_value = form_state.get('search_scope', 'all')
    
    if search_type == 'word':
        # Restore word search state
        return (
            updated_nav_data,
            no_update,  # Don't clear section search results
            current_state.get('word_search_data'),  # Restore word search results
            'word-search-tab',  # Switch to word search tab
            title_value,
            search_input_value,
            word_search_input_value,
            search_scope_value
        )
    else:
        # Restore section search state
        return (
            updated_nav_data,
            current_state.get('search_data'),  # Restore section search results
            no_update,  # Don't clear word search results
            current_state.get('active_tab', 'structure-tab'),  # Restore tab
            title_value,
            search_input_value,
            word_search_input_value,
            search_scope_value
        )

# Hotkey navigation callback
@app.callback(
    [Output('title-dropdown', 'value', allow_duplicate=True),
     Output('section-dropdown', 'value', allow_duplicate=True),
     Output('hotkey-navigation-store', 'data', allow_duplicate=True)],
    [Input('hotkey-up', 'n_clicks'),
     Input('hotkey-down', 'n_clicks'),
     Input('hotkey-left', 'n_clicks'),
     Input('hotkey-right', 'n_clicks')],
    [State('title-dropdown', 'options'),
     State('section-dropdown', 'options'),
     State('title-dropdown', 'value'),
     State('section-dropdown', 'value'),
     State('hotkey-navigation-store', 'data')],
    prevent_initial_call=True
)
def handle_hotkey_navigation(up_clicks, down_clicks, left_clicks, right_clicks, 
                           title_options, section_options, current_title, current_section, navigation_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    # Determine which button was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    action_map = {
        'hotkey-up': 'title_up',
        'hotkey-down': 'title_down',
        'hotkey-left': 'section_left',
        'hotkey-right': 'section_right'
    }
    
    action = action_map.get(button_id)
    if not action:
        return no_update, no_update, no_update
    
    print(f"DEBUG: Hotkey pressed - {action}")
    print(f"DEBUG: Current title: {current_title}, sections available: {len(section_options) if section_options else 0}")
    
    # Initialize navigation data if needed
    if not navigation_data:
        navigation_data = {'current_title_index': 0, 'current_section_index': 0}
    
    current_title_index = navigation_data.get('current_title_index', 0)
    current_section_index = navigation_data.get('current_section_index', 0)
    
    # Sync indices with current state if needed
    if current_title and title_options:
        for i, option in enumerate(title_options):
            if option['value'] == current_title:
                current_title_index = i
                break
    
    if current_section and section_options:
        for i, option in enumerate(section_options):
            if option['value'] == current_section:
                current_section_index = i
                break
    
    new_title = current_title
    new_section = current_section
    
    # Handle title navigation (Up/Down arrows)
    if action in ['title_up', 'title_down']:
        if title_options:
            if action == 'title_up':
                current_title_index = (current_title_index - 1) % len(title_options)
            else:  # title_down
                current_title_index = (current_title_index + 1) % len(title_options)
            
            new_title = title_options[current_title_index]['value']
            # Reset section to first when changing title
            current_section_index = 0
            new_section = None  # Will be set by section dropdown callback
            print(f"DEBUG: Title navigation - {action}, index: {current_title_index}, title: {new_title}")
    
    # Handle section navigation (Left/Right arrows)
    elif action in ['section_left', 'section_right']:
        if section_options and len(section_options) > 0:
            if action == 'section_left':
                current_section_index = (current_section_index - 1) % len(section_options)
            else:  # section_right
                current_section_index = (current_section_index + 1) % len(section_options)
            
            new_section = section_options[current_section_index]['value']
            print(f"DEBUG: Section navigation - {action}, index: {current_section_index}, section: {new_section}")
        else:
            print(f"DEBUG: No section options available for {action}. Current title: {current_title}")
            # If no sections available but we have a title, try to trigger section loading
            if current_title and not current_section:
                # Set index to 0 to select first section when it becomes available
                current_section_index = 0
                print(f"DEBUG: Reset section index to 0 to select first section when available")
    
    # Update navigation store
    updated_navigation_data = {
        'current_title_index': current_title_index,
        'current_section_index': current_section_index
    }
    
    return new_title, new_section, updated_navigation_data

@app.callback(
    Output('hotkey-navigation-store', 'data', allow_duplicate=True),
    [Input('title-dropdown', 'value'),
     Input('section-dropdown', 'value')],
    [State('title-dropdown', 'options'),
     State('section-dropdown', 'options'),
     State('hotkey-navigation-store', 'data')],
    prevent_initial_call=True
)
def update_navigation_indices(selected_title, selected_section, title_options, section_options, nav_data):
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]

    if not nav_data:
        nav_data = {'current_title_index': 0, 'current_section_index': 0}

    new_nav_data = nav_data.copy()

    # If the title dropdown was changed, update the title index and reset the section index.
    if triggered_input == 'title-dropdown':
        if selected_title and title_options:
            for i, option in enumerate(title_options):
                if option['value'] == selected_title:
                    new_nav_data['current_title_index'] = i
                    break
        new_nav_data['current_section_index'] = 0 # Always reset section on title change
    
    # If the section dropdown was changed, only update the section index.
    # Do not reset it if options are not ready, as this is the source of the bug.
    elif triggered_input == 'section-dropdown':
        if selected_section and section_options:
            for i, option in enumerate(section_options):
                if option['value'] == selected_section:
                    new_nav_data['current_section_index'] = i
                    break
        elif not selected_section:
            new_nav_data['current_section_index'] = 0
        # If section is selected but options are missing, DO NOTHING.
        # This prevents the reset during the hotkey race condition.

    if new_nav_data == nav_data:
        return no_update

    return new_nav_data

# AI Assistant callback (only active if AI is available)
if AI_AVAILABLE:
    @app.callback(
        [Output('ai-response-area', 'children'),
         Output('ai-conversation-store', 'data'),
         Output('ai-followup-area', 'style')],
        [Input('ai-ask-button', 'n_clicks'),
         Input('ai-simple-btn', 'n_clicks'),
         Input('ai-followup-button', 'n_clicks'),
         Input('ai-new-conversation-button', 'n_clicks')],
        [State('ai-question-input', 'value'),
         State('ai-followup-input', 'value'),
         State('search-results-store', 'data'),
         State('ai-conversation-store', 'data')],
        prevent_initial_call=True
    )
    def handle_ai_question(custom_clicks, simple_clicks, followup_clicks, new_conversation_clicks, 
                          question, followup_question, search_data, conversation_data):
        # Determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return dbc.Alert("Click a button or enter a question to get started.", color="light"), no_update, no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Check if we have section content
        if not search_data or not search_data.get('result_text'):
            return dbc.Alert("No section content available. Please navigate to a section first.", color="warning"), no_update, no_update
        
        # Initialize conversation data if needed
        if not conversation_data:
            conversation_data = {'messages': [], 'section_content': '', 'section_title': ''}
        
        # Handle new conversation button
        if button_id == 'ai-new-conversation-button':
            # Reset conversation
            new_conversation_data = {
                'messages': [],
                'section_content': search_data.get('result_text', ''),
                'section_title': search_data.get('search_query', '')
            }
            return dbc.Alert("Conversation reset. Ask a new question to begin.", color="info"), new_conversation_data, {'display': 'none'}
        
        # Get section content and title
        section_content = search_data.get('result_text', '')
        section_title = search_data.get('search_query', '')
        
        # Update conversation data if section changed
        if (conversation_data.get('section_content') != section_content or 
            conversation_data.get('section_title') != section_title):
            conversation_data = {
                'messages': [],
                'section_content': section_content,
                'section_title': section_title
            }
        
        # Determine the question to ask
        if button_id == 'ai-simple-btn':
            final_question = "Explain this section in plain English using 2-3 simple sentences."
            question_display = "Simple Explanation"
            user_question = "Get simple explanation"
        elif button_id == 'ai-ask-button' and question and question.strip():
            final_question = f"Answer this question about the USC section clearly and concisely: {question.strip()}"
            question_display = question.strip()
            user_question = question.strip()
        elif button_id == 'ai-followup-button' and followup_question and followup_question.strip():
            # Check if we have conversation history
            if not conversation_data.get('messages'):
                return dbc.Alert("No conversation history. Please ask an initial question first.", color="warning"), no_update, no_update
            
            # Check conversation limit (max 5 exchanges = 10 messages)
            if len(conversation_data['messages']) >= 10:
                return dbc.Alert("Conversation limit reached (5 exchanges). Start a new conversation.", color="warning"), no_update, no_update
            
            # Build context from conversation history
            conversation_context = "Previous conversation:\n"
            for msg in conversation_data['messages'][-6:]:  # Last 3 exchanges for context
                conversation_context += f"{msg['role']}: {msg['content']}\n"
            
            final_question = f"""Based on this conversation context and the USC section, answer this follow-up question clearly and concisely:

{conversation_context}

Follow-up question: {followup_question.strip()}

Provide a helpful answer that builds on the previous conversation."""
            question_display = followup_question.strip()
            user_question = followup_question.strip()
        else:
            return dbc.Alert("Please enter a question or click a preset button.", color="warning"), no_update, no_update
        
        try:
            # Ask the AI with performance tracking
            ai_result = ollama_integration.ask_ai_with_stats(section_content, final_question)
            
            # Create performance badge
            perf_color = "info"
            perf_text = f"🚀 {ai_result.get('response_time', 0):.1f}s"
            
            # Update conversation history
            messages = conversation_data.get('messages', [])
            messages.append({
                'role': 'user',
                'content': user_question,
                'timestamp': datetime.now().isoformat()
            })
            messages.append({
                'role': 'assistant',
                'content': ai_result.get('response', 'No response available'),
                'timestamp': datetime.now().isoformat()
            })
            
            updated_conversation_data = {
                'messages': messages,
                'section_content': section_content,
                'section_title': section_title
            }
            
            # Create conversation display
            conversation_display = []
            
            # Add conversation header
            conversation_display.append(
                html.Div([
                    html.H6(f"💬 Conversation about §{section_title}", className="text-primary mb-3"),
                    html.P(f"Messages: {len(messages)//2}/5 exchanges", className="text-muted small")
                ])
            )
            
            # Display conversation history
            for i, message in enumerate(messages):
                if message['role'] == 'user':
                    conversation_display.append(
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-user me-2"),
                                html.Strong("You:", className="text-primary")
                            ], className="mb-1"),
                            html.Div(message['content'], className="ms-4 mb-3 p-2 bg-light rounded")
                        ])
                    )
                else:  # assistant
                    # Create header with conditional performance badge
                    header_children = [
                        html.I(className="fas fa-robot me-2"),
                        html.Strong("AI:", className="text-success")
                    ]
                    # Add performance badge only to the last message
                    if i == len(messages) - 1:
                        header_children.append(dbc.Badge(perf_text, color=perf_color, className="ms-2"))
                    
                    conversation_display.append(
                        html.Div([
                            html.Div(header_children, className="mb-1"),
                            html.Div([
                                dcc.Markdown(message['content'])
                            ], className="ms-4 mb-3 p-2 bg-success bg-opacity-10 rounded border-start border-success border-3")
                        ])
                    )
            
            # Add disclaimer
            conversation_display.append(
                html.Div([
                    html.Hr(),
                    html.Small("💡 This conversation is generated by AI and should be verified with legal professionals.", 
                             className="text-muted fst-italic")
                ])
            )
            
            # Show follow-up area if we have conversation history and haven't reached limit
            followup_style = {'display': 'block'} if len(messages) < 10 else {'display': 'none'}
            
            return html.Div(conversation_display), updated_conversation_data, followup_style
            
        except Exception as e:
            return dbc.Alert(f"Error getting AI response: {str(e)}", color="danger"), no_update, no_update

# AI Performance Statistics callback
if AI_AVAILABLE:
    @app.callback(
        Output('ai-performance-stats', 'children'),
        [Input('ai-ask-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_ai_performance_stats(n_clicks):
        if not n_clicks:
            return html.Div()
        
        try:
            stats = ollama_integration.get_performance_stats()
            
            if stats['total_requests'] == 0:
                return html.Div()
            
            return dbc.Card([
                dbc.CardHeader(html.H6("⚡ Performance Stats", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Small("Total Requests:", className="text-muted"),
                            html.Div(str(stats['total_requests']), className="fw-bold")
                        ], width=3),
                        dbc.Col([
                            html.Small("Cache Hit Rate:", className="text-muted"),
                            html.Div(stats['cache_hit_rate'], className="fw-bold text-success")
                        ], width=3),
                        dbc.Col([
                            html.Small("Avg Response:", className="text-muted"),
                            html.Div(stats['avg_response_time'], className="fw-bold")
                        ], width=3),
                        dbc.Col([
                            html.Small("Fastest:", className="text-muted"),
                            html.Div(stats['fastest_response'], className="fw-bold text-info")
                        ], width=3)
                    ])
                ])
            ], className="mt-2", style={"font-size": "0.9rem"})
            
        except Exception as e:
            return html.Small(f"Stats error: {e}", className="text-muted")

# Clear follow-up input callback
if AI_AVAILABLE:
    @app.callback(
        [Output('ai-followup-input', 'value'),
         Output('ai-question-input', 'value')],
        [Input('ai-followup-button', 'n_clicks'),
         Input('ai-new-conversation-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def clear_inputs(followup_clicks, new_conversation_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return no_update, no_update
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'ai-followup-button':
            # Clear follow-up input after asking
            return "", no_update
        elif button_id == 'ai-new-conversation-button':
            # Clear both inputs when starting new conversation
            return "", ""
        
        return no_update, no_update

# Generate chunks callback
@app.callback(
    [Output("word-search-chunks-store", "data"),
     Output("word-search-results-store", "data", allow_duplicate=True)],
    [Input("download-word-search-chunks", "n_clicks")],
    [State("word-search-results-store", "data")],
    prevent_initial_call=True
)
def generate_word_search_chunks(n_clicks, word_search_data):
    """
    Generate chunks for word search results and update the results store.
    """
    if not n_clicks or not word_search_data:
        return no_update, no_update
    
    try:
        results = word_search_data.get('results', [])
        search_term = word_search_data.get('search_term', 'unknown')
        
        if not results:
            return no_update, no_update
        
        # Generate chunks
        chunks = generate_word_search_txt_chunks(results, search_term)
        
        # Create chunk data with metadata
        chunk_data = {
            'chunks': chunks,
            'search_term': search_term,
            'total_chunks': len(chunks),
            'generated_at': datetime.now().isoformat()
        }
        
        # Update word search data with chunk information
        updated_word_search_data = word_search_data.copy()
        updated_word_search_data['chunks_data'] = chunk_data
        
        return chunk_data, updated_word_search_data
        
    except Exception as e:
        print(f"Error generating chunks: {e}")
        return no_update, no_update

# Download word search results callback (single file only)
@app.callback(
    Output("download-word-search-file", "data"),
    [Input("download-word-search-results", "n_clicks")],
    [State("word-search-results-store", "data")],
    prevent_initial_call=True
)
def download_word_search_results(n_clicks, word_search_data):
    """
    Download single text file containing word search results.
    """
    if not n_clicks or not word_search_data:
        return no_update
    
    try:
        results = word_search_data.get('results', [])
        search_term = word_search_data.get('search_term', 'unknown')
        
        if not results:
            return no_update
        
        # Generate the text content (first chunk only for single download)
        txt_content = generate_word_search_txt(results, search_term)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_search_term = re.sub(r'[^\w\s-]', '', search_term).strip()
        safe_search_term = re.sub(r'[-\s]+', '_', safe_search_term)
        filename = f"usc_search_{safe_search_term}_{timestamp}.txt"
        
        return dict(content=txt_content, filename=filename)
        
    except Exception as e:
        print(f"Error generating download file: {e}")
        return no_update

# Individual chunk download callback
@app.callback(
    Output("download-word-search-file", "data", allow_duplicate=True),
    [Input({'type': 'chunk-download', 'chunk_index': ALL, 'index': ALL}, 'n_clicks')],
    [State("word-search-results-store", "data")],
    prevent_initial_call=True
)
def download_individual_chunk(n_clicks_list, word_search_data):
    """
    Download individual chunks when chunk download buttons are clicked.
    """
    ctx = dash.callback_context
    if not ctx.triggered or not word_search_data:
        return no_update
    
    # Find which chunk button was clicked
    for i, clicks in enumerate(n_clicks_list):
        if clicks and clicks > 0:
            # Get the chunk index from the triggered component
            triggered_id = ctx.triggered[0]['prop_id']
            if 'chunk-download' in triggered_id:
                # Extract chunk index from the component ID
                chunk_index = i
                
                chunks_data = word_search_data.get('chunks_data')
                if not chunks_data or not chunks_data.get('chunks'):
                    return no_update
                
                chunks = chunks_data['chunks']
                search_term = chunks_data['search_term']
                
                if chunk_index < len(chunks):
                    # Create filename for this chunk
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_search_term = re.sub(r'[^\w\s-]', '', search_term).strip()
                    safe_search_term = re.sub(r'[-\s]+', '_', safe_search_term)
                    filename = f"usc_search_{safe_search_term}_chunk{chunk_index + 1}_{timestamp}.txt"
                    
                    return dict(content=chunks[chunk_index], filename=filename)
    
    return no_update

# Download all chunks as ZIP callback
@app.callback(
    Output("download-word-search-file", "data", allow_duplicate=True),
    [Input({'type': 'download-all-chunks', 'search_term': ALL, 'index': ALL}, 'n_clicks')],
    [State("word-search-results-store", "data")],
    prevent_initial_call=True
)
def download_all_chunks_zip(n_clicks_list, word_search_data):
    """
    Download all chunks as a ZIP file when the download all chunks button is clicked.
    """
    ctx = dash.callback_context
    if not ctx.triggered or not word_search_data:
        return no_update
    
    # Check if the download all chunks button was clicked
    if any(n_clicks_list):
        chunks_data = word_search_data.get('chunks_data')
        if not chunks_data or not chunks_data.get('chunks'):
            return no_update
        
        chunks = chunks_data['chunks']
        search_term = chunks_data['search_term']
        
        if chunks:
            # Create ZIP file with all chunks
            zip_content_bytes = create_chunks_zip(chunks, search_term)
            zip_content_b64 = base64.b64encode(zip_content_bytes).decode()
            
            # Create filename for ZIP
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_search_term = re.sub(r'[^\w\s-]', '', search_term).strip()
            safe_search_term = re.sub(r'[-\s]+', '_', safe_search_term)
            filename = f"usc_search_{safe_search_term}_all_chunks_{timestamp}.zip"
            
            return dict(content=zip_content_b64, filename=filename, type="application/zip", base64=True)
    
    return no_update

if __name__ == '__main__':
    # Show performance info
    index = load_usc_index()
    if index:
        print(f"🚀 Performance Mode: Using pre-built index with {index['metadata']['total_sections']} sections")
        print("   Title switching will be near-instant!")
    else:
        print("⚠️  Compatibility Mode: Using on-demand parsing (slower)")
        print("   Run 'python build_usc_index.py' to enable fast mode")
    
    print(f"🌐 Starting server on http://127.0.0.1:8051/")
    app.run_server(debug=True, host='127.0.0.1', port=8051) 