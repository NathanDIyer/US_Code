#!/usr/bin/env python3
"""
USC Section Index Builder

This script processes all USC text files and creates a pre-built index
for fast section lookup, dramatically improving dashboard performance.

Run this script once or whenever USC files are updated.
"""

import os
import re
import json
import glob
from datetime import datetime
import sys

def extract_file_sections(file_path):
    """Extract all sections and their metadata from a USC file"""
    print(f"Processing {os.path.basename(file_path)}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    filename = os.path.basename(file_path)
    sections = {}
    
    # Extract title information from filename
    title_match = re.match(r'usc(\d+[A-Z]?)(?:_.*)?@', filename)
    title_num = title_match.group(1) if title_match else "Unknown"
    
    if '_ch' in filename:
        ch_match = re.search(r'_ch(\d+)to(\d+)', filename)
        title_label = f'Title {title_num} (Chapters {ch_match.group(1)}-{ch_match.group(2)})' if ch_match else f'Title {title_num} (Partial)'
    else:
        title_label = f'Title {title_num}'
    
    # Find all sections with their positions - exclude parenthetical citations
    # This pattern specifically avoids matching citations like (R.S. §22.) or (Act §123.)
    section_pattern = re.compile(
        r'^\s*§(\d+[A-Za-z0-9\-]*)\.[^\S\r\n]*([^\n]*)',
        re.MULTILINE,
    )
    note_heading_prefixes = (
        'CODIFICATION',
        'STATUTORY NOTES',
        'HISTORICAL AND',
        'NOTES OF',
        'PRIOR PROVISIONS',
        'EFFECTIVE DATE',
        'REFERENCES IN TEXT',
        'SHORT TITLE',
        'TRANSFER OF FUNCTIONS',
        'EXECUTIVE DOCUMENTS',
        'AMENDMENTS',
        'SAVINGS PROVISION',
        'SIMILAR PROVISIONS',
        'CONSTRUCTION',
        'APPLICABILITY',
        'SPECIAL RULES',
        'REGULATIONS',
        'REPORTS',
    )

    section_matches = []
    seen_sections = set()

    for match in section_pattern.finditer(content):
        section_num = match.group(1)

        # Ensure we only record the first occurrence of each section number so
        # later historical notes that restate the citation cannot overwrite the
        # actual section location.
        if section_num in seen_sections:
            continue

        # Get some context before the match to check for parentheses
        start_pos = max(0, match.start() - 50)
        context_before = content[start_pos:match.start()]

        # Skip if this section reference is inside parentheses by checking the
        # balance of opening and closing parentheses in the local context.
        open_parens = context_before.count('(')
        close_parens = context_before.count(')')
        if open_parens > close_parens:
            continue

        # Also skip if the immediate character before § is '('
        if match.start() > 0 and content[match.start() - 1] == '(':
            continue

        section_title = match.group(2).strip()

        if not section_title:
            # Some historical notes place the section number alone on a line and
            # move the descriptive text to the following line. Peek ahead so we
            # can decide whether this is a real heading or just a note label.
            next_line_start = match.end()
            next_newline = content.find('\n', next_line_start)
            if next_newline == -1:
                next_newline = len(content)
            next_line = content[next_line_start:next_newline].strip()

            normalized = next_line.upper()
            if not next_line or any(normalized.startswith(prefix) for prefix in note_heading_prefixes):
                continue

            section_title = next_line

        # Skip sections that are omitted, repealed, or transferred
        title_lower = section_title.lower()
        if title_lower.startswith('omitted') or title_lower.startswith('repealed') or title_lower.startswith('transferred'):
            continue

        section_start_pos = match.start()

        section_matches.append(
            {
                'number': section_num,
                'title': section_title,
                'start': section_start_pos,
            }
        )
        seen_sections.add(section_num)

    for i, match_info in enumerate(section_matches):
        section_num = match_info['number']
        section_title = match_info['title']
        section_start_pos = match_info['start']

        # Find the end position (start of next section or end of file)
        if i + 1 < len(section_matches):
            section_end_pos = section_matches[i + 1]['start']
        else:
            section_end_pos = len(content)

        # Extract the full section content for subsection analysis
        section_content = content[section_start_pos:section_end_pos]
        
        # Find subsections within this section
        subsections = {}
        subsection_matches = re.finditer(r'\(([a-zA-Z0-9]+)\)', section_content)
        
        seen_subsections = set()
        for sub_match in subsection_matches:
            sub_key = sub_match.group(1)
            
            # Avoid duplicates and very long keys
            if sub_key in seen_subsections or len(sub_key) > 10:
                continue
            seen_subsections.add(sub_key)
            
            # Get a snippet of the subsection content
            start_pos = sub_match.end()
            snippet = section_content[start_pos:start_pos+150].strip()
            
            # Clean up the snippet - stop at next subsection
            snippet = re.split(r'\([a-zA-Z0-9]+\)', snippet)[0].strip()
            
            if snippet and len(snippet) > 10:  # Only include meaningful snippets
                # Limit snippet length
                if len(snippet) > 100:
                    snippet = snippet[:100] + "..."
                subsections[sub_key] = snippet
        
        # Clean and limit section title
        clean_title = section_title[:150] + "..." if len(section_title) > 150 else section_title
        
        sections[section_num] = {
            'title': clean_title,
            'subsections': subsections,
            'file_position': section_start_pos,
            'content_length': section_end_pos - section_start_pos
        }
    
    return {
        'title_info': {
            'title_num': title_num,
            'title_label': title_label,
            'filename': filename
        },
        'sections': sections,
        'file_stats': {
            'file_size': len(content),
            'total_sections': len(sections),
            'processed_at': datetime.now().isoformat()
        }
    }

def build_complete_index():
    """Build complete index of all USC files"""
    print("Building USC Section Index...")
    print("=" * 50)
    
    # Find all USC text files
    txt_files = glob.glob("txt/usc*.txt")
    if not txt_files:
        print("No USC text files found in txt/ directory!")
        return None
    
    print(f"Found {len(txt_files)} USC files to process")
    
    complete_index = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_files': len(txt_files),
            'builder_version': '1.1'
        },
        'files': {}
    }
    
    total_sections = 0
    processed_files = 0
    
    # Process each file
    for file_path in sorted(txt_files):
        file_data = extract_file_sections(file_path)
        
        if file_data:
            filename = os.path.basename(file_path)
            complete_index['files'][filename] = file_data
            total_sections += len(file_data['sections'])
            processed_files += 1
            
            print(f"  ✓ {filename}: {len(file_data['sections'])} sections")
        else:
            print(f"  ✗ Failed to process {file_path}")
    
    # Add summary statistics
    complete_index['metadata']['total_sections'] = total_sections
    complete_index['metadata']['processed_files'] = processed_files
    
    print("=" * 50)
    print(f"Index building complete!")
    print(f"Processed: {processed_files}/{len(txt_files)} files")
    print(f"Total sections indexed: {total_sections}")
    
    return complete_index

def save_index(index_data, output_file='usc_sections_index.json'):
    """Save the index to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        # Check file size
        file_size = os.path.getsize(output_file)
        print(f"Index saved to: {output_file}")
        print(f"Index file size: {file_size / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"Error saving index: {e}")
        return False

def main():
    """Main execution function"""
    print("USC Section Index Builder")
    print("Building pre-computed index for faster dashboard performance...")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('txt'):
        print("Error: 'txt' directory not found!")
        print("Please run this script from the US_Code directory.")
        sys.exit(1)
    
    # Build the index
    index = build_complete_index()
    
    if not index:
        print("Failed to build index!")
        sys.exit(1)
    
    # Save the index
    if save_index(index):
        print()
        print("🎉 Index building successful!")
        print("Your Dash app will now load USC titles much faster.")
        print()
        print("To use the index, restart your Dash app.")
    else:
        print("Failed to save index!")
        sys.exit(1)

if __name__ == '__main__':
    main() 
