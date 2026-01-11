import re
import json
from collections import OrderedDict

def load_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except IOError:
        print(f"Error: Unable to read file at {file_path}")
    return None

def parse_usc_to_json(document):
    # Split the document into sections, only matching §number. patterns
    sections = re.split(r'(?=§\s*\d+\.)', document)
    
    usc_dict = OrderedDict()
    
    for section in sections:
        if not section.strip():
            continue
        
        # Extract section number and title, ensuring we match the dot after the number
        section_match = re.match(r'§\s*(\d+)\.\s*(.+?)(?:\n|$)', section, re.DOTALL)
        if section_match:
            section_num, section_title = section_match.groups()
            section_content = section[section_match.end():].strip()
            
            # Parse the content of the section
            section_dict = parse_section_content(section_content)
            
            usc_dict[f"§{section_num}"] = {
                "title": section_title.strip(),
                "content": section_dict
            }
    
    return usc_dict

def parse_section_content(content):
    stack = [OrderedDict()]
    current_key = None
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check for new subsection
        subsection_match = re.match(r'^(\([a-z0-9]+\))', line, re.IGNORECASE)
        if subsection_match:
            current_key = subsection_match.group(1)
            stack[-1][current_key] = OrderedDict()
            stack.append(stack[-1][current_key])
        elif current_key:
            # Add content to the current subsection
            if 'text' not in stack[-1]:
                stack[-1]['text'] = line
            else:
                stack[-1]['text'] += ' ' + line
        else:
            # If we're not in a subsection, this is probably header text
            if 'header' not in stack[0]:
                stack[0]['header'] = line
            else:
                stack[0]['header'] += ' ' + line
        
        # Check for end of current subsection
        while stack and line.endswith(')') and len(stack) > 1:
            stack.pop()
            if stack:
                current_key = list(stack[-1].keys())[-1] if stack[-1] else None
    
    return stack[0]

def save_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Example usage
input_file = "txt/usc11A@118-105.txt"
output_file = "usc_parsed.json"

document = load_document(input_file)
if document:
    usc_json = parse_usc_to_json(document)
    save_json(usc_json, output_file)
    print(f"Parsed USC has been saved to {output_file}")
else:
    print("Failed to load the document. Please check the file path and try again.")