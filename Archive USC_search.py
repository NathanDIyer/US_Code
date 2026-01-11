import re

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

def find_usc_section(document, input_string):
    section_identifier, search_sequence = parse_input(input_string)
    
    # Find the section
    section_pattern = re.compile(rf"{section_identifier}.*?(?=§\d+\.|$)", re.DOTALL)
    section_match = section_pattern.search(document)
    
    if not section_match:
        return "Section not found"
    
    section_text = section_match.group()
    current_text = section_text
    
    for i, subsection in enumerate(search_sequence):
        subsection_pattern = re.compile(rf"\({subsection}\).*?(?=\({get_next_sibling(subsection)}\)|\({get_next_parent(search_sequence, i)}\)|§\d+\.|$)", re.DOTALL)
        subsection_match = subsection_pattern.search(current_text)
        
        if not subsection_match:
            return f"Subsection {subsection} not found"
        
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

# Example usage
file_path = "txt/usc09@118-105.txt"  # Replace with the actual path to your USC document
document = load_document(file_path)

if document:
    result = find_usc_section(document, "16 (a)")
    print(result)
else:
    print("Failed to load the document. Please check the file path and try again.")