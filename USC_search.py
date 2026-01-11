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
        subsection_pattern = re.compile(rf"\({subsection}\).*?(?=\({get_next_sibling(subsection)}\)|\({get_next_parent(search_sequence, i)}\)|§\d+\.|$)", re.DOTALL)
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
    next_section_match = re.search(r"§\d+\.", document[start_index + 1:])
    
    if next_section_match:
        end_index = start_index + 1 + next_section_match.start()
        section_content = document[start_index:end_index].strip()
    else:
        # If no next section is found, extract until the end of the document
        section_content = document[start_index:].strip()
    
    return section_content

# Example usage
file_path = "txt/usc09@118-105.txt"  # Replace with the actual path to your USC document
document = load_document(file_path)

if document:
    result = find_usc_section(document, "12 (a)")
    print(result)
    
    # Example of using the function without subsection
    full_section_result = find_usc_section(document, "12")
    print(full_section_result)
else:
    print("Failed to load the document. Please check the file path and try again.")