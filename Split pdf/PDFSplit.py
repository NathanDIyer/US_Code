import math
from PyPDF2 import PdfReader, PdfWriter

def split_pdf(input_path, output_prefix):
    # Read the input PDF
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)

    # Calculate the number of pages for each section
    pages_per_section = math.ceil(total_pages / 4)

    # Create 4 new PDF writers
    writers = [PdfWriter() for _ in range(4)]

    # Distribute pages to each writer
    for i, page in enumerate(reader.pages):
        section = min(i // pages_per_section, 3)
        writers[section].add_page(page)

    # Save each section to a new file
    for i, writer in enumerate(writers):
        output_path = f"{output_prefix}_section_{i+1}.pdf"
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

# Example usage
input_pdf = "BIL.pdf"
output_prefix = "output"
split_pdf(input_pdf, output_prefix)