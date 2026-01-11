import os
from concurrent.futures import ProcessPoolExecutor
from pdfminer.high_level import extract_text
from multiprocessing import freeze_support
import traceback

def pdf_to_text(pdf_file_path, output_txt_path):
    """
    Converts a single PDF into a .txt file with relevant spacing.

    :param pdf_file_path: Path to the input PDF file.
    :param output_txt_path: Path to the output .txt file.
    """
    print(f"[INFO] Starting conversion: {pdf_file_path} -> {output_txt_path}")
    try:
        # Extract text from the PDF
        text = extract_text(pdf_file_path)
        print(f"[INFO] Successfully extracted text from: {pdf_file_path}")

        # Write the extracted text to the .txt file
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        print(f"[INFO] Written text to: {output_txt_path}")
    except Exception as e:
        print(f"[ERROR] Error converting {pdf_file_path}: {e}")
        print(traceback.format_exc())
    finally:
        print(f"[INFO] Finished processing: {os.path.basename(pdf_file_path)}")

def convert_single_pdf(args):
    """Helper function to pass multiple arguments to the ProcessPoolExecutor."""
    pdf_path, txt_path = args
    print(f"[DEBUG] Task received for conversion: {pdf_path} -> {txt_path}")
    pdf_to_text(pdf_path, txt_path)

def convert_folder_pdfs_to_txts(input_folder, output_folder, max_workers=4):
    """
    Converts all PDFs in the input folder to .txt files in the output folder using parallel processing.

    :param input_folder: Path to the folder containing PDF files.
    :param output_folder: Path to the folder where .txt files will be saved.
    :param max_workers: Maximum number of processes to run in parallel.
    """
    print(f"[INFO] Starting folder conversion: {input_folder} -> {output_folder}")
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Created output directory: {output_folder}")

    # Prepare a list of (pdf_path, txt_path) tuples for each PDF file
    tasks = []
    print(f"[DEBUG] Collecting PDF files from {input_folder}...")
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"  # Change extension to .txt
            txt_path = os.path.join(output_folder, txt_filename)
            tasks.append((pdf_path, txt_path))
            print(f"[DEBUG] Task added: {pdf_path} -> {txt_path}")

    print(f"[INFO] Number of PDFs to convert: {len(tasks)}")
    
    # Use ProcessPoolExecutor to parallelize the PDF to TXT conversion
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f"[INFO] Launching ProcessPoolExecutor with {max_workers} workers...")
        try:
            executor.map(convert_single_pdf, tasks)
            print(f"[INFO] Conversion tasks submitted to executor.")
        except KeyboardInterrupt:
            print("[WARNING] Conversion interrupted by user. Shutting down...")
        except Exception as ex:
            print(f"[ERROR] Error during parallel execution: {ex}")
            print(traceback.format_exc())
        finally:
            print(f"[INFO] All PDFs in {input_folder} have been converted to .txt files in {output_folder} using {max_workers} workers.")

# Ensure proper handling of multiprocessing on macOS/Windows
if __name__ == '__main__':
    print("[INFO] Script execution started...")
    freeze_support()  # Required for multiprocessing support on Windows/macOS

    # Example usage
    input_folder = "pdf"  # Replace with your folder path containing PDFs
    output_folder = "txt"  # Replace with the desired output folder for .txt files

    # Print paths for debugging
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Output folder: {output_folder}")

    # Convert all PDFs in the input folder to .txt files using parallel processing
    try:
        convert_folder_pdfs_to_txts(input_folder, output_folder, max_workers=8)  # Adjust max_workers based on your CPU capacity
    except Exception as e:
        print(f"[ERROR] Failed to complete folder conversion: {e}")
        print(traceback.format_exc())
    print("[INFO] Script execution completed.")
