import os
from pathlib import Path
from PIL import Image

def convert_tif_folder_to_pdf(root_directory, output_directory=None):
    """
    Convert TIF files in subfolders to PDF files.
    
    Args:
        root_directory: Path to the directory containing folders with TIF files
        output_directory: Path where PDFs will be saved (defaults to root_directory)
    """
    root_path = Path(root_directory)
    
    if not root_path.exists():
        print(f"Error: Directory '{root_directory}' does not exist")
        return
    
    # Set output directory
    if output_directory is None:
        output_path = root_path
    else:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories
    folders = [f for f in root_path.iterdir() if f.is_dir()]
    
    if not folders:
        print(f"No folders found in '{root_directory}'")
        return
    
    print(f"Found {len(folders)} folders to process\n")
    
    for folder in sorted(folders):
        # Get all TIF files in the folder
        tif_files = sorted(folder.glob('*.tif')) + sorted(folder.glob('*.tiff'))
        
        if not tif_files:
            print(f"Skipping '{folder.name}': No TIF files found")
            continue
        
        print(f"Processing '{folder.name}': {len(tif_files)} TIF file(s)")
        
        try:
            # Open all images
            images = []
            for tif_file in tif_files:
                img = Image.open(tif_file)
                # Convert to RGB if necessary (PDFs require RGB mode)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            
            # Save as PDF
            pdf_filename = output_path / f"{folder.name}.pdf"
            
            if len(images) == 1:
                images[0].save(pdf_filename, 'PDF')
            else:
                # First image is the base, rest are appended
                images[0].save(
                    pdf_filename, 
                    'PDF', 
                    save_all=True, 
                    append_images=images[1:]
                )
            
            print(f"  ✓ Created: {pdf_filename.name}\n")
            
        except Exception as e:
            print(f"  ✗ Error processing '{folder.name}': {str(e)}\n")
        finally:
            # Close all opened images
            for img in images:
                img.close()
    
    print("Conversion complete!")

if __name__ == "__main__":
    # Example usage - modify these paths as needed
    root_dir = "TIF"  # Current directory, or specify your path
    output_dir = None  # None = save PDFs in root_dir, or specify output path
    
    convert_tif_folder_to_pdf(root_dir, output_dir)