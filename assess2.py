import cv2
import numpy as np
from pathlib import Path
import sys
import csv
from datetime import datetime

class ImageQualityAnalyzer:
    """Analyzes image quality for OCR processing."""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.results = {}
    
    def analyze_skew(self):
        """Detect text skew angle using Hough Transform."""
        edges = cv2.Canny(self.image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is None:
            self.results['skew_angle'] = 0.0
            self.results['skew_status'] = "UNKNOWN"
            return
        
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            skew_angle = 0.0
        else:
            skew_angle = np.median(angles)
        
        self.results['skew_angle'] = round(skew_angle, 2)
        
        if abs(skew_angle) < 0.5:
            self.results['skew_status'] = "EXCELLENT"
        elif abs(skew_angle) < 2.0:
            self.results['skew_status'] = "GOOD"
        elif abs(skew_angle) < 5.0:
            self.results['skew_status'] = "FAIR"
        else:
            self.results['skew_status'] = "POOR"
    
    def analyze_contrast(self):
        """Measure image contrast using standard deviation."""
        std_dev = np.std(self.image)
        mean_val = np.mean(self.image)
        
        # Michelson contrast for additional metric
        min_val = np.min(self.image)
        max_val = np.max(self.image)
        michelson = (max_val - min_val) / (max_val + min_val + 1e-10)
        
        self.results['contrast_std'] = round(std_dev, 2)
        self.results['contrast_michelson'] = round(michelson, 3)
        self.results['mean_intensity'] = round(mean_val, 2)
        
        if std_dev > 60:
            self.results['contrast_status'] = "EXCELLENT"
        elif std_dev > 40:
            self.results['contrast_status'] = "GOOD"
        elif std_dev > 25:
            self.results['contrast_status'] = "FAIR"
        else:
            self.results['contrast_status'] = "POOR"
    
    def analyze_noise(self):
        """Estimate noise level using Laplacian variance."""
        laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
        noise_variance = laplacian.var()
        
        # Additional noise estimation using local standard deviation
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(self.image.astype(float), -1, kernel)
        local_std = np.sqrt(cv2.filter2D((self.image.astype(float) - local_mean)**2, -1, kernel))
        avg_noise = np.mean(local_std)
        
        self.results['noise_variance'] = round(noise_variance, 2)
        self.results['avg_local_noise'] = round(avg_noise, 2)
        
        if noise_variance < 100:
            self.results['noise_status'] = "EXCELLENT"
        elif noise_variance < 300:
            self.results['noise_status'] = "GOOD"
        elif noise_variance < 600:
            self.results['noise_status'] = "FAIR"
        else:
            self.results['noise_status'] = "POOR"
    
    def analyze_sharpness(self):
        """Measure image sharpness using Laplacian."""
        laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
        sharpness = laplacian.var()
        
        self.results['sharpness_score'] = round(sharpness, 2)
        
        if sharpness > 500:
            self.results['sharpness_status'] = "EXCELLENT"
        elif sharpness > 100:
            self.results['sharpness_status'] = "GOOD"
        elif sharpness > 50:
            self.results['sharpness_status'] = "FAIR"
        else:
            self.results['sharpness_status'] = "POOR"
    
    def check_resolution(self):
        """Check if image resolution is adequate for OCR."""
        height, width = self.image.shape
        total_pixels = height * width
        
        self.results['width'] = width
        self.results['height'] = height
        self.results['resolution'] = f"{width}x{height}"
        
        # Typical OCR needs at least 300 DPI, roughly 200+ pixels per inch of text
        if total_pixels > 2000000:  # ~2MP
            self.results['resolution_status'] = "EXCELLENT"
        elif total_pixels > 1000000:  # ~1MP
            self.results['resolution_status'] = "GOOD"
        elif total_pixels > 500000:
            self.results['resolution_status'] = "FAIR"
        else:
            self.results['resolution_status'] = "POOR"
    
    def calculate_overall_quality(self):
        """Calculate overall OCR readiness score."""
        status_scores = {
            'EXCELLENT': 4,
            'GOOD': 3,
            'FAIR': 2,
            'POOR': 1,
            'UNKNOWN': 2
        }
        
        metrics = [
            self.results.get('skew_status'),
            self.results.get('contrast_status'),
            self.results.get('noise_status'),
            self.results.get('sharpness_status'),
            self.results.get('resolution_status')
        ]
        
        scores = [status_scores.get(m, 2) for m in metrics if m]
        avg_score = np.mean(scores)
        
        if avg_score >= 3.5:
            overall = "EXCELLENT - Ready for OCR"
        elif avg_score >= 2.5:
            overall = "GOOD - Suitable for OCR"
        elif avg_score >= 1.5:
            overall = "FAIR - May need preprocessing"
        else:
            overall = "POOR - Requires significant preprocessing"
        
        self.results['overall_score'] = round(avg_score, 2)
        self.results['overall_assessment'] = overall
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if self.results['skew_status'] in ['FAIR', 'POOR']:
            recommendations.append(f"Deskew ({self.results['skew_angle']}°)")
        
        if self.results['contrast_status'] in ['FAIR', 'POOR']:
            recommendations.append("Enhance contrast")
        
        if self.results['noise_status'] in ['FAIR', 'POOR']:
            recommendations.append("Reduce noise")
        
        if self.results['sharpness_status'] in ['FAIR', 'POOR']:
            recommendations.append("Image blurry - rescan")
        
        if self.results['resolution_status'] in ['FAIR', 'POOR']:
            recommendations.append("Increase resolution (300+ DPI)")
        
        self.results['recommendations'] = "; ".join(recommendations) if recommendations else "None"
    
    def analyze(self):
        """Run all analyses."""
        self.check_resolution()
        self.analyze_skew()
        self.analyze_contrast()
        self.analyze_noise()
        self.analyze_sharpness()
        self.calculate_overall_quality()
        self.generate_recommendations()
        
        return self.results


def extract_metadata_from_path(file_path):
    """Extract volume, folio, and page from file path structure.
    
    Expected folder structure:
    Title_Document_Volume_Number___1201__Folio_Number___1/page.tif
    """
    parts = file_path.parts
    
    # Default values
    volume = ""
    folio = ""
    page = file_path.stem  # filename without extension
    
    # Get parent folder name
    if len(parts) >= 2:
        folder_name = parts[-2]
        
        # Parse folder name format: Title_Document_Volume_Number___1201__Folio_Number___1
        # Extract Volume Number and Folio Number using string parsing
        
        # Find Volume Number
        if "Volume_Number___" in folder_name:
            volume_start = folder_name.find("Volume_Number___") + len("Volume_Number___")
            volume_end = folder_name.find("__", volume_start)
            if volume_end != -1:
                volume = folder_name[volume_start:volume_end]
            else:
                volume = folder_name[volume_start:]
        
        # Find Folio Number
        if "Folio_Number___" in folder_name:
            folio_start = folder_name.find("Folio_Number___") + len("Folio_Number___")
            # Get everything after Folio_Number___ (may have trailing content)
            folio_part = folder_name[folio_start:]
            # Remove any trailing slashes or underscores
            folio = folio_part.strip("/_")
    
    return volume, folio, page


def find_tif_files(root_dir):
    """Recursively find all TIF files in directory tree."""
    root_path = Path(root_dir)
    tif_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    
    tif_files = []
    for ext in tif_extensions:
        tif_files.extend(root_path.rglob(f'*{ext}'))
    
    return sorted(tif_files)


def process_directory(root_dir, output_csv):
    """Process all TIF files in directory tree and output to CSV."""
    
    print(f"Scanning directory: {root_dir}")
    tif_files = find_tif_files(root_dir)
    
    if not tif_files:
        print("No TIF files found in directory tree.")
        return
    
    print(f"Found {len(tif_files)} TIF files to process.\n")
    
    # CSV column headers
    fieldnames = [
        'Volume',
        'Folio',
        'Page',
        'Resolution',
        'Skew Angle',
        'Skew Status',
        'Standard Deviation',
        'Michelson Contrast',
        'Mean Intensity',
        'Contrast Status',
        'Variance',
        'Avg Local Noise',
        'Noise Status',
        'Sharpness Score',
        'Sharpness Status',
        'Overall Score',
        'Overall Assessment',
        'Recommendations',
        'File Path'
    ]
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each file
        processed = 0
        failed = 0
        
        for idx, tif_file in enumerate(tif_files, 1):
            try:
                print(f"[{idx}/{len(tif_files)}] Processing: {tif_file.name}", end=' ... ')
                
                # Extract metadata from path
                volume, folio, page = extract_metadata_from_path(tif_file)
                
                # Analyze image
                analyzer = ImageQualityAnalyzer(tif_file)
                results = analyzer.analyze()
                
                # Prepare CSV row
                row = {
                    'Volume': volume,
                    'Folio': folio,
                    'Page': page,
                    'Resolution': results['resolution'],
                    'Skew Angle': results['skew_angle'],
                    'Skew Status': results['skew_status'],
                    'Standard Deviation': results['contrast_std'],
                    'Michelson Contrast': results['contrast_michelson'],
                    'Mean Intensity': results['mean_intensity'],
                    'Contrast Status': results['contrast_status'],
                    'Variance': results['noise_variance'],
                    'Avg Local Noise': results['avg_local_noise'],
                    'Noise Status': results['noise_status'],
                    'Sharpness Score': results['sharpness_score'],
                    'Sharpness Status': results['sharpness_status'],
                    'Overall Score': results['overall_score'],
                    'Overall Assessment': results['overall_assessment'],
                    'Recommendations': results['recommendations'],
                    'File Path': str(tif_file)
                }
                
                writer.writerow(row)
                processed += 1
                print("✓")
                
            except Exception as e:
                failed += 1
                print(f"✗ Error: {e}")
                
                # Write error row
                row = {
                    'Volume': volume if 'volume' in locals() else '',
                    'Folio': folio if 'folio' in locals() else '',
                    'Page': page if 'page' in locals() else tif_file.name,
                    'Resolution': 'ERROR',
                    'Skew Angle': '',
                    'Skew Status': 'ERROR',
                    'Standard Deviation': '',
                    'Michelson Contrast': '',
                    'Mean Intensity': '',
                    'Contrast Status': 'ERROR',
                    'Variance': '',
                    'Avg Local Noise': '',
                    'Noise Status': 'ERROR',
                    'Sharpness Score': '',
                    'Sharpness Status': 'ERROR',
                    'Overall Score': '',
                    'Overall Assessment': f'ERROR: {str(e)}',
                    'Recommendations': '',
                    'File Path': str(tif_file)
                }
                writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total files: {len(tif_files)}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {output_csv}")
    print(f"{'='*60}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_path> [output_csv]")
        print("Example: python script.py /path/to/images output.csv")
        print("\nIf output_csv is not specified, it will default to 'ocr_quality_report_TIMESTAMP.csv'")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    
    if not Path(root_dir).exists():
        print(f"Error: Directory not found: {root_dir}")
        sys.exit(1)
    
    if not Path(root_dir).is_dir():
        print(f"Error: Path is not a directory: {root_dir}")
        sys.exit(1)
    
    # Determine output CSV filename
    if len(sys.argv) >= 3:
        output_csv = sys.argv[2]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"ocr_quality_report_{timestamp}.csv"
    
    try:
        process_directory(root_dir, output_csv)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
