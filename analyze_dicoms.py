import os
import pydicom
from pathlib import Path
import numpy as np
import logging
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dicom_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DicomAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.stats = {
            'mri': defaultdict(list),
            'mask': defaultdict(list),
            'pairs': defaultdict(int)
        }
    
    def analyze_file(self, dcm_path, file_type):
        """Analyze a single DICOM file."""
        try:
            dcm = pydicom.dcmread(str(dcm_path))
            pixel_array = dcm.pixel_array
            
            # Get basic shape info
            shape = pixel_array.shape
            size = pixel_array.size
            dtype = pixel_array.dtype
            
            # Get value range
            min_val = float(pixel_array.min())
            max_val = float(pixel_array.max())
            unique_vals = len(np.unique(pixel_array))
            
            # Get DICOM metadata
            metadata = {
                'Rows': getattr(dcm, 'Rows', None),
                'Columns': getattr(dcm, 'Columns', None),
                'SliceThickness': getattr(dcm, 'SliceThickness', None),
                'PixelSpacing': getattr(dcm, 'PixelSpacing', None),
                'Manufacturer': getattr(dcm, 'Manufacturer', None),
            }
            
            info = {
                'shape': shape,
                'size': size,
                'dtype': dtype,
                'value_range': (min_val, max_val),
                'unique_values': unique_vals,
                'metadata': metadata
            }
            
            self.stats[file_type]['shapes'].append(shape)
            self.stats[file_type]['sizes'].append(size)
            self.stats[file_type]['dtypes'].append(dtype)
            self.stats[file_type]['value_ranges'].append((min_val, max_val))
            
            return info
            
        except Exception as e:
            logger.error(f"Error analyzing {dcm_path}: {str(e)}")
            return None
    
    def analyze_dataset(self):
        """Analyze all DICOM files in the dataset."""
        logger.info(f"Starting analysis of data in {self.data_dir}")
        
        # Get all case directories
        case_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(case_dirs)} case directories")
        
        case_stats = []
        for case_dir in tqdm(case_dirs, desc="Analyzing cases"):
            mri_path = case_dir / 'mri.dcm'
            mask_path = case_dir / 'mask.dcm'
            
            case_info = {
                'case_name': case_dir.name,
                'mri': None,
                'mask': None
            }
            
            if mri_path.exists() and mask_path.exists():
                self.stats['pairs']['complete'] += 1
                case_info['mri'] = self.analyze_file(mri_path, 'mri')
                case_info['mask'] = self.analyze_file(mask_path, 'mask')
            else:
                self.stats['pairs']['incomplete'] += 1
                logger.warning(f"Incomplete pair in {case_dir}")
            
            case_stats.append(case_info)
        
        return self.summarize_stats(case_stats)
    
    def summarize_stats(self, case_stats):
        """Generate summary statistics."""
        summary = {
            'total_cases': len(case_stats),
            'complete_pairs': self.stats['pairs']['complete'],
            'incomplete_pairs': self.stats['pairs']['incomplete'],
            'mri_shapes': {},
            'mask_shapes': {},
            'value_ranges': {
                'mri': {'min': float('inf'), 'max': float('-inf')},
                'mask': {'min': float('inf'), 'max': float('-inf')}
            }
        }
        
        # Analyze shapes
        mri_shapes = [case['mri']['shape'] for case in case_stats if case['mri']]
        mask_shapes = [case['mask']['shape'] for case in case_stats if case['mask']]
        
        for shape in mri_shapes:
            shape_str = str(shape)
            if shape_str not in summary['mri_shapes']:
                summary['mri_shapes'][shape_str] = 0
            summary['mri_shapes'][shape_str] += 1
        
        for shape in mask_shapes:
            shape_str = str(shape)
            if shape_str not in summary['mask_shapes']:
                summary['mask_shapes'][shape_str] = 0
            summary['mask_shapes'][shape_str] += 1
        
        # Find global value ranges
        for case in case_stats:
            if case['mri']:
                min_val, max_val = case['mri']['value_range']
                summary['value_ranges']['mri']['min'] = min(summary['value_ranges']['mri']['min'], min_val)
                summary['value_ranges']['mri']['max'] = max(summary['value_ranges']['mri']['max'], max_val)
            if case['mask']:
                min_val, max_val = case['mask']['value_range']
                summary['value_ranges']['mask']['min'] = min(summary['value_ranges']['mask']['min'], min_val)
                summary['value_ranges']['mask']['max'] = max(summary['value_ranges']['mask']['max'], max_val)
        
        return summary, case_stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze DICOM dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to directory containing DICOM data')
    args = parser.parse_args()
    
    analyzer = DicomAnalyzer(args.data_dir)
    summary, case_stats = analyzer.analyze_dataset()
    
    # Print summary
    print("\n=== Dataset Summary ===")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Complete pairs: {summary['complete_pairs']}")
    print(f"Incomplete pairs: {summary['incomplete_pairs']}")
    
    print("\n=== MRI Shapes ===")
    for shape, count in summary['mri_shapes'].items():
        print(f"Shape {shape}: {count} cases")
    
    print("\n=== Mask Shapes ===")
    for shape, count in summary['mask_shapes'].items():
        print(f"Shape {shape}: {count} cases")
    
    print("\n=== Value Ranges ===")
    print("MRI: ", summary['value_ranges']['mri'])
    print("Mask:", summary['value_ranges']['mask'])
    
    # Sample case details
    print("\n=== Sample Case Details ===")
    if case_stats:
        sample_case = next((case for case in case_stats if case['mri'] and case['mask']), None)
        if sample_case:
            print(f"Case name: {sample_case['case_name']}")
            print("MRI metadata:", sample_case['mri']['metadata'])
            print("Mask metadata:", sample_case['mask']['metadata'])

if __name__ == "__main__":
    main()