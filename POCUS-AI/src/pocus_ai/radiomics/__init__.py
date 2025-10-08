"""
POCUS-AI Radiomics Module

This module integrates PyRadiomics for ultrasound image feature extraction.
"""

from typing import Dict, Any, Union, List, Tuple, Optional
import os
import numpy as np
import warnings

try:
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor
    import pydicom
    import json
    import logging

    # Configure radiomics logger
    radiomics.logger.setLevel(logging.ERROR)  # Set to WARNING, ERROR, or CRITICAL to reduce verbosity
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    warnings.warn("PyRadiomics package not found. Install with 'pip install pyradiomics SimpleITK'")

class UltrasoundRadiomics:
    """Radiomics feature extractor specialized for ultrasound images"""
    
    def __init__(self, 
                 settings: Optional[Dict[str, Any]] = None,
                 config_file: Optional[str] = None):
        """
        Initialize the ultrasound radiomics extractor
        
        Args:
            settings: Dictionary of settings for PyRadiomics
            config_file: Path to a PyRadiomics configuration file
        """
        if not RADIOMICS_AVAILABLE:
            raise ImportError("PyRadiomics package is required for this functionality")
            
        self.settings = settings or {}
        self.config_file = config_file
        
        # Set default settings for ultrasound if not specified
        if not settings and not config_file:
            self.settings = {
                'binWidth': 25,  # Bin width for discretization
                'resampledPixelSpacing': None,  # Use original spacing
                'interpolator': sitk.sitkBSpline,  # B-Spline interpolation
                'verbose': False,
                'removeOutliers': 3,  # Remove outliers outside 3 standard deviations
                'normalize': True,  # Normalize the image
                
                # Features to extract
                'force2D': True,  # Most ultrasound images are effectively 2D
                'enableCExtensions': True,  # Use C extensions for speed
                
                # Feature classes to use
                'featureClass': {
                    'firstorder': None,  # First order statistics (mean, std, entropy, etc.)
                    'glcm': None,        # Gray Level Co-occurrence Matrix
                    'glrlm': None,       # Gray Level Run Length Matrix
                    'glszm': None,       # Gray Level Size Zone Matrix
                    'gldm': None,        # Gray Level Dependence Matrix
                    'ngtdm': None,       # Neighboring Gray Tone Difference Matrix
                    'shape2D': None      # 2D shape features (if segmentation mask provided)
                }
            }
            
        # Initialize the PyRadiomics extractor
        if self.config_file:
            self.extractor = featureextractor.RadiomicsFeatureExtractor(self.config_file)
        else:
            self.extractor = featureextractor.RadiomicsFeatureExtractor(**self.settings)
            
    def extract_features(self, 
                        image: Union[str, np.ndarray, sitk.Image], 
                        mask: Optional[Union[str, np.ndarray, sitk.Image]] = None,
                        dicom_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract radiomics features from an ultrasound image
        
        Args:
            image: The ultrasound image (file path, numpy array, or SimpleITK image)
            mask: Segmentation mask defining ROI (if None, entire image is used)
            dicom_metadata: DICOM metadata dictionary (for additional context)
            
        Returns:
            Dictionary of radiomics features
        """
        # Process the image input
        sitk_image = self._prepare_image(image)
        
        # Create a mask if one wasn't provided (whole image)
        if mask is None:
            mask_array = np.ones(sitk.GetArrayFromImage(sitk_image).shape, dtype=np.uint8)
            sitk_mask = sitk.GetImageFromArray(mask_array)
            sitk_mask.CopyInformation(sitk_image)  # Copy image geometry
        else:
            sitk_mask = self._prepare_mask(mask, sitk_image)
        
        # Extract features
        try:
            features = self.extractor.execute(sitk_image, sitk_mask)
            
            # Process and clean the features
            processed_features = {}
            
            # Convert to native Python types and remove diagnostic info
            for key, value in features.items():
                # Skip diagnostic features (start with "diagnostics_")
                if key.startswith('diagnostics_'):
                    continue
                    
                # Handle different value types
                if isinstance(value, (np.ndarray, np.generic)):
                    processed_features[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
                else:
                    processed_features[key] = value
                    
            # Add metadata if provided
            if dicom_metadata:
                processed_features['metadata'] = {
                    'PatientID': dicom_metadata.get('PatientID', 'Unknown'),
                    'StudyDate': dicom_metadata.get('StudyDate', 'Unknown'),
                    'Modality': dicom_metadata.get('Modality', 'US'),
                    'PixelSpacing': dicom_metadata.get('PixelSpacing', [1, 1])
                }
                
            return processed_features
            
        except Exception as e:
            warnings.warn(f"Feature extraction failed: {str(e)}")
            return {'error': str(e)}
            
    def _prepare_image(self, image: Union[str, np.ndarray, sitk.Image]) -> sitk.Image:
        """
        Convert the input image to SimpleITK format
        
        Args:
            image: Input image (file path, numpy array or SimpleITK image)
            
        Returns:
            SimpleITK image
        """
        if isinstance(image, str):
            # Check if the file is a DICOM file
            if image.lower().endswith('.dcm') or pydicom.misc.is_dicom(image):
                dcm = pydicom.dcmread(image)
                img_array = dcm.pixel_array
                sitk_image = sitk.GetImageFromArray(img_array)
                
                # Try to set spacing if available
                try:
                    if hasattr(dcm, 'PixelSpacing'):
                        sitk_image.SetSpacing([float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]), 1.0])
                except:
                    pass
                    
                return sitk_image
            else:
                # Regular image file
                return sitk.ReadImage(image)
                
        elif isinstance(image, np.ndarray):
            # Convert numpy array to SimpleITK image
            return sitk.GetImageFromArray(image)
            
        elif isinstance(image, sitk.Image):
            # Already SimpleITK image
            return image
            
        else:
            raise TypeError("Image must be a file path, numpy array, or SimpleITK image")
            
    def _prepare_mask(self, mask: Union[str, np.ndarray, sitk.Image], reference_image: sitk.Image) -> sitk.Image:
        """
        Convert the input mask to SimpleITK format and match to reference image
        
        Args:
            mask: Input mask (file path, numpy array or SimpleITK image)
            reference_image: Reference SimpleITK image to match geometry
            
        Returns:
            SimpleITK mask
        """
        if isinstance(mask, str):
            sitk_mask = sitk.ReadImage(mask)
        elif isinstance(mask, np.ndarray):
            sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
            sitk_mask.CopyInformation(reference_image)  # Copy image geometry
        elif isinstance(mask, sitk.Image):
            sitk_mask = mask
        else:
            raise TypeError("Mask must be a file path, numpy array, or SimpleITK image")
            
        # Ensure mask is binary
        if not np.array_equal(np.unique(sitk.GetArrayFromImage(sitk_mask)), [0, 1]):
            binary_mask = sitk.BinaryThreshold(sitk_mask, lowerThreshold=0.5, upperThreshold=float('inf'), 
                                             insideValue=1, outsideValue=0)
            sitk_mask = binary_mask
            
        return sitk_mask
        
    def save_features(self, features: Dict[str, Any], output_path: str) -> None:
        """
        Save extracted features to a JSON file
        
        Args:
            features: Dictionary of radiomics features
            output_path: Path to save the features
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2)
            
    @staticmethod
    def load_features(input_path: str) -> Dict[str, Any]:
        """
        Load features from a JSON file
        
        Args:
            input_path: Path to the feature file
            
        Returns:
            Dictionary of radiomics features
        """
        with open(input_path, 'r') as f:
            return json.load(f)


def extract_features(image, mask=None, settings=None, config_file=None, dicom_metadata=None):
    """
    Extract radiomics features from an ultrasound image
    
    Args:
        image: The ultrasound image (file path, numpy array, or SimpleITK image)
        mask: Segmentation mask defining ROI (if None, entire image is used)
        settings: Dictionary of settings for PyRadiomics
        config_file: Path to a PyRadiomics configuration file
        dicom_metadata: DICOM metadata dictionary (for additional context)
        
    Returns:
        Dictionary of radiomics features
    """
    if not RADIOMICS_AVAILABLE:
        warnings.warn("PyRadiomics is not available. Returning dummy features.")
        return {
            "warning": "PyRadiomics is not installed. Please install with 'pip install pyradiomics SimpleITK'",
            "firstorder_Mean": 0.0,
            "firstorder_Variance": 0.0,
            "dummy_feature": True
        }
        
    extractor = UltrasoundRadiomics(settings=settings, config_file=config_file)
    return extractor.extract_features(image, mask, dicom_metadata=dicom_metadata)