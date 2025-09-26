import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage import morphology, filters, feature
from skimage.morphology import skeletonize, thin
from skimage.filters import frangi, meijering, sato, hessian
import glob

class ThermalVeinDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_name = os.path.splitext(os.path.basename(image_path))[0]
        self.original_image = None
        self.processed_image = None
        self.vein_mask = None
        self.vein_skeleton = None

    def load_and_validate_image(self):
        """Load thermal image and validate it's suitable for processing"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Cannot load image: {self.image_path}")

        # Convert to grayscale for thermal processing
        if len(self.original_image.shape) == 3:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            self.processed_image = self.original_image.copy()

        print(f"Processing {self.image_name}: {self.processed_image.shape}")
        return True

    def thermal_preprocessing(self, save_steps=False, output_dir=None):
        """Advanced preprocessing specifically for thermal vein images"""
        if output_dir and save_steps:
            os.makedirs(output_dir, exist_ok=True)

        # Step 1: Temperature normalization for thermal imagery
        normalized = cv2.normalize(self.processed_image, None, 0, 255, cv2.NORM_MINMAX)
        if save_steps and output_dir:
            cv2.imwrite(os.path.join(output_dir, f'{self.image_name}_1_normalized.png'), normalized)

        # Step 2: Bilateral filtering to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(normalized.astype(np.uint8), 9, 75, 75)
        if save_steps and output_dir:
            cv2.imwrite(os.path.join(output_dir, f'{self.image_name}_2_bilateral.png'), bilateral)

        # Step 3: Adaptive histogram equalization for local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        if save_steps and output_dir:
            cv2.imwrite(os.path.join(output_dir, f'{self.image_name}_3_enhanced.png'), enhanced)

        # Step 4: Gaussian smoothing for vessel continuity
        smoothed = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
        if save_steps and output_dir:
            cv2.imwrite(os.path.join(output_dir, f'{self.image_name}_4_smoothed.png'), smoothed)

        self.processed_image = smoothed
        return self.processed_image

    def vessel_enhancement_multi_scale(self):
        """Multi-scale vessel enhancement using Frangi vesselness filter"""
        # Normalize image to 0-1 range for scikit-image filters
        img_norm = self.processed_image.astype(np.float64) / 255.0

        # Apply Frangi vesselness filter at multiple scales
        scales = np.arange(1, 8, 1)  # Different vessel widths
        frangi_result = frangi(img_norm, sigmas=scales, alpha=0.5,
                              beta=0.5, gamma=15, black_ridges=False)

        # Enhance using Meijering filter (alternative vessel filter)
        meijering_result = meijering(img_norm, sigmas=scales, alpha=None,
                                   black_ridges=False, mode='reflect')

        # Combine both filters for robust detection
        combined = (frangi_result + meijering_result) / 2
        combined = (combined * 255).astype(np.uint8)

        return combined, frangi_result, meijering_result

    def adaptive_thresholding_vein_segmentation(self, vessel_enhanced):
        """Advanced thresholding for vein segmentation"""
        # Otsu's thresholding
        _, otsu_thresh = cv2.threshold(vessel_enhanced, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive thresholding for local variations
        adaptive_thresh = cv2.adaptiveThreshold(vessel_enhanced, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)

        # Combine thresholding methods
        combined_thresh = cv2.bitwise_and(otsu_thresh, adaptive_thresh)

        return combined_thresh, otsu_thresh, adaptive_thresh

    def morphological_vein_refinement(self, binary_mask):
        """Morphological operations to refine vein structures"""
        # Remove noise with opening
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)

        # Close gaps in vessels
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)

        # Remove small artifacts by area filtering
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
        min_area = 50  # Minimum area for valid vein segments

        refined_mask = np.zeros_like(closed)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                refined_mask[labels == i] = 255

        return refined_mask

    def extract_vein_centerlines(self, vein_mask):
        """Extract vein centerlines using skeletonization"""
        # Convert to binary
        binary = (vein_mask > 0).astype(bool)

        # Skeletonize to get centerlines
        skeleton = skeletonize(binary)
        skeleton_img = (skeleton * 255).astype(np.uint8)

        return skeleton_img

    def create_medical_visualization(self, output_dir):
        """Create medical practitioner-friendly visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Thermal Vein Detection Analysis: {self.image_name}', fontsize=16)

        # Original thermal image
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Original Thermal Image')
        axes[0, 0].axis('off')

        # Preprocessed image
        axes[0, 1].imshow(self.processed_image, cmap='hot')
        axes[0, 1].set_title('Enhanced Thermal Image')
        axes[0, 1].axis('off')

        # Vessel enhancement result
        vessel_enhanced, _, _ = self.vessel_enhancement_multi_scale()
        axes[0, 2].imshow(vessel_enhanced, cmap='gray')
        axes[0, 2].set_title('Vessel Enhancement (Frangi+Meijering)')
        axes[0, 2].axis('off')

        # Vein segmentation
        combined_thresh, _, _ = self.adaptive_thresholding_vein_segmentation(vessel_enhanced)
        self.vein_mask = self.morphological_vein_refinement(combined_thresh)
        axes[1, 0].imshow(self.vein_mask, cmap='gray')
        axes[1, 0].set_title('Detected Vein Network')
        axes[1, 0].axis('off')

        # Vein centerlines
        self.vein_skeleton = self.extract_vein_centerlines(self.vein_mask)
        axes[1, 1].imshow(self.vein_skeleton, cmap='gray')
        axes[1, 1].set_title('Vein Centerlines')
        axes[1, 1].axis('off')

        # Overlay on original
        overlay = self.original_image.copy()
        if len(overlay.shape) == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Create colored overlay
        vein_colored = np.zeros((*self.vein_skeleton.shape, 3), dtype=np.uint8)
        vein_colored[self.vein_skeleton > 0] = [0, 255, 0]  # Green veins

        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

        alpha = 0.7
        overlay_result = cv2.addWeighted(overlay, alpha, vein_colored, 1-alpha, 0)

        axes[1, 2].imshow(overlay_result)
        axes[1, 2].set_title('Vein Overlay for Medical Assessment')
        axes[1, 2].axis('off')

        plt.tight_layout()

        # Save the analysis image
        analysis_path = os.path.join(output_dir, f'{self.image_name}_analysis.png')
        plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
        plt.close()

        return analysis_path

    def analyze_vein_characteristics(self):
        """Analyze vein characteristics for medical assessment"""
        if self.vein_mask is None or self.vein_skeleton is None:
            print("Please run full detection pipeline first")
            return

        # Calculate vein network statistics
        total_vein_area = np.sum(self.vein_mask > 0)
        total_vein_length = np.sum(self.vein_skeleton > 0)

        # Find vein segments using connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.vein_mask)

        segments = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            segments.append({
                'area': area,
                'width': width,
                'height': height,
                'aspect_ratio': max(width, height) / max(min(width, height), 1)
            })

        analysis = {
            'image_name': self.image_name,
            'total_vein_area': total_vein_area,
            'total_vein_length': total_vein_length,
            'number_of_segments': len(segments),
            'average_segment_area': np.mean([s['area'] for s in segments]) if segments else 0,
            'vein_density': total_vein_area / (self.processed_image.shape[0] * self.processed_image.shape[1]),
            'segments': segments
        }

        return analysis

def process_all_datasets():
    """Process all thermal images in the datasets directory"""
    # Setup paths
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    output_dir = os.path.join(os.path.dirname(__file__), 'batch_analysis_output')
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(datasets_dir, extension)))

    if not image_files:
        print("No image files found in datasets directory!")
        return

    print(f"Found {len(image_files)} thermal images to process")
    print("="*60)

    # Process each image
    all_analyses = []
    successful_analyses = 0

    for image_path in sorted(image_files):
        try:
            print(f"\nProcessing: {os.path.basename(image_path)}")

            # Initialize detector
            detector = ThermalVeinDetector(image_path)

            # Load and process image
            detector.load_and_validate_image()
            detector.thermal_preprocessing(save_steps=False)

            # Create visualization
            analysis_path = detector.create_medical_visualization(output_dir)

            # Analyze characteristics
            analysis = detector.analyze_vein_characteristics()
            all_analyses.append(analysis)

            print(f"[OK] Analysis saved: {os.path.basename(analysis_path)}")
            print(f"  - Vein area: {analysis['total_vein_area']} pixels")
            print(f"  - Vein segments: {analysis['number_of_segments']}")
            print(f"  - Vein density: {analysis['vein_density']:.4f}")

            successful_analyses += 1

        except Exception as e:
            print(f"[ERROR] Error processing {os.path.basename(image_path)}: {str(e)}")
            continue

    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Successfully processed: {successful_analyses}/{len(image_files)} images")
    print(f"Analysis images saved in: {output_dir}")

    # Create summary report
    if all_analyses:
        create_summary_report(all_analyses, output_dir)

    print("="*60)

def create_summary_report(analyses, output_dir):
    """Create a summary report of all analyses"""
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)

    # Calculate summary statistics
    total_areas = [a['total_vein_area'] for a in analyses]
    total_segments = [a['number_of_segments'] for a in analyses]
    densities = [a['vein_density'] for a in analyses]

    print(f"Average vein area: {np.mean(total_areas):.1f} pixels")
    print(f"Average segments per image: {np.mean(total_segments):.1f}")
    print(f"Average vein density: {np.mean(densities):.4f}")

    # Create detailed summary file
    summary_path = os.path.join(output_dir, 'batch_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("THERMAL VEIN DETECTION - BATCH ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        for analysis in analyses:
            f.write(f"Image: {analysis['image_name']}\n")
            f.write(f"  Total vein area: {analysis['total_vein_area']} pixels\n")
            f.write(f"  Total vein length: {analysis['total_vein_length']} pixels\n")
            f.write(f"  Number of segments: {analysis['number_of_segments']}\n")
            f.write(f"  Average segment area: {analysis['average_segment_area']:.1f} pixels\n")
            f.write(f"  Vein density: {analysis['vein_density']:.4f}\n")
            f.write("-" * 30 + "\n")

        f.write(f"\nOVERALL STATISTICS:\n")
        f.write(f"Average vein area: {np.mean(total_areas):.1f} pixels\n")
        f.write(f"Average segments per image: {np.mean(total_segments):.1f}\n")
        f.write(f"Average vein density: {np.mean(densities):.4f}\n")

    print(f"Detailed summary saved: {summary_path}")

if __name__ == "__main__":
    process_all_datasets()