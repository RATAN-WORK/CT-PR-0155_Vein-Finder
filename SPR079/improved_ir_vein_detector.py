import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage import morphology, filters, feature, measure
from skimage.morphology import skeletonize, thin, disk
from skimage.filters import frangi, meijering, sato, hessian
from skimage.segmentation import active_contour

class ImprovedIRVeinDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.enhanced_veins = None
        self.vein_mask = None
        self.vein_skeleton = None

    def load_and_validate_image(self):
        """Load IR image and validate it's suitable for processing"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Cannot load image: {self.image_path}")

        # Convert to grayscale for IR processing
        if len(self.original_image.shape) == 3:
            self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            self.processed_image = self.original_image.copy()

        print(f"IR Image loaded: {self.processed_image.shape}")
        return True

    def ir_preprocessing(self, save_steps=False, output_dir=None):
        """Gentle preprocessing that preserves vein information"""
        if output_dir and save_steps:
            os.makedirs(output_dir, exist_ok=True)

        # Step 1: Normalize to full intensity range
        normalized = cv2.normalize(self.processed_image, None, 0, 255, cv2.NORM_MINMAX)
        if save_steps and output_dir:
            cv2.imwrite(os.path.join(output_dir, '1_normalized.png'), normalized)

        # Step 2: Very gentle denoising to preserve vein details
        denoised = cv2.bilateralFilter(normalized.astype(np.uint8), 5, 50, 50)
        if save_steps and output_dir:
            cv2.imwrite(os.path.join(output_dir, '2_denoised.png'), denoised)

        # Step 3: Moderate contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        if save_steps and output_dir:
            cv2.imwrite(os.path.join(output_dir, '3_enhanced.png'), enhanced)

        self.processed_image = enhanced
        return self.processed_image

    def ir_vein_enhancement(self):
        """Enhanced vein detection optimized for IR images"""
        # Normalize for scikit-image filters
        img_norm = self.processed_image.astype(np.float64) / 255.0

        # Fine-tuned scales for IR vein detection
        scales = np.arange(0.5, 4.0, 0.5)  # Smaller scales for finer details

        # Frangi filter optimized for IR dark veins
        frangi_result = frangi(img_norm, sigmas=scales, alpha=0.5,
                              beta=0.5, gamma=20, black_ridges=True)

        # Meijering filter for additional vein enhancement
        meijering_result = meijering(img_norm, sigmas=scales, alpha=None,
                                   black_ridges=True, mode='reflect')

        # Sato filter for tubular structures
        sato_result = sato(img_norm, sigmas=scales, black_ridges=True, mode='reflect')

        # Combine filters with optimized weights
        combined = (0.5 * frangi_result + 0.3 * meijering_result + 0.2 * sato_result)
        combined = np.clip(combined, 0, 1)

        # Convert to 8-bit and enhance contrast
        combined_8bit = (combined * 255).astype(np.uint8)

        # Apply additional contrast enhancement to the combined result
        combined_enhanced = cv2.equalizeHist(combined_8bit)

        self.enhanced_veins = combined_enhanced
        return combined_enhanced, frangi_result, meijering_result, sato_result

    def gentle_segmentation(self, vessel_enhanced):
        """Gentle segmentation that preserves vein structure from enhancement"""

        # Method 1: Use multiple threshold levels and combine
        thresholds = [30, 40, 50, 60, 70]  # Multiple threshold levels
        masks = []

        for thresh in thresholds:
            _, binary = cv2.threshold(vessel_enhanced, thresh, 255, cv2.THRESH_BINARY)
            masks.append(binary)

        # Combine masks using voting (majority wins)
        combined_mask = np.zeros_like(vessel_enhanced)
        for i in range(vessel_enhanced.shape[0]):
            for j in range(vessel_enhanced.shape[1]):
                votes = sum([mask[i,j] > 0 for mask in masks])
                if votes >= 2:  # At least 2 out of 5 thresholds agree
                    combined_mask[i,j] = 255

        # Method 2: Percentile-based thresholding
        percentile_thresh = np.percentile(vessel_enhanced[vessel_enhanced > 0], 85)
        _, percentile_mask = cv2.threshold(vessel_enhanced, percentile_thresh, 255, cv2.THRESH_BINARY)

        # Method 3: Adaptive thresholding with larger block size
        adaptive_mask = cv2.adaptiveThreshold(vessel_enhanced, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 21, 5)

        # Combine all three methods
        final_mask = cv2.bitwise_or(combined_mask, percentile_mask)
        final_mask = cv2.bitwise_or(final_mask, adaptive_mask)

        return final_mask, combined_mask, percentile_mask, adaptive_mask

    def preserve_vein_morphology(self, binary_mask):
        """Morphological operations that preserve vein structure"""

        # Very gentle noise removal
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_tiny)

        # Connect nearby vein segments with minimal closing
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_connect)

        # Area-based filtering - keep reasonable vein segments
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected)

        # More lenient area filtering
        min_area = 15  # Smaller minimum to keep fine veins
        max_area = 15000  # Reasonable maximum

        refined_mask = np.zeros_like(connected)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Also consider aspect ratio for vein-like shapes
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(width, height) / max(min(width, height), 1)

            # Keep if area is good OR if it's elongated (vein-like)
            if (min_area <= area <= max_area) or (aspect_ratio > 3 and area > 8):
                refined_mask[labels == i] = 255

        return refined_mask

    def extract_enhanced_centerlines(self, vein_mask):
        """Extract centerlines with minimal cleaning to preserve structure"""
        # Convert to binary
        binary = (vein_mask > 0).astype(bool)

        # Skeletonize to get centerlines
        skeleton = skeletonize(binary)
        skeleton_img = (skeleton * 255).astype(np.uint8)

        # Very minimal cleaning - only remove tiny isolated pixels
        cleaned = self.minimal_skeleton_cleaning(skeleton_img, min_length=5)

        return cleaned, skeleton_img

    def minimal_skeleton_cleaning(self, skeleton, min_length=5):
        """Minimal cleaning to preserve vein structure"""
        # Find very small isolated components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton)

        cleaned = skeleton.copy()
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_length:
                cleaned[labels == i] = 0

        return cleaned

    def create_enhanced_visualization(self, output_dir):
        """Create visualization that highlights the enhanced results"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Improved IR Vein Detection - Enhanced Pipeline', fontsize=16)

        # Row 1: Original processing
        if len(self.original_image.shape) == 3:
            display_orig = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        else:
            display_orig = self.original_image
        axes[0, 0].imshow(display_orig, cmap='gray')
        axes[0, 0].set_title('Original IR Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.processed_image, cmap='gray')
        axes[0, 1].set_title('Preprocessed IR Image')
        axes[0, 1].axis('off')

        # The star of the show - enhanced veins
        vessel_enhanced, frangi_result, meijering_result, sato_result = self.ir_vein_enhancement()
        axes[0, 2].imshow(vessel_enhanced, cmap='hot')
        axes[0, 2].set_title('★ ENHANCED VEINS (Best Result) ★')
        axes[0, 2].axis('off')

        # Row 2: Individual filter results
        axes[1, 0].imshow(frangi_result, cmap='hot')
        axes[1, 0].set_title('Frangi Filter')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(meijering_result, cmap='hot')
        axes[1, 1].set_title('Meijering Filter')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(sato_result, cmap='hot')
        axes[1, 2].set_title('Sato Filter')
        axes[1, 2].axis('off')

        # Row 3: Improved segmentation results
        segmented, _, _, _ = self.gentle_segmentation(vessel_enhanced)
        self.vein_mask = self.preserve_vein_morphology(segmented)
        axes[2, 0].imshow(self.vein_mask, cmap='gray')
        axes[2, 0].set_title('Improved Vein Segmentation')
        axes[2, 0].axis('off')

        # Centerlines
        self.vein_skeleton, _ = self.extract_enhanced_centerlines(self.vein_mask)
        axes[2, 1].imshow(self.vein_skeleton, cmap='gray')
        axes[2, 1].set_title('Vein Centerlines')
        axes[2, 1].axis('off')

        # Final overlay using the enhanced veins directly
        overlay = display_orig.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

        # Create overlay using the enhanced image as intensity
        vein_colored = np.zeros((*vessel_enhanced.shape, 3), dtype=np.uint8)
        # Use the enhanced veins intensity for green channel
        vein_colored[:, :, 1] = vessel_enhanced  # Green channel intensity based on enhancement

        alpha = 0.6
        overlay_result = cv2.addWeighted(overlay, alpha, vein_colored, 1-alpha, 0)
        axes[2, 2].imshow(overlay_result)
        axes[2, 2].set_title('Enhanced Vein Overlay')
        axes[2, 2].axis('off')

        plt.tight_layout()

        # Save the analysis
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_improved_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save the enhanced veins as a separate high-quality image
        enhanced_path = os.path.join(output_dir, f'{base_name}_enhanced_veins.png')
        cv2.imwrite(enhanced_path, vessel_enhanced)

        # Save the overlay result
        overlay_path = os.path.join(output_dir, f'{base_name}_enhanced_overlay.png')
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_result, cv2.COLOR_RGB2BGR))

        return output_path, enhanced_path

    def analyze_enhanced_results(self):
        """Analysis focusing on the enhanced vein visibility"""
        if self.enhanced_veins is None:
            print("Please run enhancement pipeline first")
            return None

        # Use the enhanced veins image for analysis instead of binary mask
        enhanced_binary = self.enhanced_veins > np.percentile(self.enhanced_veins, 75)

        # Calculate statistics from enhanced image
        total_enhanced_area = np.sum(enhanced_binary)
        mean_intensity = np.mean(self.enhanced_veins[enhanced_binary])
        max_intensity = np.max(self.enhanced_veins)

        # Skeleton analysis if available
        skeleton_length = np.sum(self.vein_skeleton > 0) if self.vein_skeleton is not None else 0

        # Connected components from enhanced result
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(enhanced_binary.astype(np.uint8) * 255)

        analysis = {
            'enhanced_vein_area': total_enhanced_area,
            'enhanced_vein_coverage_percentage': (total_enhanced_area / (self.enhanced_veins.shape[0] * self.enhanced_veins.shape[1])) * 100,
            'mean_vein_intensity': mean_intensity,
            'max_vein_intensity': max_intensity,
            'skeleton_length': skeleton_length,
            'number_of_enhanced_segments': num_labels - 1,
            'enhancement_quality_score': mean_intensity / 255.0,
            'image_dimensions': self.enhanced_veins.shape
        }

        return analysis

def process_ir_image_improved(image_path, output_dir):
    """Process IR image with improved pipeline"""
    try:
        detector = ImprovedIRVeinDetector(image_path)

        # Load and process
        detector.load_and_validate_image()
        detector.ir_preprocessing(save_steps=True, output_dir=output_dir)

        # Create enhanced visualization
        viz_path, enhanced_path = detector.create_enhanced_visualization(output_dir)

        # Analyze results
        analysis = detector.analyze_enhanced_results()

        return detector, analysis, viz_path, enhanced_path

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None, None, None

def main():
    """Test the improved IR detection"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ir_datasets_dir = os.path.join(base_dir, 'datasets_ir')
    output_dir = os.path.join(os.path.dirname(__file__), 'improved_ir_output')

    os.makedirs(output_dir, exist_ok=True)

    # Get first IR image for testing
    ir_images = [f for f in os.listdir(ir_datasets_dir) if f.endswith('.png')]
    if not ir_images:
        print("No IR images found")
        return

    test_image = os.path.join(ir_datasets_dir, ir_images[0])
    print(f"Processing with improved pipeline: {test_image}")

    detector, analysis, viz_path, enhanced_path = process_ir_image_improved(test_image, output_dir)

    if analysis:
        print("\n" + "="*60)
        print("IMPROVED IR VEIN DETECTION RESULTS")
        print("="*60)
        print(f"Enhanced vein coverage: {analysis['enhanced_vein_coverage_percentage']:.2f}%")
        print(f"Mean vein intensity: {analysis['mean_vein_intensity']:.1f}/255")
        print(f"Enhancement quality score: {analysis['enhancement_quality_score']:.3f}")
        print(f"Number of enhanced segments: {analysis['number_of_enhanced_segments']}")
        print(f"\nVisualization: {viz_path}")
        print(f"Enhanced veins image: {enhanced_path}")
        print("="*60)

if __name__ == "__main__":
    main()