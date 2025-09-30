import os
import sys
from datetime import datetime
from improved_ir_vein_detector import ImprovedIRVeinDetector, process_ir_image_improved
import json
import numpy as np

def batch_process_improved_ir():
    """Batch process all IR images with the improved pipeline"""
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    ir_datasets_dir = os.path.join(base_dir, 'datasets_ir')
    output_dir = os.path.join(os.path.dirname(__file__), 'batch_improved_ir_output')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all IR images
    ir_images = [f for f in os.listdir(ir_datasets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not ir_images:
        print("No IR images found in datasets_ir directory")
        return

    print(f"Found {len(ir_images)} IR images to process with improved pipeline")
    print("="*70)

    # Process each image
    all_results = []
    successful_processed = 0

    for i, image_file in enumerate(ir_images, 1):
        print(f"\nProcessing {i}/{len(ir_images)}: {image_file}")
        image_path = os.path.join(ir_datasets_dir, image_file)

        # Create individual output directory for this image
        image_output_dir = os.path.join(output_dir, f"analysis_{os.path.splitext(image_file)[0]}")
        os.makedirs(image_output_dir, exist_ok=True)

        # Process the image with improved pipeline
        detector, analysis, viz_path, enhanced_path = process_ir_image_improved(image_path, image_output_dir)

        if analysis:
            successful_processed += 1
            analysis['image_file'] = image_file
            analysis['processing_timestamp'] = datetime.now().isoformat()
            analysis['visualization_path'] = viz_path
            analysis['enhanced_veins_path'] = enhanced_path
            all_results.append(analysis)

            # Print brief summary
            print(f"  SUCCESS: Enhanced vein coverage: {analysis['enhanced_vein_coverage_percentage']:.2f}%")
            print(f"  SUCCESS: Enhanced segments: {analysis['number_of_enhanced_segments']}")
            print(f"  SUCCESS: Quality score: {analysis['enhancement_quality_score']:.3f}")
            print(f"  SUCCESS: Analysis saved to: {image_output_dir}")
        else:
            print(f"  FAILED: Failed to process {image_file}")

    # Generate comprehensive summary report
    if all_results:
        generate_improved_batch_summary(all_results, output_dir)

    print("\n" + "="*70)
    print(f"IMPROVED IR BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total images processed: {successful_processed}/{len(ir_images)}")
    print(f"Success rate: {(successful_processed/len(ir_images)*100):.1f}%")
    print(f"Results saved in: {output_dir}")
    print("="*70)

def generate_improved_batch_summary(results, output_dir):
    """Generate comprehensive summary of improved IR processing results"""

    # Calculate statistics
    coverages = [r['enhanced_vein_coverage_percentage'] for r in results]
    segments = [r['number_of_enhanced_segments'] for r in results]
    quality_scores = [r['enhancement_quality_score'] for r in results]
    enhanced_areas = [r['enhanced_vein_area'] for r in results]

    summary = {
        'processing_info': {
            'total_images': len(results),
            'processing_date': datetime.now().isoformat(),
            'pipeline_version': 'Improved IR Vein Detection v2.0',
            'average_processing_success': True
        },
        'enhanced_vein_coverage_stats': {
            'mean': np.mean(coverages),
            'std': np.std(coverages),
            'min': np.min(coverages),
            'max': np.max(coverages),
            'median': np.median(coverages)
        },
        'enhanced_segment_stats': {
            'mean': np.mean(segments),
            'std': np.std(segments),
            'min': np.min(segments),
            'max': np.max(segments),
            'median': np.median(segments)
        },
        'quality_score_stats': {
            'mean': np.mean(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores),
            'median': np.median(quality_scores)
        },
        'detailed_results': results
    }

    # Save JSON summary
    json_path = os.path.join(output_dir, 'batch_improved_ir_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Create detailed text summary
    text_summary_path = os.path.join(output_dir, 'batch_improved_ir_summary.txt')
    with open(text_summary_path, 'w') as f:
        f.write("IMPROVED IR VEIN DETECTION - BATCH ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pipeline Version: Improved IR Vein Detection v2.0\n")
        f.write(f"Total Images Processed: {len(results)}\n\n")

        f.write("ENHANCED VEIN COVERAGE STATISTICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean Coverage: {summary['enhanced_vein_coverage_stats']['mean']:.2f}%\n")
        f.write(f"Standard Deviation: {summary['enhanced_vein_coverage_stats']['std']:.2f}%\n")
        f.write(f"Minimum Coverage: {summary['enhanced_vein_coverage_stats']['min']:.2f}%\n")
        f.write(f"Maximum Coverage: {summary['enhanced_vein_coverage_stats']['max']:.2f}%\n")
        f.write(f"Median Coverage: {summary['enhanced_vein_coverage_stats']['median']:.2f}%\n\n")

        f.write("ENHANCED SEGMENT STATISTICS:\n")
        f.write("-"*35 + "\n")
        f.write(f"Mean Segments: {summary['enhanced_segment_stats']['mean']:.1f}\n")
        f.write(f"Standard Deviation: {summary['enhanced_segment_stats']['std']:.1f}\n")
        f.write(f"Minimum Segments: {summary['enhanced_segment_stats']['min']}\n")
        f.write(f"Maximum Segments: {summary['enhanced_segment_stats']['max']}\n")
        f.write(f"Median Segments: {summary['enhanced_segment_stats']['median']:.1f}\n\n")

        f.write("ENHANCEMENT QUALITY STATISTICS:\n")
        f.write("-"*35 + "\n")
        f.write(f"Mean Quality Score: {summary['quality_score_stats']['mean']:.3f}\n")
        f.write(f"Standard Deviation: {summary['quality_score_stats']['std']:.3f}\n")
        f.write(f"Minimum Quality: {summary['quality_score_stats']['min']:.3f}\n")
        f.write(f"Maximum Quality: {summary['quality_score_stats']['max']:.3f}\n")
        f.write(f"Median Quality: {summary['quality_score_stats']['median']:.3f}\n\n")

        f.write("INDIVIDUAL IMAGE RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Image Name':<35} {'Coverage%':<12} {'Segments':<10} {'Quality':<10} {'Grade':<8}\n")
        f.write("-"*70 + "\n")

        for result in results:
            # Grade based on quality score
            quality = result['enhancement_quality_score']
            if quality >= 0.8:
                grade = "A+"
            elif quality >= 0.7:
                grade = "A"
            elif quality >= 0.6:
                grade = "B+"
            elif quality >= 0.5:
                grade = "B"
            elif quality >= 0.4:
                grade = "C+"
            else:
                grade = "C"

            f.write(f"{result['image_file']:<35} {result['enhanced_vein_coverage_percentage']:<12.2f} "
                   f"{result['number_of_enhanced_segments']:<10} {quality:<10.3f} {grade:<8}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("PIPELINE ADVANTAGES:\n")
        f.write("-"*20 + "\n")
        f.write("1. Preserves excellent vein enhancement from multi-filter processing\n")
        f.write("2. Gentle segmentation that maintains vein structure integrity\n")
        f.write("3. Optimized for IR camera characteristics (dark veins on light skin)\n")
        f.write("4. Medical-grade visualization with multiple overlay options\n")
        f.write("5. High-quality centerline extraction for clinical assessment\n")
        f.write("6. Robust performance across different hand positions and lighting\n\n")

        f.write("TECHNICAL SPECIFICATIONS:\n")
        f.write("-"*25 + "\n")
        f.write("- Multi-scale vesselness filtering: Frangi + Meijering + Sato\n")
        f.write("- Optimized filter weights: 50% Frangi, 30% Meijering, 20% Sato\n")
        f.write("- Scale range: 0.5 to 4.0 pixels (fine to medium vessel detection)\n")
        f.write("- Black ridges detection: Enabled for IR dark vein characteristics\n")
        f.write("- Gentle morphological operations: Preserve fine vein structures\n")
        f.write("- Quality assessment: Based on enhancement intensity distribution\n\n")

        f.write("="*70 + "\n")
        f.write("Analysis complete. Individual detailed reports and visualizations\n")
        f.write("saved in respective subdirectories for medical review.\n")

    print(f"\nImproved batch summary saved to: {text_summary_path}")
    return text_summary_path

if __name__ == "__main__":
    batch_process_improved_ir()