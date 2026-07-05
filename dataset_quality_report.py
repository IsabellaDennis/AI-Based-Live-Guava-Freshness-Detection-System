import os
import sys

# Ensure the app directory is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils import (
    check_class_imbalance,
    find_duplicates,
    find_corrupted_and_unreadable,
    find_blurry_images,
    get_image_dimensions
)

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

def main():
    print("=" * 60)
    print("           GUAVA DATASET QUALITY CHECK & REPORT           ")
    print("=" * 60)
    
    # 1. Class Imbalance
    imbalance = check_class_imbalance(DATASET_DIR)
    
    # 2. Corrupted and Unreadable
    corrupted, unreadable = find_corrupted_and_unreadable(DATASET_DIR)
    
    # 3. Duplicates
    duplicates = find_duplicates(DATASET_DIR)
    
    # 4. Blurry images (threshold < 100)
    blurry = find_blurry_images(DATASET_DIR, threshold=100.0)
    
    # 5. Dimensions
    dimensions = get_image_dimensions(DATASET_DIR)
    
    # Build markdown report content
    report_lines = []
    report_lines.append("# Guava Dataset Quality Check Report\n")
    
    report_lines.append("## 1. Class Imbalance Report")
    total_images = sum(imbalance.values())
    for category, count in imbalance.items():
        percentage = (count / total_images) * 100 if total_images > 0 else 0
        report_lines.append(f"- **{category}**: {count} images ({percentage:.2f}%)")
    report_lines.append(f"- **Total Images**: {total_images}\n")
    
    report_lines.append("## 2. Corrupted & Unreadable Images")
    if not corrupted and not unreadable:
        report_lines.append("- No corrupted or unreadable images detected.\n")
    else:
        if corrupted:
            report_lines.append("### Corrupted Images (PIL verify failed):")
            for item in corrupted:
                report_lines.append(f"- {os.path.basename(item['path'])}: {item['error']}")
        if unreadable:
            report_lines.append("### Unreadable Images (OpenCV read failed):")
            for item in unreadable:
                report_lines.append(f"- {os.path.basename(item['path'])}: {item['error']}")
        report_lines.append("")

    report_lines.append("## 3. Duplicate Images")
    if not duplicates:
        report_lines.append("- No duplicate images detected based on file hash comparisons.\n")
    else:
        for item in duplicates:
            dup_name = os.path.basename(item['duplicate_path'])
            orig_name = os.path.basename(item['original_path'])
            report_lines.append(f"- `{dup_name}` is a duplicate of `{orig_name}`")
        report_lines.append("")

    report_lines.append("## 4. Blurry Images (Laplacian Variance < 100)")
    report_lines.append(f"Total blurry images detected: {len(blurry)} / {total_images}\n")
    if blurry:
        report_lines.append("| Image Name | Class | Laplacian Variance |")
        report_lines.append("|---|---|---|")
        # List top 20 blurriest images
        for item in blurry[:20]:
            name = os.path.basename(item['path'])
            class_name = os.path.basename(os.path.dirname(item['path']))
            report_lines.append(f"| {name} | {class_name} | {item['variance']:.2f} |")
        if len(blurry) > 20:
            report_lines.append(f"\n*...and {len(blurry) - 20} more blurry images.*")
        report_lines.append("")

    report_lines.append("## 5. Image Resolutions Distribution")
    for res, count in sorted(dimensions.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"- **{res}**: {count} images")
    report_lines.append("")

    # Write report to markdown file
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_quality_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    # Print summary to terminal
    print("\n".join(report_lines))
    print("=" * 60)
    print(f"Report successfully saved to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
