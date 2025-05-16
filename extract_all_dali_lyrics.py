import os
import gzip
import pickle
import pandas as pd
from tqdm import tqdm
from dali_structure_extractor import extract_hierarchical_lyrics  # Handles both structures internally

def process_dali_gz_file(file_path):
    """
    Extract paragraph-line-word-note structure from a single DALI .gz file.
    Returns metadata and nested structure if valid, otherwise None.
    """
    try:
        with gzip.open(file_path, "rb") as f:
            raw = pickle.load(f)

        # Skip non-lyric dict files (e.g., ground-truth)
        if isinstance(raw, dict) or not hasattr(raw, "annotations"):
            return None

        # Convert horizontal to vertical if needed
        if hasattr(raw, "is_horizontal") and callable(raw.is_horizontal) and raw.is_horizontal():
            raw.horizontal2vertical()

        annot = raw.annotations.get("annot", {})
        nested_lyrics = extract_hierarchical_lyrics(annot)  # Auto-detects structure

        if not nested_lyrics:
            return None

        return {
            "gz_filename": os.path.basename(file_path),
            "song_id": os.path.basename(file_path).replace(".gz", ""),
            "artist": raw.info.get("artist", ""),
            "title": raw.info.get("title", ""),
            "paragraphs": nested_lyrics
        }

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None


def process_all_dali_files(dali_dir, output_excel):
    """
    Process all .gz files in the DALI dataset folder and export structured lyrics to an Excel file.
    """
    gz_files = [f for f in os.listdir(dali_dir) if f.endswith(".gz")]
    all_rows = []
    processed = 0

    for fname in tqdm(gz_files, desc="📦 Processing DALI .gz files"):
        file_path = os.path.join(dali_dir, fname)
        result = process_dali_gz_file(file_path)

        if result:
            processed += 1
            for para_idx, paragraph in enumerate(result["paragraphs"]):
                for line_idx, line in enumerate(paragraph):
                    line_text = " | ".join(" ".join(word) for word in line)
                    all_rows.append({
                        "gz_filename": result["gz_filename"],
                        "song_id": result["song_id"],
                        "artist": result["artist"],
                        "title": result["title"],
                        "para_id": para_idx,
                        "line_id": line_idx,
                        "line_text": line_text,
                        "words": line_text  # Duplicated for compatibility
                    })

    # Save results as a DataFrame
    df = pd.DataFrame(all_rows)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Extracted {len(df)} lines from {processed} valid songs.")
    print(f"📄 Output saved to: {output_excel}")


if __name__ == "__main__":
    dali_data_dir = "DALI-master/versions/v1"  # Replace with your actual path
    output_file = "dali_lyrics_structured.xlsx"
    process_all_dali_files(dali_data_dir, output_file)
