import os
from PIL import Image
from PIL.ExifTags import TAGS


def get_metadata(image_path: str) -> dict:
    """
    Extracts technical metadata from an image file.

    Returns basic file info, image properties, and EXIF data if present.

    EXIF (Exchangeable Image File Format):
    - Real camera photos contain EXIF: camera model, lens, timestamp, GPS
    - AI-generated images typically have NO EXIF data
    - However, EXIF can be stripped from real photos too
    - So missing EXIF is a supporting clue, NOT definitive proof of AI generation
    - We display this honestly in the UI with a disclaimer
    """
    result = {}

    # Basic file information
    result["filename"] = os.path.basename(image_path)
    result["file_size_kb"] = round(os.path.getsize(image_path) / 1024, 2)

    # Image properties from PIL
    with Image.open(image_path) as img:
        result["format"] = img.format          # e.g. JPEG, PNG
        result["mode"] = img.mode              # e.g. RGB, RGBA, L (greyscale)
        result["dimensions"] = f"{img.width} x {img.height}"

        # Attempt to extract EXIF data
        # _getexif() returns None if no EXIF present (common in AI images)
        exif_data = img._getexif()
        if exif_data:
            exif = {}
            for tag_id, value in exif_data.items():
                # Convert numeric tag IDs to human-readable names
                # e.g. 271 -> "Make", 272 -> "Model"
                tag = TAGS.get(tag_id, tag_id)
                exif[tag] = str(value)
            result["exif"] = exif
            result["has_exif"] = True
        else:
            result["exif"] = {}
            result["has_exif"] = False

    # Add explanatory note when EXIF is missing
    # Honest framing — missing EXIF is a clue, not proof
    if not result["has_exif"]:
        result["exif_note"] = "No EXIF data found. AI-generated images often lack EXIF metadata."

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/services/metadata_checker.py <image_path>")
    else:
        meta = get_metadata(sys.argv[1])
        for k, v in meta.items():
            print(f"{k}: {v}")