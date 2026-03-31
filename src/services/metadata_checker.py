import os
from PIL import Image
from PIL.ExifTags import TAGS


def get_metadata(image_path: str) -> dict:
    result = {}

    # Basic file info
    result["filename"] = os.path.basename(image_path)
    result["file_size_kb"] = round(os.path.getsize(image_path) / 1024, 2)

    # Image info
    with Image.open(image_path) as img:
        result["format"] = img.format
        result["mode"] = img.mode
        result["dimensions"] = f"{img.width} x {img.height}"

        # EXIF data
        exif_data = img._getexif()
        if exif_data:
            exif = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                exif[tag] = str(value)
            result["exif"] = exif
            result["has_exif"] = True
        else:
            result["exif"] = {}
            result["has_exif"] = False

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