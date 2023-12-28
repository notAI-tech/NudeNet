def scanning_process(list_of_dirs, list_of_ignored_dirs):
    from nudenet import NudeDetector
    from liteindex import DefinedIndex
    from glob import iglob
    import os
    import hashlib

    images_index = DefinedIndex(
        "images_index",
        schema={
            "image_path": "string",
            "image_size": "number",
            "image_hash": "string",
            "FEMALE_GENITALIA_COVERED": "number",
            "FACE_FEMALE": "number",
            "BUTTOCKS_EXPOSED": "number",
            "FEMALE_BREAST_EXPOSED": "number",
            "FEMALE_GENITALIA_EXPOSED": "number",
            "MALE_BREAST_EXPOSED": "number",
            "ANUS_EXPOSED": "number",
            "FEET_EXPOSED": "number",
            "BELLY_COVERED": "number",
            "FEET_COVERED": "number",
            "ARMPITS_COVERED": "number",
            "ARMPITS_EXPOSED": "number",
            "FACE_MALE": "number",
            "BELLY_EXPOSED": "number",
            "MALE_GENITALIA_EXPOSED": "number",
            "ANUS_COVERED": "number",
            "FEMALE_BREAST_COVERED": "number",
            "BUTTOCKS_COVERED": "number",
        },
        db_path="images_index.db",
    )

    detector = NudeDetector()

    for dir in list_of_dirs:
        for file in iglob(os.path.join(dir, "**", "*.*"), recursive=True):
            if os.path.isfile(file):
                if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg", ".png"]:
                    if not any(
                        ignored_dir in file for ignored_dir in list_of_ignored_dirs
                    ):
                        preds = detector.detect(file)
                        data = {p["class"]: float(p["score"]) for p in preds}
                        data["image_path"] = file
                        data["image_size"] = os.path.getsize(file)
                        data["image_hash"] = hashlib.sha256(
                            open(file, "rb").read()
                        ).hexdigest()

                        images_index.update({data["image_hash"]: data})


if __name__ == "__main__":
    scanning_process(["/Users/praneeth.bedapudi/Desktop"], [])
