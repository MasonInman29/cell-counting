from PIL import Image, ImageEnhance
from glob import glob
import pandas as pd
import os

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1200

IMAGE_DIR = "./dataset/img"
COORDINATES_DIR = "./dataset/ground_truth"
AUGMENTATIONS_DIR_BASE = "./final_augmented_dataset/"

def vertical_flip(cell_image, cell_coordinates):
    # Flip cell image vertically
    augmented_cell_image = cell_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Update cell coordinates
    cell_coordinates['Y'] = IMAGE_HEIGHT - cell_coordinates['Y']

    return augmented_cell_image, cell_coordinates

def save_augmentation(save_dir, prefix, image_num, image, coordinates):
    image.save(save_dir+"img/"+prefix+str(image_num)+".tiff")
    coordinates.to_csv(save_dir+"ground_truth/"+prefix+str(image_num)+".csv", index=False)

def main():
    metadata_df = pd.read_csv("./dataset/metadata.csv")

    augmented_metadata = []

    for row in metadata_df.itertuples(index=False):

        # Ignore test images
        if row.set == "test":
            continue

        image_num = str(row.id)

        image_path = os.path.join(IMAGE_DIR, f"{image_num}.tiff")
        cell_image = Image.open(image_path)
        cell_coordinates = pd.read_csv(os.path.join(COORDINATES_DIR, f"{image_num}.csv"))

        if cell_image.mode not in ("L", "RGB"):
            cell_image = cell_image.convert("RGB")

        row_dict = row._asdict()

        # -------- NO OP --------
        save_augmentation(
            save_dir=AUGMENTATIONS_DIR_BASE,
            prefix="",
            image_num=image_num,
            image=cell_image,
            coordinates=cell_coordinates
        )

        new_row = row_dict.copy()
        new_row["id"] = str(image_num)
        augmented_metadata.append(new_row)

        # -------- VERTICAL FLIP --------
        aug_img, new_coords = vertical_flip(cell_image, cell_coordinates)

        save_augmentation(
            save_dir=AUGMENTATIONS_DIR_BASE,
            prefix="0",
            image_num=image_num,
            image=aug_img,
            coordinates=new_coords
        )

        new_row = row_dict.copy()
        new_row["id"] = "0"+str(image_num)
        augmented_metadata.append(new_row)

        # -------- CONTRAST --------
        enhancer = ImageEnhance.Contrast(cell_image)

        save_augmentation(
            save_dir=AUGMENTATIONS_DIR_BASE,
            prefix="00",
            image_num=image_num,
            image=enhancer.enhance(1.5),
            coordinates=cell_coordinates
        )

        new_row = row_dict.copy()
        new_row["id"] = "00"+str(image_num)
        augmented_metadata.append(new_row)

        # -------- VERTICAL + CONTRAST --------
        aug_img, new_coords = vertical_flip(cell_image, cell_coordinates)
        enhancer = ImageEnhance.Contrast(aug_img)

        save_augmentation(
            save_dir=AUGMENTATIONS_DIR_BASE,
            prefix="000",
            image_num=image_num,
            image=enhancer.enhance(1.5),
            coordinates=new_coords
        )

        new_row = row_dict.copy()
        new_row["id"] = "000"+str(image_num)
        augmented_metadata.append(new_row)

    # Save augmented metadata
    augmented_df = pd.DataFrame(augmented_metadata, columns=metadata_df.columns)
    augmented_df.to_csv(os.path.join(AUGMENTATIONS_DIR_BASE, "metadata.csv"), index=False)

if __name__ == "__main__":
    os.makedirs("./final_augmented_dataset/img", exist_ok=True)
    os.makedirs("./final_augmented_dataset/ground_truth", exist_ok=True)
    main()