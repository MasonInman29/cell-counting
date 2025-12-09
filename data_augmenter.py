# Author: Vincent Lindvall
# Date: 2025-12-02

from PIL import Image, ImageEnhance
from glob import glob
import pandas as pd
import os

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1200

IMAGE_DIR = "./dataset/img"
COORDINATES_DIR = "./dataset/ground_truth"
AUGMENTATIONS_DIR_BASE = "./augmented_dataset/"

def vertical_flip(cell_image, cell_coordinates):
    # Flip cell image vertically
    augmented_cell_image = cell_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Update cell coordinates
    cell_coordinates['Y'] = IMAGE_HEIGHT - cell_coordinates['Y']

    return augmented_cell_image, cell_coordinates

def horizontal_flip(cell_image, cell_coordinates):
    # Flip cell image horizontally
    augmented_cell_image = cell_image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Update cell coordinates
    cell_coordinates['X'] = IMAGE_WIDTH - cell_coordinates['X']
    
    return augmented_cell_image, cell_coordinates

def save_augmentation(save_dir, image_num, image, coordinates):
    image.save(save_dir+"img/"+str(image_num)+".tiff")
    coordinates.to_csv(save_dir+"ground_truth/"+str(image_num)+".csv", index=False)

# Augment only the trainval cell images (ignore the test images)
def main():
    metadata_df = pd.read_csv("./dataset/metadata.csv")
    for row in metadata_df.itertuples(index=False):
        
        # Ignore test images
        if row.set == "test":
            continue
        
        image_num = row.id
        image_path = IMAGE_DIR + "/" + str(image_num) + ".tiff"

        cell_image = Image.open(image_path)
        cell_coordinates = pd.read_csv(COORDINATES_DIR+"/"+str(image_num)+".csv")

        # If necessary, convert the mode of the image. PIL's ImageEnhance is not compatible with all image modes
        if cell_image.mode not in ("L", "RGB"):
            cell_image = cell_image.convert("RGB")

        # --- Augmentations --- #

        """# Horizontally flip the images
        augmented_cell_image, new_coordinates = horizontal_flip(cell_image, cell_coordinates)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "/h_flip/",
                          image_num=image_num,
                          image=augmented_cell_image,
                          coordinates=new_coordinates
                         )

        # Vertically flip the images
        augmented_cell_image, new_coordinates = vertical_flip(cell_image, cell_coordinates)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "/v_flip/",
                          image_num=image_num,
                          image=augmented_cell_image,
                          coordinates=new_coordinates
                         )

        # Horizontally and vertically flip the images
        augmented_cell_image_1, new_coordinates_1 = horizontal_flip(cell_image, cell_coordinates)
        augmented_cell_image_2, new_coordinates_2 = vertical_flip(augmented_cell_image_1, new_coordinates_1)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "/h_and_v_flip/",
                          image_num=image_num,
                          image=augmented_cell_image_2,
                          coordinates=new_coordinates_2
                         )

        # Reduce contrast of the images
        enhancer = ImageEnhance.Contrast(cell_image)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "0_point_5_contrast/",
                          image_num=image_num,
                          image=enhancer.enhance(0.5),
                          coordinates=cell_coordinates
                         )"""
        
        """# Increase contrast of the images
        enhancer = ImageEnhance.Contrast(cell_image)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "1_point_5_contrast/",
                          image_num=image_num,
                          image=enhancer.enhance(1.5),
                          coordinates=cell_coordinates
                         )"""

        """# horizontal + vertical + contrast change
        augmented_cell_image_1, new_coordinates_1 = horizontal_flip(cell_image, cell_coordinates)
        augmented_cell_image_2, new_coordinates_2 = vertical_flip(augmented_cell_image_1, new_coordinates_1)
        enhancer = ImageEnhance.Contrast(augmented_cell_image_2)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "0_point_5_contrast_and_h_and_v_flip/",
                          image_num=image_num,
                          image=enhancer.enhance(0.5),
                          coordinates=new_coordinates_2
                         )
        
        augmented_cell_image_1, new_coordinates_1 = horizontal_flip(cell_image, cell_coordinates)
        augmented_cell_image_2, new_coordinates_2 = vertical_flip(augmented_cell_image_1, new_coordinates_1)
        enhancer = ImageEnhance.Contrast(augmented_cell_image_2)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "1_point_5_contrast_and_h_and_v_flip/",
                          image_num=image_num,
                          image=enhancer.enhance(1.5),
                          coordinates=new_coordinates_2
                         ) """
        
        """# Sharpen the Images
        enhancer = ImageEnhance.Sharpness(cell_image)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "sharpened/",
                          image_num=image_num,
                          image=enhancer.enhance(2.0),
                          coordinates=cell_coordinates
                         )

        # Blur the Images
        enhancer = ImageEnhance.Sharpness(cell_image)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "blurred/",
                          image_num=image_num,
                          image=enhancer.enhance(0.0),
                          coordinates=cell_coordinates
                         )"""

        """# v_flip + 1_point_5_contrast
        augmented_cell_image_1, new_coordinates_1 = vertical_flip(cell_image, cell_coordinates)
        enhancer = ImageEnhance.Contrast(augmented_cell_image_1)
        save_augmentation(save_dir=AUGMENTATIONS_DIR_BASE + "v_flip_and_1_point_5_contrast/",
                          image_num=image_num,
                          image=enhancer.enhance(1.5),
                          coordinates=new_coordinates_1
                         )"""

if __name__ == "__main__":
    os.makedirs("./augmented_dataset/h_flip/img", exist_ok=True)
    os.makedirs("./augmented_dataset/h_flip/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/v_flip/img", exist_ok=True)
    os.makedirs("./augmented_dataset/v_flip/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/blurred/img", exist_ok=True)
    os.makedirs("./augmented_dataset/blurred/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/sharpened/img", exist_ok=True)
    os.makedirs("./augmented_dataset/shapened/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/0_point_5_contrast/img", exist_ok=True)
    os.makedirs("./augmented_dataset/0_point_5_contrast/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/1_point_5_contrast/img", exist_ok=True)
    os.makedirs("./augmented_dataset/1_point_5_contrast/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/v_flip_and_1_point_5_contrast/img", exist_ok=True)
    os.makedirs("./augmented_dataset/v_flip_and_1_point_5_contrast/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/1_point_5_contrast_and_h_and_v_flip/img", exist_ok=True)
    os.makedirs("./augmented_dataset/1_point_5_contrast_and_h_and_v_flip/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/0_point_5_contrast_and_h_and_v_flip/img", exist_ok=True)
    os.makedirs("./augmented_dataset/0_point_5_contrast_and_h_and_v_flip/ground_truth", exist_ok=True)
    os.makedirs("./augmented_dataset/h_and_v_flip/img", exist_ok=True)
    os.makedirs("./augmented_dataset/h_and_v_flip/ground_truth", exist_ok=True)

    main()