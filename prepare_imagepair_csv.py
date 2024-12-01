import os
import pandas as pd
import itertools
import random
import argparse

def create_image_pair(dataset_path):
    dataset_dir = dataset_path

    # List all folders (each representing a person)
    folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    # Prepare lists to store the samples
    samples = []

    # Create positive samples (same person)
    positive_samples = []
    for folder in folders:
        images = os.listdir(os.path.join(dataset_dir, folder))
        # Generate all combinations of images in the same folder
        positive_pairs = list(itertools.combinations(images, 2))
        for img1, img2 in positive_pairs:
            img1_path = os.path.join(dataset_dir, folder, img1).replace("\\", "/")
            img2_path = os.path.join(dataset_dir, folder, img2).replace("\\", "/")
            positive_samples.append([img1_path, img2_path, 1])  # 1 for positive
            if len(positive_samples) >= 8000:  # Limit to 8000 positive samples
                break
        if len(positive_samples) >= 8000:
            break

    samples.extend(positive_samples)

    # Create negative samples (different persons)
    negative_samples = []
    while len(negative_samples) < 8000:
        folder1, folder2 = random.sample(folders, 2)
        img1 = random.choice(os.listdir(os.path.join(dataset_dir, folder1)))
        img2 = random.choice(os.listdir(os.path.join(dataset_dir, folder2)))
        img1_path = os.path.join(dataset_dir, folder1, img1).replace("\\", "/")
        img2_path = os.path.join(dataset_dir, folder2, img2).replace("\\", "/")
        negative_samples.append([img1_path, img2_path, 0])  # 0 for negative

    samples.extend(negative_samples)

    # Shuffle samples to mix positive and negative
    random.shuffle(samples)

    # Create a DataFrame
    df = pd.DataFrame(samples, columns=['image1', 'image2', 'label'])

    # Save to CSV
    df.to_csv('image_pairs.csv', index=False)

    print("image_pairs.csv created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a csv file with image pairs")

    parser.add_argument("--path", type=str, default="Face_Dataset" ,help="Path to dataset folder")

    args = parser.parse_args()

    #Call the funtion with parsed arguments
    create_image_pair(args.path)
