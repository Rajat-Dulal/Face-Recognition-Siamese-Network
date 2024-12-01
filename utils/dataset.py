from torch.utils.data import Dataset
from PIL import Image

class SiameseDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        # Load the CSV file
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get the image paths and label for the given index
        img1_path = self.data_frame.iloc[idx, 0]
        img2_path = self.data_frame.iloc[idx, 1]
        label = int(self.data_frame.iloc[idx, 2])

        # Load the images
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Apply transformations, if any
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Return the two images and the label
        return img1, img2, label