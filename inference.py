import os
import cv2
from utils.model import SiameseModel
from torchvision import transforms
from utils.extract_face import extract_face
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import smtplib
import imghdr
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch credentials
email_id = os.getenv('EMAIL_ID')
email_pass = os.getenv('EMAIL_PASS')

msg = EmailMessage()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = SiameseModel()

# Load the saved weights
model.load_state_dict(torch.load("model_weights_rmsprop.pth"))

# Set the model to evaluation mode
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# Function to send email to the owner
def mail_now(identity, path):
    msg['Subject'] = "Home Security Update!!"
    msg['From'] = email_id
    msg['To'] = 'rjtdulal@gmail.com'  #Keep the owner's email here
    msg.set_content(f"You have a visitor at your door.\n\nIdentification: {identity}")

    with open(path, 'rb') as m:
        file_data = m.read()
        file_type = imghdr.what(m.name)
        file_name = "Image of Individual"
        print(file_name)

    msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(email_id, email_pass)
        smtp.send_message(msg)


# Function to compare images and display the best match
def compare_images_with_all_owners(img2_faces, base_folder):
    overall_best_match_image = None
    overall_best_match_distance = float('inf')
    overall_best_owner_name = None
    overall_best_img_name = None
    overall_best_captured_face = None

    for i, img2_face in enumerate(img2_faces):
        img2_face = cv2.cvtColor(img2_face, cv2.COLOR_BGR2RGB)
        img2_face = Image.fromarray(img2_face)
        img2_face = transform(img2_face).unsqueeze(0)

        # Iterate through all owner's folders
        for owner_name in os.listdir(base_folder):
            owner_folder = os.path.join(base_folder, owner_name)

            if os.path.isdir(owner_folder):  # Check if it's a folder (i.e., an owner's folder)
                # Iterate through all images in the owner's folder
                for img_name in os.listdir(owner_folder):
                    img_path = os.path.join(owner_folder, img_name)

                    # Skip non-image files
                    if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    # Load the image from the folder
                    img1 = Image.open(img_path)
                    img1 = transform(img1).unsqueeze(0)

                    # Get embeddings of both images
                    emb1, emb2 = model(img1, img2_face)

                    # Calculate distance between the embeddings
                    distance = F.pairwise_distance(emb1, emb2)
                    print(f"Comparing with {img_name} in folder {owner_name}: Distance = {distance.item()}")

                    # Set threshold for matching
                    threshold = 0.3
                    if distance.item() < threshold:
                        # Track the best match overall
                        if distance.item() < overall_best_match_distance:
                            overall_best_match_distance = distance.item()
                            overall_best_match_image = img_path
                            overall_best_owner_name = owner_name
                            overall_best_img_name = img_name
                            overall_best_captured_face = img2_face

    # Display the result: best matching image and captured image
    if overall_best_match_image:
        identity = f"{overall_best_owner_name}\nMatching score: {overall_best_match_distance}"
        print(f"Best match found with {overall_best_img_name} from owner {overall_best_owner_name} with score {overall_best_match_distance}")
        img1 = Image.open(overall_best_match_image)

        # Display side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img1)
        axes[0].set_title(f"Best Match: {overall_best_img_name} ({overall_best_owner_name})")

        axes[1].imshow(overall_best_captured_face.squeeze(0).permute(1, 2, 0).cpu().numpy())
        axes[1].set_title("Captured Face")
        axes[0].axis('off')
        axes[1].axis('off')
        plt.show()
    else:
        identity = "Unknown!"
        print("No match found for the face!")

    return identity


# Main process: Load and process the captured image
img2_path = "Input_images/captured.jpg"

# Extract faces from the captured image
img2_faces = extract_face(img2_path)

# Specify the base folder where all owners' folders are located
base_folder = "Owner_identification"

# Compare each extracted face with all images in all owner's folders
if img2_faces:
    identity = compare_images_with_all_owners(img2_faces, base_folder)
else:
    print("No faces detected in the captured image.")

# Send mail to the owner
try:
    mail_now(identity, img2_path)
except:
    print("No Internet Access. Connect To Internet First")
