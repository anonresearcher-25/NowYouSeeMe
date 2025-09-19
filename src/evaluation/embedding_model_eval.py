import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from PIL import Image
from backbones import get_model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import cv2

# Load model
st = time.perf_counter()
model_name = "edgeface_s_gamma_05"
model = get_model(model_name)
checkpoint_path = f'../models/{model_name}.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()
en = time.perf_counter()
print(f"Model load time: {(en - st) * 1000:.2f} ms")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Image paths
image_dir = "../../input/ttp_DB"    # Two directories up

image_filenames = ["Saima.jpeg", "Afsheen.jpeg", "Hamna.jpeg"]

# Load and preprocess all images
input_batch = []
for filename in image_filenames:
    img_path = os.path.join(image_dir, filename)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch.append(img)

# Stack into a single tensor (batch_size, 3, 112, 112)
input_tensor = torch.stack(input_batch)

# Run the model once on the batch
with torch.no_grad():
    st = time.perf_counter()
    print(input_tensor.shape)
    embeddings = model(input_tensor)  # shape: (12, emb_dim)
    en = time.perf_counter()
    print(f"Batched inference time for {len(image_filenames)} images: {(en - st) * 1000:.2f} ms")

# Normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)

# Compute cosine similarity matrix
cosine_matrix = torch.matmul(embeddings, embeddings.T)

# Convert to NumPy and round for visualization
sim_matrix_rounded = np.round(cosine_matrix.numpy(), 2)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix_rounded, xticklabels=image_filenames, yticklabels=image_filenames,
            annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": 0.8})
plt.title("Cosine Similarity Between Face Embeddings", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
