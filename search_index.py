import numpy as np
import faiss
import open_clip
import torch
from PIL import Image

# File paths
index_file = "E:/University/AI_Challenge/DataSampleAIC23/vit-h14/FAISS_search/combined_features.index"
metadata_file = "E:/University/AI_Challenge/DataSampleAIC23/vit-h14/FAISS_search/metadata.npy"

# Load FAISS index and metadata
index = faiss.read_index(index_file)
all_metadata = np.load(metadata_file, allow_pickle=True)

# Load OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
clip_model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
clip_model = clip_model.to(device)
tokenizer = open_clip.get_tokenizer(model_name)

def encode_text_query(query, model, tokenizer):
    text = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).squeeze().cpu().numpy()
    return text_features

def encode_image_query(image_path, model, preprocess):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).squeeze().cpu().numpy()
    return image_features

def combine_features(text_features, image_features, text_weight=0.5, image_weight=0.5):
    # Normalize the features
    text_features /= np.linalg.norm(text_features, axis=0, keepdims=True)
    image_features /= np.linalg.norm(image_features, axis=0, keepdims=True)

    # Weighted combination of features
    combined_features = text_weight * text_features + image_weight * image_features
    combined_features /= np.linalg.norm(combined_features, axis=0, keepdims=True)  # Normalize combined features

    return combined_features

def search_index(query_features, index, metadata, k=10):
    query_features = np.array([query_features]).astype('float32')

    # Perform the search
    distances, indices = index.search(query_features, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        video_id, frame_info = metadata[idx]
        frame_number = int(frame_info['frame_idx'])
        timestamp = float(frame_info['pts_time'])
        results.append((video_id, frame_number, timestamp, dist))

    results.sort(key=lambda x: x[3], reverse=True)  # Sort by distance (higher is better for cosine similarity)
    return results

# Define your queries
text_query = "A man is holding a gun at a bank"
image_query_path = r"E:\University\AI_Challenge\DataSampleAIC23\vit-h14\search-images\maxresdefault.jpg"

# Encode the text and image queries
text_features = encode_text_query(text_query, clip_model, tokenizer)
image_features = encode_image_query(image_query_path, clip_model, preprocess)

# Combine the features with adjustable weights
combined_features = combine_features(text_features, image_features, text_weight=0.4, image_weight=0.6)

# Search the index with the combined features
ranked_results_combined = search_index(combined_features, index, all_metadata)

# Display the top results for the combined query
print("Combined Search Results:")
for video_id, frame_number, timestamp, score in ranked_results_combined[:10]:
    print(f"Video ID: {video_id}, Frame: {frame_number}, Time: {timestamp:.2f}s, Score: {score}")