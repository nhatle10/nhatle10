import os
import torch
import clip
import numpy as np
from tqdm import tqdm
import csv

print("Script started")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP model loaded")

def load_features(features_dir):
    print(f"Loading features from {features_dir}")
    features = {}
    for file in os.listdir(features_dir):
        if file.endswith('.npy'):
            try:
                video_id = file.split('.')[0]
                feature_array = np.load(os.path.join(features_dir, file))
                features[video_id] = torch.from_numpy(feature_array).float().to(device)
                print(f"Loaded features for {video_id}, shape: {features[video_id].shape}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    print(f"Loaded features for {len(features)} videos")
    return features

def load_keyframe_mappings(map_dir):
    print(f"Loading keyframe mappings from {map_dir}")
    mappings = {}
    for file in os.listdir(map_dir):
        if file.endswith('.csv'):
            try:
                video_id = file.split('.')[0]
                with open(os.path.join(map_dir, file), 'r') as f:
                    csv_reader = csv.DictReader(f)
                    mappings[video_id] = [row for row in csv_reader]
                print(f"Loaded mapping for {video_id}, entries: {len(mappings[video_id])}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    print(f"Loaded mappings for {len(mappings)} videos")
    return mappings

# Define directories
features_dir = "E:/University/AI_Challenge/DataSampleAIC23/clip-features-vit-b32-sample/clip-features"
map_dir = "E:/University/AI_Challenge/DataSampleAIC23/map-keyframes-sample/map-keyframes"

# Load features and mappings
features = load_features(features_dir)
keyframe_mappings = load_keyframe_mappings(map_dir)

# Define the text query
query = "A man is holding a gun at a bank"
print(f"Query: {query}")

# Encode text query
with torch.no_grad():
    text_features = model.encode_text(clip.tokenize(query).to(device)).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
print("Query encoded")

# Compute similarities and retrieve top matches
print("Computing similarities...")
results = []
for video_id, video_features in tqdm(features.items()):
    video_features = video_features.to(device)
    similarities = (100.0 * video_features @ text_features.T).squeeze()
    top_indices = similarities.argsort(descending=True)[:5]  # Get top 5 frames
    for idx in top_indices:
        score = similarities[idx].item()
        frame_info = keyframe_mappings[video_id][idx]
        frame_idx = frame_info['frame_idx']
        pts_time = frame_info['pts_time']
        results.append((video_id, idx, frame_idx, pts_time, score))

# Sort results by score
results.sort(key=lambda x: x[4], reverse=True)
print(f"Found {len(results)} matching frames")

# Display top 10 results
print("Top 10 results:")
for video_id, feature_idx, frame_idx, pts_time, score in results[:10]:
    print(f"Video ID: {video_id}, Feature Index: {feature_idx}, Frame Index: {frame_idx}, PTS Time: {pts_time}, Score: {score:.2f}")

print("Script finished")