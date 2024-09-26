import streamlit as st
from video_query import load_features, load_keyframe_mappings, query_video

st.set_page_config(page_title="Video Event Retrieval App", layout="wide")

with st.sidebar:
    #st.image("")
    st.title("Team Argus")
    st.info("This application allows you to query events from video")

st.title("Argus - Video Retrieval")

left_column, right_column = st.columns(2)

with left_column:
    query = st.text_input("Decribe the scene")

    st.write("Query is: ", query)

with right_column:
    image = st.file_uploader("Upload an image")

st.button("Search")

#st.title("Argus - Video Retrieval")

#query = st.text_input("Decribe the scene")

#st.write("Query is: ", query)

features_dir = r"C:\Users\admin\Downloads\clip-features-vit-b32-sample\clip-features-vit-b32-sample\clip-features"
map_dir = r"C:\Users\admin\Downloads\map-keyframes-sample\map-keyframes-sample\map-keyframes"
videos_dir = r"C:\Users\admin\Downloads\Videos_L01\video"

# Load features and mappings with caching
features = load_features(features_dir)
keyframe_mappings = load_keyframe_mappings(map_dir)

if query:
    frames = query_video(query, features, keyframe_mappings, videos_dir)
    
    st.write("Top 10 matching frames:")
    cols = st.columns(3)  # Create 3 columns for each row
    for i, frame_data in enumerate(frames):
        col = cols[i % 3]  # Select the appropriate column
        col.image(frame_data['frame'], caption=f"Video ID: {frame_data['video_id']}, Frame Index: {frame_data['frame_idx']}, PTS Time: {frame_data['pts_time']}, Score: {frame_data['score']:.2f}", use_column_width=True)
        
        # Create a new row after every 3 images
        if (i + 1) % 3 == 0 and i + 1 < len(frames):
            cols = st.columns(3)