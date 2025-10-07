import streamlit as st
import os
import subprocess

st.title("🚦 Smart Traffic Management System")

uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    input_path = os.path.join("data", uploaded_video.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success(f"Uploaded: {uploaded_video.name}")

    if st.button("Run Detection"):
        st.info("Running YOLO + OCR pipeline...")
        # Run your detection.py script with parameters
        command = [
            "python",
            "./src/detection.py",
            "--source", input_path,
            "--veh_model", "yolo11n.pt",
            "--plate_model", "./models/license_plate.pt",  # updated model
            "--out", "./outputs",
            "--conf", "0.35"
        ]
        subprocess.run(command)

        st.success("Detection complete ✅")

        output_video = "./outputs/annotated_preview.mp4"
        if os.path.exists(output_video):
            st.video(output_video)
        else:
            st.warning("No annotated video found. Please check detection logs.")

        csv_path = "./outputs/detections.csv"
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            st.dataframe(df)
            st.download_button("Download Results CSV", data=df.to_csv(index=False), file_name="detections.csv")
