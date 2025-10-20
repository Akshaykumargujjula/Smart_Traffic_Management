import streamlit as st
import os
import subprocess
import pandas as pd
import time

st.set_page_config(page_title="Smart Traffic Management", page_icon="🚦", layout="wide")

st.title("🚦 Smart Traffic Management System")
st.markdown("Upload a traffic video to detect vehicles, license plates, and count traffic")

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# File uploader
uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Save uploaded file
    input_path = os.path.join("data", uploaded_video.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.success(f"✅ Uploaded: {uploaded_video.name}")
    
    # Display original video
    with st.expander("📹 View Original Video"):
        st.video(input_path)
    
    # Detection settings
    col1, col2, col3 = st.columns(3)
    with col1:
        conf_threshold = st.slider("Vehicle Confidence", 0.1, 0.9, 0.35, 0.05)
    with col2:
        plate_conf = st.slider("Plate Confidence", 0.1, 0.9, 0.35, 0.05)
    with col3:
        skip_frames = st.slider("Frame Skip (Speed)", 1, 10, 3)
    
    if st.button("🚀 Run Detection", type="primary"):
        # Clear previous outputs
        output_video_mp4 = "./outputs/annotated_preview.mp4"
        output_video_avi = "./outputs/annotated_preview.avi"
        csv_path = "./outputs/detections.csv"
        summary_path = "./outputs/counts_summary.txt"
        
        for path in [output_video_mp4, output_video_avi, csv_path, summary_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Running YOLO + OCR pipeline...")
        progress_bar.progress(25)
        
        # Run detection script
        command = [
            "python",
            "./src/detection.py",
            "--source", input_path,
            "--veh_model", "yolo11n.pt",
            "--plate_model", "./models/license_plate.pt",
            "--out", "./outputs",
            "--conf", str(conf_threshold),
            "--plate_conf", str(plate_conf),
            "--skip", str(skip_frames),
            "--resize_width", "1280"
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            progress_bar.progress(75)
            
            if result.returncode != 0:
                st.error(f"❌ Detection failed with error:\n```\n{result.stderr}\n```")
                st.stop()
            
            progress_bar.progress(100)
            status_text.text("✅ Detection complete!")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            st.success("🎉 Detection Complete!")
            
            # Display results
            st.markdown("---")
            st.header("📊 Results")
            
            # Find which video format was created
            output_video = None
            if os.path.exists(output_video_mp4):
                output_video = output_video_mp4
            elif os.path.exists(output_video_avi):
                output_video = output_video_avi
            
            # Display annotated video
            if output_video and os.path.exists(output_video):
                st.subheader("🎥 Annotated Video")
                st.video(output_video)
                
                # Download button for video
                with open(output_video, "rb") as video_file:
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=video_file,
                        file_name=os.path.basename(output_video),
                        mime="video/mp4"
                    )
            else:
                st.warning("⚠️ No annotated video found. Check if the detection script ran successfully.")
            
            # Display traffic counts summary
            if os.path.exists(summary_path):
                st.subheader("📈 Traffic Counts")
                with open(summary_path, "r") as f:
                    summary = f.read()
                st.code(summary, language=None)
            
            # Display detection CSV
            if os.path.exists(csv_path):
                st.subheader("📋 Detection Details")
                df = pd.read_csv(csv_path)
                
                if not df.empty:
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Detections", len(df))
                    with col2:
                        st.metric("Unique Vehicles", df['track_id'].nunique() if 'track_id' in df.columns else "N/A")
                    with col3:
                        plates_detected = df['plate'].notna().sum() if 'plate' in df.columns else 0
                        st.metric("Plates Read", plates_detected)
                    
                    # Display dataframe
                    st.dataframe(df, use_container_width=True, height=300)
                    
                    # Download button for CSV
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=df.to_csv(index=False),
                        file_name="detections.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No detections found in the video.")
            else:
                st.warning("⚠️ No detection data CSV found.")
                
        except subprocess.TimeoutExpired:
            st.error("❌ Detection timed out after 5 minutes. Try with a shorter video or higher frame skip.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

else:
    st.info("👆 Please upload a traffic video to begin")
    
    # Show example usage
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload** a traffic video (MP4, MOV, or AVI format)
        2. **Adjust** detection settings if needed:
           - Vehicle/Plate Confidence: Lower = more detections (may include false positives)
           - Frame Skip: Higher = faster processing but may miss vehicles
        3. **Click** "Run Detection" to start processing
        4. **View** results:
           - Annotated video with bounding boxes
           - Traffic counts by vehicle type
           - Detailed detection log with license plates
        5. **Download** results for further analysis
        """)
        # streamlit run src/streamlit_app.py