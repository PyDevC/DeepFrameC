"""Streamlit UI for DeepFake detection."""

import streamlit as st
import requests
import tempfile
import os
from pathlib import Path
from datetime import datetime
import time

st.set_page_config(
    page_title="DeepFrameC - DeepFake Detector",
    page_icon="🔍",
    layout="wide"
)


API_URL = os.environ.get("API_URL", "http://localhost:8000")


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None


def detect_deepfake(file, threshold, num_frames, checkpoint):
    """Send detection request to API."""
    files = {"file": (file.name, file.getvalue())}
    data = {
        "threshold": threshold,
        "num_frames": num_frames,
        "checkpoint": checkpoint
    }
    
    response = requests.post(f"{API_URL}/detect", files=files, data=data)
    return response.json()


def main():
    st.title("🔍 DeepFrameC - DeepFake Detection")
    st.markdown("**Detect AI-generated videos and images**")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("⚙️ Settings")
        
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            help="Lower = more sensitive to fake content"
        )
        
        num_frames = st.slider(
            "Frames to Analyze",
            min_value=8,
            max_value=32,
            value=16,
            help="More frames = more accurate but slower"
        )
        
        checkpoint = st.selectbox(
            "Model",
            options=["best.pth", "last.pth"],
            index=0
        )
        
        st.subheader("ℹ️ About")
        st.info(
            "Upload a video or image to detect if it contains "
            "AI-generated (deepfake) content. The model analyzes "
            "visual patterns and artifacts common in synthetic media."
        )
    
    with col1:
        st.subheader("📤 Upload Media")
        
        uploaded_file = st.file_uploader(
            "Choose a video or image",
            type=["mp4", "avi", "mov", "mkv", "webm", "jpg", "jpeg", "png"],
            help="Supported: MP4, AVI, MOV, MKV, WEBM, JPG, PNG"
        )
        
        if uploaded_file is not None:
            col_preview, col_action = st.columns([1, 1])
            
            with col_preview:
                if uploaded_file.type.startswith("image/"):
                    st.image(uploaded_file, caption="Uploaded Image", width=300)
                else:
                    st.video(uploaded_file)
            
            with col_action:
                if st.button("🔍 Detect DeepFake", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        try:
                            result = detect_deepfake(
                                uploaded_file,
                                threshold,
                                num_frames,
                                checkpoint
                            )
                            
                            if result["status"] == "completed":
                                data = result["result"]
                                
                                st.success("Analysis Complete!")
                                
                                verdict_col1, verdict_col2, verdict_col3 = st.columns(3)
                                
                                with verdict_col1:
                                    if data["is_fake"]:
                                        st.error("❌ FAKE")
                                    else:
                                        st.success("✅ REAL")
                                
                                with verdict_col2:
                                    st.metric("Confidence", f"{data['confidence']:.1%}")
                                
                                with verdict_col3:
                                    st.metric("Frames", data["num_frames"])
                                
                                prob_col1, prob_col2 = st.columns(2)
                                
                                with prob_col1:
                                    st.progress(data["fake_probability"], text="Fake Probability")
                                
                                with prob_col2:
                                    st.progress(data["real_probability"], text="Real Probability")
                                
                                st.divider()
                                
                                if data.get("frame_results"):
                                    st.subheader("📊 Per-Frame Analysis")
                                    
                                    import pandas as pd
                                    df = pd.DataFrame(data["frame_results"])
                                    df.index = [f"Frame {i+1}" for i in range(len(df))]
                                    df.columns = ["Fake", "Real"]
                                    st.bar_chart(df)
                                
                                st.divider()
                                
                                with st.expander("📋 Technical Details"):
                                    st.json({
                                        "request_id": data.get("request_id", "N/A"),
                                        "model": data.get("model_name", "Unknown"),
                                        "threshold": threshold,
                                        "processing_time": f"{data['processing_time']:.2f}s",
                                        "timestamp": result["timestamp"]
                                    })
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
                        
                        except requests.exceptions.ConnectionError:
                            st.error(
                                "🔌 Cannot connect to API. "
                                "Make sure the server is running on port 8000."
                            )
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    st.divider()
    
    st.subheader("📈 Recent Results")
    
    if "results" not in st.session_state:
        st.session_state.results = []
    
    if st.session_state.results:
        import pandas as pd
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No results yet. Upload a file to start detection.")


if __name__ == "__main__":
    main()
