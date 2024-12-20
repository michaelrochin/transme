import streamlit as st
import whisper
import tempfile
import os
from pathlib import Path
import subprocess

# Page config
st.set_page_config(
    page_title="Audio Transcription",
    page_icon="🎙️",
    layout="centered"
)

# Check ffmpeg installation
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    st.write("ffmpeg is properly installed!")
except Exception as e:
    st.error("ffmpeg is not properly installed. Please contact support.")
    st.error(f"Error: {str(e)}")
    st.stop()

# Title
st.title("🎙️ Audio Transcription")
st.write("Upload your audio file and get the transcription instantly!")

# Initialize whisper model
@st.cache_resource
def load_model():
    model = whisper.load_model("tiny")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

def transcribe_audio(audio_file):
    # Create a unique temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / f"audio{os.path.splitext(audio_file.name)[1]}"
    
    try:
        # Save the uploaded file
        with open(temp_path, 'wb') as f:
            f.write(audio_file.getvalue())
        
        # Transcribe the audio
        result = model.transcribe(
            str(temp_path),
            fp16=False,
            language='en'
        )
        return result["text"]
    
    finally:
        # Clean up
        try:
            if temp_path.exists():
                temp_path.unlink()
            os.rmdir(temp_dir)
        except Exception as e:
            st.warning("Note: Temporary files will be cleaned up later.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
    help="Upload your audio file here. Supported formats: MP3, WAV, M4A, OGG, FLAC"
)

if uploaded_file is not None:
    # Add a transcribe button
    if st.button("🎯 Transcribe"):
        with st.spinner('Transcribing your audio... This might take a few minutes.'):
            try:
                # Get transcription
                transcription = transcribe_audio(uploaded_file)
                
                # Display results
                st.success("Transcription completed!")
                st.subheader("📝 Transcription:")
                st.write(transcription)
                
                # Download button
                st.download_button(
                    label="📥 Download Transcription",
                    data=transcription,
                    file_name=f"{uploaded_file.name}_transcription.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error("An error occurred during transcription. Please try again.")
                st.error(f"Error details: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using OpenAI's Whisper")