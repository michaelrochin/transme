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

# Initialize session state for transcriptions if it doesn't exist
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = {}

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
st.write("Upload multiple audio files and get their transcriptions!")

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

# Clear transcriptions button
if st.button("🗑️ Clear All Transcriptions"):
    st.session_state.transcriptions = {}
    st.experimental_rerun()

# File uploader with multiple files enabled
uploaded_files = st.file_uploader(
    "Choose audio files",
    type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
    accept_multiple_files=True,
    help="Upload your audio files here. Supported formats: MP3, WAV, M4A, OGG, FLAC"
)

if uploaded_files:
    # Add a transcribe button
    if st.button("🎯 Transcribe All Files"):
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.transcriptions:
                st.write(f"Processing: {uploaded_file.name}")
                with st.spinner(f'Transcribing {uploaded_file.name}... This might take a few minutes.'):
                    try:
                        # Get transcription
                        transcription = transcribe_audio(uploaded_file)
                        
                        # Store in session state
                        st.session_state.transcriptions[uploaded_file.name] = transcription
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}")
                        st.error(f"Error details: {str(e)}")

# Display all transcriptions
if st.session_state.transcriptions:
    st.markdown("### 📝 All Transcriptions")
    
    # Create a combined text of all transcriptions
    all_transcriptions = "\n\n".join([
        f"=== {filename} ===\n{text}"
        for filename, text in st.session_state.transcriptions.items()
    ])
    
    # Add a download button for all transcriptions
    st.download_button(
        label="📥 Download All Transcriptions",
        data=all_transcriptions,
        file_name="all_transcriptions.txt",
        mime="text/plain"
    )
    
    # Display individual transcriptions
    for filename, transcription in st.session_state.transcriptions.items():
        with st.expander(f"Transcription: {filename}", expanded=True):
            st.write(transcription)
            st.download_button(
                label=f"📥 Download This Transcription",
                data=transcription,
                file_name=f"{filename}_transcription.txt",
                mime="text/plain",
                key=f"download_{filename}"  # Unique key for each button
            )

# Footer
st.markdown("---")
st.markdown("""
### Tips:
- You can select multiple files at once
- Transcriptions are saved until you clear them
- Use the 'Clear All Transcriptions' button to start fresh
- Download individual transcriptions or all at once
""")
st.markdown("Made with ❤️ using OpenAI's Whisper")
