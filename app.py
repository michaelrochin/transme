import streamlit as st
import whisper
import tempfile
import os
from pathlib import Path
import subprocess
import soundfile as sf
import io

# Page config
st.set_page_config(
    page_title="Audio Transcription",
    page_icon="🎙️",
    layout="centered"
)

# Initialize session state for transcriptions if it doesn't exist
if 'transcriptions' not in st.session_state:
    st.session_state.transcriptions = {}

# Initialize whisper model
@st.cache_resource
def load_model():
    model = whisper.load_model("medium")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

def validate_audio(audio_file):
    """Validate audio file before processing"""
    try:
        # Try reading the audio file
        data, samplerate = sf.read(io.BytesIO(audio_file.getvalue()))
        
        # Check if file is empty
        if len(data) == 0:
            return False, "Audio file appears to be empty"
            
        # Check if duration is too short
        duration = len(data) / samplerate
        if duration < 0.1:  # Less than 0.1 seconds
            return False, "Audio file is too short"
            
        return True, "Audio file is valid"
    except Exception as e:
        return False, f"Invalid audio file: {str(e)}"

def transcribe_audio(audio_file):
    """Transcribe audio with improved error handling"""
    # First validate the audio file
    is_valid, message = validate_audio(audio_file)
    if not is_valid:
        raise ValueError(message)
    
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
        
        # Verify transcription result
        if not result or not result.get("text"):
            raise ValueError("No transcription produced")
            
        return result["text"]
    
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")
    
    finally:
        # Clean up
        try:
            if temp_path.exists():
                temp_path.unlink()
            os.rmdir(temp_dir)
        except Exception:
            pass

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
        skipped_files = []
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.transcriptions:
                st.write(f"Processing: {uploaded_file.name}")
                with st.spinner(f'Transcribing {uploaded_file.name}... This might take a few minutes.'):
                    try:
                        # Get transcription
                        transcription = transcribe_audio(uploaded_file)
                        
                        # Store in session state
                        st.session_state.transcriptions[uploaded_file.name] = transcription
                        st.success(f"Successfully transcribed: {uploaded_file.name}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        skipped_files.append((uploaded_file.name, error_msg))
                        st.error(f"Error processing {uploaded_file.name}: {error_msg}")
        
        if skipped_files:
            st.warning("Some files were skipped due to errors:")
            for filename, error in skipped_files:
                st.write(f"- {filename}: {error}")

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

# Footer with enhanced tips
st.markdown("---")
st.markdown("""
### Tips:
- Make sure your audio files are not corrupted or empty
- Audio should be clear and at least 0.1 seconds long
- If a file fails, try converting it to a different format (e.g., MP3 or WAV)
- Some very short or corrupted files might not process correctly
""")
st.markdown("Made with ❤️ using OpenAI's Whisper")
