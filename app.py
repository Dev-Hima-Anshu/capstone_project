import streamlit as st
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import tempfile
import time
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
import io

# Set page configuration
st.set_page_config(
    page_title="Audio Translation Hub",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar initially
)

# Updated CSS for Sesame-like styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;  /* Light gray background */
        font-family: 'Inter', sans-serif;  /* Modern sans-serif font */
    }
    h1, h2, h3 {
        color: #1e3d59;  /* Darker blue for headings */
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;  /* Center tabs like Sesame’s clean layout */
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;  /* Light gray for inactive tabs */
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-size: 16px;
        color: #1e3d59;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e89ae;  /* Sesame’s blue for active tab */
        color: white;
    }
    .stButton>button {
        background-color: #4e89ae;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        width: auto;  /* More natural button size */
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #326b8f;  /* Darker blue on hover */
    }
    .stSelectbox, .stTextArea {
        background-color: white;
        border-radius: 8px;
        padding: 5px;
    }
    .css-18e3th9, .css-1d391kg {
        padding-top: 2rem;  /* Reduced padding for a tighter layout */
    }
    .stAudio, .stDownloadButton {
        margin-top: 10px;
    }
    .waveform {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Available languages for translation
LANGUAGES = {
    'af': 'Afrikaans', 'sq': 'Albanian', 'am': 'Amharic', 'ar': 'Arabic',
    'hy': 'Armenian', 'az': 'Azerbaijani', 'eu': 'Basque', 'be': 'Belarusian',
    'bn': 'Bengali', 'bs': 'Bosnian', 'bg': 'Bulgarian', 'ca': 'Catalan',
    'ceb': 'Cebuano', 'ny': 'Chichewa', 'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)', 'co': 'Corsican', 'hr': 'Croatian',
    'cs': 'Czech', 'da': 'Danish', 'nl': 'Dutch', 'en': 'English',
    'eo': 'Esperanto', 'et': 'Estonian', 'tl': 'Filipino', 'fi': 'Finnish',
    'fr': 'French', 'fy': 'Frisian', 'gl': 'Galician', 'ka': 'Georgian',
    'de': 'German', 'el': 'Greek', 'gu': 'Gujarati', 'ht': 'Haitian Creole',
    'ha': 'Hausa', 'haw': 'Hawaiian', 'iw': 'Hebrew', 'hi': 'Hindi',
    'hmn': 'Hmong', 'hu': 'Hungarian', 'is': 'Icelandic', 'ig': 'Igbo',
    'id': 'Indonesian', 'ga': 'Irish', 'it': 'Italian', 'ja': 'Japanese',
    'jw': 'Javanese', 'kn': 'Kannada', 'kk': 'Kazakh', 'km': 'Khmer',
    'ko': 'Korean', 'ku': 'Kurdish (Kurmanji)', 'ky': 'Kyrgyz', 'lo': 'Lao',
    'la': 'Latin', 'lv': 'Latvian', 'lt': 'Lithuanian', 'lb': 'Luxembourgish',
    'mk': 'Macedonian', 'mg': 'Malagasy', 'ms': 'Malay', 'ml': 'Malayalam',
    'mt': 'Maltese', 'mi': 'Maori', 'mr': 'Marathi', 'mn': 'Mongolian',
    'my': 'Myanmar (Burmese)', 'ne': 'Nepali', 'no': 'Norwegian', 'ps': 'Pashto',
    'fa': 'Persian', 'pl': 'Polish', 'pt': 'Portuguese', 'pa': 'Punjabi',
    'ro': 'Romanian', 'ru': 'Russian', 'sm': 'Samoan', 'gd': 'Scots Gaelic',
    'sr': 'Serbian', 'st': 'Sesotho', 'sn': 'Shona', 'sd': 'Sindhi',
    'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'so': 'Somali',
    'es': 'Spanish', 'su': 'Sundanese', 'sw': 'Swahili', 'sv': 'Swedish',
    'tg': 'Tajik', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish',
    'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 'vi': 'Vietnamese',
    'cy': 'Welsh', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'zu': 'Zulu'
}

# Initialize translator
translator = Translator()

# Function to convert raw audio bytes to WAV
def convert_to_wav(audio_bytes):
    try:
        # Load audio from bytes (assume it might be WebM or another format)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # Export as WAV
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format='wav')
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        st.error(f"Error converting audio to WAV: {str(e)}")
        return None

# Function to visualize audio waveform
def plot_waveform(audio_data, rate):
    fig, ax = plt.subplots(figsize=(8, 1.5))  # Smaller for compactness
    ax.plot(np.linspace(0, len(audio_data) / rate, len(audio_data)), audio_data, color='#4e89ae', lw=0.8)
    ax.set_axis_off()  # Minimalist, like Sesame’s visuals
    return fig

# Function to transcribe audio
def transcribe_audio(audio_data, language='en-US'):
    r = sr.Recognizer()
    try:
        text = r.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service"

# Function to translate text
def translate_text(text, target_language):
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        return f"Translation error: {str(e)}"

# Function to generate speech from text
def text_to_speech(text, language):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(fp.name)
        return fp.name
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

# Function to convert audio file to text
def audio_file_to_text(audio_file, source_language):
    r = sr.Recognizer()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        temp_audio.write(audio_file.getvalue())
        temp_audio_path = temp_audio.name
    
    # Load audio file
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = r.record(source)
    
    # Transcribe
    try:
        text = r.recognize_google(audio_data, language=source_language)
        
        # Clean up temp file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Function to convert audio file to different format
def convert_audio_file(input_file, output_format):
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_file)
        
        # Export to desired format
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{output_format}').name
        audio.export(output_path, format=output_format)
        
        return output_path
    except Exception as e:
        st.error(f"Conversion error: {str(e)}")
        return None

# Header Section
st.title("Audio Translation Hub")
st.markdown(
    """
    **Bridging languages through voice.**  
    Experience seamless speech translation with cutting-edge AI—live, from files, or text.
    """,
    unsafe_allow_html=True
)

# Demo Section (Tabs)
st.markdown("### Try It Yourself")
tab1, tab2, tab3 = st.tabs(["🎤 Live Translation", "📁 File Translation", "✏️ Text Translation"])

# Tab 1: Live Translation
with tab1:
    st.markdown("#### Record and Translate Live")
    col1, col2 = st.columns(2)
    with col1:
        source_lang_live = st.selectbox("Source Language", options=list(LANGUAGES.items()), format_func=lambda x: x[1], key="source_live")
    with col2:
        target_lang_live = st.selectbox("Target Language", options=list(LANGUAGES.items()), format_func=lambda x: x[1], key="target_live")
    
    audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key="live_recorder")
    
    if audio:
        st.success("Recording complete!")
        
        audio_bytes = audio['bytes']
        
        # Convert to WAV format
        wav_io = convert_to_wav(audio_bytes)
        if wav_io is None:
            st.error("Failed to process recorded audio.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(wav_io.read())
                temp_audio_path = temp_audio.name
            
            # Visualize waveform
            try:
                sample_rate, audio_data = wavfile.read(temp_audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # Use first channel for stereo
                st.markdown("**Recorded Audio Visualization:**")
                st.pyplot(plot_waveform(audio_data, sample_rate))
            except Exception as e:
                st.warning(f"Could not visualize audio: {str(e)}")
            
            # Transcribe
            r = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = r.record(source)
            
            with st.spinner("Transcribing and translating..."):
                transcribed_text = transcribe_audio(audio_data, source_lang_live[0])
                
                if "Error" not in transcribed_text and "could not understand" not in transcribed_text:
                    translated_text = translate_text(transcribed_text, target_lang_live[0])
                    translated_audio_path = text_to_speech(translated_text, target_lang_live[0])
                    
                    st.markdown(f"**Original:** {transcribed_text}")
                    st.markdown(f"**Translated:** {translated_text}")
                    st.audio(translated_audio_path, format='audio/mp3')
                    
                    # Clean up translated audio file
                    if os.path.exists(translated_audio_path):
                        time.sleep(2)  # Allow audio to play before deletion
                        os.remove(translated_audio_path)
                else:
                    st.error(transcribed_text)
            
            # Clean up recorded audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

# Tab 2: File Translation
with tab2:
    st.markdown("#### Upload and Translate Audio")
    col1, col2 = st.columns(2)
    with col1:
        source_lang_file = st.selectbox("Source Language", options=list(LANGUAGES.items()), format_func=lambda x: x[1], key="source_file")
        output_format = st.selectbox("Output Format", options=["mp3", "wav", "ogg", "flac"], key="output_format")
    with col2:
        target_lang_file = st.selectbox("Target Language", options=list(LANGUAGES.items()), format_func=lambda x: x[1], key="target_file")
    
    uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "ogg", "flac"])
    
    if uploaded_file and st.button("Translate Audio"):
        with st.spinner("Processing audio file..."):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            
            # Visualize the audio waveform if it's a WAV file
            if uploaded_file.name.endswith('.wav'):
                try:
                    sample_rate, audio_data = wavfile.read(temp_file_path)
                    if len(audio_data.shape) > 1:  # Check if stereo
                        audio_data = audio_data[:, 0]  # Just use first channel
                    st.markdown("**Audio Visualization:**")
                    st.pyplot(plot_waveform(audio_data, sample_rate))
                except Exception as e:
                    st.warning(f"Could not visualize this audio file: {str(e)}")
            
            # Convert to text
            transcribed_text = audio_file_to_text(uploaded_file, source_lang_file[0])
            
            if "Error" in transcribed_text or "could not understand" in transcribed_text:
                st.error(transcribed_text)
            else:
                # Translate text
                translated_text = translate_text(transcribed_text, target_lang_file[0])
                
                # Convert translated text to speech
                translated_audio_path = text_to_speech(translated_text, target_lang_file[0])
                
                if translated_audio_path:
                    # Convert to desired output format if needed
                    if not translated_audio_path.endswith(f'.{output_format}'):
                        final_audio_path = convert_audio_file(translated_audio_path, output_format)
                        
                        # Clean up the intermediate file
                        if os.path.exists(translated_audio_path):
                            os.remove(translated_audio_path)
                    else:
                        final_audio_path = translated_audio_path
                    
                    st.markdown(f"**Original:** {transcribed_text}")
                    st.markdown(f"**Translated:** {translated_text}")
                    st.audio(final_audio_path)
                    
                    # Offer download option
                    with open(final_audio_path, "rb") as file:
                        st.download_button(
                            label=f"Download ({output_format.upper()})",
                            data=file,
                            file_name=f"translated_audio.{output_format}",
                            mime=f"audio/{output_format}"
                        )
                    
                    # Clean up temporary files
                    if os.path.exists(final_audio_path):
                        time.sleep(2)  # Give time for audio player to load the file
                        os.remove(final_audio_path)
            
            # Clean up uploaded temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# Tab 3: Text Translation
with tab3:
    st.markdown("#### Translate Your Text")
    col1, col2 = st.columns(2)
    with col1:
        source_lang_text = st.selectbox("Source Language", options=list(LANGUAGES.items()), format_func=lambda x: x[1], key="source_text")
    with col2:
        target_lang_text = st.selectbox("Target Language", options=list(LANGUAGES.items()), format_func=lambda x: x[1], key="target_text")
    
    text_input = st.text_area("Enter text", height=100)
    
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Translate"):
            if not text_input:
                st.warning("Please enter some text to translate.")
            else:
                translated_text = translate_text(text_input, target_lang_text[0])
                st.markdown(f"**Translation:** {translated_text}")
    with col4:
        if st.button("Speak"):
            if not text_input:
                st.warning("Please enter some text to translate.")
            else:
                translated_text = translate_text(text_input, target_lang_text[0])
                audio_path = text_to_speech(translated_text, target_lang_text[0])
                st.audio(audio_path, format='audio/mp3')
                
                # Clean up temporary file
                if os.path.exists(audio_path):
                    time.sleep(2)  # Give time for audio player to load the file
                    os.remove(audio_path)

# Technical Details Section
st.markdown("### How It Works")
st.markdown(
    """
    Our Audio Translation Hub leverages:
    - **Speech Recognition**: Google’s API for accurate transcription.
    - **Translation**: Neural networks for natural language conversion.
    - **Text-to-Speech**: gTTS for lifelike audio output.
    """
)

# Vision Section
st.markdown("### What’s Next")
st.markdown(
    """
    Expanding to 50+ languages, integrating with IoT devices, and enhancing real-time performance for global communication.
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    Created as a capstone project for ML, DL, and IoT course | 2025  
    Powered by open-source tools and Google APIs.
    """
)