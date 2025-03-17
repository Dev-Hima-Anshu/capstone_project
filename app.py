import streamlit as st
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import tempfile
import time
from pydub import AudioSegment
import io
import pyaudio
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Audio Translation Hub",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e89ae;
        color: white;
    }
    .stButton>button {
        color: white;
        background-color: #4e89ae;
        width: 100%;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding-top: 3.5rem;
    }
    h1, h2, h3 {
        color: #1e3d59;
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

# Function to visualize audio waveform
def plot_waveform(audio_data, rate):
    duration = len(audio_data) / rate
    time_axis = np.linspace(0, duration, len(audio_data))
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(time_axis, audio_data, color='#4e89ae', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform')
    ax.grid(True, alpha=0.3)
    
    return fig

# Function to record audio
def record_audio(duration=5):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source)
        st.write(f"Recording for {duration} seconds...")
        audio = r.listen(source, timeout=duration, phrase_time_limit=duration)
    
    return audio

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

# Main app header
st.title("🌐 Audio Translation Hub")
st.markdown("Translate speech and audio files between languages seamlessly")

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["🎤 Live Translation", "📁 File Translation", "✏️ Text Translation"])

# Tab 1: Live Translation
with tab1:
    st.header("Live Speech Translation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_lang_live = st.selectbox(
            "Source Language",
            options=list(LANGUAGES.items()),
            format_func=lambda x: x[1],
            key="source_live"
        )
        
        record_duration = st.slider("Recording Duration (seconds)", 1, 60, 5)
    
    with col2:
        target_lang_live = st.selectbox(
            "Target Language",
            options=list(LANGUAGES.items()),
            format_func=lambda x: x[1],
            key="target_live"
        )
    
    if st.button("Start Recording"):
        with st.spinner("Recording..."):
            audio_data = record_audio(record_duration)
            
        st.success("Recording complete!")
        
        with st.spinner("Transcribing..."):
            # Use the language code from the tuple
            transcribed_text = transcribe_audio(audio_data, source_lang_live[0])
        
        st.subheader("Original Text:")
        st.write(transcribed_text)
        
        with st.spinner("Translating..."):
            translated_text = translate_text(transcribed_text, target_lang_live[0])
        
        st.subheader("Translated Text:")
        st.write(translated_text)
        
        with st.spinner("Generating audio..."):
            translated_audio_path = text_to_speech(translated_text, target_lang_live[0])
            
            if translated_audio_path:
                st.audio(translated_audio_path, format='audio/mp3')
                
                # Clean up temporary file
                if os.path.exists(translated_audio_path):
                    time.sleep(2)  # Give time for audio player to load the file
                    os.remove(translated_audio_path)

# Tab 2: File Translation
with tab2:
    st.header("Audio File Translation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_lang_file = st.selectbox(
            "Source Language",
            options=list(LANGUAGES.items()),
            format_func=lambda x: x[1],
            key="source_file"
        )
        
        output_format = st.selectbox(
            "Output Format",
            options=["mp3", "wav", "ogg", "flac"],
            key="output_format"
        )
    
    with col2:
        target_lang_file = st.selectbox(
            "Target Language",
            options=list(LANGUAGES.items()),
            format_func=lambda x: x[1],
            key="target_file"
        )
    
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "ogg", "flac"])
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
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
                st.write("Audio Visualization:")
                st.pyplot(plot_waveform(audio_data, sample_rate))
            except Exception as e:
                st.warning(f"Could not visualize this audio file: {str(e)}")
        
        if st.button("Translate Audio File"):
            with st.spinner("Processing audio file..."):
                # Convert to text
                transcribed_text = audio_file_to_text(uploaded_file, source_lang_file[0])
                
                if "Error" in transcribed_text or "could not understand" in transcribed_text:
                    st.error(transcribed_text)
                else:
                    st.subheader("Original Text:")
                    st.write(transcribed_text)
                    
                    # Translate text
                    with st.spinner("Translating..."):
                        translated_text = translate_text(transcribed_text, target_lang_file[0])
                    
                    st.subheader("Translated Text:")
                    st.write(translated_text)
                    
                    # Convert translated text to speech
                    with st.spinner("Generating translated audio..."):
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
                            
                            st.subheader("Translated Audio:")
                            st.audio(final_audio_path)
                            
                            # Offer download option
                            with open(final_audio_path, "rb") as file:
                                btn = st.download_button(
                                    label=f"Download Translated Audio ({output_format.upper()})",
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
    st.header("Text Translation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_lang_text = st.selectbox(
            "Source Language",
            options=list(LANGUAGES.items()),
            format_func=lambda x: x[1],
            key="source_text"
        )
    
    with col2:
        target_lang_text = st.selectbox(
            "Target Language",
            options=list(LANGUAGES.items()),
            format_func=lambda x: x[1],
            key="target_text"
        )
    
    text_input = st.text_area("Enter text to translate", height=150)
    
    col3, col4 = st.columns(2)
    
    with col3:
        translate_button = st.button("Translate Text")
    
    with col4:
        speak_button = st.button("Speak Translation")
    
    if translate_button or speak_button:
        if not text_input:
            st.warning("Please enter some text to translate.")
        else:
            with st.spinner("Translating..."):
                translated_text = translate_text(text_input, target_lang_text[0])
            
            st.subheader("Translation:")
            st.write(translated_text)
            
            if speak_button:
                with st.spinner("Generating audio..."):
                    audio_path = text_to_speech(translated_text, target_lang_text[0])
                    
                    if audio_path:
                        st.audio(audio_path, format='audio/mp3')
                        
                        # Clean up temporary file
                        if os.path.exists(audio_path):
                            time.sleep(2)  # Give time for audio player to load the file
                            os.remove(audio_path)

# Sidebar with app info
with st.sidebar:
    st.title("About")
    st.info(
        """
        This application demonstrates audio translation 
        using machine learning capabilities.
        
        Features:
        - Live audio translation
        - Audio file translation
        - Text-to-text translation with audio playback
        
        Powered by Google's speech recognition and translation services.
        """
    )
    
    st.subheader("Translation Stats")
    st.metric("Languages Available", len(LANGUAGES))
    
    st.subheader("How It Works")
    st.markdown(
        """
        1. **Speech Recognition** converts speech to text
        2. **Machine Translation** translates the text
        3. **Text-to-Speech** converts translated text to audio
        """
    )
    
    st.subheader("ML Components")
    st.markdown(
        """
        - Speech Recognition: Uses Google's API
        - Text Translation: Neural Machine Translation
        - Audio Processing: Signal processing techniques
        """
    )

# Footer
st.markdown("---")
st.markdown("Created as a capstone project for ML, DL, and IoT course | 2025")