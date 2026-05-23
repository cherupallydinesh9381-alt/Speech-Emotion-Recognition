"""
DeepSER-GUI: Speech Emotion Recognition - Streamlit Application
Professional GUI using emotion2vec_plus_large model from Hugging Face
"""

import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
import tempfile
from emotion2vec_recognition import Emotion2VecRecognizer
try:
    from audio_recorder_streamlit import audio_recorder
except Exception:
    def audio_recorder(*args, **kwargs):
        st.warning("Optional `audio-recorder-streamlit` not installed. Install with `pip install audio-recorder-streamlit` to enable in-browser recording.")
        return None
import warnings
warnings.filterwarnings('ignore')


# Page Configuration
st.set_page_config(
    page_title="DeepSER - Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .emotion-result {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Emotion color mapping
EMOTION_COLORS = {
    'neutral': '#95A5A6',
    'calm': '#3498DB',
    'happy': '#F1C40F',
    'sad': '#3498DB',
    'angry': '#E74C3C',
    'fear': '#9B59B6',
    'disgust': '#16A085',
    'ps': '#E67E22',
    'surprise': '#E67E22',
    'boredom': '#7F8C8D'
}

# Emotion emoji mapping
EMOTION_EMOJIS = {
    'neutral': '😐',
    'calm': '😌',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'disgust': '🤢',
    'ps': '😲',
    'surprise': '😲',
    'boredom': '😑'
}

@st.cache_resource
def load_model():
    """Load emotion2vec model with caching"""
    try:
        # Force reload by clearing any old modelscope imports
        import sys
        modules_to_clear = [key for key in sys.modules.keys() if 'modelscope' in key or 'funasr' in key]
        for module in modules_to_clear:
            del sys.modules[module]
        
        model = Emotion2VecRecognizer(
            model_name='iic/emotion2vec_base',
            granularity='utterance'
        )
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, False

def predict_emotion_from_audio(model, audio_path):
    """Predict emotion from audio file using emotion2vec"""
    try:
        import traceback
        import sys
        
        # Log the attempt
        print(f"[DEBUG] Starting prediction for: {audio_path}", file=sys.stderr)
        print(f"[DEBUG] File exists: {os.path.exists(audio_path)}", file=sys.stderr)
        if os.path.exists(audio_path):
            print(f"[DEBUG] File size: {os.path.getsize(audio_path)} bytes", file=sys.stderr)
        
        result = model.predict(audio_path)
        print(f"[DEBUG] Prediction successful", file=sys.stderr)
        return result
    except Exception as e:
        import traceback
        import sys
        error_trace = traceback.format_exc()
        print(f"[ERROR] Prediction failed: {str(e)}", file=sys.stderr)
        print(f"[ERROR] Full traceback:\n{error_trace}", file=sys.stderr)
        st.error(f"Error predicting emotion: {str(e)}")
        st.code(error_trace)
        return None

def plot_waveform(audio_path):
    """Plot audio waveform"""
    try:
        audio_np, sr = sf.read(audio_path, always_2d=False)
        # stereo -> mono
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)

        fig, ax = plt.subplots(figsize=(12, 4))
        times = np.arange(audio_np.shape[0]) / sr
        ax.plot(times, audio_np, color='#1E88E5')
        ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting waveform: {e}")
        return None

def plot_spectrogram(audio_path):
    """Plot Mel spectrogram"""
    try:
        audio_np, sr = sf.read(audio_path, always_2d=False)
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)

        fig, ax = plt.subplots(figsize=(12, 4))
        Pxx, freqs, bins, im = ax.specgram(audio_np, NFFT=512, Fs=sr, noverlap=256, cmap='viridis')
        ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        fig.colorbar(im, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting spectrogram: {e}")
        return None

def plot_probability_chart(probabilities, emotions):
    """Plot emotion probability bar chart"""
    try:
        # Sort by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probs = probabilities[sorted_indices]
        sorted_emotions = [emotions[i] for i in sorted_indices]
        
        colors = [EMOTION_COLORS.get(e, '#95A5A6') for e in sorted_emotions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(sorted_emotions, sorted_probs, color=colors, alpha=0.8)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob*100:.1f}%',
                   ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
        ax.set_title('Emotion Probability Distribution', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting probability chart: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎤 DeepSER - Speech Emotion Recognition</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Deep CNN | Real-time Emotion Analysis</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Logo removed per user request
        st.header("⚙️ Configuration")
        
        # Model info
        st.info("**Model:** emotion2vec_base\n\nPre-trained transformer from Hugging Face")
        
        # Visualization options
        st.subheader("📊 Visualization Options")
        show_waveform = st.checkbox("Show Waveform", value=True)
        show_spectrogram = st.checkbox("Show Spectrogram", value=True)
        show_probabilities = st.checkbox("Show Probability Chart", value=True)
        
        # About section
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        **DeepSER** uses emotion2vec_base:
        - Transformer-based model
        - Pre-trained on large-scale emotion data
        - Supports 9 emotion categories
        - Faster inference than plus_large
        
        **Features:**
        - Real-time emotion detection
        - High accuracy predictions
        - Visual audio analysis
        """)
    
    # Load model
    with st.spinner('🔄 Loading emotion2vec model from Hugging Face...'):
        model, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ Failed to load emotion2vec model. Please check your internet connection.")
        st.stop()
    else:
        st.success("✅ emotion2vec_base model loaded successfully!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Audio", "🎙️ Record Live", "📊 Model Info", "📖 Instructions"])
    
    with tab1:
        st.subheader("Upload Your Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Supported formats: WAV, MP3, FLAC, OGG (Max 20MB)"
        )
        
        if uploaded_file is not None:
            # File info
            file_details = {
                "Filename": uploaded_file.name,
                "File Size": f"{uploaded_file.size / 1024:.2f} KB",
                "File Type": uploaded_file.type
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filename", file_details["Filename"])
            with col2:
                st.metric("Size", file_details["File Size"])
            with col3:
                st.metric("Type", file_details["File Type"])
            
            # Save uploaded file temporarily
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb')
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.close()
            tmp_path = tmp_file.name
            
            # Audio player
            st.subheader("🔊 Audio Playback")
            st.audio(uploaded_file, format='audio/wav')
            
            # Analyze button
            if st.button("🎯 Analyze Emotion", type="primary", use_container_width=True):
                with st.spinner('🔍 Analyzing emotion...'):
                    tmp_path_upload = tmp_path  # Reference the already created temp file
                    try:
                        # Predict using emotion2vec
                        result = predict_emotion_from_audio(model, tmp_path_upload)
                        
                        if result is not None:
                            predicted_emotion = result['emotion']
                            confidence = result['confidence']
                            probabilities_dict = result['probabilities']
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("🎭 Emotion Detection Results")
                            
                            # Main emotion result
                            emoji = EMOTION_EMOJIS.get(predicted_emotion, '🎭')
                            color = EMOTION_COLORS.get(predicted_emotion, '#95A5A6')
                            
                            st.markdown(
                                f'<div class="emotion-result" style="background-color: {color}20; border: 3px solid {color};">'
                                f'{emoji} <span style="color: {color};">{predicted_emotion.upper()}</span> '
                                f'<br><span style="font-size: 1.5rem;">Confidence: {confidence*100:.1f}%</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Probability chart
                            if show_probabilities:
                                st.subheader("📊 Confidence Scores")
                                # Convert dict to arrays for plotting
                                emotion_labels = list(probabilities_dict.keys())
                                emotion_probs = np.array(list(probabilities_dict.values()))
                                fig_prob = plot_probability_chart(emotion_probs, emotion_labels)
                                if fig_prob:
                                    st.pyplot(fig_prob)
                                    plt.close()
                            
                            # Visualizations
                            if show_waveform or show_spectrogram:
                                st.markdown("---")
                                st.subheader("📈 Audio Analysis")
                                
                                if show_waveform and show_spectrogram:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        fig_wave = plot_waveform(tmp_path)
                                        if fig_wave:
                                            st.pyplot(fig_wave)
                                            plt.close()
                                    with col2:
                                        fig_spec = plot_spectrogram(tmp_path)
                                        if fig_spec:
                                            st.pyplot(fig_spec)
                                            plt.close()
                                elif show_waveform:
                                    fig_wave = plot_waveform(tmp_path)
                                    if fig_wave:
                                        st.pyplot(fig_wave)
                                        plt.close()
                                elif show_spectrogram:
                                    fig_spec = plot_spectrogram(tmp_path)
                                    if fig_spec:
                                        st.pyplot(fig_spec)
                                        plt.close()
                        
                        else:
                            st.error("❌ Failed to analyze uploaded audio")
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    finally:
                        # Clean up temp file for upload
                        if tmp_path_upload and os.path.exists(tmp_path_upload):
                            try:
                                os.unlink(tmp_path_upload)
                            except:
                                pass
        
        else:
            st.info("👆 Upload an audio file to get started!")
            
            # Example section
            st.markdown("---")
            st.subheader("💡 Example Audio")
            st.markdown("""
            **Supported audio characteristics:**
            - Duration: 1-10 seconds
            - Sample rate: 22,050 Hz (auto-converted)
            - Channels: Mono (auto-converted)
            - Format: WAV, MP3, FLAC, OGG
            
            **Best results with:**
            - Clear speech
            - Minimal background noise
            - Single speaker
            """)
    
    with tab2:
        st.subheader("🎙️ Record Live Audio")
        st.markdown("Click the microphone button to start recording your voice")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            # Analyze button for recorded audio
            if st.button("🎯 Analyze Recorded Audio", type="primary", use_container_width=True):
                with st.spinner('🔍 Analyzing your voice...'):
                    tmp_path = None
                    try:
                        # Save recorded audio to temp file
                        import io
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb', dir=tempfile.gettempdir())
                        tmp_file.write(audio_bytes)
                        tmp_file.close()
                        tmp_path = os.path.abspath(tmp_file.name)
                        
                        # Verify file exists and has content
                        if not os.path.exists(tmp_path):
                            st.error(f"Temp file was not created: {tmp_path}")
                            raise FileNotFoundError(f"Temp file not found: {tmp_path}")
                        
                        file_size = os.path.getsize(tmp_path)
                        if file_size == 0:
                            st.error(f"Temp file is empty: {tmp_path}")
                            raise ValueError("Recorded audio file is empty")
                        
                        st.info(f"Debug: Audio saved to {tmp_path} ({file_size} bytes)")
                        
                        # Predict using emotion2vec
                        result = predict_emotion_from_audio(model, tmp_path)
                        
                        if result is not None:
                            predicted_emotion = result['emotion']
                            confidence = result['confidence']
                            probabilities_dict = result['probabilities']
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("🎭 Emotion Detection Results")
                            
                            # Main emotion result
                            emoji = EMOTION_EMOJIS.get(predicted_emotion, '🎭')
                            color = EMOTION_COLORS.get(predicted_emotion, '#95A5A6')
                            
                            st.markdown(
                                f'<div class="emotion-result" style="background-color: {color}20; border: 3px solid {color};">'
                                f'{emoji} <span style="color: {color};">{predicted_emotion.upper()}</span> '
                                f'<br><span style="font-size: 1.5rem;">Confidence: {confidence*100:.1f}%</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Probability chart
                            if show_probabilities:
                                st.subheader("📊 Confidence Scores")
                                emotion_labels = list(probabilities_dict.keys())
                                emotion_probs = np.array(list(probabilities_dict.values()))
                                fig_prob = plot_probability_chart(emotion_probs, emotion_labels)
                                if fig_prob:
                                    st.pyplot(fig_prob)
                                    plt.close()
                            
                            # Visualizations
                            if show_waveform or show_spectrogram:
                                st.markdown("---")
                                st.subheader("📈 Audio Analysis")
                                
                                if show_waveform and show_spectrogram:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        fig_wave = plot_waveform(tmp_path)
                                        if fig_wave:
                                            st.pyplot(fig_wave)
                                            plt.close()
                                    with col2:
                                        fig_spec = plot_spectrogram(tmp_path)
                                        if fig_spec:
                                            st.pyplot(fig_spec)
                                            plt.close()
                                elif show_waveform:
                                    fig_wave = plot_waveform(tmp_path)
                                    if fig_wave:
                                        st.pyplot(fig_wave)
                                        plt.close()
                                elif show_spectrogram:
                                    fig_spec = plot_spectrogram(tmp_path)
                                    if fig_spec:
                                        st.pyplot(fig_spec)
                                        plt.close()
                        
                        else:
                            st.error("❌ Failed to analyze recorded audio")
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    finally:
                        # Clean up temp file for recording
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
        else:
            st.info("🎤 Click the microphone button above to record your voice (3-10 seconds recommended)")
            st.markdown("""
            **Tips for best results:**
            - Speak clearly into your microphone
            - Express emotion naturally in your voice
            - Keep recording between 3-10 seconds
            - Minimize background noise
            - Use headphones to avoid feedback
            """)
    
    with tab3:
        st.subheader("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Model Architecture")
            st.markdown("""
            - **Model:** emotion2vec_base
            - **Type:** Transformer-based
            - **Input:** Raw audio waveform
            - **Framework:** PyTorch + ModelScope
            - **Output:** 9 emotion classes
            - **Activation:** Softmax probabilities
            """)
        
        with col2:
            st.markdown("### 🎯 Model Details")
            st.markdown("""
            - **Model:** emotion2vec_base
            - **Type:** Transformer-based
            - **Source:** Hugging Face ModelScope
            - **Training:** Large-scale emotion corpus
            - **Framework:** PyTorch + ModelScope
            - **Emotions:** 9 categories
            """)
        
        st.markdown("---")
        st.subheader("🎭 Supported Emotions")
        
        available_emotions = model.get_available_emotions() if model else ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'ps', 'calm']
        emotion_cols = st.columns(min(4, len(available_emotions)))
        for idx, emotion in enumerate(available_emotions):
            with emotion_cols[idx % 4]:
                emoji = EMOTION_EMOJIS.get(emotion, '🎭')
                st.markdown(f"**{emoji} {emotion.capitalize()}**")
    
    with tab4:
        st.subheader("📖 How to Use")
        
        st.markdown("""
        ### Step-by-Step Guide
        
        **Option 1: Upload Audio File**
        1. Go to "Upload Audio" tab
        2. Click "Browse files" or drag & drop
        3. Supported: WAV, MP3, FLAC, OGG (Max 20 MB)
        4. Click "Analyze Emotion" button
        5. View results with visualizations
        
        **Option 2: Record Live Audio**
        1. Go to "Record Live" tab
        2. Click the microphone button
        3. Speak with emotion (3-10 seconds)
        4. Stop recording
        5. Click "Analyze Recorded Audio"
        6. View your emotion results
        
        ### 💡 Tips
           - Listen to your uploaded file
        
        3. **Analyze Emotion**
           - Click "Analyze Emotion" button
           - emotion2vec model processes the audio
        
        4. **View Results**
           - See predicted emotion with confidence
           - Explore probability distribution
           - View waveform and spectrogram
        
        ### 💡 Tips
        
        - **Best Audio Quality:** Clear voice, minimal background noise
        - **Duration:** 3-10 seconds recommended
        - **Format:** WAV or FLAC for best quality
        - **Recording:** Speak naturally with emotional expression
        
        ### 🔬 Technology Stack
        
        - **Model:** emotion2vec_base (Hugging Face)
        - **Framework:** PyTorch + ModelScope
        - **Frontend:** Streamlit
        - **Audio:** librosa
        - **Visualization:** matplotlib, plotly
        
        ### 📚 References
        
        - [emotion2vec Model](https://huggingface.co/emotion2vec/emotion2vec_base)
        - [ModelScope](https://www.modelscope.cn/)
        - [Streamlit](https://streamlit.io)
        
        ### Tips for Best Results
        
        ✅ **DO:**
        - Use clear, clean audio
        - Record in quiet environment
        - Speak naturally with emotion
        - Use 2-5 second clips
        
        ❌ **DON'T:**
        - Use heavily processed audio
        - Include multiple speakers
        - Use excessive background noise
        - Upload very long files
        
        ### Troubleshooting
        
        **Problem:** Prediction seems incorrect
        - **Solution:** Try different audio clip, ensure clear speech
        
        **Problem:** Upload fails
        - **Solution:** Check file format and size (<20MB)
        
        **Problem:** Low confidence scores
        - **Solution:** Audio may be ambiguous or low quality
        """)
        
        st.markdown("---")
        st.info("💡 **Pro Tip:** For best results, use audio with clear emotional expression and minimal background noise.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p><strong>DeepSER-GUI v1.0</strong> | Speech Emotion Recognition System</p>
            <p>Built with Streamlit 🎈 | Powered by TensorFlow 🤖</p>
            <p>© 2025 | Master's Project | Deep Learning for Emotion Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
