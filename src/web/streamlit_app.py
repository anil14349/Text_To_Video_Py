import streamlit as st
import psutil
import platform
import torch
import plotly.graph_objects as go
from datetime import datetime
import os
from typing import Optional, Dict
import logging

# Configure page at the very beginning
st.set_page_config(
    page_title="Resume AI Assistant",
    page_icon="📄",
    layout="wide"
)

logger = logging.getLogger(__name__)

class StreamlitApp:
    def __init__(self, pipeline, tts_generator):
        self.pipeline = pipeline
        self.tts_generator = tts_generator
        
    def _get_system_metrics(self):
        """Get system metrics for display"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            gpu_used = torch.cuda.memory_allocated(0) / 1024**3
        else:
            gpu_name = "Not Available"
            gpu_memory = 0
            gpu_used = 0
            
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used / (1024**3),  # Convert to GB
            'memory_total': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'disk_used': disk.used / (1024**3),
            'disk_total': disk.total / (1024**3),
            'gpu_name': gpu_name,
            'gpu_memory': gpu_memory,
            'gpu_used': gpu_used
        }
        
    def _create_gauge_chart(self, value, title):
        """Create a gauge chart for metrics"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=200)
        return fig
        
    def _display_metrics(self):
        """Display system metrics in the sidebar"""
        metrics = self._get_system_metrics()
        
        st.sidebar.header("📊 System Metrics")
        
        # CPU and Memory Gauges
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.plotly_chart(self._create_gauge_chart(metrics['cpu_percent'], "CPU Usage %"), use_container_width=True)
        with col2:
            st.plotly_chart(self._create_gauge_chart(metrics['memory_percent'], "Memory Usage %"), use_container_width=True)
            
        # Memory Details
        st.sidebar.markdown(f"""
        💾 **Memory Details**
        - Used: {metrics['memory_used']:.1f} GB
        - Total: {metrics['memory_total']:.1f} GB
        """)
        
        # GPU Information
        st.sidebar.markdown(f"""
        🎮 **GPU Information**
        - Device: {metrics['gpu_name']}
        - Memory Used: {metrics['gpu_used']:.1f} GB
        - Total Memory: {metrics['gpu_memory']:.1f} GB
        """)
        
        # System Information
        st.sidebar.markdown(f"""
        💻 **System Information**
        - OS: {platform.system()} {platform.release()}
        - Python: {platform.python_version()}
        - Torch: {torch.__version__}
        """)
        
    def _display_model_info(self):
        """Display model information"""
        st.sidebar.header("🤖 Model Information")
        st.sidebar.markdown(f"""
        **Text Generation Model**
        - Name: {self.pipeline.model.config.model_type}
        - Parameters: {sum(p.numel() for p in self.pipeline.model.parameters())/1e6:.1f}M
        - Device: {self.pipeline.device}
        
        **TTS Model**
        - Type: ChatTTS
        - Backend: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'CPU'}
        """)
        
    def process_resume(self, file) -> Dict[str, str]:
        """Process uploaded resume file.
        
        Args:
            file: Uploaded file object from streamlit
            
        Returns:
            Dictionary containing summary sections and any errors
        """
        try:
            resume_text = self.pipeline.read_resume(file)
            if not resume_text:
                return {
                    "error": "Could not extract text from resume",
                    "overview": "Failed to process resume file"
                }
                
            summary = self.pipeline.generate_summary(resume_text)
            return summary
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return {
                "error": f"Error processing resume: {str(e)}",
                "overview": "Failed to process resume"
            }
            
    def generate_audio(self, summary: str, gender: str = "female", custom_config: dict = {}) -> bool:
        """Generate audio for summary"""
        try:
            with st.spinner("Generating audio..."):
                audio_path = self.tts_generator.generate_summary_audio(summary, gender, custom_config)
                
            if audio_path and os.path.exists(audio_path):
                # Read the audio file
                with open(audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
                return True
                
            st.warning("Could not generate audio. Please try again.")
            return False
            
        except Exception as e:
            if "rate limit exceeded" in str(e).lower():
                st.error("Rate limit exceeded for audio generation. Please try again in a few minutes.")
            else:
                st.error(f"Error generating audio: {str(e)}")
            logger.error(f"Audio generation error: {e}", exc_info=True)
            return False
    
    def run(self):
        """Run the Streamlit app."""
        st.title("Resume AI Assistant")
        st.markdown("""
        Transform your resume into a professional summary with AI-powered insights and text-to-speech capabilities.
        Upload your resume below to get started!
        """)
        
        # Display metrics and model info in sidebar
        self._display_metrics()
        self._display_model_info()
        
        # Audio settings in sidebar
        st.sidebar.header("🎧 Audio Settings")
        voice_gender = st.sidebar.radio("Voice Type", ["female", "male"], index=0)
        
        with st.sidebar.expander("Voice Customization", expanded=False):
            oral_level = st.slider("Speech Style", 0, 3, 1, 
                               help="Controls how casual or formal the speech sounds")
            laugh_level = st.slider("Expression Level", 0, 3, 0, 
                                help="Controls emotional expression in speech")
            break_level = st.slider("Pause Length", 1, 5, 3, 
                                help="Controls the length of pauses between sentences")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📎 Upload Resume")
            uploaded_file = st.file_uploader("Choose a resume file", type=["txt", "pdf", "docx"])
            
            if uploaded_file:
                with st.spinner("Processing resume..."):
                    summary = self.process_resume(uploaded_file)
                    
                    if "error" in summary:
                        st.error(summary["error"])
                        
                    if summary.get("overview"):
                        st.subheader("Summary")
                        for section, content in summary.items():
                            if section != "error" and section != "overview":
                                with st.expander(section, expanded=True):
                                    st.markdown(content)
                    
                    # Audio generation
                    st.subheader("🎧 Audio Summary")
                    if st.button("Generate Audio Summary", key="generate_audio"):
                        with st.spinner("Generating audio..."):
                            custom_config = {
                                "oral_level": oral_level,
                                "laugh_level": laugh_level,
                                "break_level": break_level,
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "top_k": 20
                            }
                            
                            self.generate_audio(
                                str(summary),
                                gender=voice_gender,
                                custom_config=custom_config
                            )
        
        with col2:
            st.subheader("📊 Processing Stats")
            if uploaded_file:
                stats_placeholder = st.empty()
                stats_placeholder.markdown(f"""
                **File Information**
                - Name: {uploaded_file.name}
                - Size: {uploaded_file.size/1024:.1f} KB
                - Type: {uploaded_file.type}
                
                **Processing Time**
                - Upload Time: {datetime.now().strftime('%H:%M:%S')}
                """)
                
                # Add a download button for the summary
                if 'summary' in locals():
                    summary_text = "\n\n".join([f"# {k}\n{v}" for k, v in summary.items()])
                    st.download_button(
                        label="Download Summary",
                        data=summary_text,
                        file_name=f"resume_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            else:
                st.info("Upload a resume to see processing statistics")
                
            # Display supported formats
            st.markdown("""
            **Supported Formats**
            - PDF (.pdf)
            - Word (.docx)
            - Text (.txt)
            
            **Best Practices**
            1. Ensure your resume is in a standard format
            2. Include clear section headers
            3. Keep formatting simple
            4. Maximum file size: 10MB
            """)

def main():
    from src.model.core.pipeline import ResumePipeline
    from utils.tts_generator import TTSGenerator
    from utils.config import Config

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.yaml')
    config = Config(config_path)

    # Initialize pipeline and TTS components with error handling
    try:
        with st.spinner("Loading AI models (this may take a few minutes on first run)..."):
            pipeline = ResumePipeline(config)
            tts_generator = TTSGenerator(config)
            
        st.success("✅ Models loaded successfully!")
        
    except Exception as e:
        if "rate limit exceeded" in str(e).lower():
            st.error("Rate limit exceeded. Please wait a few minutes and try again.")
            logger.error(f"Rate limit error: {e}")
        else:
            st.error(f"Error initializing components: {str(e)}")
            logger.error(f"Initialization error: {e}", exc_info=True)
        st.stop()

    app = StreamlitApp(pipeline, tts_generator)
    app.run()

if __name__ == "__main__":
    main()
