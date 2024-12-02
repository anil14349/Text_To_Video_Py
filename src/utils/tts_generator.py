import os
import torch
import torchaudio
import ChatTTS
from typing import Optional, Dict, Any
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress ALL warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class TTSGenerator:
    def __init__(self, config: Dict[str, Any]):
        """Initialize TTS Generator with configuration."""
        self.config = config
        self.tts_config = config.get_tts_config()
        self.paths_config = config.get_paths_config()
        
        # Set device
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        logger.info(f"Using device: {self.device}")
        self.chat = None
        self.setup_tts()
        
        # Set default sample rate
        self.sample_rate = 24000

    def setup_tts(self):
        """Initialize ChatTTS model"""
        try:
            self.chat = ChatTTS.Chat()
            self.chat.load(compile=False)
            logger.info("ChatTTS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ChatTTS model: {e}")
            self.chat = None

    def save_audio(self, audio_tensor: torch.Tensor, output_path: str) -> str:
        """Save audio tensor to file using torchaudio"""
        try:
            # Ensure audio tensor has correct shape (channels, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 2 and audio_tensor.size(0) > audio_tensor.size(1):
                audio_tensor = audio_tensor.transpose(0, 1)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save audio using torchaudio with explicit backend
            torchaudio.save(
                output_path,
                audio_tensor,
                self.sample_rate,
                backend="soundfile",
                format="wav"
            )
            
            logger.info(f"Audio saved successfully to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise

    def generate_audio(
        self,
        text: str,
        gender: str = 'female',
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Generate audio from text using ChatTTS."""
        if not self.chat:
            logger.error("Error: ChatTTS model not initialized")
            return None
            
        try:
            # Get base voice configuration
            voice_config = (
                self.tts_config.get('female_voice', {})
                if gender.lower() == 'female'
                else self.tts_config.get('male_voice', {})
            )
            
            # Merge with custom config if provided
            if custom_config:
                voice_config.update(custom_config)
            
            # Ensure required parameters have default values
            voice_config.setdefault('oral_level', 1)
            voice_config.setdefault('laugh_level', 0)
            voice_config.setdefault('break_level', 3)
            voice_config.setdefault('temperature', 0.7)
            voice_config.setdefault('top_p', 0.9)
            voice_config.setdefault('top_k', 20)

            # Generate speaker embedding
            rand_spk = self.chat.sample_random_speaker()

            # Configure inference parameters
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=rand_spk,
                temperature=voice_config['temperature'],
                top_P=voice_config['top_p'],
                top_K=voice_config['top_k']
            )

            # Configure text refinement parameters with style controls
            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt=(
                    f'[oral_{voice_config["oral_level"]}]'
                    f'[laugh_{voice_config["laugh_level"]}]'
                    f'[break_{voice_config["break_level"]}]'
                )
            )

            # Generate audio
            wavs = self.chat.infer(
                text,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code
            )

            # Convert to tensor
            audio_tensor = torch.from_numpy(wavs[0])
            
            # Generate output filename with timestamp and voice type
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"resume_summary_{gender}_{timestamp}.wav"
            
            # Save audio file
            try:
                output_dir = self.paths_config.get('audio_dir', 'audio_outputs')
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, filename)
                return self.save_audio(audio_tensor, output_path)
                
            except Exception as save_error:
                logger.error(f"Could not save audio file: {save_error}")
                return None
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None

    def generate_summary_audio(
        self,
        summary: str,
        gender: str = 'female',
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Generate audio for a resume summary with voice customization."""
        return self.generate_audio(
            text=summary,
            gender=gender,
            custom_config=custom_config
        )
