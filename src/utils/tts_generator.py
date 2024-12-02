import torch
import torchaudio
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class TTSGenerator:
    def __init__(self, config):
        self.config = config
        self.tts_config = config.get_tts_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(self.tts_config.get('output_dir', 'audio_outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TTS model
        try:
            # Get the TTS bundle
            self.bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
            
            # Get processor and models
            self.processor = self.bundle.get_text_processor()
            self.tacotron2 = self.bundle.get_tacotron2().to(self.device)
            self.vocoder = self.bundle.get_vocoder().to(self.device)
            
            logger.info("TTS models loaded successfully")
        except Exception as e:
            logger.error("Failed to initialize TTS models: %s", str(e))
            raise

    def generate_summary_audio(self, text: str, gender: str = "female") -> Optional[torch.Tensor]:
        """Generate audio from text using TTS."""
        try:
            if not text.strip():
                logger.warning("Empty text provided for TTS")
                return None

            with torch.no_grad():
                # Process text to phonemes
                tokens = self.processor(text)
                
                # Generate spectrogram
                spec = self.tacotron2.infer(tokens)
                
                # Convert to audio
                audio = self.vocoder(spec)
                
                # Adjust pitch for gender
                if gender == "male":
                    # Lower pitch for male voice
                    audio = torchaudio.functional.pitch_shift(
                        audio,
                        sample_rate=22050,
                        n_steps=-2.0
                    )

            return audio

        except Exception as e:
            logger.error("Error generating audio: %s", str(e))
            return None
