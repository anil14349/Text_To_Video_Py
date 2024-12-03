import torch
import torchaudio
import logging
import os
import ChatTTS
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
        
        # Initialize ChatTTS
        try:
            self.chat = ChatTTS.Chat()
            self.chat.load(compile=False)
            logger.info("ChatTTS model loaded successfully")
        except Exception as e:
            logger.error("Failed to initialize ChatTTS: %s", str(e))
            raise

    def generate_summary_audio(
        self,
        text: str,
        gender: str = "female",
        temperature: float = 0.3,
        top_P: float = 0.7,
        top_K: int = 20,
        laugh_level: int = 0
    ) -> Optional[torch.Tensor]:
        """Generate audio from text using ChatTTS."""
        try:
            if not text.strip():
                logger.warning("Empty text provided for TTS")
                return None

            # Set oral and break levels based on gender
            oral_level = 0 if gender.lower() == 'female' else 6
            break_level = 3 if gender.lower() == 'female' else 4

            # Generate a speaker embedding
            rand_spk = self.chat.sample_random_speaker()

            # Configure inference parameters
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=rand_spk,
                temperature=temperature,
                top_P=top_P,
                top_K=top_K
            )

            # Configure text refinement parameters
            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt=f'[oral_{oral_level}][laugh_{laugh_level}][break_{break_level}]'
            )

            # Generate audio
            wavs = self.chat.infer(
                text,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code
            )

            # Convert to torch tensor
            audio = torch.from_numpy(wavs[0])
            
            # Save the audio file
            timestamp = Path(text).stem[:30] if isinstance(text, Path) else text[:30]
            output_path = self.output_dir / f"{timestamp}_{gender}.wav"
            
            try:
                torchaudio.save(str(output_path), audio.unsqueeze(0), 24000)
            except:
                torchaudio.save(str(output_path), audio, 24000)
            
            logger.info(f"Audio saved to {output_path}")
            return audio

        except Exception as e:
            logger.error("Error generating audio: %s", str(e))
            return None
