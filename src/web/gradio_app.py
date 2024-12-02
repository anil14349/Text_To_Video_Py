import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from model.pipeline import ResumePipeline
from utils.text_processor import TextProcessor
from utils.tts_generator import TTSGenerator
from utils.evaluator import ResumeEvaluator

class GradioInterface:
    def __init__(self):
        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
        self.config = Config(config_path)
        
        # Initialize components
        self.pipeline = ResumePipeline(self.config)
        self.pipeline.setup_model()
        self.tts_generator = TTSGenerator(self.config)
        self.evaluator = ResumeEvaluator()

    def process_resume(self, file_obj, generate_audio: bool, voice_gender: str):
        try:
            # Save uploaded file temporarily
            temp_path = "temp_resume"
            with open(temp_path, "wb") as f:
                f.write(file_obj.read())
            
            # Extract and process text
            resume_text = TextProcessor.extract_text_from_file(temp_path)
            os.remove(temp_path)  # Clean up
            
            if not resume_text:
                return "Error: Could not extract text from file.", None, None
            
            # Generate summary
            summary = self.pipeline.generate_summary(resume_text)
            
            # Generate evaluation report
            report = self.evaluator.generate_report([summary])
            report_text = self.format_report(report)
            
            # Generate audio if requested
            audio_output = None
            if generate_audio:
                audio = self.tts_generator.generate_summary_audio(
                    summary,
                    gender=voice_gender
                )
                if audio is not None:
                    audio_output = (24000, audio.numpy())
            
            return summary, report_text, audio_output
            
        except Exception as e:
            return f"Error: {str(e)}", None, None

    def format_report(self, report: dict) -> str:
        """Format evaluation report for display."""
        lines = ["Evaluation Report:"]
        
        if 'readability_metrics' in report:
            lines.extend([
                "\nReadability Metrics:",
                f"Average Sentence Length: {report['readability_metrics']['avg_sentence_length']:.1f} words",
                f"Average Word Length: {report['readability_metrics']['avg_word_length']:.1f} characters",
                f"Vocabulary Size: {report['readability_metrics']['avg_vocabulary_size']:.0f} words"
            ])
        
        if 'diversity_metrics' in report:
            lines.extend([
                "\nDiversity Metrics:",
                f"Vocabulary Diversity: {report['diversity_metrics']['vocabulary_diversity']:.3f}",
                f"Phrase Repetition: {report['diversity_metrics']['phrase_repetition']:.3f}"
            ])
            
        return "\n".join(lines)

def create_interface():
    interface = GradioInterface()
    
    return gr.Interface(
        fn=interface.process_resume,
        inputs=[
            gr.File(label="Upload Resume (PDF, DOCX, or TXT)"),
            gr.Checkbox(label="Generate Audio"),
            gr.Radio(["female", "male"], label="Voice Gender", value="female")
        ],
        outputs=[
            gr.Textbox(label="Generated Summary", lines=10),
            gr.Textbox(label="Evaluation Report", lines=10),
            gr.Audio(label="Generated Audio")
        ],
        title="Resume Summarization AI",
        description="Upload a resume to generate a professional summary with optional audio narration.",
        examples=[
            ["example_resume.pdf", True, "female"],
            ["example_resume.docx", True, "male"]
        ],
        theme="default"
    )

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_port=7860)
