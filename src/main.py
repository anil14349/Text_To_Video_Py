import os
import sys
import nltk
from typing import Dict, Any
from pathlib import Path

# Add project root to PYTHONPATH first
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Group all imports at the top
from src.model.core import ResumePipeline
from src.utils import (
    Config,
    TextProcessor,
    TTSGenerator,
    ModelManager,
    ResumeEvaluator
)
from src.model.utils import load_training_data

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')

def print_evaluation_report(report: Dict[str, Any]):
    """Print the evaluation report in a formatted way."""
    print("\nEvaluation Report:")
    print("-" * 50)
    print()
    
    if 'readability_metrics' in report:
        print("Readability Metrics:")
        metrics = report['readability_metrics']
        if 'flesch_reading_ease' in metrics:
            print(f"Flesch Reading Ease: {metrics['flesch_reading_ease']:.1f}")
        if 'flesch_kincaid_grade' in metrics:
            print(f"Flesch-Kincaid Grade Level: {metrics['flesch_kincaid_grade']:.1f}")
        print(f"Average Word Length: {metrics['avg_word_length']:.1f} characters")
        print(f"Average Sentence Length: {metrics['avg_sentence_length']:.1f} characters")
        if 'avg_words_per_sentence' in metrics:
            print(f"Average Words per Sentence: {metrics['avg_words_per_sentence']:.1f} words")
    
    if 'diversity_metrics' in report:
        print("\nDiversity Metrics:")
        metrics = report['diversity_metrics']
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.2f}")

def main():
    """Main function to demonstrate resume summarization with training option."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    config = Config(config_path)
    
    # Initialize components
    pipeline = ResumePipeline(config)
    tts_generator = TTSGenerator(config)
    model_manager = ModelManager(config)
    evaluator = ResumeEvaluator()

    while True:
        print("\nResume Summarization Pipeline")
        print("1. Generate summary from resume")
        print("2. Train/Fine-tune model")
        print("3. Generate audio from summary")
        print("4. Manage model versions")
        print("5. Evaluate model performance")
        print("6. Batch process resumes")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            # Generate summary
            file_path = input("Enter the path to your resume file (PDF, DOCX, or TXT): ").strip()
            
            if not os.path.exists(file_path):
                print("Error: File not found!")
                continue

            # Extract and preprocess text
            resume_text = TextProcessor.extract_text_from_file(file_path)
            if not resume_text:
                print("Error: Could not extract text from file!")
                continue

            # Preprocess text
            resume_text = TextProcessor.preprocess_text(
                resume_text, 
                config.get_model_config().get('max_length', 1024)
            )

            print("\nGenerating summary...")
            summary = pipeline.generate_summary(resume_text)
            
            print("\nGenerated Summary:")
            print("-" * 50)
            print(summary)
            print("-" * 50)
            
            # Generate evaluation report
            report = evaluator.generate_report([summary])
            print_evaluation_report(report)
            
        elif choice == "2":
            # Train model
            data_path = input("Enter path to training data CSV (must contain 'text' and 'summary' columns): ").strip()
            if not data_path:
                data_path = '/Users/anilkumar/Downloads/UpdatedResumeDataSet 3.csv'
            if not os.path.exists(data_path):
                print("Error: File not found!")
                continue
                
            try:

                train_texts, train_summaries, val_texts, val_summaries = load_training_data(
                    data_path,
                    val_split=config.config.get('training', {}).get('validation_split', 0.2)
                )
                
                # Train the model
                pipeline.train(
                    train_texts=train_texts,
                    train_summaries=train_summaries,
                    val_texts=val_texts,
                    val_summaries=val_summaries
                )
                
                # Evaluate on validation set
                print("\nEvaluating model performance...")
                val_summaries_generated = [
                    pipeline.generate_summary(text) for text in val_texts[:10]  # Evaluate on first 10 examples
                ]
                
                report = evaluator.generate_report(
                    val_summaries_generated,
                    val_summaries[:10]
                )
                
                # Save model version with metrics
                version_name = f"model_v{len(model_manager.list_model_versions()) + 1}"
                model_manager.save_model_version(
                    pipeline.model,
                    pipeline.tokenizer,
                    report['quality_metrics'],
                    version_name
                )
                
                print("\nTraining completed successfully!")
                print_evaluation_report(report)
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                
        elif choice == "3":
            # Generate audio from summary
            print("\nAudio Generation Options:")
            print("1. Female voice")
            print("2. Male voice")
            voice_choice = input("Select voice type (1-2): ").strip()
            
            gender = 'female' if voice_choice == '1' else 'male'
            
            text = input("Enter the summary text (or press Enter to use last generated summary): ").strip()
            if not text and 'summary' in locals():
                text = summary
            elif not text:
                print("No summary available. Please enter text to convert to speech.")
                continue
            
            print(f"\nGenerating audio with {gender} voice...")
            audio = tts_generator.generate_summary_audio(text, gender=gender)
            
            if audio is not None:
                print("Audio generation completed successfully!")
            else:
                print("Failed to generate audio.")
                
        elif choice == "4":
            # Manage model versions
            print("\nModel Management Options:")
            print("1. List model versions")
            print("2. Load specific version")
            print("3. Delete version")
            
            mgmt_choice = input("Select option (1-3): ").strip()
            
            if mgmt_choice == "1":
                versions = model_manager.list_model_versions()
                print("\nAvailable Model Versions:")
                for v in versions:
                    print(f"\nVersion: {v['version']}")
                    print(f"Timestamp: {v['timestamp']}")
                    print("Metrics:")
                    for metric, value in v['metrics'].items():
                        print(f"  {metric}: {value:.3f}")
                        
            elif mgmt_choice == "2":
                version_name = input("Enter version name to load: ").strip()
                try:
                    model, tokenizer, metadata = model_manager.load_model_version(version_name)
                    pipeline.model = model
                    pipeline.tokenizer = tokenizer
                    print(f"Successfully loaded model version: {version_name}")
                except Exception as e:
                    print(f"Error loading model version: {str(e)}")
                    
            elif mgmt_choice == "3":
                version_name = input("Enter version name to delete: ").strip()
                try:
                    model_manager.delete_model_version(version_name)
                except Exception as e:
                    print(f"Error deleting model version: {str(e)}")
                    
        elif choice == "5":
            # Evaluate model performance
            print("\nEvaluation Options:")
            print("1. Evaluate on test set")
            print("2. Evaluate single summary")
            
            eval_choice = input("Select option (1-2): ").strip()
            
            if eval_choice == "1":
                test_data_path = input("Enter path to test data CSV: ").strip()
                if not os.path.exists(test_data_path):
                    print("Error: File not found!")
                    continue
                    
                try:
                    
                    test_texts, test_summaries, _, _ = load_training_data(
                        test_data_path,
                        val_split=config.config.get('training', {}).get('validation_split', 0.2)
                    )
                    
                    print("\nGenerating summaries for evaluation...")
                    generated_summaries = [
                        pipeline.generate_summary(text) for text in test_texts[:10]
                    ]
                    
                    report = evaluator.generate_report(
                        generated_summaries,
                        test_summaries[:10]
                    )
                    print_evaluation_report(report)
                    
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    
            elif eval_choice == "2":
                if 'summary' not in locals():
                    print("No summary available. Generate a summary first.")
                    continue
                    
                report = evaluator.generate_report([summary])
                print_evaluation_report(report)
                
        elif choice == "6":
            # Batch process resumes
            input_dir = input("Enter directory containing resume files: ").strip()
            if not os.path.isdir(input_dir):
                print("Error: Directory not found!")
                continue
                
            output_dir = input("Enter output directory for summaries and audio: ").strip()
            os.makedirs(output_dir, exist_ok=True)
            
            generate_audio = input("Generate audio for summaries? (y/n): ").strip().lower() == 'y'
            if generate_audio:
                gender = input("Select voice gender (male/female): ").strip().lower()
            
            print("\nProcessing resumes...")
            for filename in os.listdir(input_dir):
                if filename.endswith(('.pdf', '.docx', '.txt')):
                    try:
                        # Process resume
                        file_path = os.path.join(input_dir, filename)
                        resume_text = TextProcessor.extract_text_from_file(file_path)
                        
                        if resume_text:
                            # Generate summary
                            summary = pipeline.generate_summary(resume_text)
                            
                            # Save summary
                            base_name = os.path.splitext(filename)[0]
                            summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
                            with open(summary_path, 'w') as f:
                                f.write(summary)
                            
                            # Generate and save audio if requested
                            if generate_audio:
                                tts_generator.generate_summary_audio(
                                    summary,
                                    gender=gender,
                                    output_filename=f"{base_name}_summary.wav"
                                )
                            
                            print(f"Processed: {filename}")
                            
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
            
            print("\nBatch processing completed!")
                
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
