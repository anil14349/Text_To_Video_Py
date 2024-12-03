import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import re
from typing import Dict
import os
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configure MPS memory usage
if hasattr(torch.mps, 'set_per_process_memory_fraction'):
    torch.mps.set_per_process_memory_fraction(0.7)  # Use only 70% of available memory

logger = logging.getLogger(__name__)

class ResumePipeline:
    DEFAULT_MODEL = "gpt2"

    def __init__(self, config):
        self.config = config
        self.model_config = config.get_model_config()
        
        # Set device with memory management for MPS
        if torch.backends.mps.is_available():
            self.device = "mps"
            # Set environment variable for MPS memory
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        logger.info("Using device: %s", self.device)
        
        # Get HuggingFace token from environment
        self.hf_token = os.getenv('HUGGING_FACE_TOKEN')
        if not self.hf_token:
            logger.warning("No HuggingFace token found in environment variables")

        self.model = None
        self.tokenizer = None
        
        # Set up cache directory
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'models_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def initialize_model(self):
        """Load the model and tokenizer."""
        if self.model and self.tokenizer:
            return  # Already initialized

        try:
            model_name = self.model_config.get("name", self.DEFAULT_MODEL)
            logger.info(f"Loading model {model_name} from cache or downloading")
            
            # Try to load from local cache first
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False  # Allow download if not in cache
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False  # Allow download if not in cache
            )

            # Set padding token and model configuration
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.model.config.batch_first = True

            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error("Error initializing model: %s", str(e))
            raise

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=60, min=60, max=300))
    def setup_model(self):
        """Setup model by loading from HuggingFace Hub or initializing default."""
        try:
            # Try to load from HuggingFace Hub
            hf_repo = self.model_config.get('hf_repo')
            if hf_repo and self.hf_token:
                logger.info(f"Loading model from HuggingFace Hub: {hf_repo}")
                self.model = GPT2LMHeadModel.from_pretrained(
                    hf_repo,
                    weights_only=True,
                    token=self.hf_token
                )
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    hf_repo,
                    token=self.hf_token
                )
                
                # Set padding token and model configuration
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
                self.model.config.batch_first = True
                
                self.model.to(self.device)
                logger.info("Model loaded successfully from HuggingFace Hub")
                return
        except Exception as e:
            logger.warning(f"Could not load model from HuggingFace Hub: {str(e)}")
        
        # Fall back to default initialization
        logger.info("Falling back to default model initialization")
        self.initialize_model()

    def _create_prompt(self, text: str) -> str:
        """Create a specific and constrained prompt."""
        prompt_template = (
            "Using only the information provided in the resume below, generate a professional summary. "
            "Do not add any content that is not explicitly mentioned in the resume. Please provide a concise summary "
            "that includes the following:\n"
            "1. Candidate's current role, years of experience, and notable skills.\n"
            "2. Relevant certifications, tools used, and key accomplishments.\n"
            "3. Highlight the candidate's experience with specific projects.\n\n"
            "Resume:\n{}\n\nProfessional Summary:"
        )
        return prompt_template.format(text[:1500])

    def _encode_input(self, prompt: str, max_length=1024):
        """Encode input text for the model."""
        encoding = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    def _generate_text(self, input_ids, attention_mask, max_new_tokens=1000):
        """Generate text using constrained decoding parameters."""
        with torch.inference_mode():  # Use inference mode for better performance
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=5,  # Focus on precise generation
                no_repeat_ngram_size=3,  # Penalize repeated n-grams
                repetition_penalty=2.0,  # Penalize repeated phrases
                length_penalty=0.8,  # Avoid overly verbose outputs
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Enable caching for better performance
            )
            return outputs

    def validate_output(self, summary: str, resume: str) -> str:
        """Ensure the summary content is valid and matches the resume."""
        valid_sentences = []
        resume_lower = resume.lower()

        for sentence in summary.split('.'):
            sentence_clean = sentence.strip().lower()
            # Include the sentence only if it matches content in the resume
            if any(word in resume_lower for word in sentence_clean.split()):
                valid_sentences.append(sentence.strip())

        return '. '.join(valid_sentences)

    def generate_summary(self, text: str) -> str:
        """Generate and validate a resume summary."""
        self.setup_model()

        try:
            name = self.extract_name(text)
            prompt = self._create_prompt(text)
            encoded = self._encode_input(prompt)

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            outputs = self._generate_text(input_ids, attention_mask)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            summary = generated_text.split("Professional Summary:")[-1].strip()
            refined_summary = self.refine_summary(summary, self.model_config.get("max_lines", 6))
            validated_summary = self.validate_output(refined_summary, text)

            return f"Hi, this is {name}. {validated_summary}"
        except Exception as e:
            logger.error("Error generating summary: %s", str(e))
            raise

    def refine_summary(self, text: str, max_lines: int) -> str:
        """Refine the generated summary."""
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]

            # Remove duplicates and overly short sentences
            seen = set()
            filtered = [
                s for s in sentences if len(s.split()) >= 5 and s.lower() not in seen and not seen.add(s.lower())
            ]

            # Limit to max lines
            filtered = filtered[:max_lines]
            summary = '. '.join(filtered).strip()
            return summary + '.' if not summary.endswith('.') else summary
        except Exception as e:
            logger.error("Error refining summary: %s", str(e))
            return text

    def extract_name(self, resume: str) -> str:
        """Extract the candidate's name from the resume."""
        match = re.search(r"^\s*(\w+ \w+)", resume, re.MULTILINE)
        return match.group(1) if match else "Candidate"

    def _extract_key_points(self, resume: str) -> Dict[str, str]:
        """Dynamically extract key details from the resume."""
        key_points = {
            "name": self.extract_name(resume),
            "skills": "",
            "certifications": "",
            "companies": "",
            "projects": ""
        }

        try:
            # Extract skills from sections like "Skills" or "Technical Skills"
            skills_match = re.search(r"(Skills|Technical Skills|Tools Worked):\s*(.*?)\n", resume, re.DOTALL | re.IGNORECASE)
            if skills_match:
                key_points["skills"] = skills_match.group(2).strip().replace('\n', ', ')

            # Extract certifications from relevant sections
            certifications_match = re.search(r"(Certificates Obtained|Certifications):\s*(.*?)\n(?:\n|$)", resume, re.DOTALL | re.IGNORECASE)
            if certifications_match:
                key_points["certifications"] = certifications_match.group(2).strip().replace('\n', ', ')

            # Extract company names and roles from "Professional Experience" or similar sections
            experience_match = re.search(r"Professional Experience:(.*?)\n\n", resume, re.DOTALL | re.IGNORECASE)
            if experience_match:
                companies = re.findall(r"at ([^\n,]+)", experience_match.group(1))
                roles = re.findall(r"Working as a ([^\n,]+)", experience_match.group(1))
                key_points["companies"] = ", ".join(set(companies)) if companies else "Not available"
                key_points["roles"] = ", ".join(set(roles)) if roles else "Not available"

            # Extract project details from sections like "Project"
            projects_match = re.findall(r"PROJECT: ([^\n]+)", resume, re.IGNORECASE)
            key_points["projects"] = ", ".join(projects_match) if projects_match else "Not available"

        except Exception as e:
            logger.error("Error extracting key points from resume: %s", str(e))

        return key_points

    def train(self, train_texts: list, train_summaries: list, val_texts: list, val_summaries: list):
        """Train the model on resume data and save to HuggingFace."""
        from transformers import Trainer, TrainingArguments
        from datasets import Dataset
        import numpy as np
        
        # Initialize model first (downloads if needed)
        self.initialize_model()
        
        # Prepare datasets with smaller chunks to manage memory
        max_length = 512  # Reduced from default 1024
        
        def prepare_data(texts, summaries):
            encodings = [self._create_prompt(text) for text in texts]
            return Dataset.from_dict({
                'input_ids': [self._encode_input(text, max_length=max_length)['input_ids'].squeeze().tolist() 
                             for text in encodings],
                'attention_mask': [self._encode_input(text, max_length=max_length)['attention_mask'].squeeze().tolist() 
                                 for text in encodings],
                'labels': [self._encode_input(summary, max_length=max_length)['input_ids'].squeeze().tolist() 
                          for summary in summaries]
            })
        
        train_dataset = prepare_data(train_texts, train_summaries)
        val_dataset = prepare_data(val_texts, val_summaries)
        
        # Get training config
        training_config = self.config.config.get('training', {})
        output_dir = os.path.join(self.cache_dir, 'trained_model')
        
        # Adjust batch size based on device
        if self.device == "mps":
            batch_size = 1  # Smaller batch size for MPS
            grad_accum = 16  # Increase gradient accumulation to compensate
        else:
            batch_size = training_config.get('batch_size', 4)
            grad_accum = training_config.get('gradient_accumulation_steps', 4)
        
        # Setup training arguments with memory optimization
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get('num_epochs', 3),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=float(training_config.get('learning_rate', 2e-5)),
            weight_decay=float(training_config.get('weight_decay', 0.01)),
            logging_steps=training_config.get('logging_steps', 10),
            eval_steps=training_config.get('eval_steps', 50),
            save_steps=training_config.get('save_steps', 100),
            save_total_limit=training_config.get('save_total_limit', 3),
            fp16=False,  # Disable fp16 for MPS
            eval_strategy="steps",  # Updated parameter name
            save_strategy="steps",
            load_best_model_at_end=True,
            # Memory optimization
            gradient_checkpointing=True,
            optim="adamw_torch_fused" if self.device == "cuda" else "adamw_torch",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the model
        logger.info("Starting model training...")
        trainer.train()
        
        # Save the model locally first
        logger.info("Saving model locally...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Only after successful training, push to HuggingFace Hub
        if self.hf_token:
            try:
                hf_repo = self.model_config.get('hf_repo')
                if hf_repo:
                    logger.info(f"Pushing trained model to HuggingFace Hub: {hf_repo}")
                    trainer.push_to_hub(
                        repo_id=hf_repo,
                        token=self.hf_token,
                    )
                    logger.info("Model successfully pushed to HuggingFace Hub")
            except Exception as e:
                logger.error(f"Error pushing to HuggingFace Hub: {str(e)}")
                logger.info("Model is still saved locally at: %s", output_dir)
