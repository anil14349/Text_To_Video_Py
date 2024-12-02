import re
from typing import Optional
import PyPDF2
from docx import Document
from nltk.tokenize import sent_tokenize

class TextProcessor:
    @staticmethod
    def preprocess_text(text: str, max_length: int = None) -> str:
        """Clean and preprocess the resume text."""
        # Remove extra whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:-]', '', text)
        
        # Split into sentences and limit length if specified
        sentences = sent_tokenize(text)
        if max_length:
            sentences = sentences[:max_length]
        
        processed_text = ' '.join(sentences)
        return processed_text.strip()

    @staticmethod
    def extract_text_from_file(file_path: str) -> Optional[str]:
        """Extract text from PDF or DOCX files."""
        if not file_path:
            print("Error: No file path provided")
            return None

        try:
            text = ""
            if file_path.endswith('.pdf'):
                print("Reading PDF file...")
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = '\n'.join(
                        page.extract_text() 
                        for page in pdf_reader.pages 
                        if page.extract_text().strip()
                    )
            elif file_path.endswith('.docx'):
                print("Reading DOCX file...")
                doc = Document(file_path)
                text = '\n'.join(
                    paragraph.text.strip() 
                    for paragraph in doc.paragraphs 
                    if paragraph.text.strip()
                )
            else:
                print("Reading text file...")
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

            if not text.strip():
                print(f"Warning: No text content found in '{file_path}'")
                return None

            print(f"Successfully extracted {len(text.split())} words from the file")
            return text

        except UnicodeDecodeError:
            print(f"Error: Unable to read the file '{file_path}'. Please ensure it's a valid text file with proper encoding.")
            return None
        except Exception as e:
            print(f"Error reading file '{file_path}': {str(e)}")
            return None

    @staticmethod
    def refine_summary(text: str, max_lines: int) -> str:
        """Refine the generated summary for better readability and professionalism."""
        try:
            # Split into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Remove redundant or too short sentences
            filtered_sentences = []
            seen_content = set()
            
            for sentence in sentences:
                # Convert to lowercase for comparison
                lower_sent = sentence.lower()
                
                # Skip if too short
                if len(sentence.split()) < 3:
                    continue
                    
                # Skip if too similar to previous sentences
                similar = False
                for seen in seen_content:
                    if (lower_sent in seen) or (seen in lower_sent):
                        similar = True
                        break
                
                if not similar:
                    seen_content.add(lower_sent)
                    filtered_sentences.append(sentence)
            
            # Limit to specified number of sentences
            filtered_sentences = filtered_sentences[:max_lines]
            
            # Add professional phrases if missing
            has_experience = any('experience' in s.lower() for s in filtered_sentences)
            has_skills = any('skill' in s.lower() for s in filtered_sentences)
            
            if not has_experience and len(filtered_sentences) < max_lines:
                filtered_sentences.append("Experienced professional with a proven track record.")
            if not has_skills and len(filtered_sentences) < max_lines:
                filtered_sentences.append("Skilled in various technical and professional competencies.")
            
            # Join sentences
            refined_text = '. '.join(filtered_sentences)
            
            # Clean up common issues
            refined_text = refined_text.replace('..', '.')
            refined_text = refined_text.replace('. .', '.')
            refined_text = refined_text.strip()
            
            # Ensure proper capitalization
            refined_text = '. '.join(s.strip().capitalize() for s in refined_text.split('.') if s.strip())
            
            return refined_text + '.'
            
        except Exception as e:
            print(f"Error refining summary: {str(e)}")
            return text
