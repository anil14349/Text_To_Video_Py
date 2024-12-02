import torch
from typing import List, Dict, Any, Optional, Union
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from rouge_score import rouge_scorer
import logging

# Try to import textstat, but don't fail if it's not available
try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

logger = logging.getLogger(__name__)

class ResumeEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_metrics(self, generated_summaries: List[str], reference_summaries: List[str]) -> Dict[str, float]:
        """Calculate various metrics for generated summaries."""
        metrics = {
            'bleu': [],
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
            'length_ratio': []
        }
        
        for gen, ref in zip(generated_summaries, reference_summaries):
            # Calculate BLEU score
            try:
                bleu = sentence_bleu(
                    [word_tokenize(ref.lower())],
                    word_tokenize(gen.lower())
                )
                metrics['bleu'].append(bleu)
            except:
                metrics['bleu'].append(0.0)
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(gen, ref)
            metrics['rouge1_f'].append(rouge_scores['rouge1'].fmeasure)
            metrics['rouge2_f'].append(rouge_scores['rouge2'].fmeasure)
            metrics['rougeL_f'].append(rouge_scores['rougeL'].fmeasure)
            
            # Calculate length ratio
            ref_length = len(word_tokenize(ref))
            gen_length = len(word_tokenize(gen))
            length_ratio = gen_length / ref_length if ref_length > 0 else 0
            metrics['length_ratio'].append(length_ratio)
        
        # Calculate averages
        return {
            'avg_bleu': np.mean(metrics['bleu']),
            'avg_rouge1_f': np.mean(metrics['rouge1_f']),
            'avg_rouge2_f': np.mean(metrics['rouge2_f']),
            'avg_rougeL_f': np.mean(metrics['rougeL_f']),
            'avg_length_ratio': np.mean(metrics['length_ratio'])
        }

    def evaluate_readability(self, generated_summaries: List[Union[str, dict]]) -> Dict[str, float]:
        """
        Evaluate readability metrics for the generated summaries.
        
        Args:
            generated_summaries: List of summaries or summary dictionaries
            
        Returns:
            Dictionary containing readability metrics
        """
        metrics = {
            'avg_word_length': [],
            'avg_sentence_length': [],
            'avg_words_per_sentence': []
        }
        
        if HAS_TEXTSTAT:
            metrics.update({
                'flesch_reading_ease': [],
                'flesch_kincaid_grade': []
            })
        
        for summary in generated_summaries:
            # Handle both string and dictionary summaries
            if isinstance(summary, dict):
                if 'error' in summary:
                    continue  # Skip failed summaries
                summary_text = summary.get('overview', '')
            else:
                summary_text = summary
                
            if not summary_text:
                continue
                
            # Convert to lowercase for analysis
            text = summary_text.lower()
            
            # Tokenize
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            
            if not words or not sentences:
                continue
                
            # Calculate basic metrics
            try:
                metrics['avg_word_length'].append(sum(len(word) for word in words) / len(words))
                metrics['avg_sentence_length'].append(len(text) / len(sentences))
                metrics['avg_words_per_sentence'].append(len(words) / len(sentences))
                
                # Calculate textstat metrics if available
                if HAS_TEXTSTAT:
                    metrics['flesch_reading_ease'].append(textstat.flesch_reading_ease(text))
                    metrics['flesch_kincaid_grade'].append(textstat.flesch_kincaid_grade(text))
            except Exception as e:
                logger.warning(f"Error calculating metrics for summary: {e}")
                continue
        
        # Average the metrics
        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in metrics.items()
        }

    def evaluate_diversity(self, summaries: List[str]) -> Dict[str, float]:
        """Evaluate diversity metrics of generated summaries."""
        all_words = []
        unique_words = set()
        repeated_phrases = []
        
        for summary in summaries:
            words = word_tokenize(summary.lower())
            all_words.extend(words)
            unique_words.update(words)
            
            # Check for repeated phrases (3-grams)
            if len(words) >= 3:
                phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                repeated_phrases.extend(phrases)
        
        metrics = {
            'vocabulary_diversity': len(unique_words) / len(all_words) if all_words else 0,
            'phrase_repetition': 1 - (len(set(repeated_phrases)) / len(repeated_phrases)) if repeated_phrases else 0
        }
        
        return metrics

    def generate_report(self, generated_summaries: List[Union[str, dict]], 
                       reference_summaries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        report = {}
        
        # Quality metrics (if reference summaries are available)
        if reference_summaries:
            report['quality_metrics'] = self.calculate_metrics(
                [s if isinstance(s, str) else s.get('overview', '') for s in generated_summaries], 
                reference_summaries
            )
        
        # Readability metrics
        report['readability_metrics'] = self.evaluate_readability(generated_summaries)
        
        # Diversity metrics
        report['diversity_metrics'] = self.evaluate_diversity([s if isinstance(s, str) else s.get('overview', '') for s in generated_summaries])
        
        return report
