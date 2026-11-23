"""
Enhanced LLM Model with User Prompts and Model Persistence
Features:
- Save and load trained model weights
- Interactive prompt input from user
- Persistent storage of trained models
- Training progress tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import json
import re
import random
import base64
import io
import time
from pathlib import Path
from datetime import datetime
import requests
import urllib.parse
from PIL import Image
try:
    import pytesseract
except ImportError:
    pytesseract = None


# ============================================================================
# CONFIGURATION AND PREFERENCE MANAGER
# ============================================================================
class ConfigManager:
    """Manages user preferences and configurations"""
    
    def __init__(self, model_name, model_dir="saved_models"):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.config_file = self.model_dir / f"{model_name}_config.json"
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return self.default_config()
        return self.default_config()
    
    def default_config(self):
        """Create default configuration"""
        return {
            'model_name': self.model_name,
            'last_question_folder': None,
            'use_question_folder': False,
            'question_folder_path': None,
            'total_questions_loaded': 0,
            'last_updated': str(datetime.now())
        }
    
    def save_config(self):
        """Save configuration to file"""
        self.config['last_updated'] = str(datetime.now())
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def set_question_folder(self, folder_path):
        """Set and remember question folder"""
        if Path(folder_path).exists() and Path(folder_path).is_dir():
            self.config['question_folder_path'] = str(folder_path)
            self.config['use_question_folder'] = True
            self.save_config()
            return True
        return False
    
    def get_question_folder(self):
        """Get remembered question folder"""
        folder = self.config.get('question_folder_path')
        if folder and Path(folder).exists():
            return folder
        return None
    
    def clear_question_folder(self):
        """Clear question folder setting"""
        self.config['use_question_folder'] = False
        self.config['question_folder_path'] = None
        self.save_config()
    
    def get_config_summary(self):
        """Get configuration summary"""
        return {
            'using_question_folder': self.config['use_question_folder'],
            'folder_path': self.config['question_folder_path'],
            'questions_loaded': self.config['total_questions_loaded']
        }


class OllamaDataGenerator:
    """Generates training data using Ollama models"""
    
    def __init__(self, ollama_validator):
        self.validator = ollama_validator
    
    @staticmethod
    def extract_questions_from_file(file_path):
        """Extract questions marked with @ delimiters"""
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all text between @ markers
            parts = content.split('@')
            
            # Parts alternate: text, question, text, question, ...
            # So odd indices (1, 3, 5...) are the questions
            for i in range(1, len(parts), 2):
                question = parts[i].strip()
                if question:  # Only add non-empty questions
                    questions.append(question)
            
            return questions
        
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return []
    
    def generate_answers(self, questions, user_context="", log_path=None):
        """Generate answers for questions using Ollama.

        user_context : optional free-form instruction injected into every prompt.
        log_path     : if given, a log file is created immediately and each
                       Q&A pair is appended as it arrives so no results are
                       lost even when some questions fail.

        Uses OllamaValidator.ask_question() which streams the response and
        retries on timeout — much more robust than the old validate_text() call.
        """
        qa_pairs = []

        if not self.validator.available or not self.validator.model:
            print("❌ Ollama not available or no model selected")
            return []

        print(f"\n🔄 Generating answers using Ollama model: {self.validator.model}")
        print(f"📝 Total questions to process: {len(questions)}")
        if user_context.strip():
            print(f"📋 Context: {user_context[:80]}{'…' if len(user_context) > 80 else ''}")

        # ── Open log file IMMEDIATELY so it exists even if all calls fail ──
        log_fh = None
        if log_path:
            try:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                log_fh = open(log_path, 'w', encoding='utf-8')
                log_fh.write(
                    f"# Ollama Answer Log — {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                    f"# Model   : {self.validator.model}\n"
                    f"# Context : {user_context.strip() or '(none)'}\n"
                    f"# Questions: {len(questions)}\n"
                    f"{'='*60}\n\n"
                )
                log_fh.flush()
                print(f"📄 Log file : {log_path}")
            except Exception as log_err:
                print(f"⚠️  Could not create log file: {log_err}")
                log_fh = None
        print()

        for idx, question in enumerate(questions, 1):
            q_preview = question[:60] + ("…" if len(question) > 60 else "")
            print(f"[{idx}/{len(questions)}] Q: {q_preview}", end=" ", flush=True)

            try:
                answer = self.validator.ask_question(
                    question, context=user_context
                )
            except Exception as exc:
                answer = None
                print(f"❌ (error: {str(exc)[:40]})")

            if answer:
                qa_pairs.append({'question': question, 'answer': answer, 'index': idx})
                print("✓")
            else:
                print("❌ (no response)")

            # Write this result to log immediately — regardless of success
            if log_fh:
                try:
                    status = "✓" if answer else "✗ no response"
                    log_fh.write(f"[{idx}/{len(questions)}] {status}\n")
                    log_fh.write(f"Q: {question}\n")
                    log_fh.write(f"A: {answer if answer else '(no response)'}\n\n")
                    log_fh.flush()
                except Exception:
                    pass   # never let log I/O kill the generation loop

        # ── Close log with summary footer ─────────────────────────────────
        if log_fh:
            try:
                log_fh.write(
                    f"\n{'='*60}\n"
                    f"# Summary: {len(qa_pairs)}/{len(questions)} answers generated\n"
                )
                log_fh.close()
                print(f"\n✓ Log written: {log_path}")
            except Exception:
                pass

        return qa_pairs
    
    def format_for_training(self, qa_pairs):
        """Format Q&A pairs as training text"""
        training_text = ""
        
        for pair in qa_pairs:
            training_text += f"Q: {pair['question']}\nA: {pair['answer']}\n\n"
        
        return training_text
    
    @staticmethod
    def save_generated_data(qa_pairs, output_file):
        """Save generated Q&A pairs to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Generated Training Data from Ollama\n\n")
                
                for idx, pair in enumerate(qa_pairs, 1):
                    f.write(f"## Question {idx}\n")
                    f.write(f"Q: {pair['question']}\n")
                    f.write(f"A: {pair['answer']}\n\n")
            
            return True
        
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return False


class QuestionFormatter:
    """Helps format questions with @ delimiters"""
    
    @staticmethod
    def create_question_file(questions, output_file):
        """Create a file with questions marked with @ delimiters"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for question in questions:
                    f.write(f"@{question}@\n\n")
            
            return True
        
        except Exception as e:
            print(f"❌ Error creating file: {e}")
            return False
    
    @staticmethod
    def validate_question_file(file_path):
        """Check if file has valid @ delimiters"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count @ markers
            marker_count = content.count('@')
            
            if marker_count == 0:
                return False, "No @ markers found. Format: @question@"
            
            if marker_count % 2 != 0:
                return False, f"Odd number of @ markers ({marker_count}). Each question needs 2."
            
            # Count questions
            parts = content.split('@')
            questions = [p.strip() for p in parts[1::2] if p.strip()]
            
            return True, f"Found {len(questions)} valid questions"
        
        except Exception as e:
            return False, f"Error: {e}"



    
    @staticmethod
    def load_questions_from_folder(folder_path):
        """Load all text files from folder as questions"""
        questions = []
        folder = Path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            return []
        
        # Read all text files
        text_files = list(folder.glob('*.txt')) + list(folder.glob('*.md'))
        
        for file_path in sorted(text_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    # Split by lines and filter empty ones
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    # Add non-empty lines as potential questions
                    questions.extend(lines)
            except Exception as e:
                print(f"⚠️  Error reading {file_path.name}: {e}")
        
        return questions
    
    @staticmethod
    def generate_question_from_text(text):
        """Convert text passage to a question"""
        # Take first sentence or clause as base
        sentences = text.split('.')
        if sentences and len(sentences[0]) > 0:
            base = sentences[0].strip()
            # Make it into a question style
            if not base.endswith('?'):
                return f"Tell me about: {base.lower()}?"
            return base
        return text
    
    @staticmethod
    def validate_folder(folder_path):
        """Check if folder has valid question files"""
        folder = Path(folder_path)
        
        if not folder.exists():
            return False, "Folder does not exist"
        
        if not folder.is_dir():
            return False, "Path is not a directory"
        
        text_files = list(folder.glob('*.txt')) + list(folder.glob('*.md'))
        
        if not text_files:
            return False, "No .txt or .md files found in folder"
        
        return True, f"Found {len(text_files)} text file(s)"

# ============================================================================
# OLLAMA INTEGRATION
# ============================================================================
class OllamaValidator:
    """Validates generated text using local Ollama server"""
    
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = None
        self.available = False
        self.available_models = []
        self.check_connection()
    
    def check_connection(self):
        """Check if Ollama server is running and fetch available models."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
        except requests.exceptions.ConnectionError:
            print("⚠ Ollama server not reachable at "
                  f"{self.ollama_url} — is Ollama running?")
            self.available = False
            return
        except requests.exceptions.Timeout:
            print(f"⚠ Ollama server timed out ({self.ollama_url})")
            self.available = False
            return
        except Exception as e:
            print(f"⚠ Ollama connection error: {e}")
            self.available = False
            return

        if response.status_code != 200:
            print(f"⚠ Ollama returned HTTP {response.status_code}")
            self.available = False
            return

        # Parse JSON — handle both old and new Ollama API shapes
        try:
            data = response.json()
        except Exception as e:
            print(f"⚠ Ollama returned invalid JSON: {e}")
            self.available = False
            return

        # Ollama ≥ 0.1.14 uses {"models": [...]}
        # Older builds used {"models": [...]} too but the list items differ
        raw_models = data.get('models', [])

        models = []
        for m in raw_models:
            # Each entry is {"name": "...", "model": "...", ...}
            name = m.get('name') or m.get('model') or str(m)
            if name:
                models.append(name)

        if models:
            self.available_models = models
            self.available = True
            print(f"✓ Ollama connected — {len(models)} model(s) found:")
            for m in models:
                print(f"    • {m}")
        else:
            print("⚠ Ollama is running but has no models installed.")
            print("  Run:  ollama pull tinyllama")
            self.available = False
    
    def list_models(self):
        """Display available Ollama models and let the user select one."""
        if not self.available or not self.available_models:
            print("❌ No Ollama models available.")
            print("   Make sure Ollama is running and you have at least one model.")
            print("   Example: ollama pull tinyllama")
            return False

        print("\n" + "="*60)
        print("AVAILABLE OLLAMA MODELS")
        print("="*60)
        for i, model in enumerate(self.available_models, 1):
            print(f"  {i}. {model}")
        print()

        while True:
            choice = input(
                "Select by number, type model name, or 'skip': "
            ).strip()

            if choice.lower() == 'skip':
                return False

            # Numeric selection
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_models):
                    self.model = self.available_models[idx]
                    print(f"✓ Selected Ollama model: {self.model}")
                    return True
                else:
                    print(f"❌ Enter a number between 1 and {len(self.available_models)}")
                    continue

            # Typed model name — accept if it's in the list or if Ollama
            # confirms it exists (handles edge cases where the list lags)
            if choice in self.available_models:
                self.model = choice
                print(f"✓ Selected Ollama model: {self.model}")
                return True

            # Try a quick verify call to Ollama
            try:
                r = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": choice, "prompt": "hi", "stream": False},
                    timeout=8,
                )
                if r.status_code == 200:
                    self.model = choice
                    self.available_models.append(choice)
                    print(f"✓ Model verified and selected: {self.model}")
                    return True
                else:
                    print(f"❌ '{choice}' not found in Ollama. "
                          f"Available: {', '.join(self.available_models)}")
            except Exception as e:
                print(f"❌ Could not verify model '{choice}': {e}")
    
    def validate_text(self, generated_text, original_prompt=""):
        """Use Ollama to validate and score generated text"""
        if not self.available or not self.model:
            return None

        try:
            validation_prompt = (
                f"Evaluate this generated text for coherence, grammar, and relevance.\n"
                f"Original prompt: {original_prompt}\n"
                f"Generated text: {generated_text}\n\n"
                f"Provide a brief assessment (1-2 sentences) and a score from 1-10.\n"
                f"Format: [SCORE: X/10] Assessment here."
            )
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": validation_prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return None
        except Exception as e:
            print(f"⚠ Validation error: {e}")
            return None

    def ask_question(self, question: str, context: str = "",
                     timeout: int = 120, max_retries: int = 3):
        """Send a question directly to Ollama and return the answer text.

        Differences from validate_text():
        - Sends the question as-is (no evaluation wrapper prompt)
        - Uses streaming — chunks arrive continuously, avoiding hard timeouts
          on slow / small local models like TinyLlama
        - Retries up to max_retries times on timeout / connection errors
        - Longer default timeout (120 s) suitable for local CPU inference

        Returns the answer string, or None on failure.
        """
        if not self.available or not self.model:
            return None

        prompt = (
            f"[INSTRUCTIONS]\n{context.strip()}\n\nQuestion: {question}"
            if context.strip() else question
        )

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": True},
                    stream=True,
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    return None   # non-retriable HTTP error

                # Collect streamed JSON tokens into a single string
                full_text = ""
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                        full_text += chunk.get("response", "")
                        if chunk.get("done"):
                            break
                    except Exception:
                        continue

                return full_text.strip() if full_text.strip() else None

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as err:
                if attempt < max_retries:
                    print(f"\n  ⏳ Attempt {attempt}/{max_retries} timed out — retrying…",
                          end=" ", flush=True)
                    import time as _time
                    _time.sleep(3)
                else:
                    print(f"\n  ❌ All {max_retries} attempts failed: {err}")
                    return None

            except Exception as err:
                print(f"\n  ❌ Unexpected error: {err}")
                return None

        return None


# ============================================================================
# FEEDBACK AND SELF-IMPROVEMENT SYSTEM
# ============================================================================
class FeedbackCollector:
    """Collects and manages feedback for model improvement"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.feedback_data = []
        self.scores = []
        self.low_score_threshold = 6.0  # Score below 6/10 is considered low
    
    def add_feedback(self, question, answer, score, evaluator_comment):
        """Record an answer with its evaluation score"""
        self.feedback_data.append({
            'question': question,
            'answer': answer,
            'score': score,
            'comment': evaluator_comment
        })
        self.scores.append(score)
    
    def get_low_scoring_answers(self):
        """Get answers that scored below threshold"""
        low_scores = [
            item for item in self.feedback_data 
            if item['score'] < self.low_score_threshold
        ]
        return low_scores
    
    def get_statistics(self):
        """Get feedback statistics"""
        if not self.scores:
            return {
                'total_answers': 0,
                'average_score': 0,
                'low_score_count': 0,
                'high_score_count': 0
            }
        
        return {
            'total_answers': len(self.scores),
            'average_score': sum(self.scores) / len(self.scores),
            'low_score_count': len([s for s in self.scores if s < self.low_score_threshold]),
            'high_score_count': len([s for s in self.scores if s >= self.low_score_threshold]),
            'min_score': min(self.scores),
            'max_score': max(self.scores)
        }
    
    def export_training_data(self):
        """Convert feedback data to training format"""
        training_text = ""
        for item in self.feedback_data:
            # Format: Question: ... Answer: ...
            training_text += f"Q: {item['question']}\nA: {item['answer']}\n\n"
        return training_text


class SelfImprovingEvaluator:
    """Evaluates answers using Ollama and provides improvement suggestions"""
    
    def __init__(self, ollama_validator, user_context=""):
        self.validator = ollama_validator
        self.user_context = user_context.strip()
    
    def evaluate_answer(self, question, answer):
        """Get Ollama to evaluate the answer with detailed feedback"""
        if not self.validator.available or not self.validator.model:
            return None, None

        # Build the context block — injected only when the user supplied one
        context_block = ""
        if self.user_context:
            context_block = (
                f"\n\nTRAINING CONTEXT (follow these instructions when scoring):\n"
                f"{self.user_context}\n"
            )
        
        evaluation_prompt = f"""You are an expert evaluator. Evaluate this Q&A pair on these criteria:
- Accuracy (does it answer the question correctly?)
- Completeness (does it cover all aspects?)
- Clarity (is it well-written and understandable?)
- Relevance (is all information relevant?){context_block}

Question: {question}
Answer: {answer}

Provide:
1. A score from 1-10 (1=very poor, 10=excellent)
2. A brief comment on what could be improved

Format your response EXACTLY as:
SCORE: [number]/10
COMMENT: [your feedback]"""
        
        try:
            response = self.validator.validate_text(evaluation_prompt, question)
            if response:
                # Parse score and comment
                lines = response.strip().split('\n')
                score_line = [l for l in lines if l.startswith('SCORE:')]
                comment_lines = [l for l in lines if l.startswith('COMMENT:')]
                
                score = 5.0  # default
                comment = "No feedback"
                
                if score_line:
                    try:
                        score_str = score_line[0].replace('SCORE:', '').strip().split('/')[0]
                        score = float(score_str)
                    except:
                        pass
                
                if comment_lines:
                    comment = comment_lines[0].replace('COMMENT:', '').strip()
                
                return score, comment
            return None, None
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return None, None


# ============================================================================
# TRAINING AND Q&A LOGGER
# ============================================================================
class TrainingLogger:
    """Logs Q&A interactions, scores, and training history"""
    
    def __init__(self, model_name, model_dir="saved_models"):
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Log files
        self.qa_log_file = self.model_dir / f"{model_name}_qa_log.txt"
        self.training_log_file = self.model_dir / f"{model_name}_training_log.txt"
        self.metadata_file = self.model_dir / f"{model_name}_metadata.json"
        
        # Load or initialize metadata
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """Load training metadata from JSON"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return self.default_metadata()
        return self.default_metadata()
    
    def default_metadata(self):
        """Create default metadata structure"""
        return {
            'model_name': self.model_name,
            'total_epochs_trained': 0,
            'training_sessions': 0,
            'total_qa_logged': 0,
            'creation_date': str(datetime.now()),
            'last_updated': str(datetime.now()),
            'training_history': []
        }
    
    def save_metadata(self):
        """Save metadata to JSON"""
        self.metadata['last_updated'] = str(datetime.now())
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def log_qa(self, question, answer, score=None, source="general"):
        """Log a Q&A pair with optional score"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
{'='*80}
TIMESTAMP: {timestamp}
SOURCE: {source}
{'='*80}

QUESTION:
{question}

ANSWER:
{answer}
"""
        
        if score is not None:
            log_entry += f"\nSCORE: {score:.1f}/10\n"
        
        log_entry += f"{'='*80}\n"
        
        try:
            with open(self.qa_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            self.metadata['total_qa_logged'] += 1
            self.save_metadata()
        except Exception as e:
            print(f"⚠️  Error logging Q&A: {e}")
    
    def log_training_session(self, epochs, learning_rate, batch_size, 
                            initial_loss=None, final_loss=None, session_type="training"):
        """Log a training session"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        session_info = {
            'timestamp': timestamp,
            'type': session_type,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'initial_loss': initial_loss,
            'final_loss': final_loss
        }
        
        # Update total epochs
        self.metadata['total_epochs_trained'] += epochs
        self.metadata['training_sessions'] += 1
        self.metadata['training_history'].append(session_info)
        
        # Log to file
        log_entry = f"""
{'='*80}
TRAINING SESSION LOG
{'='*80}
Timestamp: {timestamp}
Type: {session_type}
Epochs: {epochs}
Learning Rate: {learning_rate}
Batch Size: {batch_size}
Initial Loss: {initial_loss}
Final Loss: {final_loss}
Total Epochs (Cumulative): {self.metadata['total_epochs_trained']}
{'='*80}

"""
        
        try:
            with open(self.training_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            self.save_metadata()
        except Exception as e:
            print(f"⚠️  Error logging training: {e}")
    
    def get_total_epochs(self):
        """Get total epochs trained"""
        return self.metadata['total_epochs_trained']
    
    def get_statistics(self):
        """Get logging statistics"""
        return {
            'total_epochs': self.metadata['total_epochs_trained'],
            'training_sessions': self.metadata['training_sessions'],
            'total_qa_logged': self.metadata['total_qa_logged'],
            'creation_date': self.metadata['creation_date'],
            'last_updated': self.metadata['last_updated']
        }
    
    def display_summary(self):
        """Display training and logging summary"""
        stats = self.get_statistics()
        print(f"\n📊 Model Training Summary: {self.model_name}")
        print("="*60)
        print(f"Total Epochs Trained: {stats['total_epochs']} ⏱️")
        print(f"Training Sessions: {stats['training_sessions']}")
        print(f"Q&A Interactions Logged: {stats['total_qa_logged']}")
        print(f"Creation Date: {stats['creation_date']}")
        print(f"Last Updated: {stats['last_updated']}")
        print("="*60)
    
    def save_question_folder(self, folder_path):
        """Save the question folder path to metadata"""
        self.metadata['question_folder'] = str(folder_path)
        self.save_metadata()
    
    def get_question_folder(self):
        """Get the saved question folder path"""
        return self.metadata.get('question_folder', None)


# ============================================================================
# QUESTION LOADER FOR AUTOMATED EVALUATION
# ============================================================================
class QuestionLoader:
    """Load questions from folders/files for automated evaluation"""
    
    @staticmethod
    def extract_questions_from_file(file_path):
        """Extract questions from a text file"""
        questions = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try multiple question formats
            # Format 1: One question per line
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Filter for lines that look like questions (end with ?)
            for line in lines:
                if line.endswith('?') and len(line) > 5:  # Minimum question length
                    questions.append(line)
            
            # If no "?" questions found, split by common separators
            if not questions:
                # Format 2: Questions separated by numbers (1. 2. 3. etc)
                pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    q = match.strip().split('\n')[0]  # First line of each numbered item
                    if len(q) > 5:
                        questions.append(q)
            
            # If still no questions, treat each paragraph as a question
            if not questions:
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if len(para) > 10:  # Minimum length
                        # Take first sentence
                        sentence = para.split('\n')[0].split('.')[0]
                        if len(sentence) > 5:
                            questions.append(sentence + '?')
            
            return questions
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            return []
    
    @staticmethod
    def load_questions_from_folder(folder_path):
        """Load all questions from a folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"❌ Folder not found: {folder_path}")
            return []
        
        all_questions = []
        file_count = 0
        
        # Load from all text files in folder
        for file_path in sorted(folder_path.glob('**/*.txt')):
            questions = QuestionLoader.extract_questions_from_file(file_path)
            if questions:
                all_questions.extend(questions)
                file_count += 1
                print(f"  ✓ {file_path.name}: {len(questions)} questions")
        
        # Also try markdown files
        for file_path in sorted(folder_path.glob('**/*.md')):
            questions = QuestionLoader.extract_questions_from_file(file_path)
            if questions:
                all_questions.extend(questions)
                file_count += 1
                print(f"  ✓ {file_path.name}: {len(questions)} questions")
        
        return all_questions

    @staticmethod
    def validate_folder(folder_path):
        """Check that a folder exists and holds .txt / .md question files.
        Returns (is_valid, message) — same contract as QuestionFormatter.validate_folder,
        but searches subfolders like load_questions_from_folder() does."""
        folder = Path(folder_path)
        if not folder.exists():
            return False, "Folder does not exist"
        if not folder.is_dir():
            return False, "Path is not a directory"
        text_files = list(folder.glob('**/*.txt')) + list(folder.glob('**/*.md'))
        if not text_files:
            return False, "No .txt or .md files found in folder"
        return True, f"Found {len(text_files)} text file(s)"


# ============================================================================
# OLLAMA TEACHER MODEL FOR SYNTHETIC DATA GENERATION
# ============================================================================
class OllamaTeacher:
    """Use Ollama models to generate training data for your LLM"""
    
    def __init__(self):
        self.validator = OllamaValidator()
        self.selected_model = None
    
    def get_available_models(self):
        """Get list of available Ollama models"""
        if not self.validator.available or not self.validator.available_models:
            return []
        return self.validator.available_models
    
    def select_model(self):
        """Let user select an Ollama model as teacher"""
        models = self.get_available_models()
        
        if not models:
            print("❌ No Ollama models available")
            return False
        
        print("\n" + "="*60)
        print("AVAILABLE OLLAMA MODELS (Teacher Models)")
        print("="*60)
        
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        while True:
            try:
                choice = input("\nSelect model number (or 'skip'): ").strip()
                
                if choice.lower() == 'skip':
                    return False
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(models):
                    self.selected_model = models[choice_idx]
                    print(f"✓ Selected teacher model: {self.selected_model}")
                    return True
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def read_questions_from_file(self, file_path):
        """Read questions delimited by @ symbol"""
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by @ to get questions
            parts = content.split('@')
            
            for part in parts:
                question = part.strip()
                if question and len(question) > 0:
                    questions.append(question)
            
            return questions
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return []
    
    def generate_response(self, question, temperature=0.5, system_context=""):
        """Generate response from Ollama teacher model"""
        if not self.selected_model:
            return None
        
        try:
            # Prepend system context if provided
            if system_context and system_context.strip():
                prompt = f"[CONTEXT FOR THIS RESPONSE]\n{system_context.strip()}\n\n{question}"
            else:
                prompt = f"{question}"
            
            response = requests.post(
                f"{self.validator.ollama_url}/api/generate",
                json={
                    "model": self.selected_model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            return None
        except Exception as e:
            print(f"⚠️  Error generating response: {e}")
            return None
    
    def generate_training_data(self, questions, temperature=0.5, show_progress=True,
                               system_context=""):
        """Generate training data from questions using teacher model.
        
        system_context: optional instruction block prepended to every prompt,
                        e.g. 'Answer as a pirate' or 'Focus on brief factual answers'.
        """
        training_data = ""
        successful = 0
        failed = 0
        
        print(f"\n🤖 Generating training data from {len(questions)} questions...")
        print(f"📚 Teacher model: {self.selected_model}")
        if system_context.strip():
            print(f"📋 Context: {system_context[:80]}{'…' if len(system_context) > 80 else ''}")
        print()
        
        for i, question in enumerate(questions, 1):
            if show_progress:
                print(f"[{i}/{len(questions)}] Q: {question[:60]}...", end=" ", flush=True)
            
            response = self.generate_response(question, temperature=temperature,
                                              system_context=system_context)
            
            if response:
                training_data += f"Q: {question}\nA: {response}\n\n"
                successful += 1
                if show_progress:
                    print("✓")
            else:
                failed += 1
                if show_progress:
                    print("❌")
        
        print("\n" + "="*60)
        print(f"✓ Generated {successful}/{len(questions)} Q&A pairs")
        if failed > 0:
            print(f"❌ Failed: {failed} questions")
        print("="*60)
        
        return training_data, successful, failed


# ============================================================================
# FILE PROCESSOR
# ============================================================================
class FileProcessor:
    """Handles loading and processing various file types (images, text, PDF, etc.)"""

    @staticmethod
    def load_image_as_text(image_path):
        """Load image and extract text using OCR or description"""
        try:
            img = Image.open(image_path)
            
            # Try to use pytesseract if available
            try:
                if pytesseract is not None:
                    text = pytesseract.image_to_string(img)
                    if text.strip():
                        return f"[Image OCR from {Path(image_path).name}]\n{text}"
            except Exception:
                pass
            
            # Fallback: return image info
            img_info = f"[Image: {Path(image_path).name}]\n"
            img_info += f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}\n"
            img_info += "(Note: Full OCR requires: pip install pytesseract)\n"
            return img_info
            
        except Exception as e:
            return f"[Error loading image {Path(image_path).name}]: {e}"
    
    @staticmethod
    def load_text_file(file_path):
        """Load text file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"[Text from {Path(file_path).name}]\n{content}\n"
        except Exception as e:
            return f"[Error loading {Path(file_path).name}]: {e}"
    
    @staticmethod
    def load_markdown_file(file_path):
        """Load markdown file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"[Markdown from {Path(file_path).name}]\n{content}\n"
        except Exception as e:
            return f"[Error loading {Path(file_path).name}]: {e}"
    
    @staticmethod
    def load_file(file_path):
        """Auto-detect and load any supported file type"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return f"[Error: File not found: {file_path}]"
        
        # Image files
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return FileProcessor.load_image_as_text(str(file_path))
        
        # PDF files
        elif file_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(reader.pages, 1):
                        text += f"[Page {page_num}]\n{page.extract_text()}\n"
                return f"[PDF from {file_path.name}]\n{text}\n"
            except ImportError:
                return f"[PDF from {file_path.name}]\n(Note: Install PyPDF2 with: pip install PyPDF2)\n"
            except Exception as e:
                return f"[Error reading PDF {file_path.name}]: {e}\n"
        
        # Text files
        elif file_path.suffix.lower() in ['.txt', '.log']:
            return FileProcessor.load_text_file(str(file_path))
        
        # Markdown files
        elif file_path.suffix.lower() in ['.md', '.markdown']:
            return FileProcessor.load_markdown_file(str(file_path))
        
        # JSON files
        elif file_path.suffix.lower() == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.dumps(json.load(f), indent=2)
                return f"[JSON from {file_path.name}]\n{content}\n"
            except Exception as e:
                return f"[Error loading {file_path.name}]: {e}"
        
        # CSV files
        elif file_path.suffix.lower() == '.csv':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"[CSV from {file_path.name}]\n{content}\n"
            except Exception as e:
                return f"[Error loading {file_path.name}]: {e}"
        
        else:
            # Try to read as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"[File from {file_path.name}]\n{content}\n"
            except:
                return f"[Cannot read file: {file_path.name}] (binary or unsupported format)"
    
    @staticmethod
    def load_multiple_files(file_paths):
        """Load multiple files and combine their content"""
        combined_context = ""
        for file_path in file_paths:
            content = FileProcessor.load_file(file_path)
            combined_context += content + "\n"
        
        return combined_context


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # shape [1, max_len, d_model]
        self._d_model = d_model

    def _extend_pe(self, seq_len, device):
        """Build PE for positions beyond the pre-computed buffer on the fly."""
        current = self.pe.size(1)
        extra   = seq_len - current
        position = torch.arange(current, seq_len, dtype=torch.float,
                                device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self._d_model, 2, device=device).float() *
            -(np.log(10000.0) / self._d_model)
        )
        ext = torch.zeros(1, extra, self._d_model, device=device)
        ext[0, :, 0::2] = torch.sin(position * div_term)
        ext[0, :, 1::2] = torch.cos(position * div_term)
        return torch.cat([self.pe, ext], dim=1)   # [1, seq_len, d_model]

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Input is longer than the pre-computed buffer — extend dynamically
            pe = self._extend_pe(seq_len, x.device)
        else:
            pe = self.pe
        return x + pe[:, :seq_len, :]


# ============================================================================
# SELF-ATTENTION MECHANISM
# ============================================================================
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Causal mask: position t may only attend to positions <= t.  Without
        # it every position can read the very token it is being trained to
        # predict, so the loss collapses without the model learning language.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)
        output = self.linear_out(context)
        return output


# ============================================================================
# FEEDFORWARD NETWORK
# ============================================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ============================================================================
# SIMPLE LLM MODEL
# ============================================================================
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_blocks=2, num_heads=4, 
                 max_len=100, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_blocks)
        ])
        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.blocks(x)
        logits = self.output(x)
        return logits


# ============================================================================
# CHARACTER TOKENIZER
# ============================================================================
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        """Encode text to token indices.

        Characters not in the vocabulary are silently skipped rather than
        raising a KeyError.  This makes encoding safe for any text source
        (e.g. Ollama output that may contain accented or unicode characters
        that were never seen during training).
        """
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices):
        """Decode token indices back to text.

        Out-of-range indices are skipped rather than raising a KeyError.
        """
        return ''.join(self.idx_to_char[i] for i in indices if i in self.idx_to_char)

    def encode_prompt(self, text):
        """Encode a generation prompt -> (token_ids, n_unknown).

        Unlike encode(), an unknown character is replaced by a space token
        instead of being dropped, so the prompt keeps its length.
        """
        space = self.char_to_idx.get(' ', 0)
        ids, unknown = [], 0
        for c in text:
            if c in self.char_to_idx:
                ids.append(self.char_to_idx[c])
            else:
                ids.append(space)
                unknown += 1
        return ids, unknown

    def sanitize(self, text):
        """Return a copy of *text* with every out-of-vocabulary character removed.

        Also prints a one-line warning that lists the unique unknown characters
        so the caller knows exactly what was stripped.

        Call this before building a TextDataset from any external text::

            clean = tokenizer.sanitize(ollama_output)
            dataset = TextDataset(clean, tokenizer)
        """
        unknown = sorted({c for c in text if c not in self.char_to_idx})
        if unknown:
            display = ''.join(unknown[:20])
            more    = f" … (+{len(unknown)-20} more)" if len(unknown) > 20 else ""
            print(f"  ⚠️  Removed {len(unknown)} out-of-vocab character type(s) "
                  f"from generated text: {repr(display)}{more}")
        return ''.join(c for c in text if c in self.char_to_idx)
    
    def save(self, path):
        """Save tokenizer for later use"""
        data = {
            'chars': self.chars,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path):
        """Load saved tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Files written by BPETokenizer.save() are marked "type": "bpe".  Hand
        # them to BPETokenizer so every CharTokenizer.load(...) call (including
        # exported serve.py scripts) works with either kind of tokenizer file.
        if data.get('type') == 'bpe':
            return BPETokenizer._from_dict(data)

        tokenizer = cls.__new__(cls)
        tokenizer.chars = data['chars']
        tokenizer.vocab_size = len(tokenizer.chars)
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        return tokenizer


# ============================================================================
# BYTE-PAIR-ENCODING (BPE) TOKENIZER
# ============================================================================
class BPETokenizer:
    """Byte-level BPE tokenizer (the same scheme GPT-2 uses).

    Token ids 0-255 are the raw UTF-8 bytes, so ANY text can be encoded and
    there are no out-of-vocabulary characters.  Ids 256+ are learned merges
    of two existing tokens, applied in the order they were learned.  Frequent
    character sequences (" the", "ing", ".\\n" ...) therefore become single
    tokens, so one token covers roughly 2-4 characters.

    The public interface matches CharTokenizer (encode / decode / sanitize /
    save / load / vocab_size) so the rest of the program works unchanged.
    """

    # GPT-2 style pre-tokenisation: contractions, words, numbers, punctuation
    # runs and whitespace each become their own chunk, and merges never cross
    # a chunk boundary.  Every character matches exactly one alternative, so
    # joining the chunks always reproduces the original text.
    _PATTERN = re.compile(
        r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")
    _CUT = re.compile(r"(?<=\S)\s")   # a spot that is always a chunk boundary
    _BLOCK_CHARS = 1_000_000          # characters pre-tokenised at a time
    _CACHE_LIMIT = 200_000            # max cached chunk -> token-id lists

    def __init__(self, text, vocab_size=512, verbose=True):
        """Learn merges from *text* until the vocabulary reaches *vocab_size*
        (256 byte tokens + merges).  Stops early if no pair occurs twice."""
        self.merges = []
        self._rebuild()
        self._train(text, int(vocab_size), verbose)

    # ── Lookup tables ────────────────────────────────────────────────────
    def _rebuild(self):
        """Recompute every lookup table from self.merges."""
        self.vocab_size = 256 + len(self.merges)
        self._ranks = {tuple(p): i for i, p in enumerate(self.merges)}
        vocab = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            vocab.append(vocab[a] + vocab[b])
        self._vocab = vocab
        self._cache = {}

    # ── Pre-tokenisation ─────────────────────────────────────────────────
    def _chunks(self, text):
        """Yield lists of pre-tokenised chunks, one list per ~1M-char block.

        Blocks are cut only where a chunk boundary is guaranteed (a
        non-space character followed by whitespace), so the result is
        identical to pre-tokenising the whole text at once.
        """
        n   = len(text)
        pos = 0
        while pos < n:
            end = pos + self._BLOCK_CHARS
            if end >= n:
                end = n
            else:
                m   = self._CUT.search(text, end)
                end = m.start() if m else n
            yield self._PATTERN.findall(text, pos, end)
            pos = end

    # ── Training ─────────────────────────────────────────────────────────
    def _train(self, text, vocab_size, verbose):
        import heapq
        from collections import Counter

        target = max(0, vocab_size - 256)
        counts = Counter()
        for chunk_list in self._chunks(text):
            counts.update(chunk_list)

        # Work on unique chunks weighted by frequency (far faster than
        # walking the whole corpus for every merge).
        words = [list(c.encode('utf-8', 'replace')) for c in counts]
        freqs = list(counts.values())

        pair_counts = {}     # (a, b) -> total frequency
        where       = {}     # (a, b) -> indices of words that contain it
        for wi, w in enumerate(words):
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + f
                s = where.get(p)
                if s is None:
                    where[p] = {wi}
                else:
                    s.add(wi)

        # Max-heap (via negative counts) with lazy invalidation.  Ties go to
        # the numerically smallest pair, so training is fully deterministic.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        if verbose and target:
            print(f"  🔤 Training BPE tokenizer (target vocab {vocab_size}) …")
        merges      = []
        next_report = 0.25
        while len(merges) < target and heap:
            neg_c, pair = heapq.heappop(heap)
            cur = pair_counts.get(pair, 0)
            if cur <= 0:
                continue
            if -neg_c != cur:                 # stale entry → requeue
                heapq.heappush(heap, (-cur, pair))
                continue
            if cur < 2:                       # nothing worth merging any more
                break

            a, b   = pair
            new_id = 256 + len(merges)
            merges.append(pair)
            touched = set()
            for wi in list(where.get(pair, ())):
                old = words[wi]
                new = []
                i, n = 0, len(old)
                while i < n:
                    if i < n - 1 and old[i] == a and old[i + 1] == b:
                        new.append(new_id)
                        i += 2
                    else:
                        new.append(old[i])
                        i += 1
                if len(new) == len(old):      # stale 'where' entry
                    continue
                f = freqs[wi]
                for p in zip(old, old[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                words[wi] = new
                for p in zip(new, new[1:]):
                    pair_counts[p] = pair_counts.get(p, 0) + f
                    s = where.get(p)
                    if s is None:
                        where[p] = {wi}
                    else:
                        s.add(wi)
                    touched.add(p)

            for p in touched:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
                    where.pop(p, None)

            if verbose and target >= 200 and len(merges) >= next_report * target:
                print(f"     … {len(merges)}/{target} merges")
                next_report += 0.25

        self.merges = merges
        self._rebuild()

        if verbose:
            n_tok = sum(len(w) * f for w, f in zip(words, freqs))
            ratio = len(text) / max(1, n_tok)
            note  = ""
            if len(merges) < target:
                note = f"  (stopped early — no more pairs occur twice)"
            print(f"  ✓ BPE ready: {self.vocab_size} tokens "
                  f"(256 bytes + {len(merges)} merges) | "
                  f"{ratio:.2f} chars/token{note}")

    # ── Encode / decode ──────────────────────────────────────────────────
    def _encode_chunk(self, chunk):
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit
        ids   = list(chunk.encode('utf-8', 'replace'))
        ranks = self._ranks
        while len(ids) > 1:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            a, b   = best
            new_id = 256 + best_rank
            merged = []
            i, n   = 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        """Encode text to a list of token ids.  Never drops anything."""
        out = []
        enc = self._encode_chunk
        for chunk_list in self._chunks(text):
            for chunk in chunk_list:
                out.extend(enc(chunk))
        return out

    def encode_prompt(self, text):
        """Return (token_ids, n_unknown).  Byte-level BPE has no unknown
        characters, so n_unknown is always 0."""
        return self.encode(text), 0

    def decode(self, indices):
        """Decode token ids back to text.  Out-of-range ids are skipped;
        a token sequence that stops mid-character decodes to U+FFFD."""
        vocab = self._vocab
        n     = len(vocab)
        data  = b''.join(vocab[i] for i in indices if 0 <= i < n)
        return data.decode('utf-8', errors='replace')

    def sanitize(self, text):
        """Byte-level BPE can represent every character, so nothing is ever
        removed.  Kept so callers written for CharTokenizer still work."""
        return text

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path):
        """Save tokenizer for later use"""
        data = {
            'type':       'bpe',
            'version':    1,
            'vocab_size': self.vocab_size,
            'merges':     [list(p) for p in self.merges],
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def _from_dict(cls, data):
        tok = cls.__new__(cls)
        tok.merges = [tuple(p) for p in data['merges']]
        tok._rebuild()
        return tok

    @classmethod
    def load(cls, path):
        """Load saved tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)


# ============================================================================
# DATASET
# ============================================================================
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=32):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


# ============================================================================
# MODEL MANAGER (Save/Load)
# ============================================================================
def _portable_state_dict(model):
    """Weights as CPU tensors, so the saved file loads on any machine - with or without a GPU."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


class ModelManager:
    """Handles saving and loading trained models"""
    
    def __init__(self, model_dir="saved_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, model_name):
        """Save model weights"""
        model_path = self.model_dir / f"{model_name}.pt"
        torch.save(_portable_state_dict(model), model_path)
        print(f"✓ Model saved: {model_path}")
        return model_path
    
    def load_model(self, model, model_name):
        """Load model weights"""
        model_path = self.model_dir / f"{model_name}.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
            print(f"✓ Model loaded: {model_path}")
            return True
        else:
            print(f"❌ Model not found: {model_path}")
            return False
    
    def save_backup(self, model, optimizer, epoch, loss, model_name, scheduler=None):
        """Save a training backup (overwrites previous backup for this model)."""
        backup = {
            'epoch'          : epoch,
            'model_state'    : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss'           : loss,
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'saved_at'       : str(datetime.now()),
        }
        backup_path = self.model_dir / f"{model_name}_backup.pt"
        torch.save(backup, backup_path)
        print(f"  💾 Backup saved → {backup_path.name}  (epoch {epoch + 1}, loss {loss:.4f})")

    # Keep old name as alias so existing references don't break
    def save_checkpoint(self, model, optimizer, epoch, loss, model_name, scheduler=None):
        return self.save_backup(model, optimizer, epoch, loss, model_name, scheduler)

    def load_backup(self, model, optimizer, model_name, scheduler=None, weight_decay=None):
        """Load the latest backup for this model.

        weight_decay : if given, re-applied to the optimizer after loading, because
                       Optimizer.load_state_dict() restores the *saved* value.
        """
        backup_path = self.model_dir / f"{model_name}_backup.pt"
        if backup_path.exists():
            backup = torch.load(backup_path, weights_only=False, map_location='cpu')
            model.load_state_dict(backup['model_state'])
            optimizer_reset = False
            try:
                optimizer.load_state_dict(backup['optimizer_state'])
            except (ValueError, KeyError):
                # Backups written before AdamW hold ONE Adam param group; AdamW uses two
                # (decay / no-decay), so the old running averages cannot be mapped over.
                # Weights and the LR schedule are still restored; only momentum restarts.
                optimizer_reset = True
                print("  ℹ️  Older Adam optimizer state in backup — AdamW momentum restarts "
                      "(weights and LR schedule restored).")
            if scheduler is not None and backup.get('scheduler_state') is not None:
                sched_state = dict(backup['scheduler_state'])
                # The schedule keeps one LR per param group. An old Adam backup has one
                # group, AdamW has two — copy the value to every group so the scheduler's
                # per-group lists line up (otherwise its next step() fails).
                for _key in ('base_lrs', '_last_lr'):
                    _vals = sched_state.get(_key)
                    if _vals and len(_vals) != len(optimizer.param_groups):
                        sched_state[_key] = [_vals[0]] * len(optimizer.param_groups)
                scheduler.load_state_dict(sched_state)
                if optimizer_reset:
                    # The optimizer's own LR was not restored — take it from the scheduler.
                    for _grp, _lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                        _grp['lr'] = _lr
            if weight_decay is not None:
                _adamw_apply_weight_decay(optimizer, weight_decay)
            saved_at = backup.get('saved_at', 'unknown time')
            print(f"  ✓ Backup loaded: epoch {backup['epoch'] + 1}, "
                  f"loss {backup['loss']:.4f}, saved at {saved_at}")
            return backup['epoch'], backup['loss']
        return 0, None

    # Keep old name as alias
    def load_checkpoint(self, model, optimizer, model_name, scheduler=None):
        return self.load_backup(model, optimizer, model_name, scheduler)
    
    def list_models(self):
        """List all saved models"""
        models = [m for m in self.model_dir.glob("*.pt") if not m.stem.endswith("_backup")]
        if models:
            print("\n📦 Available Models:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model.stem}")
        else:
            print("❌ No saved models found")
        return models
    
    def list_models_simple(self):
        """Return list of model names without printing"""
        models = [m for m in self.model_dir.glob("*.pt") if not m.stem.endswith("_backup")]
        return [model.stem for model in models]

    def save_tokenizer(self, tokenizer, model_name):
        """Save tokenizer vocabulary to disk alongside the model weights."""
        tok_path = self.model_dir / f"{model_name}_tokenizer.json"
        tokenizer.save(str(tok_path))
        print(f"✓ Tokenizer saved: {tok_path}")
        return tok_path

    def load_tokenizer(self, model_name):
        """Load a previously saved tokenizer (BPE or legacy char-level), or None."""
        tok_path = self.model_dir / f"{model_name}_tokenizer.json"
        if tok_path.exists():
            tok = CharTokenizer.load(str(tok_path))
            print(f"✓ Tokenizer loaded: {tok_path}")
            return tok
        print(f"❌ Tokenizer not found: {tok_path}")
        return None

    # ── Untrained-model registry ───────────────────────────────────────
    @property
    def _registry_path(self):
        return self.model_dir / "untrained_models.json"

    def _load_registry(self):
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_registry(self, registry):
        with open(self._registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

    def register_model(self, model_name, state_summary):
        """Add or update a model in the untrained registry."""
        registry = self._load_registry()
        registry[model_name] = state_summary
        self._save_registry(registry)

    def mark_trained(self, model_name):
        """Remove a model from the untrained registry (training complete)."""
        registry = self._load_registry()
        if model_name in registry:
            del registry[model_name]
            self._save_registry(registry)

    def list_unfinished(self):
        """Return dict of models that haven't finished training."""
        return self._load_registry()
    # ──────────────────────────────────────────────────────────────────


# ============================================================================
# TRAINING STATE MANAGER
# ============================================================================
class TrainingStateManager:
    """
    Persists every training config parameter to disk so a run can be resumed
    exactly from where it left off — including dataset path, LR, batch size,
    epochs completed, last loss, and status.

    File: saved_models/<model_name>_train_state.json
    """

    STATUS_UNTRAINED   = "untrained"
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_PAUSED      = "paused"
    STATUS_COMPLETE    = "complete"

    def __init__(self, model_name, model_dir="saved_models"):
        self.model_name = model_name
        self.path = Path(model_dir) / f"{model_name}_train_state.json"

    # ── Persist ───────────────────────────────────────────────────────
    def save(self, state: dict):
        state["model_name"]   = self.model_name
        state["last_updated"] = str(datetime.now())
        with open(self.path, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            with open(self.path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def update_progress(self, epochs_trained: int, last_loss: float, status: str):
        state = self.load() or {}
        state["epochs_trained"] = epochs_trained
        state["last_loss"]      = last_loss
        state["status"]         = status
        self.save(state)

    def exists(self) -> bool:
        return self.path.exists()

    # ── Pretty summary ────────────────────────────────────────────────
    @staticmethod
    def print_state(state: dict):
        print(f"  Dataset      : {state.get('dataset_path', 'unknown')}")
        print(f"  Batch size   : {state.get('batch_size', '?')}")
        print(f"  Learning rate: {state.get('learning_rate', '?')}")
        _wd_saved = state.get('weight_decay')
        _wd_txt   = f"{_wd_saved}" if _wd_saved is not None else f"{ADAMW_DEFAULT_WEIGHT_DECAY} (default)"
        print(f"  Weight decay : {_wd_txt}")
        trained = state.get('epochs_trained', 0)
        total   = state.get('num_epochs', '?')
        print(f"  Epochs       : {trained} / {total} trained")
        if state.get('last_loss') is not None:
            print(f"  Last loss    : {state['last_loss']:.4f}")
        if state.get('target_loss') is not None:
            print(f"  Target loss  : {state['target_loss']:.4f}")
        print(f"  Status       : {state.get('status', 'unknown')}")
        print(f"  Last updated : {state.get('last_updated', 'never')}")


# ============================================================================
# HELPERS
# ============================================================================
def _fmt_time(seconds):
    """Format a duration in seconds into a human-readable string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m}m"


# ============================================================================
# TRAINING SUMMARY HELPER
# ============================================================================
def _print_training_summary(model, tokenizer, model_name, state_manager):
    """Print a concise summary of a completed (or early-stopped) training run."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    # Pull epoch count and final loss from the persisted state file
    state       = state_manager.load() if state_manager else None
    epochs_done = state.get("epochs_trained", "?") if state else "?"
    last_loss   = state.get("last_loss",      None) if state else None

    # Parameter counts
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model            : {model_name}")
    print(f"  Epochs trained   : {epochs_done}")
    if last_loss is not None:
        print(f"  Final loss       : {last_loss:.4f}")
    unit = "BPE tokens" if hasattr(tokenizer, 'merges') else "characters"
    print(f"  Vocab size       : {tokenizer.vocab_size} {unit}")
    print(f"  Total parameters : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")
    print("=" * 60)


# ============================================================================
# ADAMW OPTIMIZER HELPERS
# ============================================================================
ADAMW_DEFAULT_WEIGHT_DECAY = 0.01


def _adamw_build_optimizer(model, lr, weight_decay=ADAMW_DEFAULT_WEIGHT_DECAY):
    """Build AdamW with decoupled weight decay applied to weight matrices only.

    Two param groups (both follow the same LR schedule):
      • decay    : tensors with 2+ dims — Linear / Embedding weights
      • no-decay : 1-D tensors — biases and LayerNorm weight/bias
    weight_decay = 0 behaves exactly like plain Adam.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = []
    if decay:
        groups.append({'params': decay,    'weight_decay': weight_decay, 'adamw_decay': True})
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0,          'adamw_decay': False})
    return optim.AdamW(groups, lr=lr)


def _adamw_apply_weight_decay(optimizer, weight_decay):
    """Re-apply the wanted weight decay to every param group.

    Optimizer.load_state_dict() restores the *saved* hyper-parameters, which would
    silently overwrite the weight decay chosen for this run. Groups tagged
    'adamw_decay' = False (biases, LayerNorm) always stay at 0.
    """
    for group in optimizer.param_groups:
        group['weight_decay'] = weight_decay if group.get('adamw_decay', True) else 0.0


def _adamw_ask_weight_decay(default=ADAMW_DEFAULT_WEIGHT_DECAY):
    """Ask for the AdamW weight decay (0 turns decay off). Enter keeps the default."""
    while True:
        raw = input(f"Weight decay (default {default}, 0 = off): ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if val >= 0:
                return val
            print("  ⚠️  Must be 0 or a positive number.")
        except ValueError:
            print("  ⚠️  Please enter a number (e.g. 0.01).")


# ============================================================================
# LOSS HISTORY (CSV) + LOSS GRAPH (matplotlib)
# ============================================================================
# Every finished epoch is appended to  <model>_loss_history.csv  (one file per
# model, all sessions) and this session's loss-vs-epoch graph is redrawn to
# <model>_loss_curve_session<N>.png  at every checkpoint and when training ends.
# Nothing here can stop training: any problem (matplotlib not installed, the CSV
# open in Excel, disk full ...) only prints a warning.
LOSSPLOT_ENABLED     = True    # False = record nothing and draw nothing
LOSSPLOT_SHOW_WINDOW = False   # True  = also pop the graph up in a window when
                               #         training ends (blocks until you close it)
_LOSSHIST_FIELDS = ['session', 'epoch', 'loss', 'learning_rate',
                    'weight_decay', 'epoch_seconds', 'logged_at']


def _losshist_warn(hist, key, msg):
    """Print a warning once per kind, so a locked file doesn't repeat it every epoch."""
    if key not in hist['warned']:
        hist['warned'].add(key)
        print(f"  ⚠️  {msg}")


def _losshist_begin(model_manager, model_name, start_epoch, weight_decay):
    """Open this model's loss history and return the tracker (None = feature off).

    start_epoch == 0 starts a NEW session; start_epoch > 0 continues the latest one.
    Rows for epochs that are about to be trained again are dropped first, so
    restarting from an older backup never leaves duplicate epochs behind.
    """
    if not LOSSPLOT_ENABLED or model_manager is None:
        return None
    import csv, importlib.util
    csv_path = Path(model_manager.model_dir) / f"{model_name}_loss_history.csv"
    hist = {'csv': csv_path, 'png': None, 'name': model_name, 'session': 1,
            'wd': weight_decay, 'rows': [], 'pending': [], 'warned': set(),
            'have_mpl': importlib.util.find_spec('matplotlib') is not None}

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    old = []
    try:
        if csv_path.exists():
            with open(csv_path, newline='', encoding='utf-8') as f:
                old = list(csv.DictReader(f))
    except Exception as e:
        _losshist_warn(hist, 'read', f"Could not read the earlier loss history "
                                     f"({type(e).__name__}) — new epochs are still appended.")
        old = []

    last = max([_int(r.get('session')) or 0 for r in old], default=0)
    if start_epoch > 0 and last > 0:
        hist['session'] = last                      # continue the latest session
        keep = [r for r in old
                if not (_int(r.get('session')) == last and (_int(r.get('epoch')) or 0) > start_epoch)]
        if len(keep) != len(old):                   # epochs about to be re-trained: drop old rows
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=_LOSSHIST_FIELDS, extrasaction='ignore', restval='')
                    w.writeheader()
                    w.writerows(keep)
            except OSError as e:
                _losshist_warn(hist, 'trim', f"Could not tidy the loss CSV ({type(e).__name__}) — "
                                             f"re-trained epochs may appear twice in it.")
        for r in keep:
            if _int(r.get('session')) == last and _int(r.get('epoch')) is not None:
                try:
                    hist['rows'].append({'epoch': _int(r['epoch']), 'loss': float(r['loss'])})
                except (TypeError, ValueError, KeyError):
                    pass
    elif start_epoch == 0:
        hist['session'] = last + 1                  # fresh training = new session

    hist['png'] = csv_path.with_name(f"{model_name}_loss_curve_session{hist['session']}.png")
    if hist['have_mpl']:
        hist['banner'] = f"Loss log:      {csv_path}  (+ graph: {hist['png'].name})"
    else:
        hist['banner'] = (f"Loss log:      {csv_path}  "
                          f"(graph skipped — matplotlib not installed: pip install matplotlib)")
    return hist


def _losshist_flush(hist, final=False):
    """Append rows that are not in the CSV yet. A locked file (e.g. open in Excel) only delays them."""
    if hist is None or not hist['pending']:
        return
    import csv
    path = hist['csv']
    try:
        is_new = (not path.exists()) or path.stat().st_size == 0
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=_LOSSHIST_FIELDS)
            if is_new:
                w.writeheader()
            w.writerows(hist['pending'])
        hist['pending'] = []
    except OSError as e:
        _losshist_warn(hist, 'write', f"Loss data not written yet ({type(e).__name__}) — "
                                      f"is the CSV open in another program? Retrying each epoch.")
        if final:                                   # last chance: park the rows beside the CSV
            alt = path.with_name(f"{path.stem}_unsaved_{datetime.now():%Y%m%d_%H%M%S}.csv")
            try:
                with open(alt, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=_LOSSHIST_FIELDS)
                    w.writeheader()
                    w.writerows(hist['pending'])
                print(f"  ⚠️  Loss rows saved to {alt.name} instead — merge them into the CSV later.")
                hist['pending'] = []
            except OSError:
                print("  ⚠️  Could not save the remaining loss rows.")


def _losshist_add(hist, epoch, loss, lr, seconds):
    """Record one finished epoch (epoch is 1-based) and append it to the CSV."""
    if hist is None:
        return
    hist['rows'].append({'epoch': epoch, 'loss': float(loss)})
    hist['pending'].append({
        'session': hist['session'], 'epoch': epoch, 'loss': repr(float(loss)),
        'learning_rate': repr(float(lr)), 'weight_decay': repr(float(hist['wd'])),
        'epoch_seconds': f"{seconds:.3f}",
        'logged_at': datetime.now().isoformat(timespec='seconds'),
    })
    _losshist_flush(hist)


def _losshist_draw(ax, rows, title, target_loss=None):
    """Draw loss (y axis) against epoch (x axis) onto a matplotlib Axes."""
    from matplotlib.ticker import MaxNLocator
    xs = [r['epoch'] for r in rows]
    ys = [r['loss'] if math.isfinite(r['loss']) else float('nan') for r in rows]
    ax.plot(xs, ys, color='#1f77b4', linewidth=1.8, markersize=4,
            marker='o' if len(xs) <= 60 else None, label='Training loss')
    ok = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
    if ok:
        best_x, best_y = min(ok, key=lambda p: p[1])
        ax.scatter([best_x], [best_y], s=70, color='#d62728', zorder=3,
                   label=f'Lowest: {best_y:.4f} (epoch {best_x})')
    if target_loss:
        ax.axhline(target_loss, linestyle='--', linewidth=1, color='gray',
                   label=f'Target: {target_loss:g}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')


def _losshist_plot(hist, target_loss=None, final=False):
    """Redraw this session's loss graph (PNG). final=True also finishes the CSV and reports."""
    if hist is None:
        return
    if final:
        _losshist_flush(hist, final=True)
    rows = hist['rows']
    drawn = False
    if rows and hist['have_mpl']:
        title = f"Training loss per epoch — {hist['name']} (session {hist['session']})"
        try:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(9, 5.5), dpi=130)
            _losshist_draw(fig.subplots(), rows, title, target_loss)
            fig.tight_layout()
            fig.savefig(str(hist['png']))
            drawn = True
        except Exception as e:
            _losshist_warn(hist, 'plot', f"Could not draw the loss graph ({type(e).__name__}: {e})")
        if final and drawn and LOSSPLOT_SHOW_WINDOW:
            try:
                import matplotlib.pyplot as plt
                fig2, ax2 = plt.subplots(figsize=(9, 5.5))
                _losshist_draw(ax2, rows, title, target_loss)
                fig2.tight_layout()
                plt.show()
            except Exception as e:
                _losshist_warn(hist, 'show', f"Could not open the graph window ({type(e).__name__}: {e})")
    if final and rows:
        if drawn:
            print(f"📈 Loss graph saved → {hist['png']}")
            print(f"   Loss data        → {hist['csv']}")
        else:
            print(f"📊 Loss data saved → {hist['csv']}")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_model(model, train_loader, num_epochs=10, learning_rate=0.001, 
                device='cpu', model_manager=None, model_name="my_model", 
                start_epoch=0, checkpoint_interval=5, logger=None, target_loss=None,
                min_lr_ratio=0.1, print_every=None,
                state_manager=None,
                weight_decay=ADAMW_DEFAULT_WEIGHT_DECAY):
    """Train the model.

    LR schedule (always automatic):
        Warmup  : first 10% of num_epochs  — LR rises linearly 0 → peak_lr
        Decay   : remaining 90%            — LR falls via cosine → peak_lr * min_lr_ratio

    min_lr_ratio : floor LR as fraction of peak (default 0.1 = 10% of peak).
    print_every  : kept for compatibility; every epoch is always printed.
    weight_decay : AdamW decoupled weight decay on weight matrices (0 = same as plain Adam).
    Loss history : every finished epoch is appended to <model>_loss_history.csv and the
                   loss-vs-epoch graph is redrawn (see LOSS HISTORY section; needs matplotlib).
    """

    # Always print every epoch so the user can see each one
    if print_every is None:
        print_every = 1

    # ── LR schedule: always 10 % warmup → cosine decay for remaining 90 % ──
    # warmup_epochs is always derived from num_epochs so the ratio is exact.
    # round() keeps the boundary as close to 10 % as possible.
    # e.g. 10 ep → 1 warmup (10 %), 15 ep → 2 (13 %), 150 ep → 15 (10 %)
    warmup_epochs = max(1, round(num_epochs * 0.10))
    decay_epochs  = max(1, num_epochs - warmup_epochs)   # remaining 90 %
    min_lr        = learning_rate * min_lr_ratio

    def _lr_lambda(step):
        """
        step = LambdaLR's internal last_epoch counter (0-based).

        Phase 1 — Warmup [step 0 … warmup_epochs-1]
            LR rises linearly from 0 → peak_lr  (first 10 % of total epochs)

        Phase 2 — Cosine decay [step warmup_epochs … num_epochs-1]
            LR falls smoothly: peak_lr → peak_lr * min_lr_ratio  (remaining 90 %)
        """
        if step < warmup_epochs:
            # Linear ramp: 1/warmup_epochs at step 0, 1.0 at step warmup_epochs-1
            return (step + 1) / warmup_epochs
        else:
            # Cosine: 0.0 progress at step warmup_epochs, 1.0 at step num_epochs-1
            progress = (step - warmup_epochs) / max(1, decay_epochs - 1)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            # Returns a multiplier between min_lr_ratio and 1.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    # ─────────────────────────────────────────────────────────────────────

    optimizer = _adamw_build_optimizer(model, learning_rate, weight_decay)
    # LambdaLR calls step() once at __init__, setting last_epoch = 0.
    # So _lr_lambda(0) is applied before the first training epoch.
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # ── CPU thread tuning ─────────────────────────────────────────────
    # Physical core count (halve logical count to avoid hyperthreading
    # overhead, which hurts small models more than it helps).
    phys_cores = max(1, (os.cpu_count() or 1) // 2)
    torch.set_num_threads(phys_cores)
    # Inter-op parallelism: independent ops that can run concurrently.
    try:
        torch.set_num_interop_threads(max(1, phys_cores // 2))
    except RuntimeError:
        pass   # PyTorch allows this only once per process (an earlier run already set it)

    # Warm up the thread pool NOW (before the epoch loop) so epoch 1
    # doesn't spike higher than the rest due to lazy initialisation.
    with torch.no_grad():
        _dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        try:
            model(_dummy)
        except Exception:
            pass   # shape mismatch is fine — we just want the thread pool alive
    # ──────────────────────────────────────────────────────────────────

    model.train()

    initial_loss = None
    final_loss = None
    early_stopped = False

    # ── Resume from backup if one exists ─────────────────────────────
    # LambdaLR already called step() once at init (last_epoch = 0).
    # We need to position the scheduler at exactly `start_epoch` steps.
    scheduler_already_positioned = False

    if start_epoch > 0 and model_manager:
        backup_path = model_manager.model_dir / f"{model_name}_backup.pt"
        if backup_path.exists():
            loaded_epoch, loaded_loss = model_manager.load_backup(
                model, optimizer, model_name, scheduler=scheduler,
                weight_decay=weight_decay)
            if loaded_epoch > 0:
                # load_backup already called scheduler.load_state_dict() so
                # last_epoch is correctly restored — do NOT fast-forward again.
                start_epoch = loaded_epoch + 1
                scheduler_already_positioned = True
                print(f"✓ Resumed from epoch {start_epoch}  "
                      f"(LR={optimizer.param_groups[0]['lr']:.2e})")
                if loaded_loss:
                    print(f"  Previous loss: {loaded_loss:.4f}")

    if not scheduler_already_positioned and start_epoch > 0:
        # No backup found (e.g., recursive extra-epoch call after a clean run).
        # Fast-forward the freshly-created scheduler to start_epoch steps so
        # the warmup/cosine curve continues from the right position.
        # LambdaLR already applied step 0 at init, so we need start_epoch-1
        # additional steps to land at last_epoch == start_epoch.
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"✓ LR schedule fast-forwarded to epoch {start_epoch}  "
              f"(LR={optimizer.param_groups[0]['lr']:.2e})")
    # ──────────────────────────────────────────────────────────────────

    # ── Loss history: one CSV row per epoch + loss-vs-epoch graph ─────
    loss_hist = _losshist_begin(model_manager, model_name, start_epoch, weight_decay)

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    print(f"Epochs:        {start_epoch + 1} → {num_epochs}")
    print(f"Optimizer:     AdamW (weight decay {weight_decay})")
    if loss_hist:
        print(loss_hist['banner'])
    warmup_pct = warmup_epochs / num_epochs * 100
    print(f"\n📈 LR Schedule")
    print(f"  Phase 1 — Warmup   : epochs 1 → {warmup_epochs}"
          f"  ({warmup_pct:.0f}% of total)  LR rises 0 → {learning_rate:.2e}")
    print(f"  Phase 2 — Cosine ↓ : epochs {warmup_epochs + 1} → {num_epochs}"
          f"  ({100 - warmup_pct:.0f}% of total)  LR falls → {min_lr:.2e}")
    if target_loss:
        print(f"  Early stop at loss ≤ {target_loss:.4f}")
    print(f"\n🛑 To stop & save at any time:")
    print(f"   • Press  Ctrl+C          — stops immediately after current batch")
    print(f"   • Type   q  then Enter   — stops cleanly after current epoch")
    print(f"\nBackup created every {checkpoint_interval} epoch(s) — overwrites previous backup.")
    print(f"Resume by re-running the same option and choosing 'Yes' to resume.")
    print("="*60 + "\n")

    # ── Background thread: graceful stop via 'q' + Enter ──────────────
    import threading, sys as _sys
    _stop_event = threading.Event()

    # IDLE and some other environments replace sys.stdin with an object
    # that doesn't support readline() from a background thread.
    # Only start the listener when stdin looks like a real terminal/pipe.
    _use_listener = hasattr(_sys.stdin, 'fileno')
    try:
        if _use_listener:
            _sys.stdin.fileno()   # raises in IDLE / non-real stdin
    except Exception:
        _use_listener = False

    def _stop_listener():
        while not _stop_event.is_set():
            try:
                line = _sys.stdin.readline()
                if line.strip().lower() in ('q', 'quit', 's', 'stop', 'x'):
                    print("\n⏸️  Stop requested — will save after this epoch…")
                    _stop_event.set()
                    break
            except Exception:
                break

    if _use_listener:
        _listener_thread = threading.Thread(target=_stop_listener, daemon=True)
        _listener_thread.start()
    else:
        print("ℹ️  Running in IDLE or non-terminal environment.")
        print("   Use Ctrl+C to stop training at any time.")
    # ──────────────────────────────────────────────────────────────────

    should_pause = False
    epoch_times = []  # Track time taken for each epoch
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        epoch_start_time = time.time()  # Start timing this epoch
        
        try:
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                logits = logits.view(-1, model.vocab_size)
                y = y.view(-1)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        except KeyboardInterrupt:
            # ── User pressed Ctrl+C ───────────────────────────────────
            safe_loss = total_loss / max(len(train_loader), 1)
            print(f"\n\n⏸️  Training interrupted (Ctrl+C) at epoch {epoch + 1}/{num_epochs}")
            print("💾 Saving backup…")
            if model_manager:
                model_manager.save_backup(model, optimizer, epoch, safe_loss,
                                          model_name, scheduler=scheduler)
                model_manager.save_model(model, model_name)
            if state_manager:
                state_manager.update_progress(
                    epoch + 1, safe_loss, TrainingStateManager.STATUS_PAUSED)
            if logger:
                logger.log_training_session(
                    epochs=(epoch + 1) - start_epoch,
                    learning_rate=learning_rate,
                    batch_size=train_loader.batch_size,
                    initial_loss=initial_loss,
                    final_loss=safe_loss,
                    session_type="training-paused"
                )
            print(f"✓ Backup saved at epoch {epoch + 1}")
            print(f"\n📌 To resume: use Option 8 or Option 1 → select '{model_name}' → choose Yes")
            should_pause = True
            break
            # ──────────────────────────────────────────────────────────

        except Exception as crash_err:
            # ── Unexpected crash — save backup so nothing is lost ─────
            safe_loss = total_loss / max(len(train_loader), 1)
            print(f"\n\n❌ TRAINING ERROR at epoch {epoch + 1}/{num_epochs}")
            print(f"   Error: {type(crash_err).__name__}: {crash_err}")
            print("\n💾 Saving emergency backup before exiting…")
            try:
                if model_manager:
                    model_manager.save_backup(model, optimizer, epoch, safe_loss,
                                              model_name, scheduler=scheduler)
                    model_manager.save_model(model, model_name)
                    print(f"✓ Emergency backup saved at epoch {epoch + 1}")
                if state_manager:
                    state_manager.update_progress(
                        epoch + 1, safe_loss, TrainingStateManager.STATUS_PAUSED)
            except Exception as backup_err:
                print(f"⚠️  Could not save backup: {backup_err}")
            print(f"\n📌 Resume from epoch {epoch + 1}: use Option 8 or Option 1 → '{model_name}'")
            import traceback
            print("\n── Full traceback ──")
            traceback.print_exc()
            should_pause = True
            break
            # ──────────────────────────────────────────────────────────
        
        avg_loss = total_loss / len(train_loader)

        # Track initial loss for logging
        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

        # ── Epoch timing & progress (computed BEFORE early-stop so the
        #    triggering epoch's stats are always visible to the user) ──
        current_lr    = optimizer.param_groups[0]['lr']   # LR used this epoch
        epoch_elapsed = time.time() - epoch_start_time
        epoch_times.append(epoch_elapsed)
        avg_epoch_time   = sum(epoch_times) / len(epoch_times)
        epochs_remaining = num_epochs - (epoch + 1)
        eta_seconds      = avg_epoch_time * epochs_remaining
        pct              = (epoch + 1) / num_epochs * 100
        bar_width        = 30
        filled           = int(bar_width * pct / 100)
        bar              = "█" * filled + "░" * (bar_width - filled)
        eta_str          = _fmt_time(eta_seconds) if epochs_remaining > 0 else "Done"

        # ── Per-epoch status line — printed BEFORE early-stop check ───
        print(
            f"Epoch {epoch + 1:>{len(str(num_epochs))}}/{num_epochs}"
            f"  [{bar}] {pct:5.1f}%"
            f"  Loss: {avg_loss:.4f}"
            f"  LR: {current_lr:.2e}"
            f"  {_fmt_time(epoch_elapsed)}/ep"
            f"  ETA: {eta_str}"
        )
        # ──────────────────────────────────────────────────────────────

        # ── Loss history row (before early-stop, so the last epoch is kept) ──
        _losshist_add(loss_hist, epoch + 1, avg_loss, current_lr, epoch_elapsed)

        # ── Early stopping check (after print so the loss that fires
        #    the stop is visible in the epoch line above) ──────────────
        if target_loss is not None and avg_loss <= target_loss:
            early_stopped = True
            print(f"\n🎯 Early stopping triggered — target loss reached!")
            print(f"   Epoch {epoch + 1}  |  Loss {avg_loss:.4f} ≤ target {target_loss:.4f}")
            if model_manager:
                model_manager.save_backup(model, optimizer, epoch, avg_loss,
                                              model_name, scheduler=scheduler)
                model_manager.save_model(model, model_name)
                print(f"✓ Model saved: {model_name}")
            if state_manager:
                state_manager.update_progress(
                    epoch + 1, avg_loss, TrainingStateManager.STATUS_COMPLETE)
            if model_manager:
                model_manager.mark_trained(model_name)
            break
        # ──────────────────────────────────────────────────────────────

        # ── Advance LR schedule (warmup → cosine decay) ───────────────
        scheduler.step()
        # ──────────────────────────────────────────────────────────────

        # ── Update state after every epoch ────────────────────────────
        if state_manager:
            state_manager.update_progress(
                epoch + 1, avg_loss, TrainingStateManager.STATUS_IN_PROGRESS)
        # ──────────────────────────────────────────────────────────────

        # ── Interval checkpoint save ───────────────────────────────────
        if (epoch + 1) % checkpoint_interval == 0 and model_manager:
            model_manager.save_backup(model, optimizer, epoch, avg_loss,
                                          model_name, scheduler=scheduler)
            print(f"  💾 Checkpoint saved (epoch {epoch + 1})")
            _losshist_plot(loss_hist, target_loss)   # refresh the graph at each checkpoint
        # ──────────────────────────────────────────────────────────────

        # ── Check for graceful stop request (q + Enter) ───────────────
        if _stop_event.is_set():
            print(f"\n⏸️  Stopping after epoch {epoch + 1} as requested.")
            print("💾 Saving progress…")
            if model_manager:
                model_manager.save_backup(model, optimizer, epoch, avg_loss,
                                              model_name, scheduler=scheduler)
                model_manager.save_model(model, model_name)
                print(f"✓ Checkpoint + model saved at epoch {epoch + 1}")
            if state_manager:
                state_manager.update_progress(
                    epoch + 1, avg_loss, TrainingStateManager.STATUS_PAUSED)
            if logger:
                logger.log_training_session(
                    epochs=(epoch + 1) - start_epoch,
                    learning_rate=learning_rate,
                    batch_size=train_loader.batch_size,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    session_type="training-paused",
                )
            print(f"\n📌 To resume: run the same option, pick '{model_name}', "
                  f"enter {num_epochs} epochs, then choose 'Yes' to resume.")
            should_pause = True
            break
        # ──────────────────────────────────────────────────────────────
    
    # Signal the listener thread that training is over (no more input needed)
    _stop_event.set()

    # Final loss graph + CSV flush (normal end, early stop, pause via q / Ctrl+C, crash)
    _losshist_plot(loss_hist, target_loss, final=True)

    # Only do final save/log if not paused
    if not should_pause:
        # Save final model
        if model_manager:
            model_manager.save_model(model, model_name)
        
        # Log training session
        if logger:
            actual_epochs_trained = (epoch + 1) - start_epoch if early_stopped else num_epochs - start_epoch
            session_type = "training-early-stopped" if early_stopped else "training"
            logger.log_training_session(
                epochs=actual_epochs_trained,
                learning_rate=learning_rate,
                batch_size=train_loader.batch_size,
                initial_loss=initial_loss,
                final_loss=final_loss,
                session_type=session_type
            )

        # Mark training complete in state file and registry
        if not early_stopped:      # early-stop already marked above
            if state_manager:
                state_manager.update_progress(
                    num_epochs, final_loss or 0.0, TrainingStateManager.STATUS_COMPLETE)
            if model_manager:
                model_manager.mark_trained(model_name)

        # ── Offer extra epochs if target loss was set but not reached ──
        if target_loss is not None and not early_stopped and not should_pause:
            print()
            print("="*60)
            print(f"⚠️  Target loss {target_loss:.4f} was NOT reached.")
            print(f"   Final loss after {num_epochs} epochs: {final_loss:.4f}")
            print("="*60)
            print("Options:")
            print("  • Enter a number  — train for that many MORE epochs")
            print("  • Press Enter     — stop here and keep the saved model")
            extra_input = input("Extra epochs to train: ").strip()
            if extra_input.isdigit() and int(extra_input) > 0:
                extra_epochs = int(extra_input)
                new_total    = num_epochs + extra_epochs
                print(f"\n▶  Continuing from epoch {num_epochs + 1} → {new_total}…\n")
                # Checkpoint the FINAL state first so the recursive call resumes
                # exactly from here, not from an older interval backup.
                if model_manager:
                    model_manager.save_backup(model, optimizer, num_epochs - 1,
                                              final_loss, model_name, scheduler=scheduler)
                # Recurse with updated start/total epoch counts
                return train_model(
                    model, train_loader,
                    num_epochs=new_total,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    device=device,
                    model_manager=model_manager,
                    model_name=model_name,
                    start_epoch=num_epochs,
                    checkpoint_interval=checkpoint_interval,
                    logger=logger,
                    target_loss=target_loss,
                    min_lr_ratio=min_lr_ratio,
                    print_every=print_every,
                    state_manager=state_manager,
                )
            else:
                print("✓ Stopping here — model already saved.")
        # ──────────────────────────────────────────────────────────────

    return should_pause


# ============================================================================
# INTERACTIVE GENERATION FUNCTION
# ============================================================================
def generate_text_interactive(model, tokenizer, device='cpu', max_length=100, 
                            temperature=0.7):
    """
    Interactive text generation from user prompts with optional Ollama validation
    """
    model.eval()
    
    print("\n" + "="*60)
    print("TEXT GENERATION MODE")
    print("="*60)
    
    # Ask about Ollama validation
    use_ollama = False
    validator = None
    print("\nDo you want to validate generated text using Ollama?")
    print("(Requires local Ollama server running with models installed)")
    ollama_choice = input("Use Ollama validation? (y/n): ").strip().lower()
    
    if ollama_choice == 'y':
        print("\nInitializing Ollama connection...")
        validator = OllamaValidator()
        
        if validator.available:
            # Let user select a model
            if validator.list_models():
                use_ollama = True
                print("✓ Ollama validation enabled")
            else:
                print("⚠ No model selected, continuing without validation")
        else:
            print("⚠ Ollama not available, continuing without validation")
    
    print("Type your prompt below. Type 'quit' to exit.\n")
    
    with torch.no_grad():
        while True:
            # Get user prompt
            prompt = input("\n📝 Your prompt: ").strip()
            
            if prompt.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            if not prompt:
                print("⚠️  Please enter a prompt")
                continue
            
            # Encode prompt
            try:
                tokens = tokenizer.encode(prompt)
                if not tokens:
                    print("❌ None of the characters in that prompt are in the vocabulary")
                    continue
                tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            except KeyError as e:
                print(f"❌ Character '{e}' not in vocabulary")
                continue
            
            # Generate
            print("\n⏳ Generating...", end=" ")
            
            for i in range(max_length):
                # Get prediction
                logits = model(tokens)
                next_logits = logits[0, -1, :]
                
                # Apply temperature (affects randomness)
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                
                # Keep manageable length
                if tokens.shape[1] > 100:
                    tokens = tokens[:, -100:]
            
            # Decode and display
            generated_tokens = tokens[0].cpu().numpy().tolist()
            generated_text = tokenizer.decode(generated_tokens)
            
            print("✓\n")
            print("-" * 60)
            print(generated_text)
            print("-" * 60)
            
            # Validate with Ollama if enabled
            if use_ollama and validator:
                print("\n🔍 Validating with Ollama...")
                validation_result = validator.validate_text(generated_text, prompt)
                if validation_result:
                    print("\n📊 Ollama Assessment:")
                    print("-" * 60)
                    print(validation_result)
                    print("-" * 60)
                else:
                    print("⚠ Validation could not be completed")
            
            # Show options
            print("\nOptions:")
            print("  [Enter new prompt]")
            print("  [Type 'quit' to exit]")


# ============================================================================
# QUESTION ANSWERING WITH FILE REFERENCES
# ============================================================================
def finetune_with_ollama(model, tokenizer, device='cpu', model_manager=None,
                        model_name="my_model", logger=None, user_context=""):
    """Fine-tune model using data generated from Ollama.

    Step order (new):
      1. Check Ollama + pick model
      2. Collect ALL training config (epochs, LR, batch, checkpoint, early-stop)
      3. Show config summary and confirm
      4. Load & validate question file
      5. Generate answers (with live log)
      6. Save Q&A data
      7. Fine-tune
    """
    print("\n" + "="*60)
    print("🤖 OLLAMA-POWERED FINE-TUNING")
    print("="*60)
    print("\n📋 How it works:")
    print("  1. Pick an Ollama model to generate answers")
    print("  2. Set all training parameters up-front")
    print("  3. Load your question file")
    print("  4. Ollama generates answers → fine-tuning data")
    print("  5. Fine-tune your model")
    print("\n⚠️  Requires: Ollama server running locally\n")

    # ── Step 1: Check Ollama & select model ──────────────────────────────────
    print("Checking Ollama connection…")
    validator = OllamaValidator()
    if not validator.available:
        print("❌ Ollama server not available — is Ollama running?")
        return

    print("\n" + "="*60)
    print("📂 SELECT OLLAMA MODEL FOR ANSWER GENERATION")
    print("="*60)
    if not validator.list_models():
        print("❌ No Ollama models available")
        return
    print(f"\n✓ Selected model: {validator.model}")

    # ── Step 2: Training configuration (ALL params collected here) ───────────
    print("\n" + "="*60)
    print("⚙️  TRAINING CONFIGURATION")
    print("="*60)
    print("Enter the training parameters for fine-tuning.\n")

    def _ask_int(prompt, default):
        while True:
            raw = input(f"{prompt} (default {default}): ").strip()
            if raw == "":
                return default
            try:
                val = int(raw)
                if val > 0:
                    return val
                print("  ⚠️  Must be a positive integer.")
            except ValueError:
                print("  ⚠️  Please enter a whole number.")

    def _ask_float(prompt, default):
        while True:
            raw = input(f"{prompt} (default {default}): ").strip()
            if raw == "":
                return default
            try:
                val = float(raw)
                if val > 0:
                    return val
                print("  ⚠️  Must be a positive number.")
            except ValueError:
                print("  ⚠️  Please enter a number (e.g. 0.0005).")

    epochs             = _ask_int  ("Epochs",                               10)
    batch_size         = _ask_int  ("Batch size",                            2)
    lr                 = _ask_float("Learning rate",                    0.0005)
    adamw_wd           = _adamw_ask_weight_decay()
    checkpoint_interval = _ask_int ("Save checkpoint every N epochs",        2)

    # Early stopping
    print()
    target_loss = None
    while True:
        raw = input("Early-stop target loss (Enter to skip): ").strip()
        if raw == "":
            break
        try:
            target_loss = float(raw)
            if target_loss > 0:
                print(f"  ✓ Early stop at loss ≤ {target_loss:.4f}")
                break
            print("  ⚠️  Must be a positive number.")
            target_loss = None
        except ValueError:
            print("  ⚠️  Invalid number — try again or press Enter to skip.")

    # ── Step 3: Show full config summary and let user confirm ────────────────
    print("\n" + "="*60)
    print("📋 CONFIGURATION SUMMARY")
    print("="*60)
    print(f"  Ollama model       : {validator.model}")
    print(f"  Epochs             : {epochs}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Learning rate      : {lr}")
    print(f"  Weight decay       : {adamw_wd}")
    print(f"  Checkpoint every   : {checkpoint_interval} epoch(s)")
    print(f"  Early-stop loss    : {f'≤ {target_loss:.4f}' if target_loss else 'disabled'}")
    print("="*60)
    confirm = input("\nProceed with these settings? (y/n): ").strip().lower()
    if confirm != 'y':
        print("↩️  Cancelled — returning to menu.")
        return

    # ── Step 4: Load & validate question file ────────────────────────────────
    print("\n" + "="*60)
    print("📝 LOAD QUESTION FILE")
    print("="*60)
    print("\nQuestion file format:")
    print("  @Question 1@")
    print("  @Question 2@")
    print("  @Question 3@")
    print("\nUse @ symbols to mark start/end of each question.\n")

    file_path = input("Enter path to question file: ").strip()
    file_path = os.path.expanduser(file_path)

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    is_valid, message = QuestionFormatter.validate_question_file(file_path)
    if not is_valid:
        print(f"❌ Invalid file format: {message}")
        return
    print(f"✓ {message}")

    questions = OllamaDataGenerator.extract_questions_from_file(file_path)
    if not questions:
        print("❌ No questions found in file")
        return
    print(f"✓ Extracted {len(questions)} questions")

    # ── Step 5: Generate answers ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("🔄 GENERATING ANSWERS WITH OLLAMA")
    print("="*60)

    _ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = model_manager.model_dir / f"{model_name}_ollama_answers_{_ts}.log"

    generator = OllamaDataGenerator(validator)
    qa_pairs  = generator.generate_answers(
        questions, user_context=user_context, log_path=str(log_path)
    )

    if not qa_pairs:
        print(f"❌ No Q&A pairs generated — check the log: {log_path}")
        return
    print(f"\n✓ Generated {len(qa_pairs)}/{len(questions)} Q&A pairs")

    # ── Step 6: Save generated data ───────────────────────────────────────────
    print("\n" + "="*60)
    print("💾 SAVING GENERATED DATA")
    print("="*60)

    generated_file = model_manager.model_dir / f"generated_training_{_ts}.txt"
    generator.save_generated_data(qa_pairs, str(generated_file))
    print(f"✓ Training data saved : {generated_file}")
    print(f"✓ Answer log saved    : {log_path}")

    training_text = generator.format_for_training(qa_pairs)

    # ── Step 7: Fine-tune ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🏋️  FINE-TUNING IN PROGRESS")
    print("="*60 + "\n")

    # Sanitize: Ollama may return accented letters, curly quotes, or other
    # unicode characters that were never in the original training vocabulary.
    # tokenizer.sanitize() strips those characters and warns which ones were
    # removed, so training never crashes with a KeyError.
    print("Sanitizing generated text against vocabulary…")
    training_text = tokenizer.sanitize(training_text)

    if not training_text.strip():
        print("❌ After sanitization the training text is empty.")
        print("   All Ollama output contained characters outside the model vocabulary.")
        print("   Try training the base model on richer text so its vocabulary is larger.")
        return

    dataset      = TextDataset(training_text, tokenizer, seq_len=32)
    if len(dataset) == 0:
        print(f"❌ Generated text is too short to fine-tune on: only {len(dataset.tokens)} "
              f"tokens (need at least 33).")
        return
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ft_logger    = TrainingLogger(model_name)

    paused = train_model(
        model, train_loader,
        num_epochs=epochs,
        learning_rate=lr,
        weight_decay=adamw_wd,
        device=device,
        model_manager=model_manager,
        model_name=model_name,
        checkpoint_interval=checkpoint_interval,
        logger=ft_logger,
        target_loss=target_loss,
    )

    if paused:
        print("\n⏸️  Fine-tuning paused.")
    else:
        print("\n✓ Fine-tuning complete!")
        print(f"\n📊 Run summary:")
        print(f"  Questions processed  : {len(questions)}")
        print(f"  Q&A pairs generated  : {len(qa_pairs)}")
        print(f"  Epochs trained       : {epochs}")
        print(f"  Generated data       : {generated_file}")
        print(f"  Answer log           : {log_path}")

    ft_logger.display_summary()


# ============================================================================
# FINE-TUNE ON A SAVED CONVERSATIONAL DATASET FILE  (no Ollama needed)
# ============================================================================
def finetune_on_conversational_dataset(model, tokenizer, device='cpu',
                                       model_manager=None, model_name="my_model"):
    """Fine-tune directly on a user-supplied conversational/Q&A dataset file.

    Unlike `finetune_with_ollama`, no AI is queried to generate answers —
    you provide a file that already contains the conversations you want the
    model to learn from.

    Path memory
    ───────────
    The file/folder path is loaded through `_load_training_data("conversational")`,
    which keeps its own persistent registry at
        saved_models/conversational_dataset_registry.json
    so once you've entered a path the first time, every future run lets you
    just pick it from a list — no retyping required.

    Recommended (not required) format — same block style used elsewhere
    in this program:
        Q: <question>
        A: <answer>

        Q: <question>
        A: <answer>
    Plain conversational text also works; the model simply learns to predict
    the next character, so it absorbs whatever structure the file contains.
    """
    print("\n" + "="*60)
    print("💬 TRAIN ON CONVERSATIONAL DATASET FILE")
    print("="*60)
    print("\n📋 How it works:")
    print("  1. Point to your conversational/Q&A dataset file (path is remembered)")
    print("  2. Set training parameters up-front")
    print("  3. Fine-tune your model directly on that file — no Ollama needed")
    print("\n💡 Recommended format (matches the rest of this program):")
    print("     Q: <question>")
    print("     A: <answer>")
    print("   Plain conversational text works too — any structure is learned as-is.\n")

    # ── Step 1: Training configuration (collected up-front) ──────────────────
    print("="*60)
    print("⚙️  TRAINING CONFIGURATION")
    print("="*60)
    print("Enter the training parameters for fine-tuning.\n")

    def _ask_int(prompt, default):
        while True:
            raw = input(f"{prompt} (default {default}): ").strip()
            if raw == "":
                return default
            try:
                val = int(raw)
                if val > 0:
                    return val
                print("  ⚠️  Must be a positive integer.")
            except ValueError:
                print("  ⚠️  Please enter a whole number.")

    def _ask_float(prompt, default):
        while True:
            raw = input(f"{prompt} (default {default}): ").strip()
            if raw == "":
                return default
            try:
                val = float(raw)
                if val > 0:
                    return val
                print("  ⚠️  Must be a positive number.")
            except ValueError:
                print("  ⚠️  Please enter a number (e.g. 0.001).")

    epochs              = _ask_int  ("Epochs",                               10)
    batch_size          = _ask_int  ("Batch size",                            4)
    lr                  = _ask_float("Learning rate",                    0.001)
    adamw_wd            = _adamw_ask_weight_decay()
    checkpoint_interval = _ask_int  ("Save checkpoint every N epochs",        5)

    print()
    target_loss = None
    while True:
        raw = input("Early-stop target loss (Enter to skip): ").strip()
        if raw == "":
            break
        try:
            target_loss = float(raw)
            if target_loss > 0:
                print(f"  ✓ Early stop at loss ≤ {target_loss:.4f}")
                break
            print("  ⚠️  Must be a positive number.")
            target_loss = None
        except ValueError:
            print("  ⚠️  Invalid number — try again or press Enter to skip.")

    # ── Step 2: Show config summary and confirm ───────────────────────────────
    print("\n" + "="*60)
    print("📋 CONFIGURATION SUMMARY")
    print("="*60)
    print(f"  Epochs             : {epochs}")
    print(f"  Batch size         : {batch_size}")
    print(f"  Learning rate      : {lr}")
    print(f"  Weight decay       : {adamw_wd}")
    print(f"  Checkpoint every   : {checkpoint_interval} epoch(s)")
    print(f"  Early-stop loss    : {f'≤ {target_loss:.4f}' if target_loss else 'disabled'}")
    print("="*60)
    confirm = input("\nProceed with these settings? (y/n): ").strip().lower()
    if confirm != 'y':
        print("↩️  Cancelled — returning to menu.")
        return

    # ── Step 3: Load the conversational dataset (path remembered automatically) ─
    conv_text, data_source, data_type = _load_training_data("conversational")
    if not conv_text:
        return

    # ── Step 4: Sanitize against model vocabulary ──────────────────────────────
    print("\nSanitizing dataset against model vocabulary…")
    clean_text = tokenizer.sanitize(conv_text)
    if not clean_text.strip():
        print("❌ After sanitization the dataset is empty.")
        print("   Every character in the file is outside the model's vocabulary.")
        return

    coverage = len(clean_text) / max(1, len(conv_text)) * 100
    print(f"  Vocab coverage : {coverage:.1f}%  "
          f"({len(clean_text):,} / {len(conv_text):,} chars kept)")

    dataset = TextDataset(clean_text, tokenizer, seq_len=32)
    if len(dataset) == 0:
        print("❌ Dataset too short to train with seq_len=32 "
              "(need at least 33 tokens after encoding).")
        return
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Step 5: Fine-tune ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🏋️  FINE-TUNING ON CONVERSATIONAL DATASET")
    print("="*60)
    print(f"  Dataset   : {data_source}")
    print(f"  Chars     : {len(clean_text):,}   |   Sequences: {len(dataset):,}\n")

    tsm = TrainingStateManager(model_name, model_dir=str(model_manager.model_dir))
    tsm.save({
        "dataset_path": str(data_source), "dataset_type": data_type or "file",
        "batch_size": batch_size, "num_epochs": epochs,
        "epochs_trained": 0, "learning_rate": lr,
        "weight_decay": adamw_wd,
        "target_loss": target_loss, "checkpoint_interval": checkpoint_interval,
        "last_loss": None,
        "status": TrainingStateManager.STATUS_IN_PROGRESS,
        "created_at": str(datetime.now()),
    })

    ft_logger = TrainingLogger(model_name, model_dir=str(model_manager.model_dir))

    paused = train_model(
        model, train_loader,
        num_epochs=epochs,
        learning_rate=lr,
        weight_decay=adamw_wd,
        device=device,
        model_manager=model_manager,
        model_name=model_name,
        checkpoint_interval=checkpoint_interval,
        logger=ft_logger,
        target_loss=target_loss,
        state_manager=tsm,
    )

    if paused:
        print("\n⏸️  Fine-tuning paused.")
        print(f"📌 Resume by selecting this option again on '{model_name}'.")
    else:
        print("\n✓ Fine-tuning complete!")
        _print_training_summary(model, tokenizer, model_name, tsm)
        print(f"\n📊 Run summary:")
        print(f"  Dataset used : {data_source}")

    ft_logger.display_summary()


# ============================================================================
# QUESTION ANSWERING WITH FILE REFERENCES
# ============================================================================
def answer_with_references(model, tokenizer, device='cpu', max_length=150,
                          temperature=0.7):
    """
    Answer questions using files (images, text, PDFs, etc.) as context/references
    """
    model.eval()
    
    print("\n" + "="*60)
    print("❓ QUESTION ANSWERING WITH FILE REFERENCES")
    print("="*60)
    print("\n📂 Supported file types:")
    print("  📄 Text: .txt, .log, .md, .json, .csv")
    print("  🖼️  Images: .jpg, .png, .gif, .bmp, .webp")
    print("     (Install pytesseract for text extraction from images)")
    print("  📋 Documents: .pdf (requires PyPDF2)")
    print("\n💡 Tips:")
    print("  • Reference files act as context for your questions")
    print("  • The model will generate answers based on the file contents")
    print("  • Large files are automatically truncated to avoid overwhelming the model")
    print("  • Type 'quit' to exit")
    print("\nType 'quit' to exit.\n")
    
    # Keep track of loaded references
    current_references = ""
    reference_files_list = []
    
    with torch.no_grad():
        while True:
            print("\n" + "="*60)
            
            # Manage references
            if reference_files_list:
                print(f"📎 Current References: {', '.join(reference_files_list)}")
                manage_refs = input("Load new references? (y/n): ").strip().lower()
            else:
                manage_refs = input("Load reference files? (y/n): ").strip().lower()
            
            if manage_refs == 'y':
                print("\n📁 LOAD REFERENCE FILES")
                print("-"*60)
                print("Enter file paths (one per line, empty line to finish):")
                print("Examples:")
                print("  /path/to/document.pdf")
                print("  /path/to/image.jpg")
                print("  /path/to/notes.txt")
                
                file_paths = []
                while True:
                    file_path = input("File path: ").strip()
                    if not file_path:
                        break
                    
                    expanded_path = os.path.expanduser(file_path)
                    if Path(expanded_path).exists():
                        file_paths.append(expanded_path)
                        print(f"  ✓ Added: {Path(expanded_path).name}")
                    else:
                        print(f"  ❌ File not found: {file_path}")
                
                if file_paths:
                    print("\n🔄 Processing reference files...")
                    current_references = FileProcessor.load_multiple_files(file_paths)
                    reference_files_list = [Path(fp).name for fp in file_paths]
                    
                    # Manage context size intelligently
                    max_context_chars = 2000
                    if len(current_references) > max_context_chars:
                        print(f"⚠ References truncated from {len(current_references)} to {max_context_chars} characters")
                        current_references = current_references[:max_context_chars] + "\n\n[... content truncated due to length ...]"
                    
                    print(f"✓ Loaded {len(file_paths)} file(s)")
                    print(f"✓ Reference context size: {len(current_references)} characters")
            
            # Get question from user
            print("\n" + "-"*60)
            print("❓ YOUR QUESTION")
            print("-"*60)
            question = input("Ask a question: ").strip()
            
            if question.lower() == 'quit':
                print("\n✨ Goodbye! 👋")
                break
            
            if not question:
                print("⚠️  Please enter a question")
                continue
            
            # Build context-aware prompt
            if current_references:
                full_prompt = f"""📚 REFERENCE MATERIALS:
{current_references}

---

❓ QUESTION: {question}

📝 ANSWER:"""
            else:
                full_prompt = f"❓ Question: {question}\n\n📝 Answer:"
            
            # Encode prompt with smart character handling
            print("\n⏳ Generating answer...", end=" ")
            
            try:
                tokens_list, chars_skipped = tokenizer.encode_prompt(full_prompt)
                
                if chars_skipped > 0:
                    print(f"({chars_skipped} unknown chars handled)", end=" ")
                
                tokens = torch.tensor(tokens_list, dtype=torch.long).unsqueeze(0).to(device)
            except Exception as e:
                print(f"\n❌ Error processing prompt: {e}")
                continue
            
            # Generate answer
            try:
                for i in range(max_length):
                    logits = model(tokens)
                    next_logits = logits[0, -1, :]
                    next_logits = next_logits / temperature
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                    
                    # Keep reasonable length
                    if tokens.shape[1] > 200:
                        tokens = tokens[:, -200:]
                
                # Decode and display
                generated_tokens = tokens[0].cpu().numpy().tolist()
                generated_text = tokenizer.decode(generated_tokens)
                
                print("✓")
                print("\n" + "="*60)
                print("📝 ANSWER:")
                print("="*60)
                print(generated_text)
                print("="*60)
                
                # Show reference status
                if current_references:
                    print(f"\n📎 References used: {len(reference_files_list)} file(s)")
                    print(f"   Files: {', '.join(reference_files_list)}")
                else:
                    print("\n📎 References used: None (model knowledge only)")
                
                # Options
                print("\n🔄 Next:")
                print("   • Ask another question (references stay loaded)")
                print("   • Change references (load different files)")
                print("   • Type 'quit' to exit")
                
            except Exception as e:
                print(f"\n❌ Error generating answer: {e}")
                continue


# ============================================================================
# SELF-IMPROVING Q&A WITH AUTOMATED FEEDBACK
# ============================================================================
def self_improving_qa(model, tokenizer, device='cpu', model_manager=None, 
                     model_name="my_model", max_length=150, temperature=0.7,
                     user_context="", preset_target_avg=None):
    """
    Interactive Q&A with automated evaluation and self-improvement through fine-tuning
    """
    model.eval()
    
    print("\n" + "="*60)
    print("🔄 SELF-IMPROVING Q&A WITH AUTOMATED FEEDBACK")
    print("="*60)
    print("\n💡 How it works:")
    print("  1. You ask a question (or load from folder)")
    print("  2. Model generates an answer")
    print("  3. Ollama evaluates the answer (scores 1-10)")
    print("  4. Poor answers are collected")
    print("  5. You can fine-tune on collected feedback")
    print("\n⚠️  Requires: Ollama server running with a model selected")
    if user_context.strip():
        print(f"\n📋 Active context: {user_context[:120]}{'…' if len(user_context) > 120 else ''}")
    print("Type 'quit' to exit, 'feedback' to see stats, 'improve' to fine-tune\n")
    
    # Initialize logger, feedback system, and config
    logger = TrainingLogger(model_name)
    feedback_collector = FeedbackCollector(model_name)
    config_manager = ConfigManager(model_name)
    
    # Check Ollama availability
    print("Initializing Ollama evaluator...")
    evaluator_validator = OllamaValidator()
    
    if not evaluator_validator.available:
        print("❌ Ollama server not available")
        return
    
    if not evaluator_validator.list_models():
        print("❌ No evaluator model selected")
        return
    
    evaluator = SelfImprovingEvaluator(evaluator_validator, user_context=user_context)
    print(f"✓ Using '{evaluator_validator.model}' as evaluator\n")

    # ── Target average score ──────────────────────────────────────────
    print("="*60)
    print("🎯 TARGET AVERAGE SCORE")
    print("="*60)
    print("Ollama will rate every answer from 0 to 10.")
    print("A running average is calculated across all answers.")
    print("When that average reaches your target, training pauses")
    print("and you choose whether to keep going or save and quit.")
    print()
    if preset_target_avg is not None:
        _target_input = str(preset_target_avg)      # already entered in the menu
        print(f"Using the target you entered in the menu: {_target_input}")
    else:
        _target_input = input(
            "Enter target average score (0–10), or press Enter to skip: "
        ).strip()
    target_avg_score = None
    if _target_input:
        try:
            target_avg_score = float(_target_input)
            if not 0 <= target_avg_score <= 10:
                raise ValueError
            print(f"✓ Training will pause when average score reaches {target_avg_score:.1f}/10")
        except ValueError:
            print("⚠️  Invalid value — no score target set.")
            target_avg_score = None
    else:
        print("⏭️  No target set — session runs until you type 'quit'.")
    # ─────────────────────────────────────────────────────────────────
    
    # Question folder management
    print("="*60)
    print("📁 QUESTION SOURCE CONFIGURATION")
    print("="*60)
    
    loaded_questions = []
    use_question_folder = False
    question_index = 0
    
    # Check if user has previous folder setting
    prev_folder = config_manager.get_question_folder()
    if prev_folder:
        print(f"\n📌 Remembered folder: {prev_folder}")
        use_prev = input("Use this folder again? (y/n): ").strip().lower()
        
        if use_prev == 'y':
            valid, msg = QuestionLoader.validate_folder(prev_folder)
            if valid:
                loaded_questions = QuestionLoader.load_questions_from_folder(prev_folder)
                use_question_folder = True
                print(f"✓ {msg}")
                print(f"✓ Loaded {len(loaded_questions)} questions")
            else:
                print(f"❌ {msg}")
    
    # Ask if user wants to load a new folder
    if not use_question_folder:
        print("\nDo you want to:")
        print("  1. Ask questions manually")
        print("  2. Load questions from a folder")
        
        choice = input("Select (1 or 2): ").strip()
        
        if choice == "2":
            folder_path = input("\nEnter folder path with question files (.txt or .md): ").strip()
            folder_path = os.path.expanduser(folder_path)
            
            valid, msg = QuestionLoader.validate_folder(folder_path)
            if valid:
                loaded_questions = QuestionLoader.load_questions_from_folder(folder_path)
                config_manager.set_question_folder(folder_path)
                use_question_folder = True
                print(f"\n✓ {msg}")
                print(f"✓ Loaded {len(loaded_questions)} questions from folder")
                print(f"✓ Folder setting saved for next time")
            else:
                print(f"❌ {msg}")
                use_question_folder = False
    
    print("\n" + "="*60)
    print("Starting Q&A session...")
    if use_question_folder:
        print(f"📂 Using {len(loaded_questions)} questions from folder")
        print()
        print("Mode:")
        print("  1. Manual   — you type 'next' between each question")
        print("  2. Auto     — questions run one after another automatically")
        mode_choice = input("Select mode (1 or 2, default 1): ").strip()
        auto_advance = (mode_choice == "2")
        if auto_advance:
            print("✓ Auto mode — questions will run automatically.")
            print("  Type 'pause' + Enter at any prompt to switch back to manual.")
        else:
            print("Type 'next' to use next question, 'manual' for manual input, 'quit' to exit")
    else:
        auto_advance = False
        print("📝 Manual question input mode")
        print("Type 'quit' to exit")
    print("="*60 + "\n")
    
    current_references = ""
    reference_files_list = []
    
    with torch.no_grad():
        while True:
            print("\n" + "="*60)

            # ── Auto-advance: pick next question without waiting for input ──
            if auto_advance and use_question_folder and question_index < len(loaded_questions):
                question = loaded_questions[question_index].strip()
                question_index += 1
                print(f"📂 Auto [{question_index}/{len(loaded_questions)}]: {question[:80]}")
                # Still let the user interrupt between questions
                try:
                    interrupt = input("  ↳ Press Enter to continue, 'pause' to go manual, 'quit' to exit: ").strip().lower()
                except EOFError:
                    interrupt = ""
                if interrupt == "quit":
                    user_input = "quit"
                    auto_advance = False
                elif interrupt == "pause":
                    auto_advance = False
                    print("✓ Switched to manual mode.")
                    continue
                else:
                    user_input = question    # skip normal input(); go straight to generation
            elif auto_advance and use_question_folder and question_index >= len(loaded_questions):
                print("✅ All questions in folder completed.")
                auto_advance = False
                user_input = input("Question (or 'feedback'/'improve'/'quit'): ").strip()
            # ──────────────────────────────────────────────────────────────
            elif use_question_folder and question_index < len(loaded_questions):
                print(f"📂 Question {question_index + 1}/{len(loaded_questions)}")
                user_input = input("Question (or 'next'/'manual'/'feedback'/'improve'/'quit'): ").strip()
            else:
                user_input = input("Question (or 'feedback'/'improve'/'quit'): ").strip()
            
            if user_input.lower() == 'quit':
                print("\n✨ Goodbye! Final statistics:")
                stats = feedback_collector.get_statistics()
                if stats['total_answers'] > 0:
                    print(f"  Total answers evaluated: {stats['total_answers']}")
                    print(f"  Average score: {stats['average_score']:.2f}/10")
                    print(f"  High-quality answers: {stats['high_score_count']}")
                    print(f"  Low-quality answers needing improvement: {stats['low_score_count']}")
                break
            
            elif user_input.lower() == 'feedback':
                stats = feedback_collector.get_statistics()
                print("\n📊 FEEDBACK STATISTICS")
                print("="*60)
                if stats['total_answers'] == 0:
                    print("No answers evaluated yet")
                else:
                    print(f"Total answers: {stats['total_answers']}")
                    print(f"Average score: {stats['average_score']:.2f}/10")
                    print(f"Best score: {stats['max_score']:.1f}/10")
                    print(f"Worst score: {stats['min_score']:.1f}/10")
                    print(f"High quality (≥6/10): {stats['high_score_count']}")
                    print(f"Needs improvement (<6/10): {stats['low_score_count']}")
                    
                    # Show low-scoring answers
                    low_scores = feedback_collector.get_low_scoring_answers()
                    if low_scores:
                        print(f"\n📝 Low-scoring answers to learn from:")
                        for i, item in enumerate(low_scores[:5], 1):
                            print(f"\n  {i}. Q: {item['question'][:50]}...")
                            print(f"     Score: {item['score']:.1f}/10")
                            print(f"     Issue: {item['comment'][:60]}...")
                continue
            
            elif user_input.lower() == 'manual':
                use_question_folder = False
                auto_advance = False
                question_index = 0
                print("✓ Switched to manual question input")
                continue

            elif user_input.lower() == 'auto':
                if use_question_folder and loaded_questions:
                    auto_advance = True
                    print("✓ Switched to auto mode.")
                else:
                    print("⚠️  No question folder loaded — cannot enable auto mode.")
                continue
            
            elif user_input.lower() == 'next' and use_question_folder:
                if question_index < len(loaded_questions):
                    question = loaded_questions[question_index].strip()
                    question_index += 1
                    print(f"✓ Using question {question_index}/{len(loaded_questions)}")
                else:
                    print("✓ All questions from folder used! Starting over...")
                    question_index = 0
                    if len(loaded_questions) > 0:
                        question = loaded_questions[question_index].strip()
                        question_index += 1
                    else:
                        print("❌ No questions available")
                        continue
            
            elif user_input.lower() == 'improve':
                low_scores = feedback_collector.get_low_scoring_answers()
                if not low_scores:
                    print("✓ No low-scoring answers to learn from yet!")
                    continue
                
                print(f"\n🎓 SELF-IMPROVEMENT FINE-TUNING")
                print("="*60)
                print(f"Found {len(low_scores)} low-scoring answers to learn from")
                print("\nHow many epochs to fine-tune on feedback?")
                print("(Recommended: 5-15, more if many wrong answers)")
                
                epochs_input = input("Enter epochs: ").strip()
                try:
                    improve_epochs = int(epochs_input)
                    if improve_epochs <= 0:
                        print("Invalid input")
                        continue
                except ValueError:
                    print("Invalid input")
                    continue
                
                # Create training data from feedback
                training_data = feedback_collector.export_training_data()
                
                if not training_data.strip():
                    print("No training data available")
                    continue
                
                print(f"\n🔄 Fine-tuning on {len(low_scores)} feedback examples for {improve_epochs} epochs...")
                print("This may take a moment...\n")
                
                # Create temporary dataset from feedback
                train_dataset = TextDataset(training_data, tokenizer, seq_len=100)
                if len(train_dataset) == 0:
                    # BPE packs several characters into one token, so a small
                    # feedback set can hold fewer than 100 tokens — shorten the window.
                    train_dataset = TextDataset(
                        training_data, tokenizer,
                        seq_len=max(2, len(train_dataset.tokens) // 2))
                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
                
                # Fine-tune
                optimizer = _adamw_build_optimizer(model, lr=0.0003)
                criterion = nn.CrossEntropyLoss()
                model.train()
                torch.set_grad_enabled(True)   # this whole loop runs inside torch.no_grad(); training needs gradients
                
                initial_loss = None
                final_loss = None
                
                for epoch in range(improve_epochs):
                    total_loss = 0
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        logits = logits.view(-1, model.vocab_size)
                        y = y.view(-1)
                        loss = criterion(logits, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(train_loader)
                    pct = (epoch + 1) / improve_epochs * 100
                    bar_w = 25
                    bar = "█" * int(bar_w * pct / 100) + "░" * (bar_w - int(bar_w * pct / 100))
                    print(f"  Epoch {epoch + 1}/{improve_epochs}  [{bar}] {pct:.1f}%  ({epoch + 1}/{improve_epochs} epochs)  Loss: {avg_loss:.4f}")

                    if epoch == 0:
                        initial_loss = avg_loss
                    final_loss = avg_loss

                # Save improved model
                if model_manager:
                    model_manager.save_model(model, model_name)
                    print(f"✓ Model improved and saved: {model_name}")

                # Log the self-improvement fine-tuning session
                logger.log_training_session(
                    epochs=improve_epochs,
                    learning_rate=0.0003,
                    batch_size=2,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    session_type="self-improvement-feedback"
                )

                model.eval()      # back to eval mode (dropout off) before answering again
                torch.set_grad_enabled(False)  # back to inference mode for the rest of the session
                print("✓ Self-improvement complete!")
                continue

            else:
                # A typed question (auto / 'next' modes have already set `question`)
                question = user_input
                if not question:
                    print("⚠️  Please enter a question")
                    continue

            # Regular Q&A with evaluation
            # Build prompt
            full_prompt = f"Q: {question}\nA:"
            
            # Encode prompt
            print("\n⏳ Generating answer...", end=" ")
            
            try:
                tokens_list, _ = tokenizer.encode_prompt(full_prompt)
                
                tokens = torch.tensor(tokens_list, dtype=torch.long).unsqueeze(0).to(device)
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
            
            # Generate
            for i in range(max_length):
                logits = model(tokens)
                next_logits = logits[0, -1, :]
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                if tokens.shape[1] > 200:
                    tokens = tokens[:, -200:]
            
            generated_tokens = tokens[0].cpu().numpy().tolist()
            generated_text = tokenizer.decode(generated_tokens)
            print("✓")
            
            print("\n" + "="*60)
            print("📝 ANSWER:")
            print("="*60)
            print(generated_text)
            print("="*60)
            
            # Evaluate answer
            print("\n🔍 Evaluating with Ollama...", end=" ")
            score, comment = evaluator.evaluate_answer(question, generated_text)
            
            if score is not None:
                print(f"✓\n")

                # Record feedback first so averages include this answer
                feedback_collector.add_feedback(question, generated_text, score, comment)
                logger.log_qa(question, generated_text, score=score, source="self-improving-qa")

                stats = feedback_collector.get_statistics()
                avg   = stats['average_score']
                total = stats['total_answers']

                # ── Score display ─────────────────────────────────────
                score_fill = int(round(score / 10 * 20))
                avg_fill   = int(round(avg   / 10 * 20))
                score_bar  = "█" * score_fill + "░" * (20 - score_fill)
                avg_bar    = "█" * avg_fill   + "░" * (20 - avg_fill)

                print(f"⭐ Score   [{score_bar}] {score:.1f}/10")
                print(f"📊 Average [{avg_bar}] {avg:.2f}/10  (over {total} answer{'s' if total != 1 else ''})")
                print(f"📌 Feedback: {comment}")

                if target_avg_score is not None:
                    progress_to_target = min(avg / target_avg_score * 100, 100)
                    t_fill = int(progress_to_target / 100 * 20)
                    t_bar  = "█" * t_fill + "░" * (20 - t_fill)
                    print(f"🎯 Target  [{t_bar}] {progress_to_target:.1f}% of {target_avg_score:.1f}")

                if score >= 8:
                    print("✅ Excellent answer!")
                elif score >= 6:
                    print("👍 Good answer")
                else:
                    print("⚠️  Needs improvement — added to learning dataset")
                    print(f"   Tip: type 'improve' to fine-tune on "
                          f"{len(feedback_collector.get_low_scoring_answers())} low-scoring answers")
                # ─────────────────────────────────────────────────────

                # ── Check target average ──────────────────────────────
                if target_avg_score is not None and avg >= target_avg_score:
                    print()
                    print("="*60)
                    print(f"🏆  TARGET REACHED!  Average: {avg:.2f}/10  (target: {target_avg_score:.1f})")
                    print(f"    Answers evaluated: {total}")
                    print("="*60)
                    pause_choice = input(
                        "Press Enter to keep training  —OR—  type 'quit' to save and exit: "
                    ).strip().lower()
                    if pause_choice in ('quit', 'q', 'exit'):
                        if model_manager:
                            model_manager.save_model(model, model_name)
                            print(f"\n✓ Model saved: {model_name}")
                        print(f"\n📊 Final stats:  {total} answers  |  avg {avg:.2f}/10"
                              f"  |  best {stats['max_score']:.1f}  |  worst {stats['min_score']:.1f}")
                        print("\nGoodbye! 👋")
                        return
                    else:
                        print(f"✓ Continuing — target stays at {target_avg_score:.1f}/10\n")
                # ─────────────────────────────────────────────────────
            else:
                print("❌ Could not evaluate")
            
            # Show next action suggestions
            if use_question_folder and not auto_advance:
                print(f"\n🔄 Next: 'next' for Q{question_index + 1}/{len(loaded_questions)}"
                      f" | 'auto' for auto mode | 'improve' to fine-tune | 'quit'")
            elif auto_advance:
                pass   # auto mode prints its own continue prompt at the top of the loop
            else:
                print("\n🔄 Next: Ask another question | 'improve' to fine-tune | 'quit'")


    
    return   # Session over.  Everything below is an older duplicate of the session loop above
             # (its `def` line was lost); it used to run right after 'quit' and start a second session.

    # Initialize question loader and queue
    question_queue = []
    saved_folder = logger.get_question_folder()
    
    # Ask about question folder
    print("\n" + "="*60)
    print("📂 QUESTION FOLDER (Optional)")
    print("="*60)
    
    if saved_folder and Path(saved_folder).exists():
        print(f"✓ Found saved folder: {saved_folder}")
        use_saved = input("Use this folder for questions? (y/n): ").strip().lower()
        
        if use_saved == 'y':
            print("📥 Loading questions from folder...")
            question_queue = QuestionLoader.load_questions_from_folder(saved_folder)
            if question_queue:
                print(f"✓ Loaded {len(question_queue)} questions from folder")
            else:
                print("⚠️  No questions found in folder")
        else:
            load_new = input("Load questions from a different folder? (y/n): ").strip().lower()
            if load_new == 'y':
                folder_path = input("Enter folder path (or press Enter to skip): ").strip()
                if folder_path:
                    folder_path = Path(folder_path).expanduser()
                    if folder_path.exists():
                        print("📥 Loading questions from folder...")
                        question_queue = QuestionLoader.load_questions_from_folder(folder_path)
                        if question_queue:
                            logger.save_question_folder(folder_path)
                            print(f"✓ Loaded {len(question_queue)} questions")
                            print(f"✓ Saved folder for future use")
                        else:
                            print("⚠️  No questions found in folder")
                    else:
                        print("❌ Folder not found")
    else:
        load_new = input("Load questions from a folder? (y/n): ").strip().lower()
        if load_new == 'y':
            folder_path = input("Enter folder path (or press Enter to skip): ").strip()
            if folder_path:
                folder_path = Path(folder_path).expanduser()
                if folder_path.exists():
                    print("📥 Loading questions from folder...")
                    question_queue = QuestionLoader.load_questions_from_folder(folder_path)
                    if question_queue:
                        logger.save_question_folder(folder_path)
                        print(f"✓ Loaded {len(question_queue)} questions")
                        print(f"✓ Saved folder for future use")
                    else:
                        print("⚠️  No questions found in folder")
                else:
                    print("❌ Folder not found")
    
    print("\n💡 Commands:")
    print("  • Type a question to answer it")
    print("  • Type 'next' to use next question from folder")
    print("  • Type 'feedback' to see statistics")
    print("  • Type 'improve' to fine-tune")
    print("  • Type 'quit' to exit\n")
    
    current_references = ""
    reference_files_list = []
    question_index = 0
    
    with torch.no_grad():
        while True:
            print("\n" + "="*60)
            
            # Display current question queue status
            if question_queue:
                print(f"📋 Questions in queue: {len(question_queue) - question_index}")
            
            # Check for special commands
            user_input = input("Question (or 'next'/'feedback'/'improve'/'quit'): ").strip()
            
            # Handle 'next' command to get next question from folder
            if user_input.lower() == 'next':
                if question_queue and question_index < len(question_queue):
                    question = question_queue[question_index]
                    question_index += 1
                    print(f"\n📋 Using question {question_index}/{len(question_queue)} from folder:")
                    print(f"   {question}")
                    user_input = question
                elif not question_queue:
                    print("⚠️  No questions loaded from folder")
                    continue
                else:
                    print("⚠️  All questions from folder have been used")
                    continue
            
            if user_input.lower() == 'quit':
                print("\n✨ Goodbye! Final statistics:")
                stats = feedback_collector.get_statistics()
                if stats['total_answers'] > 0:
                    print(f"  Total answers evaluated: {stats['total_answers']}")
                    print(f"  Average score: {stats['average_score']:.2f}/10")
                    print(f"  High-quality answers: {stats['high_score_count']}")
                    print(f"  Low-quality answers needing improvement: {stats['low_score_count']}")
                break
            
            elif user_input.lower() == 'feedback':
                stats = feedback_collector.get_statistics()
                print("\n📊 FEEDBACK STATISTICS")
                print("="*60)
                if stats['total_answers'] == 0:
                    print("No answers evaluated yet")
                else:
                    print(f"Total answers: {stats['total_answers']}")
                    print(f"Average score: {stats['average_score']:.2f}/10")
                    print(f"Best score: {stats['max_score']:.1f}/10")
                    print(f"Worst score: {stats['min_score']:.1f}/10")
                    print(f"High quality (≥6/10): {stats['high_score_count']}")
                    print(f"Needs improvement (<6/10): {stats['low_score_count']}")
                    
                    # Show low-scoring answers
                    low_scores = feedback_collector.get_low_scoring_answers()
                    if low_scores:
                        print(f"\n📝 Low-scoring answers to learn from:")
                        for i, item in enumerate(low_scores[:5], 1):
                            print(f"\n  {i}. Q: {item['question'][:50]}...")
                            print(f"     Score: {item['score']:.1f}/10")
                            print(f"     Issue: {item['comment'][:60]}...")
                continue
            
            elif user_input.lower() == 'improve':
                low_scores = feedback_collector.get_low_scoring_answers()
                if not low_scores:
                    print("✓ No low-scoring answers to learn from yet!")
                    continue
                
                print(f"\n🎓 SELF-IMPROVEMENT FINE-TUNING")
                print("="*60)
                print(f"Found {len(low_scores)} low-scoring answers to learn from")
                print("\nHow many epochs to fine-tune on feedback?")
                print("(Recommended: 5-15, more if many wrong answers)")
                
                epochs_input = input("Enter epochs: ").strip()
                try:
                    improve_epochs = int(epochs_input)
                    if improve_epochs <= 0:
                        print("Invalid input")
                        continue
                except ValueError:
                    print("Invalid input")
                    continue
                
                # Create training data from feedback
                training_data = feedback_collector.export_training_data()
                
                if not training_data.strip():
                    print("No training data available")
                    continue
                
                print(f"\n🔄 Fine-tuning on {len(low_scores)} feedback examples for {improve_epochs} epochs...")
                print("This may take a moment...\n")
                
                # Create temporary dataset from feedback
                train_dataset = TextDataset(training_data, tokenizer, seq_len=100)
                if len(train_dataset) == 0:
                    # BPE packs several characters into one token, so a small
                    # feedback set can hold fewer than 100 tokens — shorten the window.
                    train_dataset = TextDataset(
                        training_data, tokenizer,
                        seq_len=max(2, len(train_dataset.tokens) // 2))
                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
                
                # Fine-tune
                optimizer = _adamw_build_optimizer(model, lr=0.0003)
                criterion = nn.CrossEntropyLoss()
                model.train()
                torch.set_grad_enabled(True)   # this whole loop runs inside torch.no_grad(); training needs gradients
                
                initial_loss = None
                final_loss = None
                
                for epoch in range(improve_epochs):
                    total_loss = 0
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        logits = logits.view(-1, model.vocab_size)
                        y = y.view(-1)
                        loss = criterion(logits, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(train_loader)
                    pct = (epoch + 1) / improve_epochs * 100
                    bar_w = 25
                    bar = "█" * int(bar_w * pct / 100) + "░" * (bar_w - int(bar_w * pct / 100))
                    print(f"  Epoch {epoch + 1}/{improve_epochs}  [{bar}] {pct:.1f}%  ({epoch + 1}/{improve_epochs} epochs)  Loss: {avg_loss:.4f}")
                    
                    if epoch == 0:
                        initial_loss = avg_loss
                    final_loss = avg_loss
                
                # Save improved model
                if model_manager:
                    model_manager.save_model(model, model_name)
                    print(f"✓ Model improved and saved: {model_name}")
                
                # Log the self-improvement fine-tuning session
                logger.log_training_session(
                    epochs=improve_epochs,
                    learning_rate=0.0003,
                    batch_size=2,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    session_type="self-improvement-feedback"
                )
                
                model.eval()
                torch.set_grad_enabled(False)  # back to inference mode for the rest of the session
                print("✓ Self-improvement complete!")
                continue
            
            # Regular Q&A with evaluation
            question = user_input
            
            if not question:
                print("⚠️  Please enter a question")
                continue
            
            # Build prompt
            full_prompt = f"Q: {question}\nA:"
            
            # Encode prompt
            print("\n⏳ Generating answer...", end=" ")
            
            try:
                tokens_list, _ = tokenizer.encode_prompt(full_prompt)
                
                tokens = torch.tensor(tokens_list, dtype=torch.long).unsqueeze(0).to(device)
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
            
            # Generate
            for i in range(max_length):
                logits = model(tokens)
                next_logits = logits[0, -1, :]
                next_logits = next_logits / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                if tokens.shape[1] > 200:
                    tokens = tokens[:, -200:]
            
            generated_tokens = tokens[0].cpu().numpy().tolist()
            generated_text = tokenizer.decode(generated_tokens)
            print("✓")
            
            print("\n" + "="*60)
            print("📝 ANSWER:")
            print("="*60)
            print(generated_text)
            print("="*60)
            
            # Evaluate answer
            print("\n🔍 Evaluating with Ollama...", end=" ")
            score, comment = evaluator.evaluate_answer(question, generated_text)
            
            if score is not None:
                print(f"✓\n")

                # Record feedback first so averages include this answer
                feedback_collector.add_feedback(question, generated_text, score, comment)
                logger.log_qa(question, generated_text, score=score, source="self-improving-qa")

                stats = feedback_collector.get_statistics()
                avg   = stats['average_score']
                total = stats['total_answers']

                # ── Score display ─────────────────────────────────────
                score_fill = int(round(score / 10 * 20))
                avg_fill   = int(round(avg   / 10 * 20))
                score_bar  = "█" * score_fill + "░" * (20 - score_fill)
                avg_bar    = "█" * avg_fill   + "░" * (20 - avg_fill)

                print(f"⭐ Score   [{score_bar}] {score:.1f}/10")
                print(f"📊 Average [{avg_bar}] {avg:.2f}/10  (over {total} answer{'s' if total != 1 else ''})")
                print(f"📌 Feedback: {comment}")

                if target_avg_score is not None:
                    progress_to_target = min(avg / target_avg_score * 100, 100)
                    t_fill = int(progress_to_target / 100 * 20)
                    t_bar  = "█" * t_fill + "░" * (20 - t_fill)
                    print(f"🎯 Target  [{t_bar}] {progress_to_target:.1f}% of {target_avg_score:.1f}")

                if score >= 8:
                    print("✅ Excellent answer!")
                elif score >= 6:
                    print("👍 Good answer")
                else:
                    print("⚠️  Needs improvement — added to learning dataset")
                    print(f"   Tip: type 'improve' to fine-tune on "
                          f"{len(feedback_collector.get_low_scoring_answers())} low-scoring answers")
                # ─────────────────────────────────────────────────────

                # ── Check target average ──────────────────────────────
                if target_avg_score is not None and avg >= target_avg_score:
                    print()
                    print("="*60)
                    print(f"🏆  TARGET REACHED!  Average: {avg:.2f}/10  (target: {target_avg_score:.1f})")
                    print(f"    Answers evaluated: {total}")
                    print("="*60)
                    pause_choice = input(
                        "Press Enter to keep training  —OR—  type 'quit' to save and exit: "
                    ).strip().lower()
                    if pause_choice in ('quit', 'q', 'exit'):
                        if model_manager:
                            model_manager.save_model(model, model_name)
                            print(f"\n✓ Model saved: {model_name}")
                        print(f"\n📊 Final stats:  {total} answers  |  avg {avg:.2f}/10"
                              f"  |  best {stats['max_score']:.1f}  |  worst {stats['min_score']:.1f}")
                        print("\nGoodbye! 👋")
                        return
                    else:
                        print(f"✓ Continuing — target stays at {target_avg_score:.1f}/10\n")
                # ─────────────────────────────────────────────────────
            else:
                print("❌ Could not evaluate")
            
            print("\n🔄 Next: Ask another question, or type 'improve' to fine-tune")


# ============================================================================
# OLLAMA-COMPATIBLE LOCAL SERVER
# ============================================================================
class OllamaCompatibleServer:
    """
    Serve a trained SimpleLLM through an HTTP API that is wire-compatible
    with the Ollama REST API.  Applications that already talk to Ollama
    (including this program's own Ollama calls) can point to this server
    instead of a real Ollama instance.

    Endpoints implemented
    ─────────────────────
    GET  /api/tags          → list available models
    POST /api/generate      → text generation  (streaming + non-streaming)
    POST /api/chat          → chat format wrapper around /api/generate
    POST /api/show          → model metadata
    GET  /api/version       → fake version string
    """

    VERSION = "0.1.0-tiny-llm"

    def __init__(self, model, tokenizer, model_name: str, device: str = 'cpu',
                 port: int = 11435, temperature: float = 0.8, max_tokens: int = 200):
        self.model       = model
        self.tokenizer   = tokenizer
        self.model_name  = model_name
        self.device      = device
        self.port        = port
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._server     = None
        self._thread     = None
        model.eval()

    # ── Generation ───────────────────────────────────────────────────────────
    def _generate(self, prompt: str, max_tokens: int = None,
                  temperature: float = None) -> str:
        """Run the trained model and return a generated string."""
        import torch, math
        temperature = temperature if temperature is not None else self.temperature
        max_tokens  = max_tokens  if max_tokens  is not None else self.max_tokens

        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            return ""

        generated = list(tokens)
        with torch.no_grad():
            for _ in range(max_tokens):
                x    = torch.tensor([generated[-100:]], dtype=torch.long,
                                    device=self.device)
                logits = self.model(x)[0, -1, :]          # (vocab,)

                if temperature <= 0:
                    next_id = int(logits.argmax())
                else:
                    # Temperature scaling + softmax sampling
                    logits  = logits / temperature
                    logits  = logits - logits.max()        # numerical stability
                    probs   = torch.exp(logits)
                    probs  /= probs.sum()
                    next_id = int(torch.multinomial(probs, 1).item())

                generated.append(next_id)
                ch = self.tokenizer.decode([next_id])
                # A BPE token can span several characters, so test whether the
                # token contains a newline or ends a sentence.
                if (('\n' in ch or ch.endswith(('.', '?', '!')))
                        and len(generated) > len(tokens) + 10):
                    break

        return self.tokenizer.decode(generated[len(tokens):])

    # ── HTTP handler ─────────────────────────────────────────────────────────
    def _make_handler(self):
        """Return a BaseHTTPRequestHandler class bound to this server instance."""
        from http.server import BaseHTTPRequestHandler
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            # Silence default log_message so training output isn't drowned
            def log_message(self, fmt, *args):
                pass

            def _send_json(self, data: dict, status: int = 200):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_ndjson_line(self, data: dict):
                line = (json.dumps(data) + '\n').encode()
                self.wfile.write(line)
                self.wfile.flush()

            def _body(self) -> dict:
                length = int(self.headers.get('Content-Length', 0))
                raw    = self.rfile.read(length) if length else b'{}'
                try:
                    return json.loads(raw)
                except Exception:
                    return {}

            # ── GET ──────────────────────────────────────────────────────
            def do_GET(self):
                if self.path == '/api/tags':
                    now = datetime.now().isoformat() + 'Z'
                    self._send_json({'models': [{
                        'name'       : server_ref.model_name,
                        'model'      : server_ref.model_name,
                        'modified_at': now,
                        'size'       : 0,
                        'digest'     : 'tiny-llm-local',
                        'details'    : {'family': 'tiny-llm',
                                        'parameter_size': '~1M',
                                        'quantization_level': 'none'},
                    }]})

                elif self.path == '/api/version':
                    self._send_json({'version': OllamaCompatibleServer.VERSION})

                else:
                    self._send_json({'error': 'not found'}, 404)

            # ── POST ─────────────────────────────────────────────────────
            def do_POST(self):
                body = self._body()

                if self.path == '/api/show':
                    self._send_json({
                        'modelfile'  : f'# Tiny LLM — {server_ref.model_name}',
                        'parameters' : '',
                        'template'   : '{{ .Prompt }}',
                        'details'    : {'family': 'tiny-llm'},
                    })

                elif self.path in ('/api/generate', '/api/chat'):
                    # Unify chat → generate
                    if self.path == '/api/chat':
                        msgs   = body.get('messages', [])
                        prompt = '\n'.join(
                            f"{m.get('role','user')}: {m.get('content','')}"
                            for m in msgs)
                    else:
                        prompt = body.get('prompt', '')

                    temperature = float(body.get('options', {}).get(
                        'temperature', server_ref.temperature))
                    max_tokens  = int(body.get('options', {}).get(
                        'num_predict', server_ref.max_tokens))
                    stream      = body.get('stream', True)
                    now         = datetime.now().isoformat() + 'Z'
                    response_text = server_ref._generate(
                        prompt, max_tokens=max_tokens, temperature=temperature)

                    if stream:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/x-ndjson')
                        self.end_headers()
                        # Stream word by word for a realistic feel
                        words = response_text.split(' ')
                        for i, word in enumerate(words):
                            chunk = word + (' ' if i < len(words) - 1 else '')
                            if self.path == '/api/chat':
                                data = {'model': server_ref.model_name,
                                        'created_at': now,
                                        'message': {'role': 'assistant',
                                                    'content': chunk},
                                        'done': False}
                            else:
                                data = {'model': server_ref.model_name,
                                        'created_at': now,
                                        'response': chunk,
                                        'done': False}
                            self._send_ndjson_line(data)
                        # Final done=True line
                        done_data = {'model': server_ref.model_name,
                                     'created_at': now,
                                     'done': True,
                                     'done_reason': 'stop'}
                        if self.path == '/api/chat':
                            done_data['message'] = {'role': 'assistant',
                                                    'content': ''}
                        else:
                            done_data['response'] = ''
                        self._send_ndjson_line(done_data)
                    else:
                        base = {'model'     : server_ref.model_name,
                                'created_at': now,
                                'done'      : True,
                                'done_reason': 'stop'}
                        if self.path == '/api/chat':
                            base['message'] = {'role': 'assistant',
                                               'content': response_text}
                        else:
                            base['response'] = response_text
                        self._send_json(base)

                else:
                    self._send_json({'error': 'endpoint not implemented'}, 404)

        return _Handler

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        """Start the server in a background daemon thread."""
        import threading
        from http.server import HTTPServer
        handler   = self._make_handler()
        self._server = HTTPServer(('', self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        print(f"\n✅ Ollama-compatible server started!")
        print(f"   URL  : http://localhost:{self.port}")
        print(f"   Model: {self.model_name}")
        print(f"\n   To use in this program:  set Ollama URL to http://localhost:{self.port}")
        print(f"   To use in other apps:    point their Ollama host to localhost:{self.port}")
        print(f"\n   Available endpoints:")
        print(f"     GET  http://localhost:{self.port}/api/tags")
        print(f"     POST http://localhost:{self.port}/api/generate")
        print(f"     POST http://localhost:{self.port}/api/chat")

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()      # actually release the listening socket
            self._server = None
            print(f"\n⏹  Server stopped (port {self.port} released).")


def export_model_for_ollama(model, tokenizer, model_name: str,
                             model_manager, export_dir: str = None):
    """
    Write a portable export package so the trained model can be referenced
    from Ollama tooling.

    Files written to <export_dir>/
    ───────────────────────────────
    <model_name>.pt          — full model weights  (PyTorch)
    <model_name>_tokenizer.json  — tokenizer vocabulary
    Modelfile                — Ollama Modelfile template
    README.txt               — usage instructions
    serve.py                 — one-click server launcher
    """
    import shutil

    export_path = Path(export_dir or f"ollama_export_{model_name}")
    export_path.mkdir(parents=True, exist_ok=True)

    # 1. Copy weights
    src_weights = model_manager.model_dir / f"{model_name}.pt"
    if src_weights.exists():
        shutil.copy(src_weights, export_path / f"{model_name}.pt")
        print(f"✓ Weights copied  → {export_path / (model_name + '.pt')}")
    else:
        torch.save(_portable_state_dict(model), export_path / f"{model_name}.pt")
        print(f"✓ Weights saved   → {export_path / (model_name + '.pt')}")

    # 2. Copy tokenizer
    src_tok = model_manager.model_dir / f"{model_name}_tokenizer.json"
    if src_tok.exists():
        shutil.copy(src_tok, export_path / f"{model_name}_tokenizer.json")
        print(f"✓ Tokenizer copied → {export_path / (model_name + '_tokenizer.json')}")

    # 3. Modelfile
    modelfile_text = f"""# Ollama Modelfile for {model_name}
# Generated by Tiny LLM Program
#
# NOTE: This model uses a custom (non-GGUF) transformer architecture.
# It is served via the local Ollama-compatible server (serve.py).
# Point your Ollama client to http://localhost:11435 after running serve.py.

FROM ./{model_name}.pt

PARAMETER temperature 0.8
PARAMETER num_predict 200

SYSTEM \"\"\"
You are a language model named {model_name}, trained from scratch.
\"\"\"
"""
    (export_path / 'Modelfile').write_text(modelfile_text, encoding='utf-8')
    print(f"✓ Modelfile written → {export_path / 'Modelfile'}")

    # The exported serve.py imports THIS program file, whatever it is called now
    _src           = globals().get('__file__')
    program_module = Path(_src).stem if _src else "TINY_LLM_PROGRAM"
    program_dir    = str(Path(_src).resolve().parent) if _src else ""

    # 4. serve.py launcher
    serve_script = f'''#!/usr/bin/env python3
"""
One-click launcher: serves {model_name} through an Ollama-compatible HTTP API.
Run this script, then point any Ollama client to http://localhost:11435
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, {program_dir!r})

from {program_module} import (
    SimpleLLM, CharTokenizer, OllamaCompatibleServer
)
import threading

MODEL_NAME = "{model_name}"
WEIGHTS    = os.path.join(os.path.dirname(__file__), f"{model_name}.pt")
TOKENIZER  = os.path.join(os.path.dirname(__file__), f"{model_name}_tokenizer.json")
PORT       = 11435

tokenizer = CharTokenizer.load(TOKENIZER)
model     = SimpleLLM(vocab_size=tokenizer.vocab_size,
                      d_model=64, num_blocks=2, num_heads=4, max_len=100)
import torch
model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location='cpu'))
model.eval()

server = OllamaCompatibleServer(model, tokenizer, MODEL_NAME, port=PORT)
server.start()

print("\\nServer running. Press Ctrl+C to stop.")
try:
    threading.Event().wait()
except KeyboardInterrupt:
    server.stop()
'''
    (export_path / 'serve.py').write_text(serve_script, encoding='utf-8')
    print(f"✓ serve.py written  → {export_path / 'serve.py'}")

    # 5. README
    readme = f"""Ollama-Compatible Export: {model_name}
{'='*50}

Files
-----
  {model_name}.pt               Model weights (PyTorch)
  {model_name}_tokenizer.json   Tokenizer vocabulary
  Modelfile                     Ollama Modelfile template
  serve.py                      One-click server launcher

Quick Start
-----------
1. From this folder run:
       python serve.py

2. The server starts at:
       http://localhost:11435

3. Point any Ollama-compatible client to that URL.
   In this program ({program_module}.py), change the
   OllamaValidator URL to http://localhost:11435 and the model name
   to "{model_name}".

4. Test with curl:
       curl http://localhost:11435/api/tags
       curl -X POST http://localhost:11435/api/generate \\
            -d '{{"model":"{model_name}","prompt":"Hello","stream":false}}'

Why not native Ollama?
----------------------
Ollama natively supports GGUF models built on LLaMA/Mistral/Falcon
architectures.  This tiny LLM uses a custom transformer
that is not compatible with those formats.  The local server speaks
exactly the same HTTP API as Ollama so all Ollama client libraries work
without modification.
"""
    (export_path / 'README.txt').write_text(readme, encoding='utf-8')
    print(f"✓ README written    → {export_path / 'README.txt'}")
    print(f"\n📦 Export complete: {export_path.resolve()}")
    return str(export_path.resolve())



def _collect_user_context(feature_name: str) -> str:
    """
    Prompt the user for free-form context that will be injected into every
    Ollama call during this session.  Returns an empty string if skipped.

    Examples the user might type
    ─────────────────────────────
    • "Answer every question as briefly as possible in one sentence."
    • "You are training a model to be an expert chef. Prefer detailed
       step-by-step recipe answers. Penalise vague responses."
    • "Focus only on Python programming topics. Ignore anything off-topic."
    """
    print("\n" + "="*60)
    print(f"  OLLAMA CONTEXT FOR {feature_name.upper()}")
    print("="*60)
    print("You can give Ollama instructions on HOW to train / evaluate your")
    print("model.  This text is injected into every Ollama prompt so it")
    print("shapes both the training data it generates and how it scores")
    print("your model's answers.")
    print()
    print("Examples:")
    print('  "Answer every question in one concise sentence."')
    print('  "You are training a cooking expert. Prefer detailed recipes."')
    print('  "Focus only on Python programming. Mark off-topic answers low."')
    print()
    print("Type your context below (press Enter twice when done,")
    print("or just press Enter once to skip):")
    print("-"*60)

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            # Two consecutive blank lines → done
            lines.pop()   # remove trailing blank
            break
        if line == "" and not lines:
            # First line is blank → skip
            break
        lines.append(line)

    context = "\n".join(lines).strip()
    if context:
        print(f"\n✓ Context saved ({len(context)} chars). "
              f"Ollama will use this throughout the session.")
    else:
        print("\n⏭️  No context provided — using default evaluation criteria.")
    print("="*60)
    return context


# ============================================================================
# QUESTION-FILE SCORING HELPER
# ============================================================================
def _load_questions_from_file(path):
    """Load non-empty lines from a question file."""
    questions = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                q = line.strip()
                if q and not q.startswith('#'):
                    questions.append(q)
        print(f"✓ Loaded {len(questions)} questions from {path}")
    except Exception as e:
        print(f"❌ Could not load question file: {e}")
    return questions


def _generate_short_answer(model, tokenizer, prompt, device, max_length=200,
                            temperature=0.85, top_k=10, repetition_penalty=1.5,
                            min_new_chars=15):
    """Generate an answer string for a prompt.

    Fixes versus the old argmax-only version
    ─────────────────────────────────────────
    • Temperature sampling  — softens the logit distribution so the model
      doesn't greedily lock onto one character and repeat it endlessly
      ("kkkkkk…").  Values 0.7–1.0 give good results; lower = more focused.

    • Top-k filtering       — only the k most likely tokens are kept before
      sampling, preventing random rare characters from appearing.

    • Repetition penalty    — tokens that have appeared recently in the
      generated text have their logit divided by (penalty ^ count), making
      it progressively harder to repeat the same character.

    • Deferred stop condition (min_new_chars) — the stop characters
      ('.', '!', '?', double-newline) are only checked AFTER at least
      min_new_chars new characters have been generated.  This prevents
      the "(no output)" bug where a model trained on "Q: …\\nA: …" format
      immediately predicts '\\n' as the first token and stops.

    • Q/A prefix stripping  — if the model's output begins with the
      answer-format header ("A: " or "\\nA: "), that prefix is removed
      so only the actual answer text is returned.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    tokens = tokenizer.encode(prompt)
    if not tokens:
        return ""

    # Derive context limit from the model's PE buffer (default 100)
    pe_buf    = getattr(getattr(model, 'pos_encoding', None), 'pe', None)
    ctx_limit = int(pe_buf.size(1)) if pe_buf is not None else 100
    tokens    = tokens[-ctx_limit:]

    start_len = len(tokens)
    generated = list(tokens)
    x         = torch.tensor([generated], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_length):
            logits      = model(x)
            next_logits = logits[0, -1, :].clone().float()

            # ── Repetition penalty ──────────────────────────────────────
            if repetition_penalty > 1.0 and len(generated) > start_len:
                recent = generated[-40:]          # look at last 40 positions
                for tok_id in set(recent):
                    count   = recent.count(tok_id)
                    penalty = repetition_penalty ** count
                    # Divide positive logits / multiply negative ones, so the
                    # penalty always lowers the token's probability.
                    if next_logits[tok_id] > 0:
                        next_logits[tok_id] /= penalty
                    else:
                        next_logits[tok_id] *= penalty

            # ── Temperature scaling ─────────────────────────────────────
            next_logits = next_logits / max(temperature, 1e-8)

            # ── Top-k filtering ─────────────────────────────────────────
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                top_vals, _ = torch.topk(next_logits, k)
                threshold   = top_vals[-1]
                next_logits = next_logits.masked_fill(next_logits < threshold,
                                                      float('-inf'))

            # ── Sample next token ───────────────────────────────────────
            probs   = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            generated.append(next_id)
            x = torch.tensor([generated[-ctx_limit:]], dtype=torch.long, device=device)

            # ── Stop condition (only after min_new_chars) ───────────────
            # Measured on the decoded text (a BPE token can span several
            # characters), so min_new_chars still means characters.
            new_text  = tokenizer.decode(generated[start_len:])
            new_chars = len(new_text)
            if new_chars >= min_new_chars:
                ch = new_text[-1:]
                if ch in ('.', '!'):
                    break
                if ch == '?' and new_chars >= 30:
                    break
                # Stop on paragraph break (two consecutive newlines)
                if new_chars >= 2 and new_text.endswith('\n\n'):
                    break

    # Build raw answer text
    answer = tokenizer.decode(generated[start_len:])

    # Strip Q/A format headers the model may have prepended
    # e.g. "\nA: ", "A: ", "Answer: "
    for prefix in ('\nA: ', '\nA:', '\nAnswer: ', 'A: ', 'Answer: '):
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
            break

    return answer.strip()


def _score_model_on_questions(model, tokenizer, device, model_manager, model_name,
                               evaluator, questions,
                               low_score_threshold=6.0, extra_epochs=10,
                               train_loader=None, learning_rate=0.001):
    """
    Present each question to the model, score with Ollama (0-10),
    display running average.  If average < low_score_threshold after all
    questions, offer to train for extra_epochs more epochs automatically.

    Returns final average score (float) or None if no questions were scored.
    """
    if not questions:
        print("⚠️  No questions to score.")
        return None

    scores  = []
    bar_w   = 20

    for i, question in enumerate(questions, 1):
        print(f"\n── Question {i}/{len(questions)} " + "─"*30)
        print(f"Q: {question}")

        prompt  = f"Q: {question}\nA:"
        answer  = _generate_short_answer(model, tokenizer, prompt, device)
        print(f"A: {answer if answer else '(no output)'}")

        score   = None
        comment = ""

        if evaluator and evaluator.validator.available and evaluator.validator.model:
            score, comment = evaluator.evaluate_answer(question, answer)

        if score is None:
            # Fall back to manual scoring
            raw = input("Rate this answer 0-10 (or press Enter to skip): ").strip()
            try:
                score = float(raw)
                score = max(0.0, min(10.0, score))
            except ValueError:
                print("  Skipped.")
                continue

        scores.append(score)

        s_fill = int(score / 10 * bar_w)
        s_bar  = "█" * s_fill + "░" * (bar_w - s_fill)
        avg    = sum(scores) / len(scores)
        a_fill = int(avg   / 10 * bar_w)
        a_bar  = "█" * a_fill + "░" * (bar_w - a_fill)

        print(f"⭐ Score   [{s_bar}] {score:.1f}/10")
        print(f"📊 Average [{a_bar}] {avg:.2f}/10  ({len(scores)} answered)")
        if comment:
            print(f"📌 {comment}")

    if not scores:
        print("❌ No answers scored.")
        return None

    final_avg = sum(scores) / len(scores)
    print("\n" + "="*60)
    print(f"📊 FINAL SCORE: {final_avg:.2f}/10  ({len(scores)}/{len(questions)} questions)")
    f_fill = int(final_avg / 10 * bar_w)
    print(f"   [{'█'*f_fill}{'░'*(bar_w-f_fill)}]")
    print("="*60)

    if final_avg < low_score_threshold:
        print(f"\n⚠️  Score {final_avg:.2f} is BELOW threshold {low_score_threshold:.1f}/10")
        print(f"   Triggering {extra_epochs} additional training epoch(s) automatically…")
        if train_loader:
            train_model(
                model, train_loader,
                num_epochs=extra_epochs,
                learning_rate=learning_rate,
                device=device,
                model_manager=model_manager,
                model_name=model_name,
            )
            if model_manager:
                model_manager.save_model(model, model_name)
                print(f"✓ Model saved after retraining: {model_name}")
        else:
            print("⚠️  No training data available — cannot retrain automatically.")
    else:
        print(f"\n✅ Model PASSED!  {final_avg:.2f}/10 ≥ threshold {low_score_threshold:.1f}/10")

    return final_avg


# ============================================================================
# SHARED HELPERS FOR MENU
# ============================================================================
class DatasetRegistry:
    """
    Persists a list of dataset paths (files and folders) to disk so the user
    does not have to retype them on every run.

    Each instance is backed by its OWN JSON file under saved_models/, so
    different dataset *purposes* (training, validation, test) can keep
    separate, non-overlapping lists of saved paths — e.g. a path saved while
    loading a VALIDATION set never appears as an option for the TEST set,
    and vice versa.

    Storage: saved_models/<filename>   (default: dataset_registry.json)
    """

    def __init__(self, filename: str = "dataset_registry.json"):
        self.REGISTRY_FILE = Path("saved_models") / filename
        self.REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._data: list[dict] = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────
    def _load(self) -> list:
        if self.REGISTRY_FILE.exists():
            try:
                return json.loads(self.REGISTRY_FILE.read_text(encoding='utf-8'))
            except Exception:
                return []
        return []

    def _save(self):
        self.REGISTRY_FILE.write_text(
            json.dumps(self._data, indent=2), encoding='utf-8')

    # ── CRUD ─────────────────────────────────────────────────────────────────
    def add(self, path: str, label: str = "") -> bool:
        """Add a path. Returns False if path already registered."""
        path = str(Path(path).resolve())
        for entry in self._data:
            if entry['path'] == path:
                return False
        kind = 'folder' if Path(path).is_dir() else 'file'
        self._data.append({
            'path'    : path,
            'kind'    : kind,
            'label'   : label or Path(path).name,
            'added_at': str(datetime.now())[:19],
        })
        self._save()
        return True

    def remove(self, idx: int) -> bool:
        if 0 <= idx < len(self._data):
            removed = self._data.pop(idx)
            self._save()
            print(f"✓ Removed: {removed['path']}")
            return True
        return False

    def entries(self) -> list:
        return list(self._data)

    def valid_entries(self) -> list:
        """Return only paths that still exist on disk."""
        return [e for e in self._data if Path(e['path']).exists()]

    # ── Display ──────────────────────────────────────────────────────────────
    def display(self, entries=None):
        """Print numbered entries.  Default: every saved entry (incl. missing files).
        Pass the exact list you will index into so the printed numbers line up."""
        if entries is None:
            entries = self.entries()
        if not entries:
            print("  (no saved datasets)")
            return
        for i, e in enumerate(entries, 1):
            exists = "✓" if Path(e['path']).exists() else "✗ missing"
            print(f"  {i}. [{e['kind']:6}] {e['label']}  {exists}")
            print(f"       {e['path']}")


_dataset_registry       = DatasetRegistry("dataset_registry.json")              # training / retraining / fine-tuning / scoring
_validation_registry    = DatasetRegistry("validation_dataset_registry.json")   # validation set — kept separate
_test_registry          = DatasetRegistry("test_dataset_registry.json")         # test set — kept separate, used once
_conversational_registry = DatasetRegistry("conversational_dataset_registry.json")  # Q&A / conversation files
_mcq_registry           = DatasetRegistry("mcq_dataset_registry.json")          # multiple-choice quiz files


def _load_multiple_sources(sources: list[dict]) -> str:
    """
    Given a list of registry entries (or {'path':..., 'kind':...} dicts),
    read all files and return combined text.
    """
    all_texts = []
    total_chars = 0
    for src in sources:
        p    = Path(src['path'])
        kind = src.get('kind', 'folder' if p.is_dir() else 'file')
        if not p.exists():
            print(f"  ⚠ Skipped (not found): {p}")
            continue
        if kind == 'folder':
            files = sorted(p.glob('**/*.txt')) + sorted(p.glob('**/*.md'))
            for f in files:
                try:
                    t = f.read_text(encoding='utf-8')
                    all_texts.append(t)
                    total_chars += len(t)
                    print(f"  ✓ {f.name}  ({len(t):,} chars)")
                except Exception as e:
                    print(f"  ⚠ Skipped {f.name}: {e}")
        else:
            try:
                t = p.read_text(encoding='utf-8')
                all_texts.append(t)
                total_chars += len(t)
                print(f"  ✓ {p.name}  ({len(t):,} chars)")
            except Exception as e:
                print(f"  ⚠ Skipped {p.name}: {e}")

    combined = "\n\n".join(all_texts)
    print(f"\n✓ Combined {len(all_texts)} source(s) — {total_chars:,} chars total")
    return combined


def _load_training_data(prompt_label="training"):
    """
    Interactive training-data loader with persistent folder/file registry.

    The saved-path registry used depends on *prompt_label*:
      - "validation"     → saved_models/validation_dataset_registry.json
      - "test"           → saved_models/test_dataset_registry.json
      - "conversational" → saved_models/conversational_dataset_registry.json
      - anything else (training / retraining / fine-tuning / scoring)
                         → saved_models/dataset_registry.json

    Keeping validation and test paths in their own files means a dataset
    saved while loading one will never silently appear as an option for
    the other — important for avoiding train/val/test leakage.

    Flow
    ────
    1. Show saved datasets for THIS purpose (if any) and ask the user which to use
    2. Let the user add more paths on top (or instead) — new paths are saved
       to the same purpose-specific registry
    3. Combine everything and return (text, primary_source, type)
    """
    label_lower = prompt_label.lower()
    if label_lower == "validation":
        registry, purpose = _validation_registry, "VALIDATION"
    elif label_lower == "test":
        registry, purpose = _test_registry, "TEST"
    elif label_lower == "conversational":
        registry, purpose = _conversational_registry, "CONVERSATIONAL"
    else:
        registry, purpose = _dataset_registry, "TRAINING"

    print("\n" + "="*60)
    print(f"LOAD {prompt_label.upper()} DATA")
    print("="*60)
    print(f"📁 Saved-path file: {registry.REGISTRY_FILE}")

    saved = registry.valid_entries()
    chosen_sources: list[dict] = []

    # ── Step 1: show saved paths and ask which to use ─────────────────────
    if saved:
        print(f"\n📂 Saved {purpose} dataset locations ({len(saved)} available):\n")
        registry.display(saved)
        print()
        print("Options:")
        print("  a  — use ALL saved datasets")
        print("  numbers separated by spaces  — e.g. '1 3' to pick specific ones")
        print("  n  — skip saved datasets, add new paths only")
        print("  m  — manage saved datasets (add/remove)")
        sel = input("\nYour choice: ").strip().lower()

        if sel == 'a':
            chosen_sources = list(saved)
            print(f"✓ Using all {len(chosen_sources)} saved dataset(s).")

        elif sel == 'm':
            # ── Manage saved datasets ──────────────────────────────────
            while True:
                print(f"\n── {purpose} Dataset Registry ─────────────────────────────")
                registry.display()
                print("\n  a  — add new path")
                print("  r  — remove a path")
                print("  done — finish managing, continue")
                mc = input("Action: ").strip().lower()
                if mc == 'a':
                    _add_paths_to_registry(registry)
                elif mc == 'r':
                    idx_raw = input("Number to remove: ").strip()
                    if idx_raw.isdigit():
                        registry.remove(int(idx_raw) - 1)
                elif mc in ('done', 'd', ''):
                    break
            saved = registry.valid_entries()
            if saved:
                use_after = input("Use the updated saved datasets? (y/n): ").strip().lower()
                if use_after == 'y':
                    chosen_sources = list(saved)

        elif sel == 'n':
            pass  # skip, fall through to add new

        else:
            # Numeric selection — e.g. "1 3"
            indices = []
            for tok in sel.split():
                if tok.isdigit():
                    i = int(tok) - 1
                    if 0 <= i < len(saved):
                        indices.append(i)
            if indices:
                chosen_sources = [saved[i] for i in indices]
                print(f"✓ Selected {len(chosen_sources)} saved dataset(s).")
            else:
                print("⚠ No valid selection — continuing with no saved datasets.")

    # ── Step 2: add more / new paths ──────────────────────────────────────
    add_more = True
    if chosen_sources:
        add_raw = input("\nAdd more paths on top? (y/n): ").strip().lower()
        add_more = (add_raw == 'y')

    if add_more:
        new_paths = _add_paths_to_registry(registry, save_to_registry=True)
        chosen_sources.extend(new_paths)

    if not chosen_sources:
        print("❌ No data sources selected.")
        return None, None, None

    # ── Step 3: load and combine ──────────────────────────────────────────
    print(f"\n📖 Reading {len(chosen_sources)} source(s)…")
    combined = _load_multiple_sources(chosen_sources)
    if not combined.strip():
        print("❌ No text loaded.")
        return None, None, None

    # Primary source for session-state tracking (first chosen)
    primary      = chosen_sources[0]['path']
    primary_type = chosen_sources[0].get('kind', 'folder')
    return combined, primary, primary_type


def _add_paths_to_registry(registry: "DatasetRegistry" = None,
                           save_to_registry=True) -> list[dict]:
    """
    Prompt the user to enter one or more file/folder paths.
    Saves each to *registry* (unless save_to_registry=False).
    Defaults to the main training/scoring registry if none is given.
    Returns a list of {path, kind, label} dicts for the newly added paths.
    """
    if registry is None:
        registry = _dataset_registry
    added: list[dict] = []

    print("\nEnter paths one by one (press Enter on a blank line when done):")
    print("  Each can be a .txt FILE or a FOLDER containing .txt files.")

    while True:
        raw = input("  Path (or blank to finish): ").strip()
        if not raw:
            break

        p = Path(raw)
        if not p.exists():
            print(f"  ❌ Not found: {raw}")
            continue

        kind  = 'folder' if p.is_dir() else 'file'
        label_default = p.name
        label = input(f"  Label for this {kind} (default '{label_default}'): ").strip()
        label = label or label_default

        entry = {'path': str(p.resolve()), 'kind': kind, 'label': label}

        if save_to_registry:
            if registry.add(str(p.resolve()), label):
                print(f"  ✓ Added to registry: {p.resolve()}")
            else:
                print(f"  ℹ Already in registry: {p.resolve()}")

        added.append(entry)

    return added







# ============================================================================
# SHARED EVALUATION CORE
# ============================================================================
def _run_evaluation(model, tokenizer, device, text, seq_len=32, batch_size=4,
                    label="Evaluation"):
    """Run the model in eval-mode over *text* and return a metrics dict.

    Returns
    -------
    dict with keys: avg_loss, perplexity, accuracy, min_loss, q1_loss,
                    med_loss, q3_loss, max_loss, total_tokens, n_sequences,
                    batch_losses
    or None if the dataset is too short.
    """
    dataset     = TextDataset(text, tokenizer, seq_len=seq_len)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if len(dataset) == 0:
        print(f"❌  Dataset too short for seq_len={seq_len}. "
              f"Need at least {seq_len + 1} tokens after encoding.")
        return None

    criterion     = nn.CrossEntropyLoss()
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    batch_losses  = []
    n_batches     = len(eval_loader)
    report_every  = max(1, n_batches // 10)

    print(f"\n  Running {label} on {len(dataset):,} sequences…")
    with torch.no_grad():
        for idx, (x, y) in enumerate(eval_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size),
                y.reshape(-1)
            )
            bl = loss.item()
            total_loss    += bl
            batch_losses.append(bl)

            preds          = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens  += y.numel()

            if (idx + 1) % report_every == 0 or idx == n_batches - 1:
                pct  = (idx + 1) / n_batches
                bar  = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
                print(f"  [{bar}] {pct*100:5.1f}%  loss {bl:.4f}",
                      end="\r", flush=True)
    print()

    avg_loss   = total_loss / n_batches
    perplexity = math.exp(min(avg_loss, 500))
    accuracy   = total_correct / max(1, total_tokens) * 100

    s = sorted(batch_losses)
    n = len(s)
    return dict(
        avg_loss   = avg_loss,
        perplexity = perplexity,
        accuracy   = accuracy,
        min_loss   = s[0],
        q1_loss    = s[n // 4],
        med_loss   = s[n // 2],
        q3_loss    = s[3 * n // 4],
        max_loss   = s[-1],
        total_tokens = total_tokens,
        n_sequences  = len(dataset),
        batch_losses = batch_losses,
    )


def _perplexity_rating(ppl):
    if   ppl < 5:   return "Excellent 🏆"
    elif ppl < 15:  return "Good      ✅"
    elif ppl < 50:  return "Fair      🟡"
    elif ppl < 200: return "Poor      🔴"
    else:           return "Very Poor ❌"


def _print_metrics_box(m):
    r = _perplexity_rating(m['perplexity'])
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │  Loss         {m['avg_loss']:>10.4f}         │")
    print(f"  │  Perplexity   {m['perplexity']:>10.2f}         │")
    print(f"  │  Accuracy     {m['accuracy']:>9.2f}%         │")
    print(f"  │  Rating       {r:<20}│")
    print(f"  └─────────────────────────────────┘")
    print(f"\n  Loss distribution:")
    print(f"    Min    : {m['min_loss']:.4f}")
    print(f"    Q1     : {m['q1_loss']:.4f}")
    print(f"    Median : {m['med_loss']:.4f}")
    print(f"    Q3     : {m['q3_loss']:.4f}")
    print(f"    Max    : {m['max_loss']:.4f}")


# ============================================================================
# VALIDATION  (repeatable — tracks history, detects overfitting)
# ============================================================================
def validate_model_on_dataset(model, tokenizer, device, model_manager,
                               model_name, state_manager=None):
    """Validate the model on a held-out validation set.

    Purpose
    -------
    Validation is run *repeatedly* during development to detect overfitting:
    if validation loss is much higher than training loss, the model has memorised
    its training data but not generalised.

    Features
    --------
    • Runs the full evaluation loop (loss, perplexity, accuracy)
    • Compares val-loss to the last recorded training loss
    • Maintains a dated history file so you can track improvement across runs
    • Flags likely overfitting / underfitting automatically
    """
    print("\n" + "=" * 60)
    print("✅  VALIDATION DATASET EVALUATION")
    print("=" * 60)
    print("\nValidation checks whether the model has generalised beyond its")
    print("training data.  Run this as often as you like during development.\n")
    print("  • Val loss ≈ Train loss  → healthy generalisation")
    print("  • Val loss >> Train loss → overfitting (model memorised training data)")
    print("  • Val loss << Train loss → underfitting (model hasn't converged yet)\n")

    # ── Load validation text ─────────────────────────────────────────────────
    val_text, data_source, _ = _load_training_data("validation")
    if not val_text:
        return

    clean = tokenizer.sanitize(val_text)
    if not clean.strip():
        print("❌  Validation dataset has no characters in the model's vocabulary.")
        return

    coverage = len(clean) / max(1, len(val_text)) * 100
    print(f"  Vocab coverage : {coverage:.1f}%")

    # ── Config ───────────────────────────────────────────────────────────────
    def _ask_pos_int(prompt, default):
        while True:
            raw = input(f"  {prompt} (default {default}): ").strip()
            if not raw:
                return default
            try:
                v = int(raw)
                if v > 0:
                    return v
                print("    ⚠️  Must be a positive integer.")
            except ValueError:
                print("    ⚠️  Please enter a whole number.")

    seq_len    = _ask_pos_int("Sequence length", 32)
    batch_size = _ask_pos_int("Batch size",       4)

    # ── Run evaluation ───────────────────────────────────────────────────────
    m = _run_evaluation(model, tokenizer, device, clean,
                        seq_len=seq_len, batch_size=batch_size,
                        label="Validation")
    if m is None:
        return

    # ── Load previous validation history ────────────────────────────────────
    hist_path = model_manager.model_dir / f"{model_name}_val_history.json"
    history   = []
    if hist_path.exists():
        try:
            with open(hist_path, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    # ── Compare to training loss ─────────────────────────────────────────────
    train_loss = None
    tsm = (state_manager or
           TrainingStateManager(model_name, model_dir=str(model_manager.model_dir)))
    saved_state = tsm.load()
    if saved_state:
        train_loss = saved_state.get("last_loss")

    # ── Display ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Model    : {model_name}")
    print(f"  Dataset  : {data_source}")
    print(f"  Run at   : {timestamp}\n")
    _print_metrics_box(m)

    # Overfitting analysis
    if train_loss is not None:
        gap = m['avg_loss'] - train_loss
        print(f"\n  ── Generalisation Analysis ──────────────────────────────")
        print(f"     Training loss : {train_loss:.4f}")
        print(f"     Val loss      : {m['avg_loss']:.4f}")
        print(f"     Gap (val−train): {gap:+.4f}")
        if   gap >  1.5:  verdict = "⚠️  Strong overfitting — consider regularisation or more data"
        elif gap >  0.5:  verdict = "🟡 Mild overfitting — monitor closely"
        elif gap > -0.3:  verdict = "✅  Healthy generalisation"
        else:             verdict = "🔵 Underfitting — model may need more training"
        print(f"     Verdict       : {verdict}")
    else:
        print("\n  ℹ️   No training loss on record — train the model first to enable")
        print("       the overfitting analysis.")

    # History trend
    if history:
        print(f"\n  ── Validation History ({len(history)} previous run(s)) ──────────")
        for i, h in enumerate(history[-5:], 1):   # show last 5
            trend = ""
            if i > 1:
                prev = history[-5:][i-2]['avg_loss']
                diff = h['avg_loss'] - prev
                trend = f"  {'▲' if diff > 0 else '▼'} {abs(diff):.4f}"
            print(f"     [{h['timestamp']}]  "
                  f"loss {h['avg_loss']:.4f}  ppl {h['perplexity']:.2f}{trend}")
        if len(history) > 5:
            print(f"     … and {len(history)-5} earlier run(s)")

    # ── Save updated history ─────────────────────────────────────────────────
    history.append(dict(
        timestamp   = timestamp,
        avg_loss    = round(m['avg_loss'],   4),
        perplexity  = round(m['perplexity'], 4),
        accuracy    = round(m['accuracy'],   4),
        train_loss  = round(train_loss, 4) if train_loss is not None else None,
        dataset     = str(data_source),
        seq_len     = seq_len,
        n_sequences = m['n_sequences'],
    ))
    try:
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n  ✓  History saved : {hist_path}")
    except Exception as e:
        print(f"  ⚠️  Could not save history: {e}")

    # ── Save plain text report ───────────────────────────────────────────────
    rpt_path = (model_manager.model_dir /
                f"{model_name}_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    try:
        with open(rpt_path, 'w', encoding='utf-8') as rpt:
            rpt.write("VALIDATION REPORT\n")
            rpt.write(f"Generated  : {timestamp}\n")
            rpt.write("=" * 60 + "\n")
            rpt.write(f"Model      : {model_name}\n")
            rpt.write(f"Dataset    : {data_source}\n")
            rpt.write(f"Sequences  : {m['n_sequences']:,}\n")
            rpt.write(f"Tokens     : {m['total_tokens']:,}\n\n")
            rpt.write(f"Loss       : {m['avg_loss']:.4f}\n")
            rpt.write(f"Perplexity : {m['perplexity']:.2f}\n")
            rpt.write(f"Accuracy   : {m['accuracy']:.2f}%\n")
            rpt.write(f"Rating     : {_perplexity_rating(m['perplexity'])}\n")
            if train_loss is not None:
                rpt.write(f"\nTraining loss : {train_loss:.4f}\n")
                rpt.write(f"Gap (val-train): {m['avg_loss']-train_loss:+.4f}\n")
                rpt.write(f"Verdict       : {verdict}\n")
            rpt.write("=" * 60 + "\n")
        print(f"  ✓  Report saved  : {rpt_path}")
    except Exception as e:
        print(f"  ⚠️  Could not save report: {e}")

    print("=" * 60)


# ============================================================================
# TEST  (one-shot final benchmark — do not run repeatedly)
# ============================================================================
def test_model_on_dataset(model, tokenizer, device, model_manager, model_name):
    """Run the final test-set evaluation.

    Purpose
    -------
    The test set is a *held-out* dataset that is used EXACTLY ONCE to report
    the final unbiased performance of the model.  Using it multiple times or
    using it to make training decisions invalidates it as a benchmark.

    Features
    --------
    • Hard warning if a test report for this model already exists
    • Full evaluation loop (loss, perplexity, accuracy)
    • Sample predictions alongside ground truth
    • Detailed "test certificate" saved to disk
    """
    print("\n" + "=" * 60)
    print("🧪  TEST DATASET EVALUATION  (final benchmark)")
    print("=" * 60)
    print()
    print("  ⚠️  IMPORTANT — read before continuing:")
    print("     The test set should be used ONCE, after all training and")
    print("     hyperparameter tuning is finished.  Running it multiple")
    print("     times or using results to guide further training means")
    print("     the score no longer reflects true generalisation.\n")

    # Warn if test has been run before
    cert_dir  = model_manager.model_dir
    prev_certs = sorted(cert_dir.glob(f"{model_name}_TEST_CERTIFICATE_*.txt"))
    if prev_certs:
        print(f"  🔴  WARNING: {len(prev_certs)} test certificate(s) already exist "
              f"for this model:")
        for c in prev_certs:
            print(f"       {c.name}")
        print()
        go = input("  Run the test anyway? (yes/n): ").strip().lower()
        if go != 'yes':
            print("  ↩️  Test cancelled.")
            return
        print()

    # ── Load test text ───────────────────────────────────────────────────────
    test_text, data_source, _ = _load_training_data("test")
    if not test_text:
        return

    clean = tokenizer.sanitize(test_text)
    if not clean.strip():
        print("❌  Test dataset has no characters in the model's vocabulary.")
        return

    coverage = len(clean) / max(1, len(test_text)) * 100
    print(f"\n  Vocab coverage : {coverage:.1f}%")

    # ── Config ───────────────────────────────────────────────────────────────
    def _ask_pos_int(prompt, default):
        while True:
            raw = input(f"  {prompt} (default {default}): ").strip()
            if not raw:
                return default
            try:
                v = int(raw)
                if v > 0:
                    return v
                print("    ⚠️  Must be a positive integer.")
            except ValueError:
                print("    ⚠️  Please enter a whole number.")

    seq_len    = _ask_pos_int("Sequence length", 32)
    batch_size = _ask_pos_int("Batch size",       4)
    n_samples  = _ask_pos_int("Sample predictions to display", 5)

    # ── Run evaluation ───────────────────────────────────────────────────────
    m = _run_evaluation(model, tokenizer, device, clean,
                        seq_len=seq_len, batch_size=batch_size,
                        label="Test")
    if m is None:
        return

    # ── Generate sample predictions ──────────────────────────────────────────
    print(f"\n  Generating {n_samples} sample predictions…")
    samples = []
    model.eval()
    clean_ids = tokenizer.encode(clean)   # windows are token-based (a BPE token spans several chars)
    step = max(1, len(clean_ids) // (n_samples + 1))

    with torch.no_grad():
        for i in range(n_samples):
            start    = i * step
            snippet  = clean_ids[start : start + seq_len]
            if len(snippet) < 2:
                continue
            context_ids = snippet[:-1]       # all but last token
            expected_id = snippet[-1]        # last token is the label

            try:
                toks   = torch.tensor(
                    context_ids, dtype=torch.long
                ).unsqueeze(0).to(device)
                logits = model(toks)
                pred_idx  = logits[0, -1, :].argmax().item()
                context   = tokenizer.decode(context_ids)
                expected  = tokenizer.decode([expected_id])
                predicted = tokenizer.decode([pred_idx])
                correct   = (pred_idx == expected_id)
                samples.append(dict(
                    context   = context[-20:],   # show last 20 chars for readability
                    expected  = expected,
                    predicted = predicted,
                    correct   = correct,
                ))
            except Exception:
                continue

    # ── Display results ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("\n" + "=" * 60)
    print("TEST RESULTS  —  FINAL BENCHMARK")
    print("=" * 60)
    print(f"  Model    : {model_name}")
    print(f"  Dataset  : {data_source}")
    print(f"  Tested at: {timestamp}\n")
    _print_metrics_box(m)

    # Sample predictions table
    if samples:
        sample_acc = sum(1 for s in samples if s['correct']) / len(samples) * 100
        print(f"\n  ── Sample Predictions ({len(samples)} shown) ──────────────────")
        print(f"  {'Context (last 20 chars)':<22}  {'Expected':<9} {'Predicted':<9} {'Match'}")
        print(f"  {'-'*22}  {'-'*8} {'-'*8} {'-'*5}")
        for s in samples:
            ctx_disp = repr(s['context'])[-22:]
            mark = "✓" if s['correct'] else "✗"
            print(f"  {ctx_disp:<22}  {repr(s['expected']):<9} "
                  f"{repr(s['predicted']):<9} {mark}")
        print(f"\n  Sample-level accuracy: {sample_acc:.1f}%")

    # Final verdict
    ppl = m['perplexity']
    print(f"\n  ── Final Verdict ────────────────────────────────────────")
    print(f"     {_perplexity_rating(ppl)}   (perplexity {ppl:.2f})")
    print("=" * 60)
    print("\n  ℹ️   This test is now COMPLETE.  Avoid re-running it — use")
    print("       the Validation option (sub-option 5) for ongoing checks.")

    # ── Save test certificate ─────────────────────────────────────────────────
    cert_ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    cert_path = model_manager.model_dir / f"{model_name}_TEST_CERTIFICATE_{cert_ts}.txt"
    try:
        with open(cert_path, 'w', encoding='utf-8') as cert:
            cert.write("=" * 60 + "\n")
            cert.write("TEST CERTIFICATE\n")
            cert.write("=" * 60 + "\n")
            cert.write(f"Model          : {model_name}\n")
            cert.write(f"Test dataset   : {data_source}\n")
            cert.write(f"Tested at      : {timestamp}\n")
            cert.write(f"Sequences      : {m['n_sequences']:,}\n")
            cert.write(f"Tokens scored  : {m['total_tokens']:,}\n")
            cert.write(f"Vocab coverage : {coverage:.1f}%\n")
            cert.write(f"Seq length     : {seq_len}\n\n")
            cert.write(f"Loss           : {m['avg_loss']:.4f}\n")
            cert.write(f"Perplexity     : {m['perplexity']:.2f}\n")
            cert.write(f"Accuracy       : {m['accuracy']:.2f}%\n")
            cert.write(f"Rating         : {_perplexity_rating(m['perplexity'])}\n\n")
            cert.write("Loss distribution:\n")
            cert.write(f"  Min    : {m['min_loss']:.4f}\n")
            cert.write(f"  Q1     : {m['q1_loss']:.4f}\n")
            cert.write(f"  Median : {m['med_loss']:.4f}\n")
            cert.write(f"  Q3     : {m['q3_loss']:.4f}\n")
            cert.write(f"  Max    : {m['max_loss']:.4f}\n\n")
            if samples:
                cert.write("Sample predictions:\n")
                for s in samples:
                    mark = "CORRECT" if s['correct'] else "WRONG"
                    cert.write(f"  context={repr(s['context'])[-20:]}  "
                               f"expected={repr(s['expected'])}  "
                               f"predicted={repr(s['predicted'])}  {mark}\n")
                cert.write(f"\nSample accuracy : {sample_acc:.1f}%\n")
            cert.write("\n" + "=" * 60 + "\n")
            cert.write("NOTE: This certificate was generated by a single test run.\n")
            cert.write("      Do not use this dataset again to guide training.\n")
            cert.write("=" * 60 + "\n")
        print(f"\n  ✓  Test certificate saved: {cert_path}")
    except Exception as e:
        print(f"  ⚠️  Could not save certificate: {e}")


# ============================================================================
# DATASET SCORING  (no training — pure evaluation)
# ============================================================================
def score_model_on_dataset(model, tokenizer, device, model_manager, model_name):
    """Evaluate model quality against a user-provided dataset without any training.

    Metrics reported
    ────────────────
    • Loss        — average cross-entropy (lower is better)
    • Perplexity  — exp(loss); how surprised the model is by the text
                    (lower is better; 1.0 would be perfect prediction)
    • Accuracy    — % of next-character predictions that were exactly right

    A rating band (Excellent → Very Poor) gives a quick intuition.
    All results are saved to a dated .txt report in saved_models/.
    """
    print("\n" + "=" * 60)
    print("📊  DATASET SCORING")
    print("=" * 60)
    print("\nEvaluates your model against a dataset — no training happens.")
    print("Metrics computed:")
    print("  •  Loss        — cross-entropy (lower = better)")
    print("  •  Perplexity  — exp(loss)     (lower = better, 1.0 = perfect)")
    print("  •  Accuracy    — % of next-char predictions correct\n")

    # ── Load scoring dataset ─────────────────────────────────────────────────
    scoring_text, data_source, _ = _load_training_data("scoring")
    if not scoring_text:
        return

    # Sanitize: remove chars outside the model vocab (same fix as fine-tuning)
    clean_text = tokenizer.sanitize(scoring_text)
    if not clean_text.strip():
        print("❌  Dataset has no characters within the model's vocabulary.")
        print("    Try a dataset that matches the language the model was trained on.")
        return

    coverage = len(clean_text) / max(1, len(scoring_text)) * 100
    print(f"  Vocab coverage : {coverage:.1f}%  "
          f"({len(clean_text):,} / {len(scoring_text):,} chars kept)")

    # ── Evaluation config ────────────────────────────────────────────────────
    print()
    def _ask_pos_int(prompt, default):
        while True:
            raw = input(f"{prompt} (default {default}): ").strip()
            if not raw:
                return default
            try:
                v = int(raw)
                if v > 0:
                    return v
                print("  ⚠️  Must be a positive integer.")
            except ValueError:
                print("  ⚠️  Please enter a whole number.")

    seq_len    = _ask_pos_int("Sequence length", 32)
    batch_size = _ask_pos_int("Batch size",       4)

    # ── Build dataset ────────────────────────────────────────────────────────
    dataset     = TextDataset(clean_text, tokenizer, seq_len=seq_len)
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if len(dataset) == 0:
        print(f"❌  Dataset too short for seq_len={seq_len}. "
              f"Need at least {seq_len + 1} tokens after encoding.")
        return

    print(f"\n🔍  Scoring  [{model_name}]  on  {len(dataset):,} sequences…")
    print(f"    Source : {data_source}")
    print(f"    Chars  : {len(clean_text):,}   |   Seq len : {seq_len}   "
          f"|   Batch : {batch_size}\n")

    # ── Evaluation loop ──────────────────────────────────────────────────────
    criterion     = nn.CrossEntropyLoss()
    model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    batch_losses  = []
    n_batches     = len(eval_loader)
    report_every  = max(1, n_batches // 10)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(eval_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            # Cross-entropy loss
            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size),
                y.reshape(-1)
            )
            b_loss = loss.item()
            total_loss += b_loss
            batch_losses.append(b_loss)

            # Top-1 accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens  += y.numel()

            # Progress bar
            if (batch_idx + 1) % report_every == 0 or batch_idx == n_batches - 1:
                done  = batch_idx + 1
                pct   = done / n_batches
                bar   = "█" * int(30 * pct) + "░" * (30 - int(30 * pct))
                print(f"  [{bar}] {pct*100:5.1f}%  "
                      f"batch {done}/{n_batches}  "
                      f"loss {b_loss:.4f}",
                      end="\r", flush=True)

    print()   # newline after progress bar

    # ── Compute summary statistics ───────────────────────────────────────────
    avg_loss   = total_loss / n_batches
    perplexity = math.exp(min(avg_loss, 500))   # cap to avoid overflow
    accuracy   = total_correct / max(1, total_tokens) * 100

    # Quartile breakdown
    sorted_losses = sorted(batch_losses)
    n = len(sorted_losses)
    min_loss = sorted_losses[0]
    q1_loss  = sorted_losses[n // 4]
    med_loss = sorted_losses[n // 2]
    q3_loss  = sorted_losses[3 * n // 4]
    max_loss = sorted_losses[-1]

    # Rating
    if   perplexity < 5:    rating = "Excellent 🏆"
    elif perplexity < 15:   rating = "Good      ✅"
    elif perplexity < 50:   rating = "Fair      🟡"
    elif perplexity < 200:  rating = "Poor      🔴"
    else:                   rating = "Very Poor ❌"

    # ── Display results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCORING RESULTS")
    print("=" * 60)
    print(f"  Model          : {model_name}")
    print(f"  Dataset        : {data_source}")
    print(f"  Sequences      : {len(dataset):,}")
    print(f"  Tokens scored  : {total_tokens:,}")
    print(f"  Vocab coverage : {coverage:.1f}%")
    print()
    print(f"  ┌─────────────────────────────────┐")
    print(f"  │  Loss         {avg_loss:>10.4f}         │")
    print(f"  │  Perplexity   {perplexity:>10.2f}         │")
    print(f"  │  Accuracy     {accuracy:>9.2f}%         │")
    print(f"  │  Rating       {rating:<20}│")
    print(f"  └─────────────────────────────────┘")
    print()
    print(f"  Loss distribution across {n_batches} batches:")
    print(f"    Min    : {min_loss:.4f}")
    print(f"    Q1     : {q1_loss:.4f}")
    print(f"    Median : {med_loss:.4f}")
    print(f"    Q3     : {q3_loss:.4f}")
    print(f"    Max    : {max_loss:.4f}")
    print("=" * 60)

    # Perplexity interpretation hint
    print(f"\n  ℹ️   Perplexity guide:")
    print(f"       < 5   → model knows this text well")
    print(f"       5–15  → good generalization")
    print(f"       15–50 → partial understanding")
    print(f"       > 50  → weak fit — consider more training on similar data")

    # ── Save report ──────────────────────────────────────────────────────────
    report_path = (model_manager.model_dir /
                   f"{model_name}_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    try:
        with open(report_path, 'w', encoding='utf-8') as rpt:
            rpt.write("DATASET SCORING REPORT\n")
            rpt.write(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            rpt.write("=" * 60 + "\n")
            rpt.write(f"Model          : {model_name}\n")
            rpt.write(f"Dataset        : {data_source}\n")
            rpt.write(f"Sequences      : {len(dataset):,}\n")
            rpt.write(f"Tokens scored  : {total_tokens:,}\n")
            rpt.write(f"Seq length     : {seq_len}\n")
            rpt.write(f"Batch size     : {batch_size}\n")
            rpt.write(f"Vocab coverage : {coverage:.1f}%\n\n")
            rpt.write(f"Loss           : {avg_loss:.4f}\n")
            rpt.write(f"Perplexity     : {perplexity:.2f}\n")
            rpt.write(f"Accuracy       : {accuracy:.2f}%\n")
            rpt.write(f"Rating         : {rating}\n\n")
            rpt.write("Loss distribution:\n")
            rpt.write(f"  Min    : {min_loss:.4f}\n")
            rpt.write(f"  Q1     : {q1_loss:.4f}\n")
            rpt.write(f"  Median : {med_loss:.4f}\n")
            rpt.write(f"  Q3     : {q3_loss:.4f}\n")
            rpt.write(f"  Max    : {max_loss:.4f}\n")
            rpt.write("=" * 60 + "\n")
        print(f"\n✓  Report saved: {report_path}")
    except Exception as save_err:
        print(f"⚠️   Could not save report: {save_err}")




# ============================================================================
# MULTIPLE-CHOICE QUIZ  (ask the LLM MCQ-style questions — an alternate
# scoring method with a clean, objective right/wrong signal)
#
# Question file format (pipe-delimited, one question per line):
#     question|correct_answer|incorrect1|incorrect2|...
#
# Adapted from a user-provided quiz script. All names below are prefixed
# with "mcq_" / "_mcq_" specifically to avoid colliding with existing
# functions/methods elsewhere in this file (e.g. OllamaValidator.ask_question,
# QuestionFormatter.create_question_file) which have similar names but
# different purposes and signatures.
# ============================================================================

def mcq_choose_file(current=None):
    """
    Pick an MCQ question file using the persistent _mcq_registry so the
    user never has to retype a path once it has been used before.
    """
    saved = _mcq_registry.valid_entries()
    print(f"\n📁 Registry file: {_mcq_registry.REGISTRY_FILE}")

    if saved:
        print(f"\n📂 Saved MCQ question file(s) ({len(saved)} available):\n")
        _mcq_registry.display(saved)
        print("\nEnter a number above, type a new file path, or 'm' to manage saved entries.")
    else:
        print("\nNo saved MCQ question files yet. Type a path to load one, "
              "or create one first (option 4).")

    if current:
        prompt = f"Which file? (Enter to keep '{Path(current).name}') "
    else:
        prompt = "Which file? (blank to cancel) "
    choice = input(prompt).strip()

    if choice == "":
        return current

    if choice.lower() == 'm':
        _mcq_manage_registry()
        return mcq_choose_file(current)

    if choice.isdigit() and saved:
        idx = int(choice) - 1
        if 0 <= idx < len(saved):
            return saved[idx]['path']
        print("⚠️ Invalid number.")
        return None

    # Otherwise treat input as a path (existing file only — use option 4 to create)
    path = choice if choice.endswith(".txt") else choice + ".txt"
    if not Path(path).exists():
        print(f"⚠️ Could not find '{path}'. Use option 4 to create a new file.")
        return None

    resolved = str(Path(path).resolve())
    if _mcq_registry.add(resolved):
        print(f"✓ Added to registry: {resolved}")
    return resolved


def mcq_choose_files(current=None):
    """
    Pick one OR MORE MCQ question files using the persistent _mcq_registry,
    so questions from several files can be combined into a single quiz.

    Extends mcq_choose_file() with multi-file inputs while every existing
    single-file input still works the same way:
      <blank>      — keep current selection (or cancel if none)
      'm'          — manage saved entries (add/remove), then ask again
      'a'          — select every saved file
      '1 3 4'      — space/comma separated numbers to pick several saved files
      a single '2' — pick just that one saved file (same as before)
      a new path   — load it, then prompt for another so several brand-new
                     (not yet saved) files can be queued up in one go

    Returns a list of resolved file paths (empty list if cancelled).
    """
    saved = _mcq_registry.valid_entries()
    print(f"\n📁 Registry file: {_mcq_registry.REGISTRY_FILE}")

    if saved:
        print(f"\n📂 Saved MCQ question file(s) ({len(saved)} available):\n")
        _mcq_registry.display(saved)
        print("\nEnter a number, several numbers separated by spaces (e.g. '1 3'),")
        print("'a' for ALL saved files, a new file path, or 'm' to manage saved entries.")
    else:
        print("\nNo saved MCQ question files yet. Type a path to load one, "
              "or create one first (option 4).")

    current = current or []
    if current:
        names = ", ".join(Path(f).name for f in current)
        prompt = f"Which file(s)? (Enter to keep '{names}') "
    else:
        prompt = "Which file(s)? (blank to cancel) "
    choice = input(prompt).strip()

    if choice == "":
        return list(current)

    if choice.lower() == 'm':
        _mcq_manage_registry()
        return mcq_choose_files(current)

    if choice.lower() == 'a':
        if not saved:
            print("⚠️ No saved files to select.")
            return []
        chosen = [e['path'] for e in saved]
        print(f"✓ Selected all {len(chosen)} saved file(s).")
        return chosen

    tokens = choice.replace(",", " ").split()
    if saved and tokens and all(tok.isdigit() for tok in tokens):
        chosen = []
        for tok in tokens:
            idx = int(tok) - 1
            if 0 <= idx < len(saved):
                if saved[idx]['path'] not in chosen:
                    chosen.append(saved[idx]['path'])
            else:
                print(f"⚠️ Skipping invalid number: {tok}")
        if not chosen:
            print("⚠️ No valid selection.")
        else:
            print(f"✓ Selected {len(chosen)} saved file(s).")
        return chosen

    # Otherwise treat input as one or more NEW paths (existing files only —
    # use option 4 to create). Keep prompting so several new files can be
    # queued up in one go instead of repeating this whole menu each time.
    chosen = []
    raw = choice
    while raw:
        path = raw if raw.endswith(".txt") else raw + ".txt"
        if not Path(path).exists():
            print(f"⚠️ Could not find '{path}'. Use option 4 to create a new file.")
        else:
            resolved = str(Path(path).resolve())
            if _mcq_registry.add(resolved):
                print(f"✓ Added to registry: {resolved}")
            if resolved not in chosen:
                chosen.append(resolved)
        raw = input("Add another new file path? (blank to finish): ").strip()

    return chosen


def _mcq_manage_registry():
    """Simple add/remove management loop for the MCQ file registry."""
    while True:
        print("\n── MCQ File Registry ─────────────────────────────────────")
        _mcq_registry.display()
        print("\n  r    — remove an entry")
        print("  done — finish managing")
        mc = input("Action: ").strip().lower()
        if mc == 'r':
            idx_raw = input("Number to remove: ").strip()
            if idx_raw.isdigit():
                _mcq_registry.remove(int(idx_raw) - 1)
        elif mc in ('done', 'd', ''):
            break


def mcq_create_file():
    """Create a new, empty MCQ question file and register its path."""
    print("\n--- Create a new MCQ question file ---")
    name = input("Enter a name for the new file (e.g. history_quiz): ").strip()
    if not name:
        print("⚠️ No name entered. Cancelled.\n")
        return None
    if not name.endswith(".txt"):
        name += ".txt"

    # If the user gave a bare filename (no path separators), default to a
    # tidy location under saved_models/mcq_datasets/ instead of the CWD.
    if os.sep not in name and '/' not in name:
        target_dir = Path("saved_models") / "mcq_datasets"
        target_dir.mkdir(parents=True, exist_ok=True)
        name = str(target_dir / name)

    if os.path.exists(name):
        print(f"⚠️ '{name}' already exists. Load it instead (option 1), "
              f"or pick a different name.\n")
        return None

    Path(name).parent.mkdir(parents=True, exist_ok=True)
    open(name, "w", encoding="utf-8").close()
    resolved = str(Path(name).resolve())
    _mcq_registry.add(resolved)
    print(f"✅ Created '{resolved}' and added it to the registry.")
    print("   Use 'Add a question' to fill it in.\n")
    return resolved


def mcq_load_questions(filename):
    """
    Load MCQ questions from a pipe-delimited file.
    Each valid line: question|correct_answer|incorrect1|incorrect2|...
    Returns a list of (question, correct, [incorrect, ...]) tuples.
    """
    questions = []
    try:
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("|")
                if len(parts) >= 3:   # question + correct + at least 1 incorrect
                    question  = parts[0]
                    correct   = parts[1]
                    incorrect = parts[2:]
                    questions.append((question, correct, incorrect))
    except FileNotFoundError:
        print(f"⚠️ Could not find '{filename}'.\n")
    return questions


def mcq_add_question(filename):
    """Interactively add one MCQ question, append it to filename, and return it."""
    print("\n--- Add a new MCQ question ---")
    question = input("Enter the question: ").strip()
    correct  = input("Enter the correct answer: ").strip()

    incorrect = []
    print("Enter incorrect answers one at a time. Press Enter on a blank line when done.")
    while True:
        wrong = input(f"Incorrect answer {len(incorrect) + 1} (blank to finish): ").strip()
        if wrong == "":
            break
        incorrect.append(wrong)

    if not question or not correct or not incorrect:
        print("⚠️ A question needs text, a correct answer, and at least one "
              "incorrect answer. Not saved.\n")
        return None

    # The file uses "|" as a delimiter — strip any stray "|" from entered text
    question  = question.replace("|", "/")
    correct   = correct.replace("|", "/")
    incorrect = [w.replace("|", "/") for w in incorrect]

    line = "|".join([question, correct] + incorrect)
    with open(filename, "a", encoding="utf-8") as file:
        file.write(line + "\n")

    print(f"✅ Question saved to {filename}!\n")
    return (question, correct, incorrect)


def mcq_delete_question(filename, questions):
    """Interactively delete one question, rewrite filename, and return the updated list."""
    if not questions:
        print("⚠️ No questions loaded yet. Please load a file first.\n")
        return questions

    print("\n--- Delete a question ---")
    for i, (question, correct, incorrect) in enumerate(questions, 1):
        print(f"{i}. {question}")

    choice = input(f"Enter the number of the question to delete "
                   f"(1-{len(questions)}), or blank to cancel: ").strip()
    if choice == "":
        print("Cancelled.\n")
        return questions

    try:
        index = int(choice) - 1
        if index < 0 or index >= len(questions):
            raise IndexError
    except (ValueError, IndexError):
        print("⚠️ Invalid selection. Nothing deleted.\n")
        return questions

    removed_question, removed_correct, removed_incorrect = questions.pop(index)
    print(f"🗑️ Deleted \"{removed_question}\" along with its correct answer "
          f"({removed_correct}) and incorrect answers "
          f"({', '.join(removed_incorrect)}).\n")

    with open(filename, "w", encoding="utf-8") as file:
        for question, correct, incorrect in questions:
            line = "|".join([question, correct] + incorrect)
            file.write(line + "\n")

    return questions


def _parse_mcq_choice(raw_answer, options):
    """
    Try to determine which option (0-based index) the model's raw text
    answer refers to. Returns None if no confident match can be made.

    Checked in order of confidence:
      1) "Option N" mentioned anywhere
      2) A leading "N)" / "N." / "N:" pattern
      3) Any standalone digit within the valid option range
      4) A unique substring match against one option's own text
         (covers models that echo the option text instead of its number)
    """
    text = (raw_answer or "").strip()
    if not text:
        return None
    n = len(options)

    m = re.search(r'option\s*(\d+)', text, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            return idx

    m = re.match(r'\s*(\d+)\s*[.\):]', text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            return idx

    m = re.search(r'\b(\d+)\b', text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            return idx

    text_lower = text.lower()
    matches = [i for i, opt in enumerate(options)
              if opt.strip() and opt.strip().lower() in text_lower]
    if len(matches) == 1:
        return matches[0]

    return None


def mcq_ask_llm_question(model, tokenizer, device, question_data,
                         q_num=None, total=None, log_fh=None):
    """
    Ask the LLM a single MCQ question and score its answer.

    Mirrors the human-facing ask_question() from the original quiz script,
    but the "input()" call is replaced with a call to the LLM
    (_generate_short_answer), and the LLM's free-text reply is parsed with
    _parse_mcq_choice() to figure out which option it selected.
    """
    question, correct, incorrect = question_data

    # Show at most 3 incorrect options (4 total) — same cap as the original script
    num_incorrect_to_show = min(3, len(incorrect))
    chosen_incorrect = random.sample(incorrect, num_incorrect_to_show)
    options = chosen_incorrect + [correct]
    random.shuffle(options)

    # Build the prompt in the same "Q: ... / A:" convention used throughout
    # this program's training data, so the model recognises where to answer.
    prompt_lines = [f"Q: {question}"]
    for i, opt in enumerate(options, 1):
        prompt_lines.append(f"{i}) {opt}")
    prompt_lines.append("A:")
    full_prompt = "\n".join(prompt_lines)

    tag = f"[{q_num}/{total}] " if q_num and total else ""
    print(f"\n{tag}Question: {question}")
    for i, opt in enumerate(options, 1):
        print(f"   {i}) {opt}")

    # Short, low-drag generation — an MCQ answer should be a few characters
    raw_answer = _generate_short_answer(
        model, tokenizer, full_prompt, device,
        max_length=40, min_new_chars=1
    )

    chosen_idx = _parse_mcq_choice(raw_answer, options)
    is_correct = (chosen_idx is not None and options[chosen_idx] == correct)

    display_answer = raw_answer if raw_answer else "(no output)"
    if chosen_idx is not None:
        print(f"   🤖 Model answered: \"{display_answer}\"  "
              f"→ interpreted as option {chosen_idx + 1}")
    else:
        print(f"   🤖 Model answered: \"{display_answer}\"  "
              f"→ could not interpret a choice")

    if is_correct:
        print(f"   ✅ Correct!")
    else:
        print(f"   ❌ Wrong. Correct answer was: {correct}")

    if log_fh:
        try:
            log_fh.write(f"{tag}Q: {question}\n")
            for i, opt in enumerate(options, 1):
                marker = "  (correct)" if opt == correct else ""
                log_fh.write(f"   {i}) {opt}{marker}\n")
            log_fh.write(f"   Model raw answer : {raw_answer!r}\n")
            if chosen_idx is not None:
                log_fh.write(f"   Interpreted as   : option {chosen_idx + 1}\n")
            else:
                log_fh.write(f"   Interpreted as   : (unparseable)\n")
            log_fh.write(f"   Result           : {'CORRECT' if is_correct else 'WRONG'}\n\n")
            log_fh.flush()
        except Exception:
            pass   # never let log I/O break the quiz loop

    return {
        'question'  : question,
        'correct'   : correct,
        'options'   : options,
        'raw_answer': raw_answer,
        'chosen_idx': chosen_idx,
        'is_correct': is_correct,
    }


def _run_mcq_quiz_session(model, tokenizer, device, model_manager, model_name,
                          questions, source_file):
    """Run every loaded MCQ question against the model, tally the score,
    print a results summary, and save a full log + report to disk."""
    print("\n" + "=" * 60)
    print(f"RUNNING QUIZ  —  {len(questions)} question(s)")
    print("=" * 60)

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = model_manager.model_dir / f"{model_name}_mcq_quiz_{ts}.log"

    log_fh = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, 'w', encoding='utf-8')
        log_fh.write(f"MCQ QUIZ LOG — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log_fh.write(f"Model      : {model_name}\n")
        log_fh.write(f"Source file: {source_file}\n")
        log_fh.write(f"Questions  : {len(questions)}\n")
        log_fh.write("=" * 60 + "\n\n")
        log_fh.flush()
        print(f"📄 Log file: {log_path}")
    except Exception as e:
        print(f"⚠️ Could not create log file: {e}")
        log_fh = None

    results = []
    for i, q_data in enumerate(questions, 1):
        result = mcq_ask_llm_question(
            model, tokenizer, device, q_data,
            q_num=i, total=len(questions), log_fh=log_fh
        )
        results.append(result)

    score      = sum(1 for r in results if r['is_correct'])
    total      = len(results)
    pct        = (score / total * 100) if total else 0.0
    unanswered = sum(1 for r in results if r['chosen_idx'] is None)

    if   pct >= 80: rating = "Excellent 🏆"
    elif pct >= 60: rating = "Good      ✅"
    elif pct >= 40: rating = "Fair      🟡"
    elif pct >= 20: rating = "Poor      🔴"
    else:           rating = "Very Poor ❌"

    print("\n" + "=" * 60)
    print("QUIZ RESULTS")
    print("=" * 60)
    print(f"  Model         : {model_name}")
    print(f"  Source file   : {source_file}")
    print(f"  Score         : {score}/{total}  ({pct:.1f}%)")
    print(f"  Unparseable   : {unanswered}  (model didn't clearly pick an option)")
    print(f"  Rating        : {rating}")
    print("=" * 60)

    if log_fh:
        try:
            log_fh.write("=" * 60 + "\n")
            log_fh.write("SUMMARY\n")
            log_fh.write(f"Score              : {score}/{total} ({pct:.1f}%)\n")
            log_fh.write(f"Unparseable answers: {unanswered}\n")
            log_fh.write(f"Rating             : {rating}\n")
            log_fh.write("=" * 60 + "\n")
            log_fh.close()
            print(f"\n✓ Quiz log saved: {log_path}")
        except Exception:
            pass


def run_mcq_quiz(model, tokenizer, device, model_manager, model_name):
    """
    Top-level MCQ feature entry point — file management + running the quiz.

    Serves as an additional way of scoring the model: instead of judging
    free-form generated text, this gives a clean, objective right/wrong
    signal by asking the model to pick from a fixed set of options.
    """
    current_file  = None       # "active" file — target for add (2) / delete (3) / create (4)
    current_files: list = []   # every file currently contributing loaded questions
    questions: list = []

    print("\n" + "=" * 60)
    print("🧩  MULTIPLE-CHOICE QUIZ  (ask the LLM MCQ-style questions)")
    print("=" * 60)
    print("\nQuestion file format (one question per line):")
    print("  question|correct_answer|incorrect1|incorrect2|...")
    print("\nThe model is shown up to 4 options and must pick one — a clean,")
    print("objective way to score how well it has learned the material.")

    while True:
        print("\n" + "-" * 60)
        if current_files:
            print(f"Current file(s): {', '.join(current_files)}  "
                  f"({len(questions)} question(s) loaded)")
            if len(current_files) > 1:
                print(f"  ℹ️  Add/Delete (2/3) act on ONE active file: {Path(current_file).name}")
        else:
            print("No file selected yet.")
        print("  1. Load / select question file(s)")
        print("  2. Add a question to current file")
        print("  3. Delete a question from current file")
        print("  4. Create a new question file")
        print("  5. Run quiz — ask the LLM all loaded questions")
        print("  6. Back to fine-tuning menu")
        mcq_choice = input("\nSelect option (1-6): ").strip()

        if mcq_choice == "1":
            filenames = mcq_choose_files(current=current_files)
            if filenames:
                questions = []
                for fname in filenames:
                    questions.extend(mcq_load_questions(fname))
                current_files = filenames
                current_file  = filenames[-1]
                if len(filenames) == 1:
                    print(f"✓ {len(questions)} question(s) loaded from '{filenames[0]}'")
                else:
                    names = ", ".join(Path(f).name for f in filenames)
                    print(f"✓ {len(questions)} question(s) loaded from "
                          f"{len(filenames)} files: {names}")

        elif mcq_choice == "2":
            if not current_file:
                print("⚠️ No question file selected. Load or create one first.\n")
            else:
                new_q = mcq_add_question(current_file)
                if new_q:
                    questions.append(new_q)

        elif mcq_choice == "3":
            if not current_file:
                print("⚠️ No question file selected. Load one first.\n")
            else:
                questions = mcq_delete_question(current_file, questions)

        elif mcq_choice == "4":
            new_file = mcq_create_file()
            if new_file:
                current_file  = new_file
                current_files = [new_file]
                questions = []

        elif mcq_choice == "5":
            if not questions:
                print("⚠️ No questions loaded. Load or create a file with "
                      "questions first (options 1 or 4).\n")
            else:
                source_display = ", ".join(current_files) if len(current_files) > 1 else current_file
                _run_mcq_quiz_session(model, tokenizer, device, model_manager,
                                      model_name, questions, source_display)

        elif mcq_choice == "6":
            break
        else:
            print("❌ Invalid option. Please select 1-6.")


def _ask_question_file_scoring(model, tokenizer, device, model_manager, model_name,
                                train_loader=None, lr=0.001):
    """
    Offer user option to run question-file scoring against the model.
    If score is below threshold, trigger extra training automatically.
    When no train_loader is provided (e.g. called from Option 3), ask the
    user to supply training data before retraining.
    """
    print("\n" + "="*60)
    print("QUESTION-FILE SCORING  (optional)")
    print("="*60)
    print("Score your model on a set of questions (0-10 per answer).")
    print("If average score is below your threshold, extra training starts.")
    use_qs = input("\nScore the model using a question file? (y/n): ").strip().lower()
    if use_qs != 'y':
        return

    qs_path = input("Path to question file (.txt, one question per line): ").strip()
    questions = _load_questions_from_file(qs_path)
    if not questions:
        return

    # Threshold
    try:
        threshold = float(input("Minimum average score to PASS (0-10, default 6): ").strip() or "6")
        threshold = max(0.0, min(10.0, threshold))
    except ValueError:
        threshold = 6.0

    # Extra epochs if below threshold
    try:
        extra_ep = int(input("Extra epochs to train if score is BELOW threshold (default 10): ").strip() or "10")
        extra_ep = max(1, extra_ep)
    except ValueError:
        extra_ep = 10

    # Optional Ollama evaluator
    evaluator = None
    use_ollama = input("Use Ollama to score answers automatically? (y/n): ").strip().lower()
    if use_ollama == 'y':
        v = OllamaValidator()
        if v.available and v.list_models():
            context = input("Optional scoring context for Ollama (or Enter to skip): ").strip()
            evaluator = SelfImprovingEvaluator(v, user_context=context)
        else:
            print("⚠️  Ollama not available — falling back to manual scoring.")

    final_avg = _score_model_on_questions(
        model, tokenizer, device, model_manager, model_name,
        evaluator, questions,
        low_score_threshold=threshold,
        extra_epochs=extra_ep,
        train_loader=train_loader,
        learning_rate=lr,
    )

    # If no train_loader was supplied but the score was low, ask for data now
    if (final_avg is not None and
            final_avg < threshold and
            train_loader is None):
        print(f"\n⚠️  Score {final_avg:.2f} is below threshold {threshold:.1f}")
        print("   No training data was loaded. Load data now to trigger retraining?")
        load_now = input("Load training data and retrain? (y/n): ").strip().lower()
        if load_now == 'y':
            training_text, data_source, data_type = _load_training_data("retraining")
            if training_text:
                try:
                    bs = max(1, int(input("Batch size (default 4): ").strip() or "4"))
                except ValueError:
                    bs = 4
                try:
                    new_lr = float(input(f"Learning rate (default {lr}): ").strip() or str(lr))
                except ValueError:
                    new_lr = lr
                dataset      = TextDataset(training_text, tokenizer, seq_len=32)
                new_loader   = DataLoader(dataset, batch_size=bs, shuffle=True)
                print(f"\n▶  Retraining for {extra_ep} epochs…")
                train_model(
                    model, new_loader,
                    num_epochs=extra_ep,
                    learning_rate=new_lr,
                    device=device,
                    model_manager=model_manager,
                    model_name=model_name,
                )
                if model_manager:
                    model_manager.save_model(model, model_name)
                    print(f"✓ Model saved after retraining: {model_name}")


# ============================================================================
# MAIN MENU
# ============================================================================
def main_menu():

    print("\n" + "="*60)
    print("🤖  TINY LLM — INTERACTIVE INTERFACE")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model_manager = ModelManager()

    # ── Shared helpers ────────────────────────────────────────────────────────
    def _pick_saved_model(prompt="Select model: "):
        """Return model_name chosen from saved models, or None."""
        saved = model_manager.list_models_simple()
        if not saved:
            print("❌ No saved models found.  Train a model first (Option 1).")
            return None
        print()
        for i, m in enumerate(saved, 1):
            print(f"  {i}. {m}")
        raw = input(f"\n{prompt}").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(saved):
                return saved[idx]
        if raw in saved:
            return raw
        print(f"❌ '{raw}' not found.")
        return None

    def _load_model_and_tokenizer(model_name):
        """Load tokenizer + model weights. Returns (model, tokenizer) or (None, None)."""
        tok_path = model_manager.model_dir / f"{model_name}_tokenizer.json"
        if not tok_path.exists():
            print(f"❌ Tokenizer not found for '{model_name}'.")
            return None, None
        tokenizer = CharTokenizer.load(str(tok_path))
        model = SimpleLLM(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_blocks=2, num_heads=4, max_len=100
        ).to(device)
        model_manager.load_model(model, model_name)
        print(f"✓ Loaded '{model_name}'  (vocab {tokenizer.vocab_size})")
        return model, tokenizer
    # ─────────────────────────────────────────────────────────────────────────

    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("  1. Train a new model")
        print("  2. Fine-tune existing model")
        print("  3. Load and generate with existing model")
        print("  4. Export / Serve model for Ollama")
        print("  5. List saved models")
        print("  6. Exit")
        choice = input("\nSelect option (1-6): ").strip()

        # ════════════════════════════════════════════════════════════════
        # OPTION 1 — TRAIN A NEW MODEL
        # ════════════════════════════════════════════════════════════════
        if choice == "1":
            print("\n" + "="*60)
            print("TRAIN A NEW MODEL")
            print("="*60)

            training_text, data_source, data_type = _load_training_data("training")
            if not training_text:
                continue

            model_name = input("\nModel name: ").strip() or "default_model"

            # ── Check for saved session / backup ──────────────────────
            backup_path = model_manager.model_dir / f"{model_name}_backup.pt"
            tsm_check   = TrainingStateManager(model_name, model_dir=str(model_manager.model_dir))
            saved_state = tsm_check.load()

            if (saved_state and
                    saved_state.get("status") != TrainingStateManager.STATUS_COMPLETE and
                    backup_path.exists()):
                print(f"\n📋 Saved training session found:")
                TrainingStateManager.print_state(saved_state)
                if input("\nReload all saved parameters and resume? (y/n): ").strip().lower() == 'y':
                    data_source   = saved_state["dataset_path"]
                    data_type     = saved_state.get("dataset_type", "file")
                    batch_size    = saved_state["batch_size"]
                    epochs        = saved_state["num_epochs"]
                    lr            = saved_state["learning_rate"]
                    adamw_wd      = saved_state.get("weight_decay", ADAMW_DEFAULT_WEIGHT_DECAY)
                    target_loss   = saved_state.get("target_loss")
                    chk_interval  = saved_state.get("checkpoint_interval", 5)
                    start_epoch   = saved_state.get("epochs_trained", 0)
                    try:
                        if data_type == "folder":
                            training_text = "\n\n".join(
                                p.read_text(encoding='utf-8')
                                for p in sorted(Path(data_source).glob("*.txt")))
                        else:
                            training_text = Path(data_source).read_text(encoding='utf-8')
                        print(f"✓ Dataset reloaded ({len(training_text):,} chars)")
                    except Exception as e:
                        print(f"❌ Could not reload dataset: {e}")
                        continue
                    tokenizer    = CharTokenizer.load(
                        str(model_manager.model_dir / f"{model_name}_tokenizer.json"))
                    model        = SimpleLLM(vocab_size=tokenizer.vocab_size,
                                            d_model=64, num_blocks=2, num_heads=4,
                                            max_len=100).to(device)
                    model_manager.load_model(model, model_name)
                    dataset      = TextDataset(training_text, tokenizer, seq_len=32)
                    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    logger       = TrainingLogger(model_name, model_dir=str(model_manager.model_dir))
                    print(f"\n▶  Resuming epoch {start_epoch + 1} → {epochs}")
                    paused = train_model(model, train_loader, num_epochs=epochs,
                        learning_rate=lr, device=device, model_manager=model_manager,
                        model_name=model_name, start_epoch=start_epoch,
                        checkpoint_interval=chk_interval, logger=logger,
                        target_loss=target_loss, state_manager=tsm_check,
                        weight_decay=adamw_wd)
                    print("\n⏸️  Paused." if paused else "\n✓ Complete!")
                    logger.display_summary()
                    if not paused:
                        _print_training_summary(model, tokenizer, model_name, tsm_check)
                    continue
            # ──────────────────────────────────────────────────────────

            # Config
            print("\n" + "="*60)
            print("TRAINING CONFIGURATION")
            print("="*60)
            try:
                epochs = max(1, int(input("Epochs (default 10): ").strip() or 10))
            except ValueError:
                epochs = 10
            try:
                lr = float(input("Learning rate (default 0.001): ").strip() or 0.001)
            except ValueError:
                lr = 0.001
            adamw_wd = _adamw_ask_weight_decay()
            try:
                batch_size = max(1, int(input("Batch size (default 4): ").strip() or 4))
            except ValueError:
                batch_size = 4
            try:
                chk_interval = max(1, int(input("Backup every N epochs (default 5): ").strip() or 5))
            except ValueError:
                chk_interval = 5
            try:
                bpe_vocab = max(256, int(input("BPE vocabulary size, min 256 (default 512): ").strip() or 512))
            except ValueError:
                bpe_vocab = 512

            # Early stop
            tl_raw = input("Early-stop target loss (Enter to skip): ").strip()
            target_loss = None
            try:
                if tl_raw:
                    target_loss = float(tl_raw)
                    print(f"✓ Early stop at loss ≤ {target_loss:.4f}")
            except ValueError:
                pass

            # Build tokenizer + model + loader
            tokenizer    = BPETokenizer(training_text, vocab_size=bpe_vocab)
            model        = SimpleLLM(vocab_size=tokenizer.vocab_size,
                                     d_model=64, num_blocks=2, num_heads=4,
                                     max_len=100).to(device)
            dataset      = TextDataset(training_text, tokenizer, seq_len=32)
            if len(dataset) == 0:
                print(f"❌ Training text is too short: it encodes to only {len(dataset.tokens)} "
                      f"tokens (need at least 33). Use more text or a smaller BPE vocabulary.")
                continue
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Determine start_epoch from backup
            start_epoch = 0
            if backup_path.exists():
                if input(f"\n⚠  Backup found for '{model_name}'. Resume? (y/n): ").strip().lower() == 'y':
                    start_epoch = 1
                    # The backup was trained with the SAVED tokenizer.  Keep using it instead of
                    # the one just rebuilt (a different BPE vocab size would no longer fit the
                    # backup, and saving the new tokenizer would overwrite the old one).
                    saved_tok_path = model_manager.model_dir / f"{model_name}_tokenizer.json"
                    if saved_tok_path.exists():
                        tokenizer    = CharTokenizer.load(str(saved_tok_path))
                        model        = SimpleLLM(vocab_size=tokenizer.vocab_size,
                                                 d_model=64, num_blocks=2, num_heads=4,
                                                 max_len=100).to(device)
                        dataset      = TextDataset(training_text, tokenizer, seq_len=32)
                        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Save session state
            tsm = TrainingStateManager(model_name, model_dir=str(model_manager.model_dir))
            tsm.save({"dataset_path": str(data_source), "dataset_type": data_type,
                      "batch_size": batch_size, "num_epochs": epochs,
                      "epochs_trained": start_epoch, "learning_rate": lr,
                      "weight_decay": adamw_wd,
                      "target_loss": target_loss, "checkpoint_interval": chk_interval,
                      "last_loss": None,
                      "status": TrainingStateManager.STATUS_IN_PROGRESS,
                      "created_at": str(datetime.now())})
            model_manager.register_model(model_name,
                {"status": TrainingStateManager.STATUS_IN_PROGRESS,
                 "epochs_trained": start_epoch, "num_epochs": epochs,
                 "dataset_path": str(data_source), "last_loss": None})
            logger = TrainingLogger(model_name, model_dir=str(model_manager.model_dir))
            model_manager.save_tokenizer(tokenizer, model_name)

            paused = train_model(model, train_loader, num_epochs=epochs,
                learning_rate=lr, device=device, model_manager=model_manager,
                model_name=model_name, start_epoch=start_epoch,
                checkpoint_interval=chk_interval, logger=logger,
                target_loss=target_loss, state_manager=tsm,
                weight_decay=adamw_wd)
            print("\n⏸️  Training paused." if paused else "\n✓ Training complete!")
            logger.display_summary()
            if not paused:
                _print_training_summary(model, tokenizer, model_name, tsm)

        # ════════════════════════════════════════════════════════════════
        # OPTION 2 — FINE-TUNE EXISTING MODEL
        # ════════════════════════════════════════════════════════════════
        elif choice == "2":
            while True:
                print("\n" + "="*60)
                print("FINE-TUNE EXISTING MODEL")
                print("="*60)
                print("  1. Train more epochs on new or old data")
                print("  2. Ollama model-based feedback learning")
                print("  3. Use already-trained LLM for reinforcement feedback learning")
                print("  4. Score model on dataset  (no training — pure evaluation)")
                print("  5. Validate model          (repeatable — overfitting check)")
                print("  6. Test model              (one-shot final benchmark)")
                print("  7. Train on conversational dataset file  (no Ollama needed)")
                print("  8. Multiple-choice quiz    (ask the LLM MCQ-style questions)")
                print("  9. Back to main menu")
                sub = input("\nSelect sub-option (1-9): ").strip()

                # ── 2-1: MORE EPOCHS ─────────────────────────────────
                if sub == "1":
                    print("\n" + "="*60)
                    print("FINE-TUNE: MORE EPOCHS")
                    print("="*60)

                    model_name = _pick_saved_model("Model to fine-tune: ")
                    if not model_name:
                        continue

                    model, tokenizer = _load_model_and_tokenizer(model_name)
                    if model is None:
                        continue

                    # Check for resume
                    tsm_ft   = TrainingStateManager(model_name,
                                                    model_dir=str(model_manager.model_dir))
                    saved_ft = tsm_ft.load()
                    start_epoch = 0
                    training_text, data_source, data_type = None, None, None
                    batch_size, lr, epochs, chk_interval, target_loss = 4, 0.001, 10, 5, None

                    if (saved_ft and
                            saved_ft.get("status") != TrainingStateManager.STATUS_COMPLETE and
                            (model_manager.model_dir / f"{model_name}_backup.pt").exists()):
                        print("\n📋 Previous session:")
                        TrainingStateManager.print_state(saved_ft)
                        if input("Resume from saved parameters? (y/n): ").strip().lower() == 'y':
                            data_source  = saved_ft["dataset_path"]
                            data_type    = saved_ft.get("dataset_type", "file")
                            batch_size   = saved_ft["batch_size"]
                            epochs       = saved_ft["num_epochs"]
                            lr           = saved_ft["learning_rate"]
                            adamw_wd     = saved_ft.get("weight_decay", ADAMW_DEFAULT_WEIGHT_DECAY)
                            target_loss  = saved_ft.get("target_loss")
                            chk_interval = saved_ft.get("checkpoint_interval", 5)
                            start_epoch  = saved_ft.get("epochs_trained", 0)
                            try:
                                if data_type == "folder":
                                    training_text = "\n\n".join(
                                        p.read_text(encoding='utf-8')
                                        for p in sorted(Path(data_source).glob("*.txt")))
                                else:
                                    training_text = Path(data_source).read_text(encoding='utf-8')
                            except Exception as e:
                                print(f"❌ Could not reload data: {e}")
                                training_text = None

                    if training_text is None:
                        # Ask for new or old data
                        use_new = input("\nLoad new training data? (y = new file, n = use previous): ").strip().lower()
                        if use_new == 'y':
                            training_text, data_source, data_type = _load_training_data("fine-tuning")
                            if not training_text:
                                continue
                        elif saved_ft and saved_ft.get("dataset_path"):
                            try:
                                dp, dt = saved_ft["dataset_path"], saved_ft.get("dataset_type", "file")
                                if dt == "folder":
                                    training_text = "\n\n".join(
                                        p.read_text(encoding='utf-8')
                                        for p in sorted(Path(dp).glob("*.txt")))
                                else:
                                    training_text = Path(dp).read_text(encoding='utf-8')
                                data_source, data_type = dp, dt
                                print(f"✓ Reloaded previous dataset ({len(training_text):,} chars)")
                            except Exception as e:
                                print(f"❌ Could not reload: {e}")
                                continue
                        else:
                            print("❌ No previous dataset found. Load new data.")
                            continue

                        # Config
                        try:
                            epochs = max(1, int(input("Extra epochs (default 10): ").strip() or 10))
                        except ValueError:
                            epochs = 10
                        try:
                            lr = float(input("Learning rate (default 0.001): ").strip() or 0.001)
                        except ValueError:
                            lr = 0.001
                        adamw_wd = _adamw_ask_weight_decay()
                        try:
                            batch_size = max(1, int(input("Batch size (default 4): ").strip() or 4))
                        except ValueError:
                            batch_size = 4
                        try:
                            chk_interval = max(1, int(input("Backup every N epochs (default 5): ").strip() or 5))
                        except ValueError:
                            chk_interval = 5
                        tl_raw = input("Early-stop target loss (Enter to skip): ").strip()
                        target_loss = None
                        try:
                            if tl_raw:
                                target_loss = float(tl_raw)
                        except ValueError:
                            pass

                    dataset      = TextDataset(training_text, tokenizer, seq_len=32)
                    if len(dataset) == 0:
                        print(f"❌ Training text is too short: it encodes to only "
                              f"{len(dataset.tokens)} tokens (need at least 33).")
                        continue
                    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                    tsm_ft.save({"dataset_path": str(data_source), "dataset_type": data_type,
                                 "batch_size": batch_size, "num_epochs": epochs + start_epoch,
                                 "epochs_trained": start_epoch, "learning_rate": lr,
                                 "weight_decay": adamw_wd,
                                 "target_loss": target_loss, "checkpoint_interval": chk_interval,
                                 "last_loss": None,
                                 "status": TrainingStateManager.STATUS_IN_PROGRESS,
                                 "created_at": str(datetime.now())})
                    logger = TrainingLogger(model_name, model_dir=str(model_manager.model_dir))
                    total_epochs = epochs + start_epoch

                    paused = train_model(model, train_loader,
                        num_epochs=total_epochs, learning_rate=lr, device=device,
                        model_manager=model_manager, model_name=model_name,
                        start_epoch=start_epoch, checkpoint_interval=chk_interval,
                        logger=logger, target_loss=target_loss, state_manager=tsm_ft,
                        weight_decay=adamw_wd)
                    print("\n⏸️  Paused." if paused else "\n✓ Fine-tuning complete!")
                    logger.display_summary()
                    if not paused:
                        _print_training_summary(model, tokenizer, model_name, tsm_ft)

                    # Optional question-file scoring
                    _ask_question_file_scoring(model, tokenizer, device,
                                               model_manager, model_name,
                                               train_loader=train_loader, lr=lr)

                # ── 2-2: OLLAMA FEEDBACK LEARNING ────────────────────
                elif sub == "2":
                    print("\n" + "="*60)
                    print("OLLAMA MODEL-BASED FEEDBACK LEARNING")
                    print("="*60)

                    # Connect to Ollama first
                    validator = OllamaValidator()
                    if not validator.available:
                        print("❌ Ollama not running. Start Ollama and retry.")
                        continue

                    model_name = _pick_saved_model("Local model to fine-tune: ")
                    if not model_name:
                        continue
                    model, tokenizer = _load_model_and_tokenizer(model_name)
                    if model is None:
                        continue

                    rl_context = _collect_user_context("Ollama Feedback Learning")
                    finetune_with_ollama(model, tokenizer, device=device,
                                         model_manager=model_manager,
                                         model_name=model_name,
                                         logger=TrainingLogger(model_name),
                                         user_context=rl_context)

                    # Optional question-file scoring after Ollama fine-tuning
                    _ask_question_file_scoring(model, tokenizer, device,
                                               model_manager, model_name)

                # ── 2-3: RL WITH ALREADY-TRAINED LLM ─────────────────
                elif sub == "3":
                    print("\n" + "="*60)
                    print("REINFORCEMENT FEEDBACK LEARNING (LLM as teacher)")
                    print("="*60)

                    # Ollama evaluator
                    evaluator_v = OllamaValidator()
                    if not evaluator_v.available:
                        print("❌ Ollama not running. Start Ollama and retry.")
                        continue

                    model_name = _pick_saved_model("Model to improve: ")
                    if not model_name:
                        continue
                    model, tokenizer = _load_model_and_tokenizer(model_name)
                    if model is None:
                        continue

                    rl_context = _collect_user_context("RL Feedback Learning")
                    target_avg = None
                    raw_avg = input("Pause when average score reaches (0-10, Enter to skip): ").strip()
                    try:
                        target_avg = float(raw_avg)
                    except ValueError:
                        pass

                    self_improving_qa(model, tokenizer, device=device,
                                      model_manager=model_manager,
                                      model_name=model_name,
                                      user_context=rl_context,
                                      preset_target_avg=target_avg)

                    # Optional question-file scoring after RL session
                    _ask_question_file_scoring(model, tokenizer, device,
                                               model_manager, model_name)

                # ── 2-4: DATASET SCORING ─────────────────────────────────
                elif sub == "4":
                    print("\n" + "="*60)
                    print("SCORE MODEL ON DATASET")
                    print("="*60)

                    model_name = _pick_saved_model("Model to score: ")
                    if not model_name:
                        continue

                    model, tokenizer = _load_model_and_tokenizer(model_name)
                    if model is None:
                        continue

                    score_model_on_dataset(
                        model, tokenizer, device, model_manager, model_name
                    )

                # ── 2-5: VALIDATE ─────────────────────────────────────
                elif sub == "5":
                    print("\n" + "="*60)
                    print("VALIDATE MODEL ON VALIDATION DATASET")
                    print("="*60)

                    model_name = _pick_saved_model("Model to validate: ")
                    if not model_name:
                        continue

                    model, tokenizer = _load_model_and_tokenizer(model_name)
                    if model is None:
                        continue

                    tsm_val = TrainingStateManager(
                        model_name, model_dir=str(model_manager.model_dir)
                    )
                    validate_model_on_dataset(
                        model, tokenizer, device, model_manager,
                        model_name, state_manager=tsm_val
                    )

                # ── 2-6: TEST ─────────────────────────────────────────
                elif sub == "6":
                    print("\n" + "="*60)
                    print("TEST MODEL ON TEST DATASET  (final benchmark)")
                    print("="*60)

                    model_name = _pick_saved_model("Model to test: ")
                    if not model_name:
                        continue

                    model, tokenizer = _load_model_and_tokenizer(model_name)
                    if model is None:
                        continue

                    test_model_on_dataset(
                        model, tokenizer, device, model_manager, model_name
                    )

                # ── 2-7: TRAIN ON CONVERSATIONAL DATASET FILE ────────
                elif sub == "7":
                    print("\n" + "="*60)
                    print("TRAIN ON CONVERSATIONAL DATASET FILE")
                    print("="*60)

                    model_name = _pick_saved_model("Model to fine-tune: ")
                    if not model_name:
                        continue

                    model, tokenizer = _load_model_and_tokenizer(model_name)
                    if model is None:
                        continue

                    finetune_on_conversational_dataset(
                        model, tokenizer, device=device,
                        model_manager=model_manager, model_name=model_name
                    )

                    # Optional question-file scoring after training
                    _ask_question_file_scoring(model, tokenizer, device,
                                               model_manager, model_name)

                # ── 2-8: MULTIPLE-CHOICE QUIZ ────────────────────────
                elif sub == "8":
                    model_name = _pick_saved_model("Model to quiz: ")
                    if not model_name:
                        continue

                    model, tokenizer = _load_model_and_tokenizer(model_name)
                    if model is None:
                        continue

                    run_mcq_quiz(model, tokenizer, device, model_manager, model_name)

                elif sub == "9":
                    break
                else:
                    print("❌ Invalid sub-option. Please select 1-9.")

        # ════════════════════════════════════════════════════════════════
        # OPTION 3 — LOAD AND GENERATE
        # ════════════════════════════════════════════════════════════════
        elif choice == "3":
            print("\n" + "="*60)
            print("LOAD AND GENERATE")
            print("="*60)

            model_name = _pick_saved_model("Model to load: ")
            if not model_name:
                continue

            model, tokenizer = _load_model_and_tokenizer(model_name)
            if model is None:
                continue

            # Optional Ollama answer-checking
            evaluator = None
            use_ollama = input("\nUse Ollama to check/score generated answers? (y/n): ").strip().lower()
            if use_ollama == 'y':
                ov = OllamaValidator()
                if ov.available and ov.list_models():
                    ctx = input("Optional scoring context (Enter to skip): ").strip()
                    evaluator = SelfImprovingEvaluator(ov, user_context=ctx)
                    print("✓ Ollama scoring enabled.")
                else:
                    print("⚠️  Ollama unavailable — generating without scoring.")

            # Generate loop
            print("\nType a prompt and press Enter. Type 'quit' to exit.")
            if evaluator:
                print("Generated answers will be scored by Ollama automatically.")
            print("─"*60)

            while True:
                prompt = input("\nPrompt: ").strip()
                if prompt.lower() == 'quit':
                    break

                answer = _generate_short_answer(model, tokenizer, prompt, device, max_length=200)
                print(f"\n📝 Generated: {answer if answer else '(no output)'}")

                if evaluator:
                    score, comment = evaluator.evaluate_answer(prompt, answer)
                    if score is not None:
                        s_fill = int(score / 10 * 20)
                        print(f"⭐ Score [{' '*0}{'█'*s_fill}{'░'*(20-s_fill)}] {score:.1f}/10")
                        if comment:
                            print(f"📌 {comment}")

            # Optional question-file scoring
            _ask_question_file_scoring(model, tokenizer, device,
                                       model_manager, model_name)

        # ════════════════════════════════════════════════════════════════
        # OPTION 4 — EXPORT / SERVE MODEL FOR OLLAMA
        # ════════════════════════════════════════════════════════════════
        elif choice == "4":
            print("\n" + "="*60)
            print("EXPORT / SERVE MODEL FOR OLLAMA")
            print("="*60)
            print("  1. Start local Ollama-compatible server")
            print("     (other apps can use your model via the Ollama API)")
            print("  2. Export model package")
            print("     (copies weights + tokenizer + Modelfile + launcher)")
            print("  3. Back to main menu")
            sub4 = input("\nSelect (1-3): ").strip()

            if sub4 not in ("1", "2"):
                continue

            model_name = _pick_saved_model("Model to export/serve: ")
            if not model_name:
                continue
            model, tokenizer = _load_model_and_tokenizer(model_name)
            if model is None:
                continue

            if sub4 == "1":
                # ── Start local Ollama-compatible server ──────────────
                try:
                    port_raw = input(
                        "Port to listen on (default 11435, "
                        "real Ollama uses 11434): "
                    ).strip()
                    port = int(port_raw) if port_raw.isdigit() else 11435
                except ValueError:
                    port = 11435

                try:
                    temp_raw = input("Generation temperature (default 0.8): ").strip()
                    temperature = float(temp_raw) if temp_raw else 0.8
                except ValueError:
                    temperature = 0.8

                try:
                    tok_raw = input("Max tokens per response (default 200): ").strip()
                    max_tok = int(tok_raw) if tok_raw.isdigit() else 200
                except ValueError:
                    max_tok = 200

                server = OllamaCompatibleServer(
                    model, tokenizer, model_name,
                    device=device, port=port,
                    temperature=temperature, max_tokens=max_tok,
                )
                server.start()

                print("\nServer is running in the background.")
                print("Press Enter to stop the server and return to the menu.")
                try:
                    input()
                except (KeyboardInterrupt, EOFError):
                    pass
                finally:
                    server.stop()

            elif sub4 == "2":
                # ── Export package ────────────────────────────────────
                default_dir = f"ollama_export_{model_name}"
                dir_raw = input(
                    f"Export folder (default: {default_dir}): "
                ).strip()
                export_dir = dir_raw if dir_raw else default_dir
                export_model_for_ollama(model, tokenizer, model_name,
                                        model_manager, export_dir=export_dir)

        # ════════════════════════════════════════════════════════════════
        # OPTION 5 — LIST SAVED MODELS
        # ════════════════════════════════════════════════════════════════
        elif choice == "5":
            model_manager.list_models()

        # ════════════════════════════════════════════════════════════════
        # OPTION 6 — EXIT
        # ════════════════════════════════════════════════════════════════
        elif choice == "6":
            print("\nGoodbye! 👋")
            break

        else:
            print("❌ Invalid option. Please select 1-6.")


# ============================================================================
# RUN THE PROGRAM
# ============================================================================
if __name__ == "__main__":
    main_menu()
