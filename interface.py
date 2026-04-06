import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import jiwer
from nltk.translate.bleu_score import sentence_bleu
import torchaudio
import spacy
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

class SpeechAnalysisPipeline:
    def __init__(self):
        st.set_page_config(page_title="Speech Analysis Pipeline", layout="wide")
        self.initialize_session_state()
        self.load_models()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'transcriptions' not in st.session_state:
            st.session_state.transcriptions = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = None

    def load_models(self):
        """Load all required models"""
        try:
            with st.spinner('Loading transcription model...'):
                from transcription_model import TranscriptionModel
                self.transcriber = TranscriptionModel()
                st.success("Transcription model loaded!")
                
            with st.spinner('Loading analysis models...'):
                from mt5_model import MT5Processor
                from xlmr_model import XLMRProcessor
                from tinybert_model import TinyBERTProcessor
                
                self.mt5 = MT5Processor()
                self.xlmr = XLMRProcessor()
                self.tinybert = TinyBERTProcessor()
                st.success("Analysis models loaded!")
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            raise

    def main(self):
        """Main application interface"""
        st.title("Speech Analysis Pipeline")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select Stage",
            ["Transcription", "Slots & Intents", "Comparison"]
        )

        # Page routing
        if page == "Transcription":
            self.transcription_page()
        elif page == "Slots & Intents":
            self.slots_intents_page()
        else:
            self.comparison_page()

    def transcription_page(self):
        """Audio transcription or upload interface"""
        st.header("Transcription / Upload English Text")

        # Dataset selection
        dataset_choice = st.selectbox(
            "Select Dataset",
            ["Common Voice", "Lahaja"],
            key="dataset_choice"
        )
        st.session_state.selected_dataset = dataset_choice

        st.markdown("**Option 1: Upload Audio Files**")
        uploaded_files = st.file_uploader(
            f"Upload Audio Files for {dataset_choice}",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            key="audio_uploader"
        )

        st.markdown("**Option 2: Upload Pre-Transcribed File (TSV/CSV, must have a 'text' column in English)**")
        uploaded_trans_file = st.file_uploader(
            "Upload Transcribed File",
            type=["tsv", "csv"],
            accept_multiple_files=False,
            key="trans_file_uploader"
        )

        # If user uploads a transcribed file, use that
        if uploaded_trans_file is not None:
            ext = Path(uploaded_trans_file.name).suffix
            if ext == ".csv":
                df = pd.read_csv(uploaded_trans_file)
            else:
                df = pd.read_csv(uploaded_trans_file, sep="\t")
            # Automatically rename 'transcription' to 'text' if needed
            if "text" not in df.columns and "transcription" in df.columns:
                df = df.rename(columns={"transcription": "text"})
            if "text" not in df.columns:
                st.error("Uploaded file must have a 'text' column (or 'transcription' column) with English transcriptions.")
            else:
                st.session_state.transcriptions = df
                st.session_state.slots_intents_results = None
                st.session_state.analysis_results = {}
                st.session_state.comparison_results = None
                st.success("Transcriptions loaded from file!")
                self.display_transcriptions()

        # If user uploads audio files, process them
        if uploaded_files:
            with st.spinner('Processing audio files...'):
                results = []
                progress_bar = st.progress(0)
                for i, file in enumerate(uploaded_files):
                    try:
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        # Transcribe
                        result = self.transcriber.transcribe(temp_path)
                        results.append({
                            "filename": file.name,
                            "dataset": dataset_choice,
                            **result
                        })
                        os.remove(temp_path)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                if results:
                    df = pd.DataFrame(results)
                    # If Lahaja, translate to English
                    if dataset_choice == "Lahaja":
                        st.info("Translating Lahaja transcriptions to English...")
                        df["text"] = self.translate_lahaja(df["text"].tolist())
                    st.session_state.transcriptions = df
                    st.session_state.slots_intents_results = None
                    st.session_state.analysis_results = {}
                    st.session_state.comparison_results = None
                    st.success("Transcriptions ready!")
                    self.display_transcriptions()

        st.info("Upload audio files or a transcribed file to proceed.")

    def translate_lahaja(self, texts):
        # Stub: Replace with actual translation logic or import
        # For now, just return the same texts (no-op)
        # You can import and call your translation function here
        return texts

    def slots_intents_page(self):
        """Extract and display slots and intents for transcriptions or uploaded file"""
        st.header("Slots & Intents Extraction")

        # Allow user to upload a transcribed file here as well
        st.markdown("**Option: Upload Transcribed File (TSV/CSV, must have a 'text' column in English)**")
        uploaded_trans_file = st.file_uploader(
            "Upload Transcribed File for Slot/Intent Extraction",
            type=["tsv", "csv"],
            accept_multiple_files=False,
            key="slots_intents_trans_file_uploader"
        )
        if uploaded_trans_file is not None:
            ext = Path(uploaded_trans_file.name).suffix
            if ext == ".csv":
                df = pd.read_csv(uploaded_trans_file)
            else:
                df = pd.read_csv(uploaded_trans_file, sep="\t")
            # Automatically rename 'transcription' to 'text' if needed
            if "text" not in df.columns and "transcription" in df.columns:
                df = df.rename(columns={"transcription": "text"})
            if "text" not in df.columns:
                st.error("Uploaded file must have a 'text' column (or 'transcription' column) with English transcriptions.")
            else:
                st.session_state.transcriptions = df
                st.session_state.slots_intents_results = None
                st.session_state.analysis_results = {}
                st.session_state.comparison_results = None
                st.success("Transcriptions loaded from file!")
                self.display_transcriptions()

        # Ensure transcriptions are available
        if "transcriptions" not in st.session_state or st.session_state.transcriptions is None:
            st.warning("Please complete transcription or upload a transcribed file first!")
            return

        # Model selection for single extraction
        model_choice = st.selectbox(
            "Select Model",
            ["MT5", "XLM-R", "TinyBERT"],
            key="slots_intents_model_choice"
        )

        if st.button("Extract Slots & Intents"):
            with st.spinner(f'Extracting slots and intents with {model_choice}...'):
                texts = st.session_state.transcriptions["text"].tolist()
                if model_choice == "MT5":
                    results = [self.mt5.process_text(t) for t in texts]
                elif model_choice == "XLM-R":
                    results = [self.xlmr.process_text(t) for t in texts]
                else:
                    results = [self.tinybert.process_text(t) for t in texts]
                st.session_state.slots_intents_results = results

        # Show results if available
        if "slots_intents_results" in st.session_state:
            results = st.session_state.slots_intents_results
            df_results = pd.DataFrame(results)
            # Convert lists/dicts to pretty strings for display
            for col in ["intents", "slots"]:
                if col in df_results.columns:
                    df_results[col] = df_results[col].apply(lambda x: json.dumps(x, ensure_ascii=False, indent=2) if isinstance(x, (list, dict)) else str(x))
            st.subheader(f"{model_choice} - Slots & Intents")
            st.dataframe(df_results[[col for col in df_results.columns if col in ["text", "intents", "slots"]]])

        # --- NEW: Button to run all models and store in analysis_results for comparison page ---
        st.markdown("---")
        st.markdown("### Run Analysis for All Models (Required for Comparison Page)")
        if st.button("Run Analysis for All Models"):
            with st.spinner("Running all models on current transcriptions..."):
                texts = st.session_state.transcriptions["text"].tolist()
                analysis_results = {}
                analysis_results["MT5"] = [self.mt5.process_text(t) for t in texts]
                analysis_results["XLM-R"] = [self.xlmr.process_text(t) for t in texts]
                analysis_results["TinyBERT"] = [self.tinybert.process_text(t) for t in texts]
                st.session_state.analysis_results = analysis_results
                st.success("Analysis results for all models are ready! You can now use the Comparison page.")

    def comparison_page(self):
        """Model Results & Distribution Graphs interface"""
        st.header("Model Results & Distribution Graphs")

        if not st.session_state.analysis_results:
            st.warning("Please run analysis with different models first!")
            return

        # Model selection for results panel
        model_choice = st.selectbox(
            "Select Model for Results Panel",
            list(st.session_state.analysis_results.keys()),
            key="results_model_choice"
        )

        # Show distribution graphs for the selected model
        if model_choice:
            st.subheader(f"{model_choice} Distribution Graphs")
            results = st.session_state.analysis_results[model_choice]
            self.plot_intent_distribution(results, model_choice)
            self.plot_confidence_distribution(results, model_choice)
            # Show slot distribution as well
            self.plot_slot_distribution(results, model_choice)

        # Optionally, show model comparison
        if st.button("Show Model Comparison"):
            with st.spinner('Generating comparison...'):
                try:
                    comparison = self.generate_comparison()
                    st.session_state.comparison_results = comparison
                    self.display_comparison()
                except Exception as e:
                    st.error(f"Error in comparison: {str(e)}")

    def display_transcriptions(self):
        """Display transcription results"""
        st.subheader("Transcription Results")
        st.dataframe(st.session_state.transcriptions)
        
        if st.button("Save Transcriptions"):
            try:
                st.session_state.transcriptions.to_csv(
                    "transcriptions.tsv", 
                    sep="\t", 
                    index=False
                )
                st.success("Transcriptions saved!")
            except Exception as e:
                st.error(f"Error saving transcriptions: {e}")
                st.write(st.session_state.transcriptions.head())

    def display_comparison(self):
        """Display model comparison results"""
        comparison = st.session_state.comparison_results
        
        st.subheader("Model Performance Comparison")
        
        # Display metrics
        cols = st.columns(len(comparison["models"]))
        for i, (model, metrics) in enumerate(comparison["models"].items()):
            with cols[i]:
                st.markdown(f"**{model}**")
                st.metric("Avg Intents", f"{metrics['avg_intents']:.2f}")
                st.metric("Avg Confidence", f"{metrics['avg_confidence']:.2f}")
                st.metric("Avg Slots", f"{metrics['avg_slots']:.2f}")
        
        # Visualizations
        self.plot_model_comparison()

    @staticmethod
    def calculate_metrics(reference_file, transcriptions) -> Dict[str, float]:
        """Calculate transcription quality metrics"""
        # Read reference text
        if reference_file.name.endswith('.tsv'):
            ref_df = pd.read_csv(reference_file, sep='\t')
            # Try to match on filename if possible
            if 'filename' in ref_df.columns and 'filename' in transcriptions.columns:
                merged = pd.merge(transcriptions, ref_df, on='filename', suffixes=('_hyp', '_ref'))
                hyp_texts = merged['text_hyp'].astype(str).tolist()
                ref_texts = merged['text_ref'].astype(str).tolist()
            else:
                # fallback: just use order
                ref_texts = ref_df.iloc[:, 0].astype(str).tolist()
                hyp_texts = transcriptions['text'].astype(str).tolist()
        else:
            # Assume plain text, one line per reference
            ref_texts = reference_file.read().decode('utf-8').splitlines()
            hyp_texts = transcriptions['text'].astype(str).tolist()

        # Truncate to shortest length
        n = min(len(ref_texts), len(hyp_texts))
        ref_texts = ref_texts[:n]
        hyp_texts = hyp_texts[:n]

        st.write("Reference texts:", ref_texts)
        st.write("Hypothesis texts:", hyp_texts)

        # Calculate metrics
        wer = jiwer.wer(ref_texts, hyp_texts)
        cer = jiwer.cer(ref_texts, hyp_texts)
        bleu = sum(sentence_bleu([ref.split()], hyp.split()) for ref, hyp in zip(ref_texts, hyp_texts)) / n

        return {'wer': wer, 'cer': cer, 'bleu': bleu}

    def generate_comparison(self) -> Dict[str, Any]:
        """Generate model comparison results"""
        comparison = {"models": {}}
        
        for model, results in st.session_state.analysis_results.items():
            metrics = {
                "avg_intents": np.mean([len(r["intents"]) for r in results]),
                "avg_confidence": np.mean([i["confidence"] for r in results for i in r["intents"]]),
                "avg_slots": np.mean([len(r["slots"]) for r in results])
            }
            comparison["models"][model] = metrics
        
        return comparison

    @staticmethod
    def plot_intent_distribution(results: List[Dict], model_name: str):
        """Plot intent distribution"""
        # Count intents
        from collections import Counter
        all_intents = [intent['intent'] for r in results if 'intents' in r for intent in r['intents']]
        intent_counts = Counter(all_intents)
        if not intent_counts:
            st.info("No intents found to plot.")
            return
        fig = px.bar(x=list(intent_counts.keys()), y=list(intent_counts.values()),
                     labels={'x': 'Intent', 'y': 'Count'},
                     title=f"Intent Distribution for {model_name}")
        st.plotly_chart(fig)

    @staticmethod
    def plot_confidence_distribution(results: List[Dict], model_name: str):
        """Plot confidence distribution"""
        all_confidences = [intent['confidence'] for r in results if 'intents' in r for intent in r['intents'] if 'confidence' in intent]
        if not all_confidences:
            st.info("No confidence scores found to plot.")
            return
        fig = px.histogram(all_confidences, nbins=20, labels={'value': 'Confidence'},
                           title=f"Confidence Score Distribution for {model_name}")
        st.plotly_chart(fig)

    @staticmethod
    def plot_slot_distribution(results: List[Dict], model_name: str):
        """Plot slot type distribution"""
        from collections import Counter
        all_slots = [slot['type'] for r in results if 'slots' in r for slot in r['slots']]
        slot_counts = Counter(all_slots)
        if not slot_counts:
            st.info("No slots found to plot.")
            return
        fig = px.bar(x=list(slot_counts.keys()), y=list(slot_counts.values()),
                     labels={'x': 'Slot Type', 'y': 'Count'},
                     title=f"Slot Type Distribution for {model_name}")
        st.plotly_chart(fig)

    def plot_model_comparison(self):
        """Plot model comparison visualizations"""
        # Implementation here
        pass

if __name__ == "__main__":
    pipeline = SpeechAnalysisPipeline()
    pipeline.main() 