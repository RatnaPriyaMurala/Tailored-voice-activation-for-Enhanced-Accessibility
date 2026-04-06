import pandas as pd
from jiwer import wer, cer
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def calculate_metrics(hypothesis, reference):
    """Calculate WER, CER, and BLEU Score."""
    try:
        # Word Error Rate (WER)
        error_rate = wer(reference, hypothesis)
        
        # Character Error Rate (CER)
        char_error_rate = cer(reference, hypothesis)
        
        # BLEU Score
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()
        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
        
        return error_rate, char_error_rate, bleu_score
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return None, None, None

def evaluate_transcriptions(transcriptions_file, reference_file, output_file):
    """Evaluate transcription accuracy using WER, CER, and BLEU."""
    try:
        # Load transcription and reference files
        trans_df = pd.read_csv(transcriptions_file, sep='\t')
        ref_df = pd.read_csv(reference_file, sep='\t')

        # Print column names for debugging
        logging.info(f"Columns in transcriptions file: {trans_df.columns}")
        logging.info(f"Columns in reference file: {ref_df.columns}")

        # Ensure both files have the 'file_name' column
        if 'file_name' not in trans_df.columns:
            logging.error("'file_name' column missing in transcriptions file!")
            return None

        # Merge transcription and reference data on 'file_name' and 'path' columns
        merged_df = pd.merge(trans_df, ref_df, left_on="file_name", right_on="path")
        logging.info(f"Merged DataFrame size: {len(merged_df)}")  # Verify number of rows

        metrics = []
        for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Calculating metrics"):
            # Reference sentence may be in 'sentence' column in reference file
            wer_score, cer_score, bleu_score = calculate_metrics(row["transcription"], row["sentence"])
            metrics.append({
                "file_name": row["file_name"],
                "wer": wer_score,
                "cer": cer_score,
                "bleu": bleu_score
            })

        # Create results DataFrame
        results_df = pd.DataFrame(metrics)

        # Save detailed results to file
        results_df.to_csv(output_file, sep='\t', index=False)
        logging.info(f"Detailed results saved to: {output_file}")

        # Summary statistics
        logging.info("\nAccuracy Metrics Summary:")
        logging.info(f"WER (Mean): {results_df['wer'].mean():.4f}")
        logging.info(f"CER (Mean): {results_df['cer'].mean():.4f}")
        logging.info(f"BLEU (Mean): {results_df['bleu'].mean():.4f}")

        return results_df
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        return None

if __name__ == "__main__":
    # File paths
    TRANSCRIPTIONS_FILE = r"C:\Users\badug\Desktop\transcriptions.tsv"  # Replace with your transcription file path
    REFERENCE_FILE = r"C:\Users\badug\Downloads\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en\validated.tsv"  # Replace with your reference file path
    OUTPUT_FILE = r"C:\Users\badug\Desktop\evaluation_results.tsv"  # Output file for results

    # Run evaluation
    results = evaluate_transcriptions(TRANSCRIPTIONS_FILE, REFERENCE_FILE, OUTPUT_FILE)
    if results is not None:
        logging.info("Evaluation completed successfully!")
