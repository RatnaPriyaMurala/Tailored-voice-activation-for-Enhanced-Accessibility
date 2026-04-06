import os
import whisper
import pandas as pd
from tqdm import tqdm
import gc
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def verify_audio_files(input_folder):
    """Verify audio files exist and return list of valid files."""
    all_files = os.listdir(input_folder)
    valid_files = []
    invalid_files = []

    for filename in all_files:
        if filename.endswith('.mp3'):
            full_path = os.path.abspath(os.path.join(input_folder, filename))  # Absolute path
            if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                valid_files.append(filename)
            else:
                invalid_files.append(filename)

    logging.info(f"Total files in directory: {len(all_files)}")
    logging.info(f"Valid MP3 files: {len(valid_files)}")
    if invalid_files:
        logging.warning(f"Invalid MP3 files: {len(invalid_files)}")
    
    return valid_files

def transcribe_audio_folder(input_folder, output_file, model_size="base", custom_prompt=None):
    """Transcribe all valid MP3 files in a folder."""
    if not os.path.isdir(input_folder):
        logging.error(f"Input folder not found: {input_folder}")
        return None
    
    logging.info(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    
    audio_files = verify_audio_files(input_folder)
    if not audio_files:
        logging.warning("No valid audio files found to process!")
        return None
    
    results = []

    for audio_file in tqdm(audio_files, desc="Transcribing"):
        try:
            audio_path = os.path.abspath(os.path.join(input_folder, audio_file))
            if not os.path.isfile(audio_path):
                continue

            # Transcribe with a custom prompt
            result = model.transcribe(audio_path, prompt=custom_prompt)

            # Filter transcription to remove sections with background noise
            transcription = filter_noise(result)

            results.append({
                'file_name': audio_file,
                'transcription': transcription.strip()
            })

            # Progressive saving
            if len(results) % 100 == 0:
                logging.info(f"Saving batch of 100 transcriptions to {output_file}...")
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(output_file, sep='\t', index=False, mode='a', header=not os.path.exists(output_file))
                results = []  # Reset the list after saving

            del result  # Free memory
            gc.collect()
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {str(e)}")

    # Final save for any remaining results
    if results:
        logging.info(f"Saving final batch of {len(results)} transcriptions...")
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(output_file, sep='\t', index=False, mode='a', header=not os.path.exists(output_file))
    else:
        logging.warning("No transcriptions to save!")

    if os.path.exists(output_file):
        logging.info(f"Transcriptions successfully saved to: {output_file}")
        return pd.read_csv(output_file, sep='\t')
    else:
        logging.error(f"Failed to save transcriptions to: {output_file}")
        return None

def filter_noise(result):
    """Filter transcription to remove background noise."""
    filtered_segments = []
    for segment in result["segments"]:
        if segment["no_speech_prob"] < 0.5:  # Only include if confidence in speech is high
            filtered_segments.append(segment["text"])
    return " ".join(filtered_segments)

if __name__ == "__main__":
    INPUT_FOLDER = r"C:\Users\badug\Downloads\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en\small_clips"
    OUTPUT_FILE = r"C:\Users\badug\Desktop\transcriptions.tsv"

    # Domain-specific prompt
    CUSTOM_PROMPT = "Transcribe speech with focus on names and places like 'Hertford', 'Oxford', and 'Lawnswood'."

    try:
        files = os.listdir(INPUT_FOLDER)
        logging.info(f"Total files found: {len(files)}")
    except Exception as e:
        logging.error(f"Error accessing input folder: {str(e)}")
        exit(1)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = transcribe_audio_folder(INPUT_FOLDER, OUTPUT_FILE, model_size="small", custom_prompt=CUSTOM_PROMPT)

    if df is not None and not df.empty:
        logging.info("Sample of transcriptions:")
