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
        if filename.endswith('.wav'):  # Change to .wav since Lahaja dataset uses .wav files
            full_path = os.path.abspath(os.path.join(input_folder, filename))  # Absolute path
            if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                valid_files.append(filename)
            else:
                invalid_files.append(filename)

    logging.info(f"Total files in directory: {len(all_files)}")
    logging.info(f"Valid WAV files: {len(valid_files)}")
    if invalid_files:
        logging.warning(f"Invalid WAV files: {len(invalid_files)}")
    
    return valid_files

def segregate_audio_files(input_folder):
    """Segregate audio files based on language from filenames."""
    language_files = {}

    # Assume the filename starts with the language name, e.g., 'english_001.wav'
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            # Extract language name from the filename (modify if necessary)
            language = filename.split('_')[0]
            if language not in language_files:
                language_files[language] = []
            language_files[language].append(filename)

    return language_files

def transcribe_audio_by_language(input_folder, output_file, model_size="base", custom_prompt=None):
    """Transcribe audio files segregated by language."""
    if not os.path.isdir(input_folder):
        logging.error(f"Input folder not found: {input_folder}")
        return None
    
    logging.info(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    
    # Get the segregated audio files
    language_files = segregate_audio_files(input_folder)
    if not language_files:
        logging.warning("No audio files found to process!")
        return None
    
    results = []

    for language, files in language_files.items():
        logging.info(f"Processing {language} language files...")

        for audio_file in tqdm(files, desc=f"Transcribing {language}"):
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
                    'language': language,
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
    # Path to your Lahaja dataset folder
    INPUT_FOLDER = r"C:\Users\badug\Downloads\lahaja\lahaja\small_clips"
    OUTPUT_FILE = r"C:\Users\badug\Desktop\transcriptions_lahaja.tsv"

    # Domain-specific prompt
    CUSTOM_PROMPT = "Transcribe speech in different languages with focus on non-native accents."

    try:
        files = os.listdir(INPUT_FOLDER)
        logging.info(f"Total files found: {len(files)}")
    except Exception as e:
        logging.error(f"Error accessing input folder: {str(e)}")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Start the transcription process
    df = transcribe_audio_by_language(INPUT_FOLDER, OUTPUT_FILE, model_size="small", custom_prompt=CUSTOM_PROMPT)

    # Check and display a sample of the transcriptions
    if df is not None and not df.empty:
        logging.info("Sample of transcriptions:")
        print(df.head())
