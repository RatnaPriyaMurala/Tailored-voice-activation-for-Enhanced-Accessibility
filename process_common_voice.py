import os
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split


# ---- CONFIGURATION ---- #
DATA_DIR = r"C:\Users\badug\Downloads\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en"
AUDIO_DIR = os.path.join(DATA_DIR, "clips")
OUTPUT_AUDIO_DIR = os.path.join(DATA_DIR, "preprocessed_clips")
VALIDATED_TSV = os.path.join(DATA_DIR, "validated.tsv")
OUTPUT_TRAIN_TSV = os.path.join(DATA_DIR, "train.tsv")
OUTPUT_VAL_TSV = os.path.join(DATA_DIR, "val.tsv")
OUTPUT_TEST_TSV = os.path.join(DATA_DIR, "test.tsv")

TARGET_SAMPLE_RATE = 16000  # Common ASR input sample rate
MIN_DURATION = 1.0  # Minimum audio duration in seconds
MAX_DURATION = 10.0  # Maximum audio duration in seconds


# ---- LOAD DATA ---- #
def load_metadata(tsv_path):
    """Load metadata from a TSV file."""
    print(f"Loading metadata from {tsv_path}...")
    data = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(data)} samples.")
    return data


# ---- FILTER DATA ---- #
def filter_metadata(data, audio_dir, min_duration=MIN_DURATION, max_duration=MAX_DURATION):
    """Filter metadata based on audio file existence, duration, and transcript validity."""
    print("Filtering metadata...")
    filtered = data.copy()

    # Remove rows where audio files are missing
    filtered = filtered[filtered['path'].apply(lambda x: os.path.exists(os.path.join(audio_dir, x)))]
    
    # Filter by duration
    #filtered = filtered[(filtered['duration'] >= min_duration) & (filtered['duration'] <= max_duration)]
    
    # Remove invalid transcripts
    filtered = filtered.dropna(subset=['sentence'])
    
    print(f"Filtered dataset size: {len(filtered)}")
    return filtered


# ---- TEXT NORMALIZATION ---- #
def normalize_text(text):
    """Normalize transcript text."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def normalize_sentences(data):
    """Normalize sentences in the metadata."""
    print("Normalizing sentences...")
    data['sentence'] = data['sentence'].apply(normalize_text)
    return data


# ---- AUDIO PREPROCESSING ---- #
def preprocess_audio(input_path, output_path, target_sr=TARGET_SAMPLE_RATE):
    """Preprocess audio: resample to target sample rate and save."""
    y, sr = librosa.load(input_path, sr=target_sr)
    sf.write(output_path, y, target_sr)


def preprocess_audio_files(data, input_dir, output_dir):
    """Process and save all audio files in the dataset."""
    print("Preprocessing audio files...")
    for idx, row in data.iterrows():
        input_path = os.path.join(input_dir, row['path'])
        output_path = os.path.join(output_dir, row['path'])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
        preprocess_audio(input_path, output_path)
    print("Audio preprocessing complete!")


# ---- SPLIT DATA ---- #
def split_data(data):
    """Split metadata into train, validation, and test sets."""
    print("Splitting data into train, validation, and test sets...")
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    print(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    return train, val, test


# ---- SAVE METADATA ---- #
def save_metadata(data, output_path):
    """Save metadata to a TSV file."""
    data.to_csv(output_path, sep="\t", index=False)
    print(f"Saved metadata to {output_path}")


# ---- MAIN WORKFLOW ---- #
def main():
    # Step 1: Load metadata
    metadata = load_metadata(VALIDATED_TSV)
    
    # Step 2: Filter metadata
    filtered_metadata = filter_metadata(metadata, AUDIO_DIR)
    
    # Step 3: Normalize sentences
    normalized_metadata = normalize_sentences(filtered_metadata)
    
    # Step 4: Preprocess audio files
    preprocess_audio_files(normalized_metadata, AUDIO_DIR, OUTPUT_AUDIO_DIR)
    
    # Step 5: Split data into train, validation, and test sets
    train, val, test = split_data(normalized_metadata)
    
    # Step 6: Save split metadata
    save_metadata(train, OUTPUT_TRAIN_TSV)
    save_metadata(val, OUTPUT_VAL_TSV)
    save_metadata(test, OUTPUT_TEST_TSV)


if __name__ == "__main__":
    main()
