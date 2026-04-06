import os
import pandas as pd
import whisper
from tqdm import tqdm

def process_audio_files(validated_tsv, clips_dir, output_file, model_size="base", max_files=None):
    """Process audio files and create dataset"""
    # Load validated.tsv
    print("Loading validated.tsv...")
    df = pd.read_csv(validated_tsv, sep='\t')
    
    # Verify each file exists
    print("Verifying files...")
    valid_files = []
    for idx, row in df.iterrows():
        file_path = os.path.join(clips_dir, row['path'])
        if os.path.isfile(file_path):
            valid_files.append(row)
    
    df = pd.DataFrame(valid_files)
    print(f"Found {len(df)} valid files")
    
    if max_files:
        df = df.head(max_files)
    
    print(f"\nTotal files to process: {len(df)}")
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)
    
    # Process files
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            # Get audio file path
            audio_path = os.path.join(clips_dir, row['path'])
            
            # Transcribe audio
            result = model.transcribe(audio_path)
            
            # Store results
            results.append({
                'file_name': row['path'],
                'transcribed_text': result['text'].strip(),
                'original_text': row['sentence'],
                'age': row.get('age', ''),
                'gender': row.get('gender', ''),
                'accent': row.get('accent', '')
            })
            
            # Save intermediate results
            if len(results) > 0:  # Save after each successful transcription
                interim_df = pd.DataFrame(results)
                interim_df.to_csv(output_file, sep='\t', index=False)
                print(f"\nSaved {len(results)} transcriptions...")
            
        except Exception as e:
            print(f"\nError processing {row['path']}: {str(e)}")
            continue
    
    # Save final results
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_file, sep='\t', index=False)
        print(f"\nFinal dataset saved to {output_file}")
        print(f"Total transcriptions: {len(results)}")
        return final_df
    else:
        print("No files were processed successfully!")
        return None

if __name__ == "__main__":
    # Configure paths
    DATA_DIR = r"C:\Users\badug\Downloads\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en"
    CLIPS_DIR = os.path.join(DATA_DIR, "clips")
    VALIDATED_TSV = os.path.join(DATA_DIR, "validated.tsv")
    OUTPUT_FILE = os.path.join(DATA_DIR, "processed_dataset.tsv")
    
    # Process a small batch first (for testing)
    MAX_FILES = 2  # Start with just 2 files to test
    
    print("Starting audio processing...")
    print(f"Using clips directory: {CLIPS_DIR}")
    print(f"Using validated.tsv: {VALIDATED_TSV}")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    
    # Process the audio files
    df = process_audio_files(
        validated_tsv=VALIDATED_TSV,
        clips_dir=CLIPS_DIR,
        output_file=OUTPUT_FILE,
        max_files=MAX_FILES
    )
    
    # Verify the output
    if df is not None:
        print("\nSample of processed data:")
        print(df.head())