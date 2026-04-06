import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from langdetect import detect, DetectorFactory
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure consistent language detection results
DetectorFactory.seed = 0

def detect_language(text):
    """
    Detect the language of the given text using langdetect.
    """
    try:
        return detect(text)
    except Exception as e:
        logging.warning(f"Language detection failed for text: {text} - {str(e)}")
        return "unknown"

def translate_text(texts, model_name):
    """
    Translate a list of texts into English using MarianMT.
    """
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    translations = []
    for text in tqdm(texts, desc="Translating"):
        try:
            tokenized = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
            translated = model.generate(**tokenized)
            english_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            translations.append(english_text)
        except Exception as e:
            logging.error(f"Error translating text: {text} - {str(e)}")
            translations.append("")
    return translations

def process_and_translate(transcription_file, output_file):
    """
    Process the transcription file, detect languages, and translate non-English text to English.
    """
    try:
        df = pd.read_csv(transcription_file, sep='\t')
        logging.info(f"Loaded {len(df)} rows from {transcription_file}")

        # Add a column for detected language
        logging.info("Detecting languages...")
        df['detected_language'] = df['transcription'].apply(lambda x: detect_language(x) if isinstance(x, str) else "unknown")

        # Filter rows where language is not English and valid text exists
        non_english_rows = df[(df['detected_language'] != 'en') & (df['detected_language'] != 'unknown')]
        texts_to_translate = non_english_rows['transcription'].tolist()
        logging.info(f"Found {len(texts_to_translate)} non-English transcriptions to translate.")

        if texts_to_translate:
            # Choose the appropriate model for translation
            model_name = "Helsinki-NLP/opus-mt-mul-en"  # Multilingual to English model
            translations = translate_text(texts_to_translate, model_name)

            # Add translations back to the dataframe
            df.loc[(df['detected_language'] != 'en') & (df['detected_language'] != 'unknown'), 'english_translation'] = translations
        else:
            logging.warning("No valid non-English transcriptions found for translation.")
            df['english_translation'] = None

        # Save the updated DataFrame
        df.to_csv(output_file, sep='\t', index=False)
        logging.info(f"Translations saved to {output_file}")

        return df
    except Exception as e:
        logging.error(f"Error processing and translating transcriptions: {str(e)}")
        return None

if __name__ == "__main__":
    # File paths
    TRANSCRIPTION_FILE = r"C:\Users\badug\OneDrive\Desktop\transcriptions_lahaja.tsv"
    TRANSLATED_FILE = r"C:\Users\badug\OneDrive\Desktop\translated_transcriptions_with_language.tsv"

    # Process and translate
    try:
        translated_df = process_and_translate(TRANSCRIPTION_FILE, TRANSLATED_FILE)

        if translated_df is not None:
            logging.info("Sample Translations:")
            print(translated_df.head())
    except Exception as e:
        logging.error(f"An error occurred during the main execution: {str(e)}")
