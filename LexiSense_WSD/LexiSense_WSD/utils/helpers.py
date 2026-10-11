import io
import pandas as pd


def load_uploaded(file):
    if file.name.lower().endswith('.csv'):
        df = pd.read_csv(file)
        if 'sentence' not in df.columns:
            raise ValueError("CSV must contain a 'sentence' column.")
        if 'target_word' not in df.columns:
            df['target_word'] = ''
        return df[['sentence','target_word']].fillna('')
    if file.name.lower().endswith('.txt'):
        text = file.getvalue().decode('utf-8', errors='replace')
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        return pd.DataFrame({'sentence': rows, 'target_word': [''] * len(rows)})
    raise ValueError("Unsupported file type. Please upload CSV or TXT.")


def results_to_csv(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode('utf-8')
