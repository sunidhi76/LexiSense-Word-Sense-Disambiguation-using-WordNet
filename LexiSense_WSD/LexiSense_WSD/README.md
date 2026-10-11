# LexiSense – Word Sense Disambiguation using WordNet

LexiSense is an interactive Word Sense Disambiguation (WSD) application built with Python, Streamlit, NLTK WordNet and a WordNet-based Lesk algorithm. It determines which meaning of an ambiguous word best fits its surrounding context and exposes the evidence behind the decision.

## Objectives
- Retrieve real candidate senses from NLTK WordNet.
- Apply Basic and Enhanced Lesk-style contextual overlap scoring.
- Provide explainable sense predictions rather than hard-coded answers.
- Support single sentences and user-uploaded CSV/TXT datasets.
- Explore WordNet definitions, synonyms and lexical relations.
- Present results in a polished, presentation-ready Streamlit interface.

## Features
- Home dashboard and guided WSD pipeline.
- Single Sentence Analyzer with automatic ambiguity suggestions.
- Basic Lesk and Enhanced Lesk algorithms.
- Candidate sense cards, scores and matching evidence.
- WordNet Explorer with synsets, definitions, synonyms, hypernyms, hyponyms, antonyms and examples.
- CSV/TXT Dataset Analyzer.
- Downloadable annotated results.
- Interactive Plotly charts.
- Analysis & Statistics dashboard.
- Friendly handling of missing WordNet resources and invalid inputs.
- Automatic NLTK corpus setup; no API keys required.

## System architecture
`Streamlit UI → preprocessing → WordNet candidate generation → Lesk scoring → ranked senses → explainable visualization`

## WSD methodology
1. Tokenize and normalize the sentence.
2. Remove common stopwords for lexical comparison.
3. Retrieve all WordNet synsets for the target word.
4. Build a lexical signature for each synset from its definition, lemmas and examples.
5. Enhanced Lesk additionally incorporates hypernym/hyponym definitions and slightly weights lemma matches.
6. Compute contextual overlap for every candidate.
7. Rank candidates and display the best sense plus supporting evidence.

The displayed confidence is a comparative share of candidate scores, not a calibrated statistical probability.

## Dataset format
CSV:
```csv
sentence,target_word
"The bank approved my loan.",bank
"The fisherman sat beside the bank.",bank
```

TXT: one sentence per line. If a CSV contains only `sentence`, LexiSense attempts to suggest an ambiguous target word automatically.

A built-in dataset with 30+ natural examples is included in `data/sample_wsd_dataset.csv`.

## Installation – Windows / VS Code
```powershell
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch, the app automatically checks/downloads the NLTK resources it needs (`wordnet`, `omw-1.4`, `punkt`, `punkt_tab`, and `stopwords`). Internet is only needed for this initial NLTK corpus acquisition if those resources are absent.

## Example input
Sentence: `The fisherman sat on the bank of the river.`
Target: `bank`
Algorithm: `Enhanced Lesk`

Expected behavior: the river-side sense should receive contextual evidence associated with the river context. Exact ranking can vary with the installed WordNet corpus/version because LexiSense genuinely queries WordNet rather than hard-coding the answer.

## Project structure
```text
LexiSense_WSD/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── sample_wsd_dataset.csv
├── src/
│   ├── __init__.py
│   ├── wsd_engine.py
│   ├── wordnet_utils.py
│   ├── preprocessing.py
│   └── visualizations.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── assets/
    └── README.txt
```

## Limitations
Classic Lesk relies on lexical overlap and can fail when a sentence uses paraphrases or sparse context. WordNet's sense inventory also reflects lexical distinctions rather than modern domain-specific meanings. The score shown in the interface should not be interpreted as a probability.

## Future enhancements
- POS-aware candidate filtering.
- WordNet semantic similarity features.
- Context embeddings and transformer-based reranking.
- Gold-standard evaluation metrics such as accuracy and F1.
- Larger benchmark datasets and confusion analysis.
- Richer interactive semantic graphs.

## Verification checklist
The packaged application is designed to verify imports, sample data loading, WordNet lookup, single-sentence analysis, CSV/TXT processing, CSV download, WordNet exploration, Plotly visualizations and sidebar navigation from the project root.
