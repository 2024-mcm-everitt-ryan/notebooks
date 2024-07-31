general = {
    'marker': "<",
    'FT': [
        {'model': 'BERT base uncased', 'f1': 0.38, 'precision': 0.90, 'recall': 0.24},
        {'model': 'BERT large uncased', 'f1': 0.60, 'precision': 0.88, 'recall': 0.45},
        {'model': 'RoBERTa base', 'f1': 0.62, 'precision': 0.75, 'recall': 0.53},
        {'model': 'RoBERTa large', 'f1': 0.26, 'precision': 0.65, 'recall': 0.16},
        {'model': 'Flan T5 XL', 'f1': 0.68, 'precision': 0.79, 'recall': 0.60},
        {'model': 'Phi3 3.8B 4k', 'f1': 0.57, 'precision': 0.65, 'recall': 0.51},
        {'model': 'Gemma2-9B', 'f1': 0.60, 'precision': 0.84, 'recall': 0.46},
        {'model': 'Llama3-8B', 'f1': 0.67, 'precision': 0.82, 'recall': 0.56},
       # {'model': 'Phi3-7B 8k', 'f1': 0.00, 'precision': 0.00, 'recall': 0.00}
    ],
    'PROMPT': {
        'pZS': [
            {'model': 'GPT-4', 'f1': 0.31, 'precision': 0.57, 'recall': 0.21},
            {'model': 'Gemma2-9B', 'f1': 0.44, 'precision': 0.35, 'recall': 0.59},
            {'model': 'Llama3-8B', 'f1': 0.36, 'precision': 0.30, 'recall': 0.46},
            {'model': 'Phi3-7B 8k', 'f1': 0.27, 'precision': 0.23, 'recall': 0.35}
        ],
        'pFS': [
            {'model': 'Gemma2-9B', 'f1': 0.34, 'precision': 0.21, 'recall': 0.86},
            {'model': 'Llama3-8B', 'f1': 0.31, 'precision': 0.22, 'recall': 0.54},
            {'model': 'Phi3-7B 8k', 'f1': 0.24, 'precision': 0.20, 'recall': 0.31}
        ],
        'pCOT': [
            {'model': 'Gemma2-9B', 'f1': 0.37, 'precision': 0.26, 'recall': 0.69},
            {'model': 'Llama3-8B', 'f1': 0.33, 'precision': 0.37, 'recall': 0.30},
            {'model': 'Phi3-7B 8k', 'f1': 0.35, 'precision': 0.28, 'recall': 0.49}
        ],
        'pSC': [
            {'model': 'Gemma2-9B', 'f1': 0.38, 'precision': 0.26, 'recall': 0.69},
            {'model': 'Llama3-8B', 'f1': 0.33, 'precision': 0.37, 'recall': 0.30},
            {'model': 'Phi3-7B 8k', 'f1': 0.36, 'precision': 0.28, 'recall': 0.50}
        ]
    }
}
