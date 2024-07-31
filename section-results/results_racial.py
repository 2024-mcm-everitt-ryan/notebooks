racial = {
    'marker': 's',
    'FT': [
        {'model': 'BERT base uncased', 'f1': 0.85, 'precision': 0.89, 'recall': 0.81},
        {'model': 'BERT large uncased', 'f1': 0.87, 'precision': 0.87, 'recall': 0.86},
        {'model': 'RoBERTa base', 'f1': 0.82, 'precision': 0.83, 'recall': 0.81},
        {'model': 'RoBERTa large', 'f1': 0.85, 'precision': 0.87, 'recall': 0.84},
        {'model': 'Flan T5 XL', 'f1': 0.83, 'precision': 0.89, 'recall': 0.79},
        {'model': 'Phi3 3.8B 4k', 'f1': 0.85, 'precision': 0.93, 'recall': 0.79},
        {'model': 'Gemma2-9B', 'f1': 0.86, 'precision': 0.90, 'recall': 0.82},
        {'model': 'Llama3-8B', 'f1': 0.84, 'precision': 0.89, 'recall': 0.80},
        #{'model': 'Phi3-7B 8k', 'f1': 0.00, 'precision': 0.00, 'recall': 0.00}
    ],
    'PROMPT': {
        'pZS': [
            {'model': 'GPT-4', 'f1': 0.64, 'precision': 0.95, 'recall': 0.49},
            {'model': 'Gemma2-9B', 'f1': 0.70, 'precision': 0.81, 'recall': 0.62},
            {'model': 'Llama3-8B', 'f1': 0.66, 'precision': 0.89, 'recall': 0.53},
            {'model': 'Phi3-7B 8k', 'f1': 0.73, 'precision': 0.91, 'recall': 0.61}
        ],
        'pFS': [
            {'model': 'Gemma2-9B', 'f1': 0.71, 'precision': 0.75, 'recall': 0.68},
            {'model': 'Llama3-8B', 'f1': 0.73, 'precision': 0.76, 'recall': 0.70},
            {'model': 'Phi3-7B 8k', 'f1': 0.62, 'precision': 0.95, 'recall': 0.46}
        ],
        'pCOT': [
            {'model': 'Gemma2-9B', 'f1': 0.69, 'precision': 0.78, 'recall': 0.62},
            {'model': 'Llama3-8B', 'f1': 0.71, 'precision': 0.89, 'recall': 0.59},
            {'model': 'Phi3-7B 8k', 'f1': 0.72, 'precision': 0.86, 'recall': 0.62}
        ],
        'pSC': [
            {'model': 'Gemma2-9B', 'f1': 0.69, 'precision': 0.78, 'recall': 0.62},
            {'model': 'Llama3-8B', 'f1': 0.71, 'precision': 0.90, 'recall': 0.59},
            {'model': 'Phi3-7B 8k', 'f1': 0.74, 'precision': 0.89, 'recall': 0.62}
        ]
    }
}
