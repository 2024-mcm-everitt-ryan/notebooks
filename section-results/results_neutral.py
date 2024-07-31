neutral = {
    'marker': 'v',
    'FT': [
        {'model': 'BERT base uncased', 'f1': 0.47, 'precision': 0.32, 'recall': 0.86},
        {'model': 'BERT large uncased', 'f1': 0.47, 'precision': 0.32, 'recall': 0.88},
        {'model': 'RoBERTa base', 'f1': 0.50, 'precision': 0.38, 'recall': 0.72},
        {'model': 'RoBERTa large', 'f1': 0.48, 'precision': 0.33, 'recall': 0.89},
        {'model': 'Flan T5 XL', 'f1': 0.53, 'precision': 0.37, 'recall': 0.93},
        {'model': 'Phi3 3.8B 4k', 'f1': 0.44, 'precision': 0.30, 'recall': 0.90},
        {'model': 'Gemma2-9B', 'f1': 0.49, 'precision': 0.34, 'recall': 0.86},
        {'model': 'Llama3-8B', 'f1': 0.50, 'precision': 0.35, 'recall': 0.90},
       # {'model': 'Phi3-7B 8k', 'f1': 0.00, 'precision': 0.00, 'recall': 0.00}
    ],
    'PROMPT': {
        'pZS': [
            {'model': 'GPT-4', 'f1': 0.33, 'precision': 0.21, 'recall': 0.71},
            {'model': 'Gemma2-9B', 'f1': 0.35, 'precision': 0.48, 'recall': 0.28},
            {'model': 'Llama3-8B', 'f1': 0.34, 'precision': 0.22, 'recall': 0.70},
            {'model': 'Phi3-7B 8k', 'f1': 0.36, 'precision': 0.29, 'recall': 0.47}
        ],
        'pFS': [
            {'model': 'Gemma2-9B', 'f1': 0.35, 'precision': 0.51, 'recall': 0.26},
            {'model': 'Llama3-8B', 'f1': 0.40, 'precision': 0.31, 'recall': 0.56},
            {'model': 'Phi3-7B 8k', 'f1': 0.36, 'precision': 0.23, 'recall': 0.81}
        ],
        'pCOT': [
            {'model': 'Gemma2-9B', 'f1': 0.39, 'precision': 0.56, 'recall': 0.30},
            {'model': 'Llama3-8B', 'f1': 0.35, 'precision': 0.23, 'recall': 0.81},
            {'model': 'Phi3-7B 8k', 'f1': 0.33, 'precision': 0.30, 'recall': 0.35}
        ],
        'pSC': [
            {'model': 'Gemma2-9B', 'f1': 0.41, 'precision': 0.60, 'recall': 0.31},
            {'model': 'Llama3-8B', 'f1': 0.35, 'precision': 0.23, 'recall': 0.81},
            {'model': 'Phi3-7B 8k', 'f1': 0.34, 'precision': 0.32, 'recall': 0.36}
        ]
    }
}
