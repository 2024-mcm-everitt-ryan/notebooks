masculine = {
    'marker': "^",
    'FT': [
        {'model': 'BERT base uncased', 'f1': 0.61, 'precision': 0.72, 'recall': 0.54},
        {'model': 'BERT large uncased', 'f1': 0.67, 'precision': 0.82, 'recall': 0.56},
        {'model': 'RoBERTa base', 'f1': 0.67, 'precision': 0.78, 'recall': 0.59},
        {'model': 'RoBERTa large', 'f1': 0.64, 'precision': 0.76, 'recall': 0.55},
        {'model': 'Flan T5 XL', 'f1': 0.74, 'precision': 0.83, 'recall': 0.68},
        {'model': 'Phi3 3.8B 4k', 'f1': 0.61, 'precision': 0.95, 'recall': 0.45},
        {'model': 'Gemma2-9B', 'f1': 0.66, 'precision': 0.65, 'recall': 0.66},
        {'model': 'Llama3-8B', 'f1': 0.63, 'precision': 0.62, 'recall': 0.65},
       # {'model': 'Phi3-7B 8k', 'f1': 0.00, 'precision': 0.00, 'recall': 0.00}
    ],
    'PROMPT': {
        'pZS': [
            {'model': 'GPT-4', 'f1': 0.52, 'precision': 0.74, 'recall': 0.40},
            {'model': 'Gemma2-9B', 'f1': 0.43, 'precision': 0.29, 'recall': 0.86},
            {'model': 'Llama3-8B', 'f1': 0.31, 'precision': 0.94, 'recall': 0.19},
            {'model': 'Phi3-7B 8k', 'f1': 0.50, 'precision': 0.54, 'recall': 0.47}
        ],
        'pFS': [
            {'model': 'Gemma2-9B', 'f1': 0.61, 'precision': 0.53, 'recall': 0.72},
            {'model': 'Llama3-8B', 'f1': 0.42, 'precision': 0.73, 'recall': 0.30},
            {'model': 'Phi3-7B 8k', 'f1': 0.43, 'precision': 0.71, 'recall': 0.31}
        ],
        'pCOT': [
            {'model': 'Gemma2-9B', 'f1': 0.50, 'precision': 0.35, 'recall': 0.90},
            {'model': 'Llama3-8B', 'f1': 0.35, 'precision': 0.75, 'recall': 0.23},
            {'model': 'Phi3-7B 8k', 'f1': 0.53, 'precision': 0.58, 'recall': 0.49}
        ],
        'pSC': [
            {'model': 'Gemma2-9B', 'f1': 0.52, 'precision': 0.36, 'recall': 0.91},
            {'model': 'Llama3-8B', 'f1': 0.33, 'precision': 0.74, 'recall': 0.21},
            {'model': 'Phi3-7B 8k', 'f1': 0.53, 'precision': 0.58, 'recall': 0.49}
        ]
    }
}
