disability = {
    'marker': "v",
    'FT': [
        {'model': 'BERT base uncased', 'f1': 0.54, 'precision': 0.79, 'recall': 0.41},
        {'model': 'BERT large uncased', 'f1': 0.56, 'precision': 0.91, 'recall': 0.40},
        {'model': 'RoBERTa base', 'f1': 0.63, 'precision': 0.87, 'recall': 0.50},
        {'model': 'RoBERTa large', 'f1': 0.60, 'precision': 0.88, 'recall': 0.45},
        {'model': 'Flan T5 XL', 'f1': 0.55, 'precision': 0.89, 'recall': 0.40},
        {'model': 'Phi3 3.8B 4k', 'f1': 0.60, 'precision': 0.97, 'recall': 0.44},
        {'model': 'Gemma2-9B', 'f1': 0.66, 'precision': 0.95, 'recall': 0.50},
        {'model': 'Llama3-8B', 'f1': 0.64, 'precision': 0.97, 'recall': 0.47},
       # {'model': 'Phi3-7B 8k', 'f1': 0.00, 'precision': 0.00, 'recall': 0.00}
    ],
    'PROMPT': {
        'pZS': [
            {'model': 'GPT-4', 'f1': 0.59, 'precision': 0.62, 'recall': 0.56},
            {'model': 'Gemma2-9B', 'f1': 0.49, 'precision': 0.74, 'recall': 0.36},
            {'model': 'Llama3-8B', 'f1': 0.42, 'precision': 0.64, 'recall': 0.31},
            {'model': 'Phi3-7B 8k', 'f1': 0.47, 'precision': 0.72, 'recall': 0.35}
        ],
        'pFS': [
            {'model': 'Gemma2-9B', 'f1': 0.28, 'precision': 0.67, 'recall': 0.17},
            {'model': 'Llama3-8B', 'f1': 0.63, 'precision': 0.60, 'recall': 0.66},
            {'model': 'Phi3-7B 8k', 'f1': 0.31, 'precision': 0.88, 'recall': 0.19}
        ],
        'pCOT': [
            {'model': 'Gemma2-9B', 'f1': 0.53, 'precision': 0.82, 'recall': 0.39},
            {'model': 'Llama3-8B', 'f1': 0.50, 'precision': 0.68, 'recall': 0.40},
            {'model': 'Phi3-7B 8k', 'f1': 0.62, 'precision': 0.66, 'recall': 0.59}
        ],
        'pSC': [
            {'model': 'Gemma2-9B', 'f1': 0.51, 'precision': 0.81, 'recall': 0.38},
            {'model': 'Llama3-8B', 'f1': 0.52, 'precision': 0.70, 'recall': 0.41},
            {'model': 'Phi3-7B 8k', 'f1': 0.63, 'precision': 0.67, 'recall': 0.59}
        ]
    }
}
