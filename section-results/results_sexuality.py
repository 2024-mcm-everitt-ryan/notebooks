sexuality = {
    'marker': 'p',
    'FT': [
        {'model': 'BERT base uncased', 'f1': 0.87, 'precision': 0.94, 'recall': 0.81},
        {'model': 'BERT large uncased', 'f1': 0.85, 'precision': 0.97, 'recall': 0.76},
        {'model': 'RoBERTa base', 'f1': 0.92, 'precision': 0.96, 'recall': 0.89},
        {'model': 'RoBERTa large', 'f1': 0.86, 'precision': 0.97, 'recall': 0.78},
        {'model': 'Flan T5 XL', 'f1': 0.88, 'precision': 0.96, 'recall': 0.81},
        {'model': 'Phi3 3.8B 4k', 'f1': 0.84, 'precision': 0.95, 'recall': 0.75},
        {'model': 'Gemma2-9B', 'f1': 0.85, 'precision': 0.97, 'recall': 0.75},
        {'model': 'Llama3-8B', 'f1': 0.84, 'precision': 0.95, 'recall': 0.75},
       # {'model': 'Phi3-7B 8k', 'f1': 0.00, 'precision': 0.00, 'recall': 0.00}
    ],
    'PROMPT': {
        'pZS': [
            {'model': 'GPT-4', 'f1': 0.40, 'precision': 1.00, 'recall': 0.25},
            {'model': 'Gemma2-9B', 'f1': 0.57, 'precision': 0.97, 'recall': 0.40},
            {'model': 'Llama3-8B', 'f1': 0.28, 'precision': 0.93, 'recall': 0.16},
            {'model': 'Phi3-7B 8k', 'f1': 0.64, 'precision': 0.70, 'recall': 0.60}
        ],
        'pFS': [
            {'model': 'Gemma2-9B', 'f1': 0.57, 'precision': 0.83, 'recall': 0.44},
            {'model': 'Llama3-8B', 'f1': 0.55, 'precision': 0.97, 'recall': 0.39},
            {'model': 'Phi3-7B 8k', 'f1': 0.56, 'precision': 0.89, 'recall': 0.41}
        ],
        'pCOT': [
            {'model': 'Gemma2-9B', 'f1': 0.57, 'precision': 0.97, 'recall': 0.40},
            {'model': 'Llama3-8B', 'f1': 0.22, 'precision': 1.00, 'recall': 0.12},
            {'model': 'Phi3-7B 8k', 'f1': 0.69, 'precision': 0.79, 'recall': 0.61}
        ],
        'pSC': [
            {'model': 'Gemma2-9B', 'f1': 0.57, 'precision': 0.97, 'recall': 0.40},
            {'model': 'Llama3-8B', 'f1': 0.22, 'precision': 1.00, 'recall': 0.12},
            {'model': 'Phi3-7B 8k', 'f1': 0.69, 'precision': 0.79, 'recall': 0.61}
        ]
    }
}
