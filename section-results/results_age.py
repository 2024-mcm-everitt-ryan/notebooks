age = {
    'marker': "o",
    'FT': [
        {'model': 'BERT base uncased', 'f1': 0.66, 'precision': 0.84, 'recall': 0.54},
        {'model': 'BERT large uncased', 'f1': 0.63, 'precision': 0.87, 'recall': 0.50},
        {'model': 'RoBERTa base', 'f1': 0.63, 'precision': 0.80, 'recall': 0.51},
        {'model': 'RoBERTa large', 'f1': 0.61, 'precision': 0.86, 'recall': 0.47},
        {'model': 'Flan T5 XL', 'f1': 0.71, 'precision': 0.89, 'recall': 0.59},
        {'model': 'Phi3 3.8B 4k', 'f1': 0.55, 'precision': 0.89, 'recall': 0.40},
        {'model': 'Gemma2-9B', 'f1': 0.65, 'precision': 0.72, 'recall': 0.60},
        {'model': 'Llama3-8B', 'f1': 0.63, 'precision': 0.91, 'recall': 0.49},
       # {'model': 'Phi3-7B 8k', 'f1': 0.00, 'precision': 0.00, 'recall': 0.00}
    ],
    'PROMPT': {
        'pZS': [
            {'model': 'GPT-4o', 'f1': 0.70, 'precision': 0.75, 'recall': 0.66},
            {'model': 'Gemma2-9B', 'f1': 0.58, 'precision': 0.72, 'recall': 0.49},
            {'model': 'Llama3-8B', 'f1': 0.33, 'precision': 0.74, 'recall': 0.21},
            {'model': 'Phi3-7B 8k', 'f1': 0.64, 'precision': 0.62, 'recall': 0.66}
        ],
        'pFS': [
            {'model': 'Gemma2-9B', 'f1': 0.54, 'precision': 0.61, 'recall': 0.49},
            {'model': 'Llama3-8B', 'f1': 0.56, 'precision': 0.65, 'recall': 0.50},
            {'model': 'Phi3-7B 8k', 'f1': 0.28, 'precision': 0.93, 'recall': 0.16}
        ],
        'pCOT': [
            {'model': 'Gemma2-9B', 'f1': 0.56, 'precision': 0.66, 'recall': 0.49},
            {'model': 'Llama3-8B', 'f1': 0.45, 'precision': 0.78, 'recall': 0.31},
            {'model': 'Phi3-7B 8k', 'f1': 0.56, 'precision': 0.47, 'recall': 0.70}
        ],
        'pSC': [
            {'model': 'Gemma2-9B', 'f1': 0.57, 'precision': 0.67, 'recall': 0.49},
            {'model': 'Llama3-8B', 'f1': 0.44, 'precision': 0.76, 'recall': 0.31},
            {'model': 'Phi3-7B 8k', 'f1': 0.57, 'precision': 0.47, 'recall': 0.71}
        ]
    }
}


