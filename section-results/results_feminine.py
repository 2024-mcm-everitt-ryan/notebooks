feminine = {
    'marker': "^",
    'FT': [
        {'model': 'BERT base uncased', 'f1': 0.94, 'precision': 0.95, 'recall': 0.94},
        {'model': 'BERT large uncased', 'f1': 0.94, 'precision': 0.95, 'recall': 0.93},
        {'model': 'RoBERTa base', 'f1': 0.93, 'precision': 0.93, 'recall': 0.94},
        {'model': 'RoBERTa large', 'f1': 0.96, 'precision': 0.97, 'recall': 0.94},
        {'model': 'Flan T5 XL', 'f1': 0.91, 'precision': 0.92, 'recall': 0.90},
        {'model': 'Phi3 3.8B 4k', 'f1': 0.93, 'precision': 0.99, 'recall': 0.89},
        {'model': 'Gemma2-9B', 'f1': 0.95, 'precision': 0.99, 'recall': 0.91},
        {'model': 'Llama3-8B', 'f1': 0.92, 'precision': 0.99, 'recall': 0.86},
      #  {'model': 'Phi3-7B 8k', 'f1': 0.00, 'precision': 0.00, 'recall': 0.00}
    ],
    'PROMPT': {
        'pZS': [
            {'model': 'GPT-4', 'f1': 0.85, 'precision': 0.87, 'recall': 0.82},
            {'model': 'Gemma2-9B', 'f1': 0.88, 'precision': 0.80, 'recall': 0.97},
            {'model': 'Llama3-8B', 'f1': 0.84, 'precision': 0.85, 'recall': 0.84},
            {'model': 'Phi3-7B 8k', 'f1': 0.87, 'precision': 0.81, 'recall': 0.95}
        ],
        'pFS': [
            {'model': 'Gemma2-9B', 'f1': 0.82, 'precision': 0.74, 'recall': 0.94},
            {'model': 'Llama3-8B', 'f1': 0.69, 'precision': 0.55, 'recall': 0.95},
            {'model': 'Phi3-7B 8k', 'f1': 0.83, 'precision': 0.73, 'recall': 0.96}
        ],
        'pCOT': [
            {'model': 'Gemma2-9B', 'f1': 0.83, 'precision': 0.73, 'recall': 0.96},
            {'model': 'Llama3-8B', 'f1': 0.82, 'precision': 0.79, 'recall': 0.85},
            {'model': 'Phi3-7B 8k', 'f1': 0.89, 'precision': 0.84, 'recall': 0.95}
        ],
        'pSC': [
            {'model': 'Gemma2-9B', 'f1': 0.83, 'precision': 0.73, 'recall': 0.96},
            {'model': 'Llama3-8B', 'f1': 0.82, 'precision': 0.79, 'recall': 0.85},
            {'model': 'Phi3-7B 8k', 'f1': 0.89, 'precision': 0.84, 'recall': 0.95}
        ]
    }
}
