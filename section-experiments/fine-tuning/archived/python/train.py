# Based on https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

import platform
import sys
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import HfFolder
from huggingface_hub import ModelCard, EvalResult, ModelCardData
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


def main():
    # Parameters

    hf_site_id = os.environ.get('HF_SITE_ID', '2024-mcm-everitt-ryan')
    dataset_name = os.environ.get('DATASET_NAME','job-bias-synthetic-human-benchmark')
    dataset_id = os.environ.get('DATASET_ID', f'{hf_site_id}/{dataset_name}')
    base_model_id = os.environ.get('BASE_MODEL_ID', sys.argv[1].strip())
    base_model_name = os.environ.get('BASE_MODEL_NAME', base_model_id.split('/')[-1])
    model_id =  os.environ.get('MODEL_ID', f'{base_model_name}-job-bias-classifier')

    hub_model_id = os.environ.get('HF_HUB_MODEL_ID', f'{hf_site_id}/{model_id}')

    lora = True if os.environ.get('LORA', 'false').lower() == 'true' else False
    fp16 = True if os.environ.get('FP16', 'false').lower() == 'true' else False

    batch_size = int(os.environ.get('BATCH_SIZE', '6'))

    # Use value slightly smaller than pretraining lr value & close to LoRA standard
    learning_rate=float(os.environ.get('LEARNING_RATE', '5e-5'))
    num_train_epochs = int(os.environ.get('NUM_TRAIN_EPOCHS', '5'))
    weight_decay = float(os.environ.get('WEIGHT_DECAY', '0.01'))
    metric_for_best_model = os.environ.get('METRIC_FOR_BEST_MODEL', "f1_micro")

    # Save some memory at the expense of training
    # See https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one
    gradient_checkpointing = True if os.environ.get('GRADIENT_CHECKPOINTING', 'true').lower() == 'true' else False

    multi_label_metrics_threshold=float(os.environ.get('MULTI_LABEL_METRICS_THRESHOLD', '0.5'))


    print(f'hf_site_id: {hf_site_id}')
    print(f'dataset_id: {dataset_id}')
    print(f'base_model_id: {base_model_id}')
    print(f'model_id: {model_id}')
    print(f'hub_model_id: {hub_model_id}')

    print(f'multi_label_metrics_threshold: {multi_label_metrics_threshold}')


    print(f'lora: {lora}')
    print(f'fp16: {fp16}')
    print(f'gradient_checkpointing: {gradient_checkpointing}')
    print(f'learning_rate: {learning_rate}')
    print(f'batch_size: {batch_size}')
    print(f'num_train_epochs: {num_train_epochs}')
    print(f'weight_decay: {weight_decay}')
    print(f'metric_for_best_model: {metric_for_best_model}')

    if lora:
        # TODO
        lora_r = 2
        lora_alpha = 16
        lora_dropout = 0.1
        lora_bias = "none",


    ###### Dataset ######

    dataset = load_dataset(dataset_id)
    column_names = dataset['train'].column_names

    print(f"Columns: {dataset.num_columns}")
    print(f"Rows: {dataset.num_rows}")
    print(f"Column Names: {column_names}")

    text_col = 'text'
    label_cols = [col for col in column_names if col.startswith('label_')]
    labels = [label.replace("label_", "") for label in label_cols]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    print(f"Text column: {text_col}")
    print(f"Label columns: {label_cols}")
    print(f"Labels: {labels}")

    # Remove all columns apart from the two needed for multi-class classification
    keep_columns = ['context_id', 'synthetic', text_col] + label_cols
    for split in ["train", "val", "test"]:
        dataset[split] = dataset[split].remove_columns(
            [col for col in dataset[split].column_names if col not in keep_columns])

    # Merge train,val, test into one dataframe
    df = pd.concat([
        dataset['train'].to_pandas(),
        dataset['val'].to_pandas(),
        dataset['test'].to_pandas()])

    print(f"{df.synthetic.value_counts().to_string()}")
    for col in label_cols:
        print(f"\n{df[col].value_counts().to_string()}")

    # Longest phrase
    longest_text = df[text_col].apply(lambda x: (len(x), x)).max()[1]
    print(f"Longest text:\n==============\n{longest_text}\n==============")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_prefix_space=True)

    max_char = len(longest_text)
    max_words = len(longest_text.split())
    max_tokens = len(tokenizer.encode(longest_text))

    print(f'Max characters: {max_char}')
    print(f'Max words: {max_words}')
    print(f'Max tokens: {max_tokens}')

    tokenizer_max_length = min(max_tokens, tokenizer.model_max_length)
    print(f'tokenizer_max_length: {tokenizer_max_length}')

    def preprocess_data(sample):
        # take a batch of texts
        text = sample[text_col]
        # encode them
        encoding = tokenizer(text, truncation=True, max_length=tokenizer_max_length, padding="max_length")
        # encoding = tokenizer(text, truncation=True, max_length=tokenizer_max_length, padding=True)
        # add labels
        labels_batch = {k: sample[k] for k in sample.keys() if k in label_cols}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(label_cols)))
        # fill numpy array
        for idx, label in enumerate(label_cols):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding

    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch")

    ###### Model ######

    model = AutoModelForSequenceClassification.from_pretrained(base_model_id,
                                                               problem_type="multi_label_classification",
                                                               num_labels=len(label_cols),
                                                               id2label=id2label,
                                                               label2id=label2id)

    if lora:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=2, lora_alpha=16, lora_dropout=0.1, bias="none",
        ))

    if lora:
        print(model.print_trainable_parameters())

    ###### # Define Metrics ######

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= multi_label_metrics_threshold)] = 1
        # finally, compute metrics
        y_true = labels

        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

        f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        f1_samples = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
        f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

        precision_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
        recall_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc_micro = roc_auc_score(y_true=y_true, y_score=y_pred, average='micro')
        # return as dictionary
        metrics = {
            f'f1_micro': f1_micro,
            f'f1_macro': f1_macro,
            f'f1_samples': f1_samples,
            f'f1_weighted': f1_weighted,
            f'precision_micro': precision_micro,
            f'recall_micro': recall_micro,
            f'roc_auc_micro': roc_auc_micro,
            'accuracy': accuracy
        }
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    ###### Train ######

    optimiser = 'paged_adamw_8bit'  # Use paged optimizer to save memory

    args = TrainingArguments(
        model_id,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        # optim=optimiser,
        # lr_scheduler_type="cosine",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        # push_to_hub=True,
        # output_dir=repository_id,
        # logging_dir=f"{model_id}/logs",
        # logging_strategy="steps",
        # logging_steps=10,
        # warmup_steps=500,
        # warmup_ratio=0.1,
        # max_grad_norm=0.3,
        # save_total_limit=2,
        # report_to="tensorboard",
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=hub_model_id,
        hub_token=HfFolder.get_token(),
    )

    # early_stop = transformers.EarlyStoppingCallback(10, 1.15)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        # For padding a batch of examples to the maximum length seen in the batch
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        # tokenizer=tokenizer,
        #   callbacks=[early_stop]
    )

    model.config.use_cache = False  # Silence the warnings.
    trainer.train()

    ###### Evaluate ######
    test_results = trainer.evaluate(eval_dataset=encoded_dataset['test'])
    print(f'evaluation (test) results: {test_results}')

    ###### Save Model ######
    # save locally
    model.save_pretrained(model_id)

    # push to the hub
    model.push_to_hub(repo_id=hub_model_id, token=HfFolder.get_token())
    tokenizer.push_to_hub(repo_id=hub_model_id, token=HfFolder.get_token())

    ###### Update Model Card ######

    eval_results = []
    for k, v in test_results.items():
        eval_results.append(EvalResult(
            task_type='multi_label_classification',
            dataset_type='mix_human-eval_synthetic',
            dataset_name=dataset_id,
            metric_type=k.replace("eval_", "", 1),
            metric_value=v))

    direct_use = """
        ```python
        from transformers import pipeline

        pipe = pipeline("text-classification", model="${hub_model_id}", return_all_scores=True)

        results = pipe("Join our dynamic and fast-paced team as a Junior Marketing Specialist. We seek a tech-savvy and energetic individual who thrives in a vibrant environment. Ideal candidates are digital natives with a fresh perspective, ready to adapt quickly to new trends. You should have recent experience in social media strategies and a strong understanding of current digital marketing tools. We're looking for someone with a youthful mindset, eager to bring innovative ideas to our young and ambitious team. If you're a recent graduate or early in your career, this opportunity is perfect for you!")
        print(results)
        ```
        >> [[
        {'label': 'age', 'score': 0.9883460402488708}, 
        {'label': 'disability', 'score': 0.00787709467113018}, 
        {'label': 'feminine', 'score': 0.007224376779049635}, 
        {'label': 'general', 'score': 0.09967829287052155}, 
        {'label': 'masculine', 'score': 0.0035264550242573023}, 
        {'label': 'racial', 'score': 0.014618005603551865}, 
        {'label': 'sexuality', 'score': 0.005568435415625572}
        ]]
        """
    direct_use = direct_use.replace('${hub_model_id}', hub_model_id, -1)

    card_data = ModelCardData(
        model_id=model_id,
        model_name=model_id,
        model_description="The model is a multi-label classifier designed to detect various types of bias within job descriptions.",
        base_model=base_model_id,
        language='en',
        license='apache-2.0',
        developers="Tristan Everitt and Paul Ryan",
        model_card_authors='See developers',
        model_card_contact='See developers',
        repo="https://gitlab.computing.dcu.ie/everitt2/2024-mcm-everitt-ryan",
        eval_results=eval_results,
        compute_infrastructure=f'{platform.system()} {platform.release()} {platform.processor()}',
        # hardware_requirements=f"CPUs: {psutil.cpu_count()}, Memory: {psutil.virtual_memory().total} bytes",
        software=f'Python {platform.python_version()}',
        hardware_type=platform.machine(),
        hours_used='N/A',
        cloud_provider='N/A',
        cloud_region='N/A',
        co2_emitted='N/A',
        datasets=[dataset_id],
        direct_use=direct_use
    )

    card = ModelCard.from_template(card_data)

    card.push_to_hub(repo_id=hub_model_id, token=HfFolder.get_token())


if __name__ == "__main__":
    main()
