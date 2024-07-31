# Adapted from the following, but changed to handle multi-label
# https://github.com/VanekPetr/flan-t5-text-classifier/blob/main/classifier/AutoModelForSeq2SeqLM/flan-t5-finetuning.py

import warnings
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from transformers import Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    classification_report, confusion_matrix
import nltk
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
import numpy as np
from nltk import sent_tokenize
from typing import List, Tuple
from datasets import Dataset
import pandas as pd
import sys
from accelerate import Accelerator

from transformers import TrainerCallback
from huggingface_hub import ModelCard, EvalResult, ModelCardData, HfFolder
import platform




# Dataset
hf_site_id = '2024-mcm-everitt-ryan'
dataset_id = f'{hf_site_id}/job-bias-synthetic-human-benchmark'
# dataset_id = f'{hf_site_id}/job-bias-synthetic-human-verified'

# Model
base_model_id = f'google/flan-t5-{sys.argv[1]}'
base_model_name = base_model_id.split('/')[-1]
model_id = f'{base_model_name}-seq2seq-job-bias-classifier'
hub_model_id = f'{hf_site_id}/{model_id}'

print(base_model_id)

# Training
num_train_epochs = 10
batch_size = 1
learning_rate = 3e-4
# learning_rate = 5e-5

# Regularisation
dropout_rate = 0.1
weight_decay = 0.0001

# Misc
seed = 2024
results_output_dir = 'results'
logging_dir = 'logs'
push_hf=False



#####################################################

warnings.filterwarnings('ignore')


#####################################################
dataset = load_dataset(dataset_id)
column_names = dataset['train'].column_names

text_col = 'text'
label_cols = [col for col in column_names if col.startswith('label_')]

labels = [label.replace("label_", "") for label in label_cols]

id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

# Remove all columns apart from the two needed for multi-class classification
keep_columns = ['id', text_col] + label_cols
for split in ["train", "val", "test"]:
    dataset[split] = dataset[split].remove_columns(
        [col for col in dataset[split].column_names if col not in keep_columns])

for type in ['train', 'val', 'test']:
    dataset[type] = dataset[type].shuffle(seed=seed)  # .select(range(10))

print(dataset)


#####################################################


# Prepare target sequences for T5
def create_target_sequence(example):
    labels = [key.replace('label_', '') for key, value in example.items() if key.startswith('label_') and value]
    labels = ','.join(labels)
    labels = labels.strip()
    return labels

def preprocess_function(sample: Dataset, padding: str = "max_length") -> dict:
    """Preprocess the dataset."""
    inputs = [item for item in sample["text"]]
    labels = [item for item in sample["labels"]]

    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )

    labels = tokenizer(
        text_target=labels, max_length=max_target_length, padding=padding, truncation=True
    )

    if padding == "max_length":
        labels["input_ids"] = [
            [(la if la != tokenizer.pad_token_id else -100) for la in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["text"], truncation=True),
    batched=True,
    remove_columns=dataset['train'].column_names,
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")


# Add target sequence to the dataset
dataset = dataset.map(lambda x: {'labels': create_target_sequence(x)},
                      remove_columns=[col for col in dataset['train'].column_names if col.startswith('label_')])

# Tokenise targets
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["labels"], truncation=True),
    batched=True,
    remove_columns=dataset['train'].column_names,
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")


tokenized_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=["text", "labels"]
)
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

nltk.download("punkt")



#####################################################

config = AutoConfig.from_pretrained(base_model_id, dropout_rate=dropout_rate)
model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model_id,
    config=config
)

print(model)

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8
)

def postprocess_text(labels: List[str], preds: List[str]) -> Tuple[List[str], List[str]]:
    """Helper function to postprocess text"""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return labels, preds


def compute_metrics(eval_predictions):
    y_hat, y = eval_predictions

    # Replace -100 in the labels .
    y = np.where(y != -100, y, tokenizer.pad_token_id)

    if isinstance(y_hat, tuple):
        y_hat = y_hat[0]

    # print(y)
    # print('--------------------')
    # print(y_hat)

    y_str = tokenizer.batch_decode(y, skip_special_tokens=True)
    y_hat_str = tokenizer.batch_decode(y_hat, skip_special_tokens=True)

    y_str, y_hat_str = postprocess_text(y_str, y_hat_str)

    # print('--------------------')
    # print(y_str)
    # print('--------------------')
    # print(y_hat_str)
    # print('--------------------')

    # Flatten the list of labels
    true_flat = [label.strip() for sublist in [t.split(',') for t in y_str] for label in sublist]
    pred_flat = [label.strip() for sublist in [p.split(',') for p in y_hat_str] for label in sublist]

    # print(true_flat)
    # print('--------------------')
    # print(pred_flat)

    # Convert to binary format for multi-label metrics
    # unique_labels = list(set(true_flat + pred_flat))  # This will include out-of-scope (not a label) predictions
    unique_labels = list(set(true_flat))

    # Remove the blank label (no bias)
    unique_labels = [label for label in unique_labels if label != '' and label is not None]

    y_true = [[1 if label in t else 0 for label in unique_labels] for t in y_str]
    y_pred = [[1 if label in p else 0 for label in unique_labels] for p in y_hat_str]

    y_true_str = [[label if label in t else 0 for label in unique_labels] for t in y_str]
    y_pred_str = [[label if label in p else 0 for label in unique_labels] for p in y_hat_str]
    # unique_labels = ['no_bias' if not label else label for label in unique_labels]

    print('\n------------------ Confusion Matrix ------------------')
    conf_matrix = confusion_matrix(np.asarray(y_true).argmax(axis=1), np.asarray(y_pred).argmax(axis=1))
    df_cm = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
    print(df_cm)
    print('\n--------- Classification Report ------------------')
    print(classification_report(y_true, y_pred, target_names=unique_labels))  # , target_names=list(id2label.values())))

    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_samples = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    precision_micro = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall_micro = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc_micro = roc_auc_score(y_true=y_true, y_score=y_pred, average='micro')

    # for i in range(1): #range(len(y_true_str)):
    #    yt  = [num for num in y_true_str[i] if num != 0]
    #    yth  = [num for num in y_pred_str[i] if num != 0]
    #    print(f't: {yt}')
    #    print(f'p: {yth}')
    #    print(f't: {y_true[i]}')
    #    print(f'p: {y_pred[i]}')
    #    print(f"accuracy: {accuracy_score(y_true=y_true[i], y_pred=y_pred[i])}")
    #    print(f"precision: {precision_score(y_true=y_true[i], y_pred=y_pred[i], average='micro')}")
    #    print(f"recall: {recall_score(y_true=y_true[i], y_pred=y_pred[i], average='micro')}")
    #    print('----------------------')

    metrics = {
        'accuracy': accuracy,
        f'f1_micro': f1_micro,
        f'f1_macro': f1_macro,
        f'f1_samples': f1_samples,
        f'f1_weighted': f1_weighted,
        f'precision_micro': precision_micro,
        f'recall_micro': recall_micro,
        f'roc_auc_micro': roc_auc_micro}
    return metrics

#####################################################

training_args = Seq2SeqTrainingArguments(
    output_dir=results_output_dir,
    # logging_dir=logging_dir,  # logging & evaluation strategies
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    fp16=False,  # Overflows with fp16
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    # weight_decay=weight_decay
    # report_to="tensorboard",
    # push_to_hub=True,
    # hub_strategy="every_save",
    # hub_model_id=REPOSITORY_ID,
    # hub_token=HfFolder.get_token(),
)


# early_stop = transformers.EarlyStoppingCallback(10, 1.15)
class PrintClassificationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        print("----------------------------------------------------------")


accelerator = Accelerator()
train_dataset, eval_dataset, model = accelerator.prepare(
    tokenized_dataset["train"], tokenized_dataset["val"], model
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[PrintClassificationCallback]
)

model.config.use_cache = False  # Silence the warnings.
trainer.train()

#####################################################

test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])
print(test_results)



df = pd.DataFrame(list(test_results.items()), columns=['Metric', 'Value'])
print(df.to_string(index=False))

#####################################################

if push_hf:
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
