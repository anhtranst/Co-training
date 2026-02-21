from comet_ml import Experiment
import os
import sys
import gc
import random
import time
import torch
import argparse
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
from transformers import (
    set_seed,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
    RobertaTokenizer,
    BertTokenizer,
    CLIPTokenizer,
    get_scheduler,
    logging as transformers_logging
)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from pathlib import Path
import logging as lg

from dotenv import load_dotenv
load_dotenv()

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils import delete_saved_models, log_message, str2bool
from src.data_processor import TextDataset
from src.models import BERTweet, ClassifyCLIP, BERTweetLarge, RoBERTa
from src.loss import SmoothCrossEntropyLoss
# from gen_init_weights import WeightGenerator
# from co_training_parallel import CoTrainer
# from fine_tune_models import DualModelTrainer
from trainer_classes import WeightGenerator, CoTrainer, DualModelTrainer

import optuna

# Constants
ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 128 # usually 300 but CLIP model - ValueError: Sequence length must be less than max_position_embeddings (got `sequence length`: 300 and max_position_embeddings: 77 
EPOCH_PATIENCE = 5

# Dataset configurations
LABELED_SAMPLES = {
    'informative': [2, 10, 20, 40, 100, 200, 300, 400, 500, 1000],
    'humanitarian': [2, 10, 20, 40, 100, 200, 300, 400, 500],
    'humanitarian8': [10, 20, 50, 100],
    'humanitarian9': [10, 20, 50, 100],
    'humanitarian10': [10, 20, 50, 100],
}

NUM_CLASSES = {
    'informative': 2,
    'humanitarian': 5,
    'humanitarian8': 8,
    'humanitarian9': 9,
    'humanitarian10': 10,

}

NUM_CLASSES_MAPPING = {
    'canada_wildfires_2016': 'humanitarian8',
    'hurricane_irma_2017': 'humanitarian9',
    'hurricane_harvey_2017': 'humanitarian9',
    'kerala_floods_2018': 'humanitarian9',
    'hurricane_dorian_2019': 'humanitarian9',
    'hurricane_maria_2017': 'humanitarian9',
    'hurricane_florence_2018': 'humanitarian9',
    'kaikoura_earthquake_2016': 'humanitarian9',
    'california_wildfires_2018': 'humanitarian10',
    'cyclone_idai_2019': 'humanitarian10',
}

# Model mapping for easier reference
HF_MODEL_MAPPING = {
    "gpt4o": "GPT-4o",
    "gpt4o-mini": "GPT-4o-Mini",
    "llama-3.2-11b-vi": "Llama-3.2-11B-Vision-Instruct",
    "phi-3": "Phi-3-medium-4k",
    "phi-3-128k": "Phi-3-medium-128k",
    "mistral-7b": "Mistral-7B-Instruct",
    "llama-3-8b": "Llama-3.1-8B",
    "llama-3-70b": "Llama-3.3-70B",
    "roberta": "roberta-base"
}

PLM_ID_MAPPING = {
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "deberta-base": "microsoft/deberta-base",
    "deberta-large": "microsoft/deberta-large",
    "bert-base": "bert-base-uncased",
    "bert-large": "bert-large-uncased",
    "bert-tweet": "vinai/bertweet-base",
    "bertweetl": "vinai/bertweet-large",
    "clip": "openai/clip-vit-base-patch32"
}

few_shot_samples_per_class = {
    'informative': 2,
    'humanitarian': 5,
    "humanitarian8": 8,
    "humanitarian9": 9,
    "humanitarian10": 10
}

plm_ids = list(PLM_ID_MAPPING.keys())
llm_ids = list(HF_MODEL_MAPPING.keys())
datasets = list(LABELED_SAMPLES.keys())

def get_exponential_decay_ratio(num_classes, imbalance_ratio=10):
    cls_indices = np.arange(num_classes)
    ratios = imbalance_ratio ** (-cls_indices / (num_classes - 1))
    return ratios / ratios.sum()

def create_text_dataloader(dataframe, tokenizer, dataset_name, batch_size, max_len):
    """Create a DataLoader for a given dataset."""
    dataset_obj = TextDataset(dataframe, tokenizer, max_len, dataset=dataset_name)
    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)

def create_image_dataloader(dataframe, tokenizer, dataset_name, batch_size, max_len):
    """Create a DataLoader for a given dataset."""
    dataset_obj = TextDataset(dataframe, tokenizer, max_len, dataset=dataset_name)
    return DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Do NOT decode using tokenizer. Compare integers directly.
    acc = accuracy_score(labels, predictions)
    
    # Use 'weighted' or 'macro' depending on class imbalance
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }


def initialize_models(num_classes, args):
    """Initialize models based on PLM type."""
    if "clip" == args.plm_id:
        print(f"Using CLIP model: {args.plm_id}")
        model_1 = ClassifyCLIP(num_classes=num_classes, single_modality=True, text_embed=True, image_embed=False)
        model_2 = ClassifyCLIP(num_classes=num_classes, single_modality=True, text_embed=True, image_embed=False)
    elif "bert-tweet" == args.plm_id:
        print(f"Using Bert Tweet model: {args.plm_id}")
        model_1 = BERTweet(num_classes=num_classes, args=args)
        model_2 = BERTweet(num_classes=num_classes, args=args)
    elif "bertweetl" == args.plm_id:
        print(f"Using Bert Tweet model: {args.plm_id}")
        model_1 = BERTweetLarge(num_classes=num_classes, args=args)
        model_2 = BERTweetLarge(num_classes=num_classes, args=args)
    elif "roberta-base" == args.plm_id:
        print(f"Using Bert Tweet model: {args.plm_id}")
        model_1 = RoBERTa(num_classes=num_classes, args=args)
        model_2 = RoBERTa(num_classes=num_classes, args=args)
    return model_1, model_2


def setup_optimization(model_1, model_2, dataloaders, training_params, criterion_class=nn.CrossEntropyLoss):
    """Set up optimizers, schedulers and criterion for training."""
    criterion = criterion_class(reduction='none')
    learning_rate = training_params['learning_rate']
    num_epochs = training_params['num_epochs']
    wd = training_params.get('weight_decay', 0.01)

    train_dataloader_1 = dataloaders['train_dataloader_1']
    train_dataloader_2 = dataloaders['train_dataloader_2']
    
    optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=learning_rate, weight_decay=wd)
    optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=learning_rate, weight_decay=wd)
    
    num_training_steps_1 = num_epochs * len(train_dataloader_1)
    num_training_steps_2 = num_epochs * len(train_dataloader_2)
    
    lr_scheduler_1 = get_scheduler(
        name="linear", 
        optimizer=optimizer_1, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps_1
    )
    
    lr_scheduler_2 = get_scheduler(
        name="linear", 
        optimizer=optimizer_2, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps_2
    )
    
    optimizer_params = {
        'criterion': criterion,
        'optimizer_1': optimizer_1,
        'optimizer_2': optimizer_2,
        'num_training_steps_1': num_training_steps_1,
        'num_training_steps_2': num_training_steps_2,
        'lr_scheduler_1': lr_scheduler_1,
        'lr_scheduler_2': lr_scheduler_2
    }
    
    return optimizer_params

def calculate_ece(confidences, predictions, true_labels, n_bins=15):
    """
    Calculates Expected Calibration Error (ECE).
    """
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Get samples in this bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        if np.sum(in_bin) == 0:
            continue
            
        # Accuracy: Fraction where prediction matched true label
        # We only care about the accuracy of the samples falling in this bin
        bin_preds = predictions[in_bin]
        bin_labels = true_labels[in_bin]
        accuracy = np.mean(bin_preds == bin_labels)
        
        # Confidence: Average predicted probability in this bin
        avg_confidence = np.mean(confidences[in_bin])
        
        # Weighted difference
        fraction = np.sum(in_bin) / total_samples
        ece += fraction * np.abs(avg_confidence - accuracy)
        
    return ece

def evaluate_models(model_1, model_2, eval_dataloader, device_1, device_2, num_classes):
    """Evaluate ensembled models on provided dataloader."""
    model_1.eval()
    model_2.eval()
    y_true = []
    y_pred = []
    y_conf = []

    with torch.no_grad():
        for batch in eval_dataloader:
            # Process on first device
            batch_1 = {k: v.to(device_1) for k, v in batch.items()}
            outputs_1 = model_1(input_ids=batch_1['input_ids'], attention_mask=batch_1['attention_mask'])
            outputs_1 = outputs_1.logits if hasattr(outputs_1, 'logits') else outputs_1
            val_probs_1 = torch.nn.functional.softmax(outputs_1, dim=-1)

            # Process on second device
            batch_2 = {k: v.to(device_2) for k, v in batch.items()}
            outputs_2 = model_2(input_ids=batch_2['input_ids'], attention_mask=batch_2['attention_mask'])
            outputs_2 = outputs_2.logits if hasattr(outputs_2, 'logits') else outputs_2
            val_probs_2 = torch.nn.functional.softmax(outputs_2, dim=-1)
            
            # Ensemble predictions
            # IMPORTANT: We must divide by 2 to keep probabilities between 0 and 1
            val_probs = val_probs_1.cpu() + val_probs_2.cpu()
            
            # Get max probability (confidence) and the predicted class
            batch_conf, out_ensembled = torch.max(val_probs, dim=1)            
            out_ensembled = out_ensembled.cpu().detach().numpy()
            batch_conf = batch_conf.cpu().detach().numpy() 
            
            # Collect data
            y_pred.extend(out_ensembled.tolist())
            y_conf.extend(batch_conf.tolist())       
            y_true.extend(batch_1['labels'].cpu().numpy().tolist())
                
    cur_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    all_res = {}

    precision_ma, recall_ma, f1_ma, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro")
    
    all_res['f1_macro'] = f1_ma

    # Calculate ECE
    ece_score = calculate_ece(y_conf, y_pred, y_true)

    all_res['ece'] = ece_score

    return cur_f1, ece_score, all_res

def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Co-Training Script")
    parser.add_argument("--dataset", type=str,  choices=datasets, help="Dataset name")
    parser.add_argument("--humaid", type=str, help="Dataset name")
    parser.add_argument("--labeled_sample_idx", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="Index for labeled samples")
    parser.add_argument("--hf_model_id_short", type=str, choices=llm_ids, help="Short ID for the Hugging Face model")
    parser.add_argument("--seed", type=int, default=1234, choices=[1234, 4567, 8998], help="Random seed for reproducibility")
    parser.add_argument("--plm_id", type=str, default="roberta-base", choices=plm_ids, help="PLM (bert-base, roberta-base, deberta-base, etc.)")
    parser.add_argument("--pseudo_label_shot", type=int, default=0, help="Number of pseudo labeled samples")
    parser.add_argument("--few_shot", action="store_true", default=False, help="Use few-shot prompted pseudolabels.")
    parser.add_argument("--single_set", action="store_true", default=False, help="Use single training set for both models")
    parser.add_argument("--no_co_training", action="store_true", default=False, help="Disable co-training")
    parser.add_argument("--metric_combination", type=str, default='cv', choices=["cv", "cc"], help="Metric combination method")
    parser.add_argument("--exp_name", type=str, default="lg-cotr", help="Experiment name")
    parser.add_argument("--use_correct_labels_only", type=str2bool, default=False, help="Use correct labels only")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3", help="Comma-separated list of CUDA device IDs to use (e.g., 0,1)")
    parser.add_argument("--imb_training", action="store_true", default=False, help="Use imbalanced training")

    parser.add_argument("--setup_local_logging", action="store_true", default=False, help="Setup local logging")
    parser.add_argument("--comet_ml", action="store_true", default=False, help="Use comet_ml for experiment tracking")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb for experiment tracking")

    parser.add_argument("--event", type=str, default=None, help="Event name for humaid dataset")
    parser.add_argument("--lbcl", type=str, default=None, help="Labeled count per class string/int for humaid dataset")
    parser.add_argument("--set_num", type=str, default=None, help="Set number string for humaid dataset")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--epoch_patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides default)")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="Accumulation steps (overrides default)")
    parser.add_argument("--optuna_trials", type=int, default=20, help="Number of Optuna trials for hyperparameter search. Set 0 to disable.")

    args = parser.parse_args()
    
    #args.pseudo_label_shot = few_shot_samples_per_class[args.dataset] if args.few_shot else 0
    
    return args

def set_environment(args):
    """Set environment variables and random seeds."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    transformers_logging.set_verbosity_error()
    
    # Set random seeds
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device configuration
    if torch.cuda.device_count() >= 2:
        device_1 = torch.device("cuda:1")
        device_2 = torch.device("cuda:1")
    else:
        device_1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_2 = device_1
        
    return device_1, device_2

def setup_local_logging_old(args):
    """Set up logging to file and console."""
    if not args.setup_local_logging:
        return None
    
    log_dir = f"{ROOT}/output/{args.humaid}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    output_log_path = os.path.join(log_dir, f"log_{args.exp_name}.txt")
    
    lg.basicConfig(
        filename=output_log_path,
        filemode='w',
        level=lg.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = lg.getLogger()
    return logger

def setup_local_logging(args):
    """Set up logging to BOTH file and console."""
    if not args.setup_local_logging:
        # Even if local logging is off, ensure Optuna prints to console
        optuna.logging.enable_default_handler()
        return None
    
    log_dir = f"{ROOT}/output/{args.humaid}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    output_log_path = os.path.join(log_dir, f"log_{args.exp_name}.txt")
    
    # 1. Get the root logger
    logger = lg.getLogger()
    logger.setLevel(lg.INFO)
    
    # Clear existing handlers to avoid duplicate logs if function is called twice
    if logger.hasHandlers():
        logger.handlers.clear()

    # 2. Create Formatters
    formatter = lg.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 3. Add File Handler (Writes to log.txt)
    file_handler = lg.FileHandler(output_log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 4. Add Console Handler (Writes to Terminal/Optuna output)
    console_handler = lg.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 5. Explicitly tell Optuna to use this configuration
    optuna.logging.enable_propagation()  # Ensures Optuna logs go up to our root logger
    optuna.logging.disable_default_handler() # Prevents double logging
    
    return logger

def setup_comet_experiment(args):
    """Set up Comet ML experiment."""
    if not args.comet_ml:
        return None
    
    experiment = Experiment(
        api_key="yQyP31kIndUJxLb9RyzYZW8P6",
        project_name="llmcot",
        workspace="bharanibala"
    )
    experiment.set_name(f"{args.dataset}_{args.saved_model_name_suffix}")
    return experiment

def load_dataset_helper(args):
    """
    Helper function to load datasets based on dataset type.
    args.use_correct_labels_only, N, args.dataset, args.humaid
    
    """
    
    def json2pd(filepath):
        return pd.read_json(filepath, orient='index') # was index previously in the original code
    
    def load_data(file_name):
        return pd.read_csv(file_name, sep='\t')
    
    if args.dataset == 'informative':
        label_map = {
            'not_informative': 0, 
            'informative': 1
        }
    elif args.dataset == 'humanitarian':
        label_map = {
            "affected_individuals": 0,
            "rescue_volunteering_or_donation_effort": 1,
            "infrastructure_and_utility_damage": 2,
            "other_relevant_information": 3,
            "not_humanitarian": 4,
        }
    elif args.dataset == 'humanitarian8':
        label_map = {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "not_humanitarian":3,
                "other_relevant_information":4,
                "requests_or_urgent_needs":5,
                "rescue_volunteering_or_donation_effort":6,
                "sympathy_and_support":7,
        }
    elif args.dataset == 'humanitarian9':
        label_map = {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "injured_or_dead_people":3,
                "not_humanitarian":4,
                "other_relevant_information":5,
                "requests_or_urgent_needs":6,
                "rescue_volunteering_or_donation_effort":7,
                "sympathy_and_support":8,
        }
    elif args.dataset == 'humanitarian10':
        label_map = {
                "caution_and_advice":0,
                "displaced_people_and_evacuations":1,
                "infrastructure_and_utility_damage":2,
                "injured_or_dead_people":3,
                "missing_or_found_people":4,
                "not_humanitarian":5,
                "other_relevant_information":6,
                "requests_or_urgent_needs":7,
                "rescue_volunteering_or_donation_effort":8,
                "sympathy_and_support":9,
        }
    
    valid_tsv = f"/home/b/bharanibala/datasets/humaid_data/data/{args.humaid}/{args.humaid}_dev.tsv"
    validationSet = load_data(valid_tsv)
    validationSet['label'] = validationSet['class_label'].map(label_map)

    
    test_tsv = f"/home/b/bharanibala/datasets/humaid_data/data/{args.humaid}/{args.humaid}_test.tsv"
    testingSet = load_data(test_tsv)
    testingSet['label'] = testingSet['class_label'].map(label_map)

    trainingSet_1 = load_data(f"/home/b/bharanibala/datasets/humaid_data/cotrain/{args.humaid}/labeled_{args.lbcl}_set{args.set_num}_part1.tsv")
    trainingSet_2 = load_data(f"/home/b/bharanibala/datasets/humaid_data/cotrain/{args.humaid}/labeled_{args.lbcl}_set{args.set_num}_part2.tsv")
    auto_labeled_data = load_data(f"/home/b/bharanibala/datasets/humaid_data/cotrain/{args.humaid}/unlabeled_{args.lbcl}_set{args.set_num}.tsv")
    
    # Set labels appropriately
    trainingSet_1['label'] = trainingSet_1['class_label'].map(label_map)
    trainingSet_2['label'] = trainingSet_2['class_label'].map(label_map)
    auto_labeled_data['class_label'] = auto_labeled_data['class_label'].map(label_map)
    auto_labeled_data['label'] = auto_labeled_data['predicted_label'].map(label_map)

    if args.use_correct_labels_only:
        auto_labeled_data = auto_labeled_data[auto_labeled_data['label'] == auto_labeled_data['class_label']]
    return trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data

def optuna_objective(trial):
    """One full run: Init -> Co-Train -> Fine-Tune"""

    st = time.time()

    # Parse command line arguments
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    co_epochs = trial.suggest_int("co_train_epochs", 5, 20)
    epoch_patience = trial.suggest_int("epoch_patience", 4, 10)

    #args = parse_arguments()
    
    # Set up environment and devices
    device_1, device_2 = set_environment(args)
    hf_model_name = HF_MODEL_MAPPING[args.hf_model_id_short]
    
    args.pseudo_label_shot = args.lbcl
    
    # Set up experiment name
    # args.exp_name = "lg-cotr"
    
    # Set up paths
    saved_model_name_suffix = f"_{args.exp_name}_{args.hf_model_id_short}_{args.pseudo_label_shot}_shot_{args.plm_id}_{args.lbcl}_seed_{args.seed}"
            
    args.saved_model_name_suffix = saved_model_name_suffix
    
    # Set up directories
    data_dir = os.path.join(ROOT, 'data')
    saved_model_dir = f"{ROOT}/saved_models/{args.dataset}/{args.exp_name}"
    processed_dir = f"{ROOT}/processed/{args.dataset}/{args.hf_model_id_short}"
    # save_dir = os.path.join(processed_dir, f'N_{N}')
    
    args.saved_model_dir = saved_model_dir
    
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir, exist_ok=True)
    
    # Set batch size based on dataset and args.plm_id
    BATCH_SIZE = bs
    
    log_message(message=f"Using devices: {device_1}, {device_2}", args=args)
    log_message(message=f"Devices: {device_1}, {device_2}", args=args)
    
    log_message(message=f'Starting log', args=args)
    log_message(message=f'Dataset: {args.dataset}, Event: {args.humaid}, N: {args.lbcl}, Seed: {args.seed}, HF Model: {hf_model_name}, NumShots: {args.pseudo_label_shot}, PLM: {args.plm_id}', args=args)
    
    
    trainingSet_1, trainingSet_2, testingSet, validationSet, auto_labeled_data = load_dataset_helper(args)    
    
    # If not using multiset, make both training sets the same
    if args.single_set:
        trainingSet_1 = pd.concat([trainingSet_1, trainingSet_2], ignore_index=True)
        trainingSet_2 = trainingSet_1.copy()
    
    # Initialize tokenizer
    if args.plm_id == "clip":
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        #CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", do_lower_case=False)
    elif args.plm_id == "bert-tweet":
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", do_lower_case=False)
    elif args.plm_id == "bertweetl":
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", do_lower_case=False)
    elif "roberta-base" == args.plm_id:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
    elif "roberta-large" == args.plm_id:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=False)
    elif "bert-base" == args.plm_id:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif "deberta-base" == args.plm_id:
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base', do_lower_case=False)
    else:
        print(f"Tokenizer for {args.plm_id} not recognized. Defaulting to RoBERTa tokenizer.")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
    
    # tokenizer = AutoTokenizer.from_pretrained(PLM_ID_MAPPING[args.plm_id], do_lower_case=False)
    
    # Create dataloaders
    dataloaders = {
        'train_dataloader_1': create_text_dataloader(trainingSet_1, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN),
        'train_dataloader_2': create_text_dataloader(trainingSet_2, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN),
        'val_dataloader': create_text_dataloader(validationSet, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN),
        'test_dataloader': create_text_dataloader(testingSet, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN),
        'auto_label_dataloader': create_text_dataloader(auto_labeled_data, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN)
    }

    # Set up hyperparameters
    hyper_params = {
        'BATCH_SIZE': bs,
        'MAX_LEN': MAX_LEN,
        'EPOCH_PATIENCE': epoch_patience
    }

    # Training parameters
    training_params = {
        'num_epochs': co_epochs,
        'learning_rate': lr,
        'accumulation_steps': max(1, 64 // BATCH_SIZE),
        'weight_decay': wd
    }

    log_message(f"Learning Rate: {lr}\nWeight Decay: {wd}\nBatch Size: {bs}\nNo. Epochs: {co_epochs}\nEpoch Patience: {epoch_patience}\n Accumulation Steps: {training_params['accumulation_steps']}", args=args)

    # Initialize models and Set up optimizers and criterion for initial weight generation
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    optimizer_params = setup_optimization(model_1, model_2, dataloaders, training_params, criterion_class=nn.CrossEntropyLoss)
    
    
    # Generate initial weights
    log_message(message='Generating initial weights', args=args)
    generator = WeightGenerator(
        args=args,
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params=hyper_params,
        devices=(device_1, device_2),
        models=(model_1, model_2),
        auto_labeled_data=auto_labeled_data,
        # metric_combination='cv'
    )
    init_df = generator.generate_weights()
    
    # Re-initialize models for co-training and Set up optimizers with SmoothCrossEntropyLoss for co-training
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    optimizer_params = setup_optimization(model_1, model_2, dataloaders, training_params, criterion_class=SmoothCrossEntropyLoss)
    
    # Add init_df to dataloaders
    dataloaders['init_df_dataloader'] = create_text_dataloader(init_df, tokenizer, args.dataset, BATCH_SIZE, MAX_LEN)
    
    # Co-training
    log_message(message='Starting co-training', args=args)
    trainer = CoTrainer(
        args=args,
        models={'model_1': model_1, 'model_2': model_2},
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params=hyper_params,
        devices=[device_1, device_2],
        init_df=init_df,
        #metric_combination='cv'
    )
    co_training_df = trainer.train()
    #save the co_training_df to a file
    co_training_df.to_csv(os.path.join(saved_model_dir, f'co_training_df{saved_model_name_suffix}.csv'), index=False)    
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    del model_1
    del model_2
    gc.collect()
    
    # Load co-trained models
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    model_1_path = f'{saved_model_dir}/co_trained_model_1{saved_model_name_suffix}.pt'
    model_2_path = f'{saved_model_dir}/co_trained_model_2{saved_model_name_suffix}.pt'
    
    model_1.load_state_dict(torch.load(model_1_path))
    model_2.load_state_dict(torch.load(model_2_path))
    
    delete_saved_models(model_1_path)
    delete_saved_models(model_2_path)
    
    # Set up fine-tuning parameters
    training_params['num_epochs'] = 100
    hyper_params['EPOCH_PATIENCE'] = 10
    
    # Set up optimizers for fine-tuning
    optimizer_params = setup_optimization(
        model_1, model_2, 
        dataloaders,
        training_params
    )
    
    # Fine-tune models
    log_message(message='Fine-tuning models', args=args)
    dual_trainer = DualModelTrainer(
        args=args,
        dataloaders=dataloaders,
        training_params=training_params,
        optimizer_params=optimizer_params,
        hyper_params=hyper_params,
        devices=(device_1, device_2),
        models=(model_1, model_2)
    )
    dual_trainer.train()
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    del model_1
    del model_2
    gc.collect()
    
    # Load fine-tuned models
    model_1, model_2 = initialize_models(NUM_CLASSES[args.dataset], args)
    model_1_path = f'{saved_model_dir}/final_model_1{saved_model_name_suffix}.pt'
    model_2_path = f'{saved_model_dir}/final_model_2{saved_model_name_suffix}.pt'
    
    model_1.load_state_dict(torch.load(model_1_path))
    model_2.load_state_dict(torch.load(model_2_path))
    
    delete_saved_models(model_1_path)
    delete_saved_models(model_2_path)
    
    model_1.to(device_1)
    model_2.to(device_2)
    
    # Evaluate models
    eval_split = 'val' if args.dataset in ['humanitarian', 'informative'] else 'test'
    eval_dataloader = dataloaders['test_dataloader']
    
    cur_f1, ece, results_list = evaluate_models(model_1, model_2, eval_dataloader, device_1, device_2, NUM_CLASSES[args.dataset])
    
    # Log and print final results
    result_msg = (f"\n\nHf Model: {hf_model_name} PLM: {args.plm_id} Dataset: {args.dataset}, NumShots: {args.pseudo_label_shot}, "
                 f"N: {args.lbcl} {eval_split.capitalize()} SEED: {args.seed} F1: {cur_f1:.4f}, "
                 f"{eval_split.capitalize()} ECE: {ece:.4f}")
    
    log_message(message=result_msg, args=args)
    
    all_res_msg = f"All results: {results_list}" 
    log_message(message=all_res_msg, args=args)

    msg = f"\nTotal time taken: {time.time() - st:.2f} seconds"
    log_message(message=msg, args=args)

    return cur_f1

# --- MAIN EXECUTION ---

def main():

    st = time.time()

    global args
    args = parse_arguments()

    logger = setup_local_logging(args)
    args.logger = logger

    # Setup Optuna Study
    log_message(f"\n[Optuna] Starting hyperparameter search with {args.optuna_trials} trials.", args=args)
    study_name = f"study_{args.dataset}_{args.humaid}"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name
    )
    
    # Trigger Bayesian Optimization
    study.optimize(optuna_objective, n_trials=args.optuna_trials)
    
    log_message("\n[BEST TRIAL RESULTS]", args=args)
    log_message(f"F1 Score: {study.best_value:.4f}", args=args)
    log_message(f"Params: {study.best_params}", args=args)
    for k, v in study.best_trial.params.items():
        log_message(f"  {k}: {v}", args=args)

    msg = f"\nTotal time taken: {time.time() - st:.2f} seconds"
    log_message(message=msg, args=args)

if __name__ == "__main__":
    main()