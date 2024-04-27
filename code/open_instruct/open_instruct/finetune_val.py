#!/usr/bin/env python
# coding=utf-8


import copy
import argparse
import logging
import math
import os
import random
import datasets
import torch
import time
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from datasets import load_dataset,Features,Value
from tqdm.auto import tqdm
import json
from datasets import load_metric


import numpy as np

from sklearn.metrics import (
    classification_report
)


import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM
)
from peft import LoraConfig, TaskType, get_peft_model

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--val_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--metrics_file", type=str, default=None, help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--save_merged_lora_model",
        action="store_true",
        help="If passed, will merge the lora modules and save the entire model.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the eval dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Whether do eval during training",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default=None,
        help="The name of the report project in wandb.",
    )
    parser.add_argument(
        "--report_tags",
        nargs='+',
        type=str,
        default=[],
        help="The list of tags of the report project in wandb.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
        if args.val_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`val_file` should be a json/jsonl file."
    return args

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
        
        
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def encode_with_prompt_completion_format_val(example, tokenizer, max_seq_length):
    # Since ACE has longer seq length, we don't set the max_seq_length
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
        
    trigger = tokenizer(example['trigger'], return_tensors='pt', max_length=2, truncation=True).input_ids
        
    example_text = example_text + tokenizer.eos_token
    # Cannot add the <s> token, and also need to do left padding for val
    # Prepare
    # tokenizer.padding_side = "left"
    # tokenizer.add_eos_token = False
    tokenized_example = tokenizer(example_text, return_tensors='pt')
    input_ids = tokenized_example.input_ids
    input_ids = input_ids[:, :-1] # Remove the <s> token 
    # Restore
    # tokenizer.padding_side = "right"
    # tokenizer.add_eos_token = True
    
    # The label here is trash, ignore it
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt')
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    
    
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        "data_id": example['data_id'],
        'trigger': trigger.flatten(),
        "event_type": example['event_type']
    }



        

def main():
    args = parse_args()

    # A hacky way to make llama work with flash attention
    # if args.use_flash_attn:
    #     from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    #     replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            if args.val_file is not None:
                data_files["val"] = args.val_file
            else:
                # We will add the fake val file here even if it's not specified
                data_files["val"] = args.train_file
            if args.test_file is not None:
                data_files["test"] = args.test_file
            else:
                # We will add the fake val file here even if it's not specified
                data_files["test"] = args.train_file
        dataset_args["features"] = Features({
            'Event definition': Value(dtype='string', id=None),
            'Event type': Value(dtype='string', id=None),
            'prompt': Value(dtype='string', id=None),
            'completion': Value(dtype='string', id=None),
            'trigger': Value(dtype='string', id=None),
            'Event type': Value(dtype='string', id=None),
            'Event definition': Value(dtype='string', id=None),
            'data_id': Value(dtype='int32', id=None),
            'event_type': Value(dtype='int32', id=None)})
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
        
        
        
        
    special_tokens = ['<trigger>', '<sep>', '<Trigger>']
    tokenizer.add_tokens(special_tokens)
    for tk, tkid in zip(special_tokens, tokenizer(special_tokens).input_ids):
        print(f"Special Token {tk}: {tkid}")
        print(f"Special Token decoded {tkid}: {tokenizer.decode(tkid)}")
    

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)


    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
        encode_function_val = partial(
            encode_with_prompt_completion_format_val,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    
    # Before mapping, get the ground truth
    val_ground_truth = []
    if args.val_file is not None:
        for d in raw_datasets['val']:
            # if d["trigger"] == "<trigger>":
            #     debug_ids = tokenizer.encode(d["trigger"], add_special_tokens=False)
            #     print(f"debug tokenizer.encode {debug_ids}")
            val_ground_truth.append(tokenizer.encode(d["trigger"], add_special_tokens=False)[0])

    # print(f"debug val_ground_truth {val_ground_truth}")
        
    test_ground_truth = []
    if args.test_file is not None:
        for d in raw_datasets['test']:
            test_ground_truth.append(tokenizer.encode(d["trigger"], add_special_tokens=False)[0])
        
        
    
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())
    train_dataset = lm_datasets["train"]
    
    raw_datasets_copy = copy.deepcopy(raw_datasets)
    raw_datasets_copy['train'] = raw_datasets_copy['val']
    with accelerator.main_process_first():
        lm_datasets_val = raw_datasets_copy.map(
            encode_function_val,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask", "trigger"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets_val.set_format(type="pt")
    
    
    val_dataset = lm_datasets_val["val"]
    test_dataset = lm_datasets_val["test"]
    
    
    
    
    
    

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    for index in random.sample(range(len(val_dataset)), 3):
        logger.info(f"Sample {index} of the validation set: {val_dataset[index]}.")
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )
    # For validation
    
    if args.val_file is not None:
        val_dataloader = DataLoader(
            val_dataset, 
            shuffle=False, 
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=args.per_device_eval_batch_size
        )
    # For test
    if args.test_file is not None:
        test_dataloader = DataLoader(
            test_dataset, 
            shuffle=False, 
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=args.per_device_eval_batch_size
        )
        
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader,  lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.val_file:
        val_dataloader = accelerator.prepare_data_loader(val_dataloader, device_placement=True)
    if args.test_file:
        test_dataloader = accelerator.prepare_data_loader(test_dataloader, device_placement=True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        init_kwargs = {}
        if args.report_to == "wandb":
            # See https://docs.wandb.ai/ref/python/init for all available args
            init_kwargs = {
                "wandb": {
                    "name": args.report_name,
                    "tags": args.report_tags,
                    }
                }
        accelerator.init_trackers("open_instruct", experiment_config, init_kwargs=init_kwargs)


    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            
            path = os.path.basename(args.resume_from_checkpoint)
            
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real stepsp
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    def compute_metric(prediction, ground_truth, split = "val"):
        # TODO: Add F1 ground truth here, currently using Accuracy
        
        # print(f"debug prediction {len(prediction)}")
        # print(f"debug ground_truth {len(ground_truth)}")
        print("---------------------Pred")
        print(prediction[:3])
        print("---------------------GT")
        print(ground_truth[:3])
        exit()
        # negative_class =
        total_labels = []
        total_prelabels = []
        
        Correct = 0
        for pred, gt in zip(prediction, ground_truth):
            if gt == 32000:
                print(f"debug Pred: {pred}, GT: {gt}")
            # print(f"Pred: {tokenizer.decode(pred)}, GT: {tokenizer.decode(gt)}")
            if gt == 32000:
                total_labels.append(0)
                if pred == 32000:
                    total_prelabels.append(0)
                else:
                    total_prelabels.append(1)
            else:
                total_labels.append(1)
                if pred == gt:
                    total_prelabels.append(1)
                else:
                    total_prelabels.append(0)
            if pred == gt:
                Correct += 1
        
        total_prelabels = np.array(total_prelabels)
        total_labels = np.array(total_labels)
                
        classifi_report = classification_report(
            total_labels, total_prelabels, target_names=[0, 1], output_dict=True
        )
        positive_precision = classifi_report[1]["precision"]
        positive_recall = classifi_report[1]["recall"]
        positive_f1_score = classifi_report[1]["f1-score"]
        negative_precision = classifi_report[0]["precision"]
        negative_recall = classifi_report[0]["recall"]
        negative_f1 = classifi_report[0]["f1-score"]

                
                
        return {
            f"{split}_Accuracy": Correct/len(prediction),
            f"{split}_Positive_Precision": positive_precision,
            f"{split}_Positive_Recall": positive_recall,
            f"{split}_Positive_F1": positive_f1_score,
            f"{split}_Negative_Precision": negative_precision,
            f"{split}_Negative_Recall": negative_recall,
            f"{split}_Negative_F1": negative_f1,
            }
                
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                # First try to add eval here
                if args.val_file is not None:
                    if args.eval_steps is None:
                        args.eval_steps = args.max_train_steps//args.num_train_epochs # Set to the last step of training epochs
                        logger.info(f"args.eval_steps is None. Set to do eval after each epoch, which is {args.eval_steps}")
                    if completed_steps % args.eval_steps == 0:
                        # Prepare Eval
                        model.eval()
                        scores_metric = load_metric(args.metrics_file, num_process=accelerator.num_processes, process_id=accelerator.process_index)
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                            logger.info(f"Start doing evaluation(val) at step: {completed_steps}")#
                            # Get model output for val set
                            predictions = []
                            # For val, do left padding
                            # Prepare 
                            tokenizer.padding_side = "left"
                            for eval_batch in tqdm(val_dataloader):
                                # print(f"debug eval_batch['trigger'] {eval_batch['trigger']}")
                                val_gt = eval_batch['trigger'][:, 1].squeeze().tolist()
                                val_event_type = eval_batch['event_type'].squeeze().tolist()
                                val_data_id = eval_batch['data_id'].squeeze().tolist()
                                
                                del eval_batch['trigger']
                                del eval_batch['event_type']
                                del eval_batch['data_id']
                                
                                input_ids = eval_batch["input_ids"][0]
                                
                                # print(f"debug eval_batch input_ids {input_ids}")
                                # print(f"debug eval_batch input_ids {tokenizer.decode(input_ids)}")
                                # print(f"debug eval_batch val_gt {val_gt[0]}")
                                # print(f"debug eval_batch val_gt {tokenizer.decode(val_gt[0])}")

                                # print(eval_batch.input_ids[0])
                                # print(eval_batch.attention_mask[0])
                                # print(tokenizer.decode([32003]))
                                outputs = model(**eval_batch, use_cache=False, return_dict = True)
                                batch_predict = torch.argmax(outputs.logits[:,-1,:], dim = -1).cpu().tolist()
                                
                                # print(f"debug outputs {batch_predict[0]}")
                                # print(f"debug outputs {batch_predict}")
                                # print(f"debug outputs {tokenizer.decode(batch_predict[0])}")

                                scores_metric.add_batch(predictions=batch_predict, references=val_gt, event_type=val_event_type, data_id=val_data_id)
                            # Print the last batch
                            print("Decode Prediction of Last Batch: ", tokenizer.decode(batch_predict[:10]))
                            # Restore
                            tokenizer.padding_side = "right"
                            print(f"finish inference")
                            scores = scores_metric.compute()
                            
                            if accelerator.process_index == 0:
                                
                                report_scores = {}
                                for key in scores:
                                    report_scores[f"val_{key}"] = scores[key]
                                
                                logger.info(f" [Validation] Step: {completed_steps}, Epoch: {epoch}, Scores: {json.dumps(report_scores)}")
                                if args.with_tracking:
                                    accelerator.log(
                                        report_scores,
                                        step=completed_steps,
                                    )
                        model.train()
                # First try to add eval here
                if args.test_file is not None:
                    if args.eval_steps is None:
                        args.eval_steps = args.max_train_steps//args.num_train_epochs # Set to the last step of training epochs
                        logger.info(f"args.eval_steps is None. Set to do eval after each epoch, which is {args.eval_steps}")
                    if completed_steps % args.eval_steps == 0:
                        # Prepare Eval
                        model.eval()
                        scores_metric = load_metric(args.metrics_file, num_process=accelerator.num_processes, process_id=accelerator.process_index)
                        with torch.no_grad():
                            torch.cuda.empty_cache()
                            logger.info(f"Start doing evaluation(test) at step: {completed_steps}")#
                            # Get model output for val set
                            predictions = []
                            # For val, do left padding
                            # Prepare 
                            tokenizer.padding_side = "left"
                            for eval_batch in tqdm(test_dataloader):
                                
                                test_gt = eval_batch['trigger'][:, 1].squeeze().tolist()
                                test_event_type = eval_batch['event_type'].squeeze().tolist()
                                test_data_id = eval_batch['data_id'].squeeze().tolist()
                                
                                del eval_batch['trigger']
                                del eval_batch['event_type']
                                del eval_batch['data_id']
                                
                                outputs = model(**eval_batch, use_cache=False, return_dict = True)
                                batch_predict = torch.argmax(outputs.logits[:,-1,:], dim = -1).cpu().tolist()
                                scores_metric.add_batch(predictions=batch_predict, references=test_gt, event_type=test_event_type, data_id=test_data_id)
                            # Restore
                            print("Decode Prediction of Last Batch: ", tokenizer.decode(batch_predict[:10]))
                            tokenizer.padding_side = "right"
                            scores = scores_metric.compute()
                            
                            if accelerator.process_index == 0:
                                
                                report_scores = {}
                                for key in scores:
                                    report_scores[f"test_{key}"] = scores[key]
                                
                                logger.info(f" [Test] Step: {completed_steps}, Epoch: {epoch}, Scores: {json.dumps(report_scores)}")
                                if args.with_tracking:
                                    accelerator.log(
                                        report_scores,
                                        step=completed_steps,
                                    )
                        model.train()
                # Put this to the end to enable eval after training is done
                if completed_steps >= args.max_train_steps:
                    break
        # if args.checkpointing_steps == "epoch":
        #     output_dir = f"epoch_{epoch}"
        #     if args.output_dir is not None:
        #         output_dir = os.path.join(args.output_dir, output_dir)
        #     accelerator.save_state(output_dir)
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                unwrapped_model = accelerator.unwrap_model(model)
                # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
                # Otherwise, sometimes the model will be saved with only part of the parameters.
                # Also, accelerator needs to use the wrapped model to get the state_dict.
                state_dict = accelerator.get_state_dict(model)
                if args.use_lora:
                    # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
                    # and has its own save_pretrained function for only saving lora modules.
                    # We have to mannually specify the is_main_process outside the save_pretrained function.
                    if accelerator.is_main_process:
                        unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
                else:
                    unwrapped_model.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
                    )

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        state_dict = accelerator.get_state_dict(model)
        if args.use_lora:
            # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
            # and has its own save_pretrained function for only saving lora modules.
            # We have to mannually specify the is_main_process outside the save_pretrained function.
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
        else:
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
            )
        


if __name__ == "__main__":
    main()
