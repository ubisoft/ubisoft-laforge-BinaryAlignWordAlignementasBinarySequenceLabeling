import torch
from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    HfArgumentParser, 
    AdamW, 
    get_linear_schedule_with_warmup
)

from binaryalign.utils import collate_fn_span, load_files, evaluation_span_bidirectional
from binaryalign.dataset import SpanTokenDataset
from binaryalign.models import AutoModelForBinaryTokenClassification

from dataclasses import dataclass, field
from typing import Optional
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator
import os

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScriptArguments:
    # dataset args
    train_path: str = field(default=None, metadata={"help": "Path to the training dataset file."})
    eval_path: str = field(default=None, metadata={"help": "Path to the evaluation dataset file."})
    gold_one_index: Optional[bool] = field(default=True, metadata={"help": "Indicate whether the gold alignments are one-indexed. Default is True."})
    ignore_possible_alignments: Optional[bool] = field(default=False, metadata={"help": "Ignore possible alignments during training if set to True."})
    do_train: Optional[bool] = field(default=False, metadata={"help": "Set to True to perform training."})
    do_eval: Optional[bool] = field(default=False, metadata={"help": "Set to True to perform evaluation."})
    
    # model args
    model_path: Optional[str] = field(default="microsoft/mdeberta-v3-base", metadata={"help": "Path to the model (either from Hugging Face or a local path)."})
    is_pretrained: Optional[bool] = field(default=False, metadata={"help": "Indicate whether the model has been fine-tuned on word alignment data before."})
    dropout_rate: Optional[float] = field(default=.1, metadata={"help": "Dropout rate for the model."})
    tk2word_prob: Optional[str] = field(default="max", metadata={"help": "Method to aggregate token to word probabilities. Options: 'last', 'first', 'mean', 'min', 'max'."})
    threshold: Optional[float] = field(default=.5, metadata={"help": "Threshold for binary classification."})
    bidirectional_combine_type: Optional[str] = field(default="avg", metadata={"help": "Method to combine bidirectional alignments. Options: 'union', 'intersection', 'bidi_avg', 'avg'."})
    
    # training args
    num_train_epochs: Optional[int] = field(default=5, metadata={"help": "Number of training epochs."})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "Learning rate for the optimizer."})
    weight_decay: Optional[float] = field(default=1e-2, metadata={"help": "Weight decay for the optimizer."})
    bs: Optional[int] = field(default=8, metadata={"help": "Batch size for training."})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "Warmup ratio for learning rate scheduling."})
    logging_steps: Optional[int] = field(default=100, metadata={"help": "Number of steps between logging events."})
    save_strategy: Optional[str] = field(default='end', metadata={"help": "Strategy for saving checkpoints. Options: 'epoch', 'end'."})
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Enable gradient checkpointing to save memory if set to True."})
    mixed_precision: Optional[str] = field(default="fp16", metadata={"help": "Mixed precision training mode. Options: 'fp16', 'bf16'."})
    save_path: str = field(default=None, metadata={"help": "Directory to save the model checkpoints."})
    log_dir: str = field(default=None, metadata={"help": "Directory to save training logs."})
    name: str = field(default=None, metadata={"help": "Name of the run, used for saving and logging."})
    log_with: str = field(default="tensorboard", metadata={"help": "Logging tool to use. Options: 'tensorboard', 'wandb'."})
    
    
def main(args: ScriptArguments):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not args.do_train and not args.do_eval:
        raise ValueError(
            "You should set at least 'do_eval' or 'do_train' as True."
        )
    
    # keep track of experiments with tensorboard logs
    if args.do_train:
        save_path = os.path.join(args.save_path, args.name)
        args.save_path = save_path
        log_dir = os.path.join(args.log_dir, args.name)
        if args.do_train:
            accelerator = Accelerator(
                mixed_precision= args.mixed_precision,
                log_with=args.log_with,
                project_dir=log_dir
                )
            accelerator.init_trackers(".")

        
    print("Loading the model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    config.classifier_dropout = args.dropout_rate
    model = AutoModelForBinaryTokenClassification.from_pretrained(args.model_path, config=config)
            
    context_sep = " [WORD_SEP] "
    if not args.is_pretrained:
        print("Adding a special token")
        special_tokens_dict = {"additional_special_tokens": [context_sep]}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))          
       
    
    print("Loading the dataset") 
    if args.do_train:
        lines_src, lines_tgt, gold_lines = load_files(args.train_path)
    eval_lines_src, eval_lines_tgt, eval_gold_lines = load_files(args.eval_path)
    collator = partial(collate_fn_span, tokenizer=tokenizer)
    
    
    if args.do_train:
        train_dataset = SpanTokenDataset(
            tokenizer,
            lines_src, 
            lines_tgt,
            gold_lines,
            gold_one_index=args.gold_one_index,
            ignore_possible_alignments=args.ignore_possible_alignments,
            context_sep=context_sep,
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collator,
            batch_size=args.bs
        )
        number_of_steps = len(train_dataloader) * args.num_train_epochs
        warmup_steps = args.warmup_ratio*number_of_steps
        
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if (not (any(nd in n for nd in no_decay)))],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if ((any(nd in n for nd in no_decay)))],
            "weight_decay": 0.0},
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=number_of_steps
        )
        
        model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, train_dataloader)
        
    
    if args.do_eval:
        if not args.do_train:
            model = model.to(device)
            
        eval_dataset = SpanTokenDataset(
            tokenizer,
            eval_lines_src, 
            eval_lines_tgt,
            eval_gold_lines,
            gold_one_index=args.gold_one_index,
            ignore_possible_alignments=False,
            do_inference=True,
            context_sep=context_sep,
        )
        
        sure, possible = eval_dataset.sure, eval_dataset.possible
        print(len(sure), len(possible))
    
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size = 1,
            shuffle=False
        )
    
    evaluate_fn = partial(evaluation_span_bidirectional, bidirectional_combine_type=args.bidirectional_combine_type, tk2word_prob=args.tk2word_prob)
    
    if args.do_eval:
        
        metrics = evaluate_fn(
            dataloader=eval_dataloader,
            model=model,
            threshold=args.threshold,
            sure = sure,
            possible=possible,
            batch_size=args.bs
        )
            
        if args.do_train:
            accelerator.log(
                {
                    "precision": metrics[0],
                    "recall": metrics[1],
                    "aer": metrics[2],
                },
                step=0
            )
    
    if args.do_train:
        
        accelerator.wait_for_everyone()
        accelerator.print("*** Start Training ***")
        accelerator.print(" Num examples = ", len(train_dataset))
        accelerator.print(" Num Epochs = ", args.num_train_epochs)
        accelerator.print( " Batch Size per device = ", args.bs)
        accelerator.print(" Total optimization steps = ", number_of_steps)
        pbar = tqdm(int(number_of_steps), total=number_of_steps, disable=not accelerator.is_local_main_process)
        global_step, globalstep_last_logged = 0, 0
        total_loss_scalar = 0.
        best_aer = 100
    
        for epoch in range(args.num_train_epochs):
            for batch in train_dataloader:
                batch = {k:v.to(model.device) for k,v in batch.items()}

                loss = model(**batch).loss
                    
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                
                global_step += 1
                total_loss_scalar += round(loss.item(), 4)
                
                if global_step%args.logging_steps==0:
                    
                    tr_loss = round(total_loss_scalar/(global_step-globalstep_last_logged), 4)
                    globalstep_last_logged = global_step
                    total_loss_scalar = 0.
                    accelerator.log({'loss': tr_loss}, step=global_step)
                    accelerator.print(tr_loss)
                    
                pbar.update(1)
                
            if args.do_eval:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    metrics = evaluate_fn(
                        dataloader=eval_dataloader,
                        model=model,
                        threshold=args.threshold,
                        sure = sure,
                        possible=possible,
                        batch_size=args.bs
                    )
                    aer = metrics[2]
                    accelerator.log(
                        {
                            "precision": metrics[0],
                            "recall": metrics[1],
                            "aer": metrics[2],
                        },
                        step=global_step
                    )
                
            accelerator.wait_for_everyone()
            if args.save_strategy=="epoch" and epoch!=(args.num_train_epochs-1) and aer<best_aer:
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.print("Found a better model. Saving the model")
                save_epoch_path = os.path.join(args.save_path, f"checkpoint_{epoch}")
                unwrapped_model.save_pretrained(save_epoch_path)
                tokenizer.save_pretrained(save_epoch_path)
                best_aer = aer
        
        
        accelerator.print("*** End of Training ***")
        accelerator.print("*** Saving the model... ***")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
            
            
if __name__=="__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)