import torch
from typing import List, Tuple
from tqdm import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import itertools
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from collections import defaultdict
import numpy as np
import torch.nn as nn


AGG_FN = {
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "last": lambda x: x[-1],
    "first": lambda x: x[0],
}


def load_files(path: str):
    """
    Load source, target, and gold alignment files.

    Parameters:
        path (str): 
            The base path to the files. The function expects files with extensions .src, .tgt, and .talp.

    Returns:
        Tuple of lists containing lines from the source, target, and gold alignment files, respectively.
    """
    with open(path + ".src", encoding="utf-8") as fs:
        lines_src = fs.readlines()
    with open(path + ".tgt", encoding="utf-8") as ft:
        lines_tgt = ft.readlines()
    with open(path + ".talp", encoding="utf-8") as g:
        gold_lines = g.readlines()
        
    assert len(lines_src)==len(lines_tgt)==len(gold_lines)
        
    return lines_src, lines_tgt, gold_lines


def collate_fn_span(examples: List[dict], tokenizer: PreTrainedTokenizer = None):
    """
    Collate function for preparing batches of binary token classification examples for BinaryAlign.

    Parameters:
        examples (List[dict]): 
            A list of examples where each example is a dictionary containing input_ids, attention_mask, and labels.
        tokenizer (PreTrainedTokenizer, *optional*): 
            A tokenizer for padding purposes.

    Returns:
        dict: 
            A dictionary containing padded input_ids, attention_mask, and labels for the batch.
            If examples are tuples, returns a dictionary with keys suffixed by 1 and 2 for each part of the tuple.
    """
    def _get_examples(examples):
        example1 = []
        example2 = []
        for example in examples:
            example1.append(example[0])
            example2.append(example[1])
        return example1, example2
    
    def _produce_batch(examples):
        input_ids = [torch.tensor(x["input_ids"]) for x in examples]
        attention_mask= [torch.tensor(x["attention_mask"]) for x in examples]
        labels = [torch.tensor(x["labels"]) for x in examples]
            
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
    if isinstance(examples[0], tuple):
        examples1, examples2 = _get_examples(examples)
        batch1 = _produce_batch(examples1)
        batch2 = _produce_batch(examples2)
    else:
        return _produce_batch(examples)
    
    
    return {
        **{f"{key}1":value for key,value in batch1.items()},
        **{f"{key}2":value for key,value in batch2.items()},
    }


def parse_single_alignment(string, one_indexed=False, use_reverse: bool = False):
    """
    Parse a single alignment string into a tuple of integers.

    Parameters:
        string (`str`): 
            The alignment string.
        one_indexed (`bool`, *optional*): 
            Whether the indices in the alignment string are one-indexed. Default is False.
        use_reverse (`bool`, *optional*): 
            Whether to reverse the order of the parsed indices. Default is False.

    Returns:
        tuple: 
        A tuple of two integers representing the alignment.
    """
    assert ('-' in string or 'p' in string) and 'Bad Alignment separator'
    a, b = string.replace('p', '-').split('-')
    a, b = int(a), int(b)

    if one_indexed:
        a = a - 1
        b = b - 1
        
    if use_reverse:
        return b, a
    else:
        return a, b


@torch.no_grad
def evaluation_span_bidirectional(
    dataloader: DataLoader,
    model: PreTrainedModel, 
    threshold: float,
    sure: List[str],
    possible: List[str],
    batch_size: int,
    bidirectional_combine_type: str,
    tk2word_prob: str
) -> Tuple[float]:
    
    model.eval()
    softmax = nn.Softmax(dim=-1)
    sigmoid = nn.Sigmoid()
    hypothesis = []
    
    for sample in tqdm(dataloader, desc="Evaluating"):
        
        def _get_sample_probs(sample, reverse: bool = False) -> Dict[tuple, float]:
            sample_preds = defaultdict(list)
            all_input_ids = sample["input_ids"][0].split(batch_size)
            all_attention_mask = sample["attention_mask"][0].split(batch_size)
            bpe2word_map = sample["bpe2word_map"][0].tolist()
            
            count = 0
            for input_ids, attention_mask in zip(all_input_ids, all_attention_mask):
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)
                
                logits = model(input_ids, attention_mask=attention_mask).logits
                if logits.shape[2]==2:
                    probs = softmax(logits)[:,:, 1]
                else:
                    probs = sigmoid(logits)[:,:, 0]
                
                for index, sentence_probs in enumerate(probs):
                    index = count+index
                    for word_number, word_prob in enumerate(sentence_probs):
                        tgt = bpe2word_map[word_number]
                        if tgt!=-1:
                            if reverse:
                                sample_preds[(tgt, index)].append(word_prob.item())
                            else:
                                sample_preds[(index, tgt)].append(word_prob.item())
                        
                count+=batch_size

            return {k:AGG_FN[tk2word_prob](v) for k,v in sample_preds.items()}
        
        def _combine_preds(preds1: Dict[tuple, float], preds2: Dict[tuple, float], bidirectional_combine_type: str) -> set:
            current_preds = set()
            if bidirectional_combine_type=="union":
                preds1 = set({k:v for k,v in preds1.items() if v>threshold})
                preds2 = set({k:v for k,v in preds2.items() if v>threshold})
                preds1 = set(preds1)
                preds2 = set(preds2)
                current_preds = preds1.union(preds2)
            elif bidirectional_combine_type=="intersection":
                preds1 = set({k:v for k,v in preds1.items() if v>threshold})
                preds2 = set({k:v for k,v in preds2.items() if v>threshold})
                preds1 = set(preds1)
                preds2 = set(preds2)
                current_preds = preds1.intersection(preds2)
            elif bidirectional_combine_type=="avg":
                current_preds_with_probs = {k: (preds1.get(k, 0) + preds2.get(k, 0))/2 for k in set(preds1) | set(preds2)}
                current_preds = set({k:v for k,v in current_preds_with_probs.items() if v>threshold})
            elif bidirectional_combine_type=="bidi_avg":
                preds1 = {k:v for k,v in preds1.items() if v>threshold}
                preds2 = {k:v for k,v in preds2.items() if v>threshold}
                current_preds_with_probs = {k: (preds1.get(k, 0) + preds2.get(k, 0))/2 for k in set(preds1) | set(preds2)}
                current_preds = set({k:v for k,v in current_preds_with_probs.items() if v>threshold})
            else:
                raise ValueError(
                    f"{bidirectional_combine_type} not supported!"
                )
            
            return current_preds
        
        if isinstance(sample, list):
            sample1, sample2 = sample
            sample_preds1 = _get_sample_probs(sample1, reverse=False)
            sample_preds2 = _get_sample_probs(sample2, reverse=True)
            current_preds = _combine_preds(sample_preds1, sample_preds2, bidirectional_combine_type)
        else:
            sample_preds = _get_sample_probs(sample, reverse=False)
            current_preds = set({k:v for k,v in sample_preds.items() if v>threshold})
                    
        hypothesis.append(current_preds)
        
            
    print(len(sure), len(possible), len(hypothesis))
    metrics = calculate_metrics(sure, possible, hypothesis)
    aer = round(metrics[2],4)
    
    print("*** AER: ***", aer)
    print("*** Precision: ***", round(metrics[0],4))
    print("*** Recall: ***", round(metrics[1],4))
    print("*** F1 score: ***", round(metrics[3],4))
    model.train()
        
    return metrics



def calculate_metrics(array_sure, array_possible, array_hypothesis):
    """
    Function taken from https://github.com/lilt/alignment-scripts/blob/master/scripts/aer.py
    """
    
    assert len(array_sure) == len(array_possible) == len(array_hypothesis)

    sum_a_intersect_p, sum_a_intersect_s, sum_s, sum_a = 4 * [0.0]

    for S, P, A in itertools.zip_longest(array_sure, array_possible, array_hypothesis):
        
        sum_a += len(A)
        sum_s += len(S)
        sum_a_intersect_p += len(A.intersection(P))
        sum_a_intersect_s += len(A.intersection(S))

    if sum_a!=0:
        precision = sum_a_intersect_p / sum_a
        recall = sum_a_intersect_s / sum_s
        aer = 1.0 - ((sum_a_intersect_p + sum_a_intersect_s) / (sum_a + sum_s))
    else:
        precision = 0.
        recall= 0.
        aer= 1.
        
    if precision+recall>0:
        f1_score = (2*precision*recall)/(precision+recall)
    else:
        f1_score = 0.

    return precision, recall, aer, f1_score