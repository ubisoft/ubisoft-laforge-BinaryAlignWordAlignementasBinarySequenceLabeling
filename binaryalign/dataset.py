import torch
import itertools
from typing import Optional
from transformers import PreTrainedTokenizer
from typing import List
from tqdm import tqdm
from binaryalign.utils import parse_single_alignment


class SpanTokenDataset:
    r"""
    Initialize SpanTokenDataset.
    
    This function is adapted from the AccAlign repository:
    https://github.com/sufenlp/AccAlign/blob/main/aligner/sent_aligner.py

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer to use.
        lines_src (`List[str]`):
            A list of strings containing the source sentences.
        lines_tgt (`List[str]`):
            A list of strings containing the target sentences.
        gold_lines (`List[str]`, *optional*):
            A list of strings containing the gold alignments.
        gold_one_index (`bool`, *optional*):
            A boolean indicating whether the gold list indexing starts at 1. 
        ignore_possible_alignments (`bool`, defaults to ):
            A boolean indicating whether we ignore the possible alignments or not. Only recommended for training purposes.
        context_sep (`str`, defaults to ` [WORD_SEP] `):
            A token used to surround the word to get the alignment of.
        do_inference (`bool`, defaults to `False`):
            A boolean indicating whether this dataset is used for inference or training.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        lines_src: List[str],
        lines_tgt: List[str],
        gold_lines: Optional[List[str]] = None,
        gold_one_index: Optional[bool] = False,
        ignore_possible_alignments: Optional[bool] = False,
        context_sep: Optional[str] = " [WORD_SEP] ",
        do_inference: Optional[bool] = False,
    ):
        
        self.ignore_possible_alignments = ignore_possible_alignments
        self.gold_one_index = gold_one_index
        self.context_sep = context_sep
        self.tokenizer = tokenizer
        self.data, self.data_2 = [], []
        self.sure, self.possible = [], []
        self.do_inference = do_inference
        
        self.prepare(lines_src, lines_tgt, gold_lines, self.data)
        self.prepare(lines_src, lines_tgt, gold_lines, self.data_2, use_reverse=True)
        if not self.do_inference:
            self.data = self.data + self.data_2
        
        
    def prepare(
        self,
        lines_src: List[str],
        lines_tgt: List[str],
        gold_lines: List[str],
        data: List[dict],
        use_reverse: bool = False
        ) -> None:
        pbar = tqdm(total=len(lines_src))
        
        for _, (line_src, line_tgt, gold_line) in enumerate(zip(lines_src, lines_tgt, gold_lines)):
            pbar.update(1)
            examples = []
            
            if use_reverse:
                line_src, line_tgt = line_tgt, line_src
                
            sent_src, sent_tgt = line_src.strip().split(), line_tgt.strip().split()
                
            if len(sent_src) == 0 or len(sent_tgt) == 0:
                continue
            
            for i, _ in enumerate(sent_src):
                examples.append(sent_src[:i] + [self.context_sep] + [sent_src[i]] + [self.context_sep] + sent_src[i+1:])
                
            
            token_src, token_tgt = [[self.tokenizer.tokenize(word)  for word in sent_src] for sent_src in examples], [self.tokenizer.tokenize(word) for word in sent_tgt]
                
            wid_src, wid_tgt = [[self.tokenizer.convert_tokens_to_ids(x) for x in token ] for token in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
            
            input_ids_src = torch.tensor([self.tokenizer.prepare_for_model(list(itertools.chain(*wid)), max_length=512, truncation=True)["input_ids"] for wid in wid_src])
            input_ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                            max_length=512, truncation=True)['input_ids'][1:]
            
            input_ids_tgt = input_ids_tgt.repeat(len(input_ids_src),1)
        
            bpe2word_map_tgt = []
            for i, word_list in enumerate(token_tgt):
                bpe2word_map_tgt += [i for _ in word_list]
            bpe2word_map_tgt = torch.tensor(bpe2word_map_tgt + [-1])
            bpe2word_map_src = torch.ones_like(input_ids_src[0])*-1
            
            input_ids = torch.cat((input_ids_src, input_ids_tgt), axis=-1)[:, :512]
            labels_src = torch.ones_like(input_ids_src)*-100
            labels_tgt = torch.zeros_like(input_ids_tgt)
            
            if not use_reverse:
                self.sure.append(set())
                self.possible.append(set())
            
            for src_tgt in gold_line.strip().split():
                
                if 'p' in src_tgt:
                    if self.ignore_possible_alignments:
                        continue
                    wsrc, wtgt = src_tgt.split('p')
                else:
                    wsrc, wtgt = src_tgt.split('-')

                if use_reverse:
                    wtgt, wsrc= (int(wsrc), int(wtgt)) if not self.gold_one_index else (
                        int(wsrc) - 1, int(wtgt) - 1)
                else:
                    wsrc, wtgt= (int(wsrc), int(wtgt)) if not self.gold_one_index else (
                        int(wsrc) - 1, int(wtgt) - 1)
                   
                if wsrc<len(labels_tgt):
                    labels_tgt[wsrc,:] = torch.where(bpe2word_map_tgt==wtgt, 1, labels_tgt[wsrc,:])
                           
                if not use_reverse:
                    # only do it for the sure alignment
                    sure_alignment = True if '-' in src_tgt else False
                    alignment_tuple = parse_single_alignment(src_tgt, self.gold_one_index)

                    if sure_alignment:
                        self.sure[-1].add(alignment_tuple)
                    if sure_alignment or not self.ignore_possible_alignments:
                        self.possible[-1].add(alignment_tuple)
                
            labels = torch.cat((labels_src, labels_tgt),1)
            if input_ids[0, -1]==self.tokenizer.eos_token_id:
                labels[:,-1] = -100
            
            if self.do_inference:
                bpe2word_map = torch.cat((bpe2word_map_src, bpe2word_map_tgt), 0)
                data.append(
                        {
                            "input_ids": input_ids,
                            "attention_mask": torch.ones_like(input_ids),
                            "labels": labels,
                            "bpe2word_map": bpe2word_map
                        }
                    )
            else:
                for input_id, label in zip(input_ids.tolist(), labels.tolist()):
                    data.append(
                        {
                            "input_ids": input_id,
                            "attention_mask": [1]* len(input_id),
                            "labels": label[:512]
                        }
                    )
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        if self.do_inference:
            return self.data[item], self.data_2[item]
        else:
            return self.data[item]
