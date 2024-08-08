# BinaryAlign: Word Alignment as Binary Sequence Labeling

## Introduction
**BinaryAlign** reformulates word alignment as a set of binary sequence labeling tasks. It outperforms existing approaches in both high and low-resource language settings, providing a unified approach to word alignment. This repository contains the code and models for BinaryAlign as described in [our paper](https://arxiv.org/pdf/2407.12881) accepted to the main conference of ACL 2024.

## Datasets

### Format

* source (.src)
```
He has a sofa .
```

* target (.tgt)
```
Il a un canapé .
```

* gold alignment (.talp)
```
1-1 2-2 3-3 4-4 5-5 
```

### Data

We used the same datasets as https://github.com/sufenlp/AccAlign.

## Training

```shell
bash train.sh
```

## Evaluation

```shell
bash eval.sh
```

## Checkpoints

| Training Languages |Link |
| ------------- | ------------- |
| Align6 |  models/ |
| deen |   models/ |
| roen |   models/ |
| fren |   models/ |
| zhen |   models/ |
| jaen |   models/ |

## Citation

```
@article{latouche2024binaryalign,
  title={BinaryAlign: Word Alignment as Binary Sequence Labeling},
  author={Latouche, Gaetan Lopez and Carbonneau, Marc-Andr{\'e} and Swanson, Ben},
  journal={arXiv preprint arXiv:2407.12881},
  year={2024}
}
```

## License

TO ADD