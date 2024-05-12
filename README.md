# Code completion for Kotlin


## Task

This project aims to explore on teh performance improvement of fine-tuning LLMs T LLMs (specifically Phi-1.5) for the code completion task in underrepresented programming languages (specifically Kotlin language). 

## Dataset

There are two datasets used in this project: 
1. CodeXGLUE dataset used for evaluation of code completion in Python.
2. Custom Kotlin dataset parsed from the jetbrains/Kotlin repository.

The datasets adhere to the same format:
```
{
  "input": "<s> from __future__ import absolute_import , division , print_function <EOL> from . _ithreads import AlreadyQuit <EOL> class Quit ( object ) : <EOL>",
  "labels": " "def __init__ ( self ) :""
}
```

## Dataset parsing

You can use the following code snippet to run datasets parsing:
```shell
python data_parser.py --type python kotlin
```

## Evaluation

To evaluate the code completion predictions, two metrics are calculated: Levenstein Edit Similarity and Exact Match score, as suggested by the CodeXGLUE authors.

You can use the following code snippet to run evaluations:
```shell
python evaluator.py --answers path/to/answers --predictions path/to/predictions
```

## Fine tuning pipeline

You can use the following code snippet to run fine tuning:

```shell
python trainer.py --model_name microsoft/phi-1_5 --num_train_epochs 10 --train_batch_size 2 \
  --eval_batch_size 2 --learning_rate 0.00001 --gradient_accumulation_steps 2
```

## Results

### py150

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
|  Phi-1.5 pre-train                                           |    0.203   |      40.35       |
| Phi-1.5 fine-tuned                                           |   X   |       X       |

### Custom Kotlin

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
| Phi-1.5 pre-train                                           |    0.093   |      45.832        |
| Phi-1.5 fine-tuned                                           |    X   |       X       |


