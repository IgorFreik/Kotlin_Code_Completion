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
  "input": "",
  "labels": ""
}
```

## Evaluation

To evaluate the code completion predictions, two metrics are calculated: Levenstein Edit Similarity and Exact Match score, as suggested by the CodeXGLUE authors.

You can use the following code snippet to run evaluations:
```shell
export <VARS>

python evaluator/main.py <VARS>
```

## Fine tuning pipeline

You can use the following code snippet to run fine tuning:

```shell
export <VARS>

python sft/main.py <VARS>
```

## Results

### py150

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
|  Phi-1.5 pre-train                                           |    X   |      X       |
| Phi-1.5 fine-tuned                                           |   X   |       X       |

### Custom Kotlin

| Model                                                 |     EM     |  Edit similarity  |
| ----------------------------------------------------- | :--------: | :---------------: |
| Phi-1.5 pre-train                                           |    X   |       X       |
| Phi-1.5 fine-tuned                                           |    X   |       X       |


