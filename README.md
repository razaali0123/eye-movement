Predicting General and Passage Level Text Comprehension from Scanpaths using Transformers Architecture
====================================================================================================================




## About the Project
In this project, I devised a dual-sequence model that simultaneously handles
the processing of both fixated words and eye-movement scan paths. The measures to be predicted includes
inferring text comprehension, general text comprehension, determining how
hard was reading the text for the reader and finally whether the participant was a
native English speaker or not. An ablation study and qualitative analysis support
an in-depth understanding of the model’s behavior. we used multiple different architectures and later the compared the results with the previous state of the art model. The models can be listed as follow:

* Pretrained DistillBERT
* Unaligned DistillBERT
* Trained DistillBERT by aggregating Embeddings
   * Sum of sub-token Embedding
   * Mean of sub-token Embedding
* LSTM Model
* Ablation study
* Pretrained DistillBERT with only first sub-token

## Reproduce the experiments

### Clone this repository
You can clone this repository by using
```bash
git clone https://github.com/razaali0123/eye-movement.git
```


### Download the data
You can download the publicly available data here

```bash
git clone https://github.com/ahnchive/SB-SAT
```

### Install packages
Install all required python packages via:
```bash
pip install -r requirements.txt
```
### Extract data

I executed my code using Google Colab, so I'll provide the paths accordingly. you have to replace the <NAME_OF_MODEL> with the options of the models given above.

You can create the data splits using:
```bash
python /content/eye-movement/utils/generate_text_sequence_splits_<NAME_OF_MODEL.py> --seq_len 100 --scale True --preprocess_text True
```

### Running the Model
Then you can directly start training the choosen model using the command. you have to replace the <NAME_OF_MODEL> with the options of the models given above.


If you provide multiple values for either seq_len_list or dropout_list, the script will automatically initiate a process to fine-tune the model based on those values.

```bash
python /content/eye-movement/nn/<NAME_OF_MODEL.py> --save_dir "/content/eye-movement/nn/results/"  -seq_len_list 10 20 --dropout_list 0.1 0.5
```

Note: Running the experiments, especially on CPU, will take some time.

### Previous State-of-the-art-Model

This repo provides the code for comparing the results with the previous state of the art model which are presented in the following paper. [Inferring Native and Non-Native Human Reading Comprehension and Subjective Text Difficulty from Scanpaths in Reading](https://dl.acm.org/doi/abs/10.1145/3517031.3529639).

To compare the results, I used the exact same evaluation protocol as used in this paper.



