# MolReGPT
Empowering Molecule Discovery for Molecule-Caption Translation with Large Language Models: A ChatGPT Perspective

## Requirements

```
transformers == 4.30.0
torch == 1.13.1+cu117
rdkit == 2022.09.5
fcd == 1.1
rank_bm25 == 0.2.2
sentence_transformers == 2.2.2
openai == 0.27.2
```

## Usage

### For start from scratch, please follow the following steps to test MolReGPT.

Step 1: Input your API_KEY in `query_chatgpt.py` and run the following command to query the results of MolReGPT.

```
python query_chatgpt.py --tgt_folder ./results/new_results/ --model gpt-3.5-turbo --n_shot 10 --m2c_method morgan --c2m_method bm25
```

Step 2: Run the following command to merge the multi-processing results.

```
python merge_transfer.py --file_path ./results/new_results/ --merge
```

Step 3: Run the evaluation scripts to get the metrics

```
python naive_test.py --pro_folder ./results/new_results/
python ./evaluations/mol_text2mol_metric.py --input_file ./results/new_results/caption2smiles_example.txt
python ./evaluations/text_text2mol_metric.py --input_file ./results/new_results/smiles2caption_example.txt
```

### For convenience, we also provide the processed results, including all the results of MolReGPT mentioned in the paper.
Step 1: Run the following command to transfer results for testing.

```
python merge_transfer.py --file_path ./results/gpt-4-0314/
```

Step 2: Run the evaluation scripts to get the metrics

```
python naive_test.py --pro_folder ./results/gpt-4-0314/
python ./evaluations/mol_text2mol_metric.py --input_file ./results/gpt-4-0314/caption2smiles_example.txt
python ./evaluations/text_text2mol_metric.py --input_file ./results/gpt-4-0314/smiles2caption_example.txt
```


## Customized Inference
If you wanna use examples provided in ChEBI dataset, you could run the `demo_full.py` script to get the results.

```
python demo_full.py
```

If you wanna try customized text captions, you could run the `demo_c2m.py` script to get the results.

```
python demo_c2m.py
```

# Have fun with MolReGPT!
