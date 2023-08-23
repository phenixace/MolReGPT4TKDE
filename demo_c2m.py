import requests
import json
import openai
import os
import time
import random
import re
import rank_bm25
import argparse
from sentence_transformers import SentenceTransformer, util
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
from rdkit import DataStructs

api_key = "YOUR_API_KEY"
# openai.organization = "YOUR_ORG_ID"
openai.api_key = api_key


def retrieve_m2c_zero_prompts():
    template = "Task Format: \n" \
        + "```\n" \
        + "Instruction: Given the SMILES representation of a molecule, predict the caption of the molecule.\n" \
        + "Input: [MOLECULE_MASK]\n" \
        + "```\n" \
        + "\n" \
        + "Your output should be: \n" \
        + "```\n" \
        + "{\"caption\": \"[CAPTION_MASK]\"}\n" \
        + "```\n" \
        + "\n"


    head_prompt = "You are now working as an excellent expert in chemisrty and drug discovery. Given the SMILES representation of a molecule, your job is to predict the caption of the molecule. The molecule caption is a sentence that describes the molecule, which mainly describes the molecule's structures, properties, and production.\n" \
        + "\n" \
        + template + "\n" \
        + "Your response should only be in the JSON format above; THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. "
    return head_prompt


def retrieve_m2c_prompts(examples):

    def get_template(num):
        template = "Example {}: \n".format(num+1) \
            + "```\n" \
            + "Instruction: Given the SMILES representation of a molecule, predict the caption of the molecule.\n" \
            + "Input: {}\n".format(examples[num]["molecule"]) \
            + "```\n" \
            + "\n" \
            + "Your output should be: \n" \
            + "```\n" \
            + "{\"caption\": \"" + examples[num]["caption"] + "\"}\n" \
            + "```\n" \
            + "\n"

        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        example_prompts += get_template(i)

    head_prompt = "You are now working as an excellent expert in chemisrty and drug discovery. Given the SMILES representation of a molecule, your job is to predict the caption of the molecule. The molecule caption is a sentence that describes the molecule, which mainly describes the molecule's structures, properties, and production.\n" \
        + "\n" \
        + example_prompts + "\n" \
        + "Your response should only be in the JSON format above; THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. "
    return head_prompt

def retrieve_c2m_zero_prompts():
    template = "Task Format\n" \
        + "```\n" \
        + "Instruction: Given the caption of a molecule, predict the SMILES representation of the molecule.\n" \
        + "Input: [CAPTION_MASK]\n" \
        + "```\n" \
        + "\n" \
        + "Your output should be: \n" \
        + "```\n" \
        + "{\"molecule\": \"[MOLECULE_MASK]\"}\n" \
        + "```\n" \
        + "\n"
        
    head_prompt = "You are now working as an excellent expert in chemisrty and drug discovery. Given the caption of a molecule, your job is to predict the SMILES representation of the molecule. The molecule caption is a sentence that describes the molecule, which mainly describes the molecule's structures, properties, and production. You can infer the molecule SMILES representation from the caption.\n" \
        + "\n" \
        + template + "\n" \
        + "Your response should only be in the exact JSON format above; THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. "
    return head_prompt


def retrieve_c2m_prompts(examples):

    def get_template(num):
        template = "Example {}: \n".format(num+1) \
            + "```\n" \
            + "Instruction: Given the caption of a molecule, predict the SMILES representation of the molecule.\n" \
            + "Input: {}\n".format(examples[num]["caption"]) \
            + "```\n" \
            + "\n" \
            + "Your output should be: \n" \
            + "```\n" \
            + "{\"molecule\": \"" + examples[num]["molecule"] + "\"}\n" \
            + "```\n" \
            + "\n"
        
        return template
    
    example_prompts = ""
    for i in range(len(examples)):
        example_prompts += get_template(i)
        
    head_prompt = "You are now working as an excellent expert in chemisrty and drug discovery. Given the caption of a molecule, your job is to predict the SMILES representation of the molecule. The molecule caption is a sentence that describes the molecule, which mainly describes the molecule's structures, properties, and production. You can infer the molecule SMILES representation from the caption.\n" \
        + "\n" \
        + example_prompts + "\n" \
        + "Your response should only be in the exact JSON format above; THERE SHOULD BE NO OTHER CONTENT INCLUDED IN YOUR RESPONSE. "
    return head_prompt

def sentenceBERT_similarity(caption, caption_corpus):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    query_embedding = model.encode([caption], convert_to_tensor=True)
    caption_embeddings = model.encode(caption_corpus, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, caption_embeddings)[0]
    cos_scores = cos_scores.cpu().detach().numpy()
    return cos_scores


def get_examples(file, n_shot, input=None, c2m_method="random"):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    lines = lines[1:]
    molecule_corpus = []
    caption_corpus = []
    for line in lines:
        line = line.strip().strip("\n").strip()
        molecule_corpus.append(line.split("\t")[1])
        caption_corpus.append(line.split("\t")[2])

    def remove_punctuation(text):
        text = text.replace("-", " ")
        text = text.replace(",", " ")
        text = text.replace(".", "")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = re.sub(r' +', ' ', text)
        return text


    cap_examples = []
    if c2m_method == "bm25":
        # retrieve caption examples
        tokenized_caption_corpus = []
        for doc in caption_corpus:
            doc = remove_punctuation(doc)
            tokenized_caption_corpus.append(doc.split(" "))

        bm25 = rank_bm25.BM25Okapi(tokenized_caption_corpus)
        query = input["caption"]
        query = remove_punctuation(query)
        tokenized_query = query.split(" ")
        # print(tokenized_query)

        doc_scores = bm25.get_scores(tokenized_query)
        candidates = [i for i in range(len(doc_scores))]
        candidates = sorted(candidates, key=lambda i: doc_scores[i], reverse=True)
        candidates = candidates[:n_shot]
        for candidate in candidates:
            cap_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    elif c2m_method == "sentencebert":
        # retrieve caption examples
        doc_scores = sentenceBERT_similarity(input["caption"], caption_corpus)
        candidates = [i for i in range(len(doc_scores))]
        candidates = sorted(candidates, key=lambda i: doc_scores[i], reverse=True)
        candidates = candidates[:n_shot]
        for candidate in candidates:
            cap_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    elif c2m_method == "random":
        candidates = random.sample(range(len(lines)), n_shot)
        for candidate in candidates:
            cap_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    return cap_examples



if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="./cap2mol_trans_raw/")
    parser.add_argument("--n_shot", type=int, default=10
                        )
    parser.add_argument("--c2m_method", type=str, default="bm25")

    args = parser.parse_args()

    # print args
    print("===== args =====")
    print("data_folder: {}".format(args.data_folder))
    print("n_shot: {}".format(args.n_shot))
    print("c2m_method: {}".format(args.c2m_method))
    print("================")


    # iterate the file folder
    data_folder = args.data_folder
    n_shot = args.n_shot
    c2m_method = args.c2m_method
    


    example_file = data_folder + "train.txt"
    with open(example_file, 'r') as f:
        temp_lines = f.readlines()
        
    temp_lines = temp_lines[1:]
    
    caption = input("Please input the molecule caption:")

    input = {"caption": caption}

    if n_shot != 0:
        cap_examples = get_examples(example_file, n_shot, input=input, c2m_method=c2m_method)
        caption2molecule = retrieve_c2m_prompts(cap_examples)
    else:
        caption2molecule = retrieve_c2m_zero_prompts()

    try:    
        res2 = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": caption2molecule},
                            {"role": "user", "content": "Input: " + caption + "\n"},
                        ]
                    )
        response2 = res2['choices'][0]['message']['content'].strip('\n')
        print("="*5 + "RAW" + "="*5)
        print(response2)

        template = re.compile(r'\{.*\}')
        try:
            response2 = json.loads(template.findall(response2)[0])
        except:
            response2 = template.findall(response2)[0]
            response2 = {"molecule": response2.split(":")[1].strip('\n').strip("}").strip().strip("\"")}

        try:
            response2 = response2["molecule"]
        except:
            try:
                response2 = response2["smiles"]
            except:
                response2 = response2["predicted_smiles"]
        print("="*5 + "PROCESSED" + "="*5)
        print(response2)
    except Exception as e:
        print(e)
        # query too long
        if "maximum context length" in str(e):
            length_ex = [len(cap_example["molecule"]) + len(cap_example["caption"]) for cap_example in cap_examples]
            temp_n = length_ex.index(max(length_ex))
            cap_examples.pop(temp_n)
            caption2molecule = retrieve_c2m_prompts(cap_examples)
        else:
            try:
                length_ex = [len(cap_example["molecule"]) + len(cap_example["caption"]) for cap_example in cap_examples]
                temp_n = length_ex.index(max(length_ex))
                cap_examples.pop(temp_n)
                caption2molecule = retrieve_c2m_prompts(cap_examples)
            except:
                pass
