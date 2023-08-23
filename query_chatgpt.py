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

import multiprocessing as mp

# initialize
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


def get_examples(file, n_shot, input=None, m2c_method="random", c2m_method="random", molecule_rdkits=None):
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

    mol_examples = []
    if m2c_method == "bm25":
        # retrieve molecule examples
        tokenized_molecule_corpus = [list(doc) for doc in molecule_corpus]
        bm25 = rank_bm25.BM25Okapi(tokenized_molecule_corpus)
        query = input["molecule"]
        tokenized_query = list(query)
        # print(tokenized_query)

        doc_scores = bm25.get_scores(tokenized_query)
        candidates = [i for i in range(len(doc_scores))]
        candidates = sorted(candidates, key=lambda i: doc_scores[i], reverse=True)
        candidates = candidates[:n_shot]

        for candidate in candidates:
            mol_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    elif m2c_method == "morgan":
        # retrieve molecule examples
        input_mol = Chem.MolFromSmiles(input["molecule"])
        mol_scores = []
        for mol in molecule_rdkits:
            mol_scores.append(DataStructs.FingerprintSimilarity(FingerprintMols.FingerprintMol(mol), FingerprintMols.FingerprintMol(input_mol)))

        candidates = [i for i in range(len(mol_scores))]
        candidates = sorted(candidates, key=lambda i: mol_scores[i], reverse=True)
        candidates = candidates[:n_shot]

        for candidate in candidates:
            mol_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})
    
    elif m2c_method == "random":
        candidates = random.sample(range(len(lines)), n_shot)
        for candidate in candidates:
            mol_examples.append({"molecule": molecule_corpus[candidate], "caption": caption_corpus[candidate]})

    return cap_examples, mol_examples


if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="./cap2mol_trans_raw/")
    parser.add_argument("--tgt_folder", type=str, default="./results/new_results/")
    parser.add_argument("--model", type=str, default="gpt-4-0314")
    parser.add_argument("--n_shot", type=int, default=10)
    parser.add_argument("--process", type=int, default=8)
    parser.add_argument("--file", type=str, default="test.txt")
    parser.add_argument("--m2c_method", type=str, default="morgan")
    parser.add_argument("--c2m_method", type=str, default="bm25")
    parser.add_argument("--failure_allowance", type=int, default=10)

    args = parser.parse_args()

    # print args
    print("===== args =====")
    print("data_folder: {}".format(args.data_folder))
    print("tgt_folder: {}".format(args.tgt_folder))
    print("model: {}".format(args.model))
    print("n_shot: {}".format(args.n_shot))
    print("process: {}".format(args.process))
    print("file: {}".format(args.file))
    print("m2c_method: {}".format(args.m2c_method))
    print("c2m_method: {}".format(args.c2m_method))
    print("================")


    # iterate the file folder
    data_folder = args.data_folder
    tgt_folder = args.tgt_folder

    # if tgt_folder does not exist, create one
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    n_shot = args.n_shot
    process = args.process
    m2c_method = args.m2c_method
    c2m_method = args.c2m_method

    # rdkit_molecules = None
    file = args.file

    raw_file = data_folder + file
    with open(raw_file, 'r') as f:
        lines = f.readlines()

    lines = lines[1:]


    example_file = data_folder + "train.txt"
    with open(example_file, 'r') as f:
        temp_lines = f.readlines()

    temp_lines = temp_lines[1:]
    if m2c_method == "morgan":
        rdkit_molecules = []
        for temp_line in temp_lines:
            temp_line = temp_line.strip().strip("\n").strip()
            rdkit_molecules.append(Chem.MolFromSmiles(temp_line.split("\t")[1]))


    def run(n):
        name = mp.current_process().name
        print('Process', name, 'starting')

        tgt_file = tgt_folder + file.split(".")[0] + "_Part{}.txt".format(n)
        full_tgt = tgt_folder + file.split(".")[0] + "_Full_Part{}.txt".format(n)

        exist_caps_items = []
        exist_caps = []
        exist_mols_items = []
        exist_mols = []
        if os.path.exists(tgt_file):
            with open(tgt_file, 'r') as f:
                tgt_lines = f.readlines()

            for item in tgt_lines:
                item = item.strip().strip('\n').strip()
                if item.split('\t')[1] != "N/A":
                    exist_caps_items.append(item.split('\t')[0])
                    exist_caps.append(item.split('\t')[1])
                
                if item.split('\t')[2] != "N/A":
                    exist_mols_items.append(item.split('\t')[0])
                    exist_mols.append(item.split('\t')[2])


        # iterate the lines
        all_items = {}

        for line in lines[int(len(lines)*(n-1)/args.process):int(len(lines)*n/args.process)]:

            line = line.strip()
            if line == "":
                continue
            # retrieve the prompts
            ids = line.split("\t")[0]
            molecule = line.split("\t")[1]
            caption = line.split("\t")[2]
            input = {"molecule": molecule, "caption": caption}
            # read the examples
            molecule_rdkits = None if m2c_method != "morgan" else rdkit_molecules

            if ids in exist_caps_items and ids in exist_mols_items:
                all_items[ids] = {'caption': exist_caps[exist_caps_items.index(ids)], 'molecule': exist_mols[exist_mols_items.index(ids)]}
                continue
            elif ids in exist_caps_items:
                all_items[ids] = {'caption': exist_caps[exist_caps_items.index(ids)], 'molecule': "N/A"}
            elif ids in exist_mols_items:
                all_items[ids] = {'caption': "N/A", 'molecule': exist_mols[exist_mols_items.index(ids)]}
            else:
                all_items[ids] = {'caption': "N/A", 'molecule': "N/A"}

            if n_shot != 0:
                # caption-molecule retrieval
                cap_examples, mol_examples = get_examples(example_file, n_shot, input=input, m2c_method=m2c_method, c2m_method=c2m_method, molecule_rdkits=molecule_rdkits)

                # prompt management
                molecule2caption = retrieve_m2c_prompts(mol_examples)
                caption2molecule = retrieve_c2m_prompts(cap_examples)
            else:
                molecule2caption = retrieve_m2c_zero_prompts()
                caption2molecule = retrieve_c2m_zero_prompts()

            if all_items[ids]['caption'] == "N/A":
                # error allowance
                fail_num = 0
                while True:              
                    try:
                        # in-context learning
                        res1 = openai.ChatCompletion.create(
                                    model=args.model,
                                    messages=[
                                            {"role": "system", "content": molecule2caption},
                                            {"role": "user", "content": "Input: " + molecule + "\n"},
                                        ]
                                    )    
                        # format check
                        response1 = res1['choices'][0]['message']['content'].strip('\n')
                        print('Mol2Cap', response1)

                        template = re.compile(r'\{.*\}')
                        response1 = json.loads(template.findall(response1)[0])
                        response1 = response1["caption"]

                        all_items[ids] = {'caption': response1, 'molecule': all_items[ids]['molecule']}

                        fail_num = 0
                        break
                    except Exception as e:              
                        print(e)
                        if n_shot == 0:
                            fail_num += 1
                        else:                        
                            # query too long, n shot degradation -> n-1 shot, do not count as failure
                            if "maximum context length" in str(e):
                                length_ex = [len(mol_example["molecule"]) + len(mol_example["caption"]) for mol_example in mol_examples]
                                temp_n = length_ex.index(max(length_ex))
                                mol_examples.pop(temp_n)
                                molecule2caption = retrieve_m2c_prompts(mol_examples)
                            elif "list index out of range" in str(e):
                                length_ex = [len(mol_example["molecule"]) + len(mol_example["caption"]) for mol_example in mol_examples]
                                temp_n = length_ex.index(max(length_ex))
                                mol_examples.pop(temp_n)
                                molecule2caption = retrieve_m2c_prompts(mol_examples)
                            else:
                                fail_num += 1
                                time.sleep(60)

                        # reach the error allowance skip the example
                        if fail_num > args.failure_allowance:
                            all_items[ids] = {'caption': "N/A", 'molecule': all_items[ids]['molecule']}
                            fail_num = 0
                            break

            if all_items[ids]['molecule'] == "N/A":
                fail_num = 0
                while True:
                    try:    
                        res2 = openai.ChatCompletion.create(
                                    model=args.model,
                                    messages=[
                                            {"role": "system", "content": caption2molecule},
                                            {"role": "user", "content": "Input: " + caption + "\n"},
                                        ]
                                    )
                        response2 = res2['choices'][0]['message']['content'].strip('\n')
                        print('Cap2Mol:', response2)

                        template = re.compile(r'\{.*\}')

                        try:
                            response2 = json.loads(template.findall(response2)[0])
                        except:
                            # format correction
                            response2 = template.findall(response2)[0]
                            response2 = {"molecule": response2.split(":")[1].strip('\n').strip("}").strip().strip("\"")}

                        try:
                            response2 = response2["molecule"]
                        except:
                            # format correction
                            try:
                                response2 = response2["smiles"]
                            except:
                                response2 = response2["predicted_smiles"]

                        all_items[ids] = {'caption': all_items[ids]['caption'], 'molecule': response2}

                        
                        fail_num = 0

                        break
                    except Exception as e:
                        print(e)

                        # query too long
                        if n_shot == 0:
                            fail_num += 1
                        else:
                            if "maximum context length" in str(e):
                                length_ex = [len(cap_example["molecule"]) + len(cap_example["caption"]) for cap_example in cap_examples]
                                temp_n = length_ex.index(max(length_ex))
                                cap_examples.pop(temp_n)
                                caption2molecule = retrieve_c2m_prompts(cap_examples)
                            elif "list index out of range" in str(e):
                                length_ex = [len(cap_example["molecule"]) + len(cap_example["caption"]) for cap_example in cap_examples]
                                temp_n = length_ex.index(max(length_ex))
                                cap_examples.pop(temp_n)
                                caption2molecule = retrieve_c2m_prompts(cap_examples)
                            else:
                                time.sleep(60)
                                fail_num += 1

                        if fail_num > args.failure_allowance:
                            all_items[ids] = {'caption': all_items[ids]['caption'], 'molecule': "N/A"}
                            with open(tgt_file, 'a+') as f:
                                f.write(ids + '\t' + all_items[ids]['caption'] + "\t" + "N/A" + "\n")
                            fail_num = 0
                            break
        
            # write the log results
            with open(tgt_file, 'a+') as f:
                f.write(ids + '\t' + all_items[ids]['caption'] + "\t" + all_items[ids]['molecule'] + "\n")


        with open(full_tgt, 'w+') as f:
            for i in range(len(all_items.keys())):
                ids = list(all_items.keys())[i]
                f.write(ids + '\t' + all_items[ids]['caption'] + '\t' + all_items[ids]['molecule'] + '\n')


    # initial multiprocessing
    pool = mp.Pool(processes=args.process)
    for i in range(1, args.process+1):
        pool.apply_async(run, args=(i, ))
    pool.close()
    pool.join()
    