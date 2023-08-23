import os
from argparse import ArgumentParser
from rdkit import Chem



parser = ArgumentParser()
parser.add_argument('--raw', type=str, default='./cap2mol_trans_raw/test.txt')
parser.add_argument('--file_path', type=str, default='./results/gpt4-0314/')
parser.add_argument('--parts', type=int, default=5)
parser.add_argument('--merge', type=bool, default=False)


args = parser.parse_args()

gound_truth_file = args.raw


#ground truth  
'''
ground_truth[0]  CID
ground_truth[1]  SMILES
ground_truth[2]  description
'''
with open(gound_truth_file, 'r+',encoding='utf-8') as f:
    lines = f.readlines()
    ground_truth = []
    for line in lines[1:]:
        ground_truth.append(line.strip().split('\t'))


# print(ground_truth[0])


def merge_file(file_path):
    with open(file_path+'test.txt', 'w+',encoding='utf-8') as f:
        f.write('CID'+'\t'+'molecule2caption'+'\t'+'caption2molecule\n')

    for i in range(1, args.parts+1):
        file_name = 'test_Full_Part'+str(i)+'.txt'
        with open(file_path+file_name, 'r+',encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(file_path+'test.txt', 'a+',encoding='utf-8') as f:
            for line in lines[1:]:
                line.strip().strip('\n').strip()
                f.write(line+'\n')



def get_example(file_path, file_name='test.txt'): 
    # mol2caption and caption2mol test data
    with open(file_path+file_name, 'r+',encoding='utf-8') as f:
        lines = f.readlines()
        mol2caption = []
        caption2mol = []
        '''
        line[1]  molecule2caption
        line[2]  caption2molecule
        '''
        for line in lines[1:]:
            try:
                mol2caption.append(line.strip().split('\t')[1])
                # caption2mol.append(line.strip().split('\t')[2])
                cap2mol = line.strip().split('\t')[2].replace('".','')
                cap2mol = cap2mol.replace('"','')
                # mol = Chem.MolFromSmiles(cap2mol)
                # if mol is None:
                #     print(line.strip().split('\t')[0])
                #     continue
                caption2mol.append(cap2mol)
            except Exception as e:
                print('------------------------------------')
                print(e)
                print(file_path)
                print(line.strip().split('\t'))
                print('------------------------------------')
                continue


    # build caption2mol example.txt
    with open(file_path +'/caption2smiles_example.txt', 'w+',encoding='utf-8') as f:
        f.write('description'+'\t'+'ground truth'+'\t'+'output\n')
        for i in range(len(caption2mol)):
            f.write(ground_truth[i][2]+'\t'+ground_truth[i][1]+'\t'+caption2mol[i]+'\n')
    '''
    ground_truth[0]  CID
    ground_truth[1]  SMILES
    ground_truth[2]  description
    '''
    # build mol2caption example.txt
    with open(file_path +'/smiles2caption_example.txt', 'w+',encoding='utf-8') as f:
        f.write('SMILES'+'\t'+'ground truth'+'\t'+'output\n')
        for i in range(len(mol2caption)):
            f.write(ground_truth[i][1]+'\t'+ground_truth[i][2]+'\t'+mol2caption[i]+'\n')


if args.merge:
    print("********* merge file ********")
    merge_file(args.file_path)

print("******** convert file *******")
try:
    get_example(args.file_path, 'test.txt')
except Exception as e:
    print(e)

print("********* finish ************")