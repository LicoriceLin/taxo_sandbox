from Bio import SeqIO
from pathlib import Path
from Bio import Entrez
from warnings import warn

import pickle as pkl
import tqdm
from pathlib import Path
import pandas as pd
from typing import List,Dict
import re
import xmltodict
Entrez.email='zfevan.deng@mail.utoronto.ca'

def fetch_parse(i:str):
    '''
    i: genbank accession
    '''
    xml_file=Entrez.efetch(db='nuccore',id=i,rettype="gb", retmode="xml")
    data_dict = xmltodict.parse(xml_file.read())
    return data_dict

def robust_split(s:str,split:str)->List[str]:
    '''
    split and remove blankspace
    '''
    return [i.strip() for i in s.split(split)]

def get_genbank_id(accession:str)->Dict[str,str]:
    '''
    accession: entry in VMR
    '''
    assert isinstance(accession,str),f'accession: {accession} is not a str!'
    o={}
    holder_token=0
    bracket_pattern = r'\s*\(.*?\)\s*'
    accession = re.sub(bracket_pattern, '', accession).strip()
    if ';' in accession:
        for sub_a in robust_split(accession,';'):
            if ':' in sub_a:
                subk,subv=robust_split(sub_a,':')
                o[subk.replace(' ','')]=subv
            else:
                o[f'{holder_token}']=sub_a
                holder_token+=1
    else:
        if ":" in accession:
            subk,subv=robust_split(accession,':')
            o[subk.replace(' ','')]=subv
        else:
            o['_']=accession
    return o

def get_correct_name(name:str)->str:
    """
    some names block contains multiple name sep by ';'
    only use the first one
    warning: this name is not unique!
    """
    assert isinstance(name,str),f'valid input: {name}'
    if ';' in name:
        o=robust_split(name,';')[0]
    else:
        o=name
        # warn(f'multiple names: {name},use the first one {o}')
    if '=' in o:
        o=robust_split(o,'=')[0]
    if ',' in o:
        o=robust_split(o,',')[0]
    o=o.replace(' ','_').replace(':','#').replace(r'/','!')
    return o
    
def is_multiple_access(access:Dict[str,str])->bool:
    ''' 
    helper for `get_genbank_id`'s output 
    `False`: one seg; `True`:c segs 
    '''
    if '_' in access:
        return False
    else:
        return True
    
def get_file_stem(virus_series:pd.Series,
    access_col='true_access',name_col='true_name')->List[str]:
    '''
    vmr['true_access']=vmr['Virus GENBANK accession'].apply(get_genbank_id)
    vmr['true_name']=vmr['Virus name abbreviation(s)'].apply(get_correct_name)
    '''
    _=virus_series
    access:Dict[str,str]= _[access_col]
    if not is_multiple_access(_[access_col]):
        return [f"{_[name_col]}||{access['_']}"]
    else:
        return [f"{_[name_col]}|{k}|{v}" for k,v in access.items()]

def group_genbankid(access_series:pd.Series):
    '''
    access_series: vmr['true_access']
    
    pkl.dump(group_genbankid(vmr['true_access']),open('data/access_group.pkl','wb'))
    ''' 
    id_groups={}
    for i in access_series:
        if len(i)==0:
            genbank_id=list(i.values())[0]
            id_groups[genbank_id]=genbank_id
        else:
            genbank_id=list(i.values())[0]
            for v in i.values():
                id_groups[v]=genbank_id
    return id_groups