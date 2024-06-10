# %%
'''
Input in version 2 is some random RNA seg (~300bp, +/- based on hhblits performance) and all its hhblits seqs

Data Fetch methods:
Fasta obj in n4j -> 
'''

# %%
import concurrent.futures
import numpy as np
import pandas as pd
import random 
import torch
from tempfile import TemporaryDirectory
from neomodel.sync_.core import Database
from neomodel.integration.pandas import to_dataframe
from misc.hhblits_annotation import hhblits_annotation,group_hit
from typing import Optional,List
# from tempfile import TemporaryDirectory
from subprocess import run
from Bio import SeqIO
from Bio.SeqRecord import Seq,SeqRecord
from pathlib import Path
import shutil,os
from io import BytesIO
from torch.utils.data import IterableDataset,get_worker_info
from dataclasses import dataclass,field
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
# from taxo_train_1 import OrderManager
import logging
logger=logging.getLogger()
from hierataxo import LocalRandomGenerator
# %%

# %%
# def annot_seg(seg:str):
#     pass


@dataclass
class HyperParamManager:
    log_seglen_loc:float=2.5
    log_seglen_scal:float=0.3
    seglen_cap:float=1200
    max_diamond_hit:int=5
    diamond_thread:int=8


def diamond_search(seg:str,
                   diamond='/home/rnalab/zfdeng/pgg/taxo_sandbox/diamond',
                   db='/home/rnalab/zfdeng/pgg/taxo_sandbox/all.dmnd',
                   thread:int=4):
    with TemporaryDirectory() as tdir:
        SeqIO.write(SeqRecord(Seq(seg),id='idholder',
                name='nameholder',description='desholder'),
            f'{tdir}/in.fa','fasta')
        o=run([Path(diamond).absolute(),'blastx',
             '-q','in.fa',
             '-d',Path(db).absolute(),
             '--masking','0',
             '-p',str(thread),
             '-k1',
             '--ultra-sensitive',
             '-f','6',
             'qseqid', 'qstart', 'qend', 'qlen','qstrand',
             'sseqid', 'sstart', 'send', 'slen',
             'pident','evalue','cigar',
             'qseq_translated','full_qseq','full_qseq_mate'
             ],cwd=tdir,capture_output=True) #stdout=open('o.pro','w'),
        # shutil.copytree(src=tdir,dst='./tmp')
        # return o
        names=('qseqid qstart qend qlen qstrand sseqid sstart send '
                    'slen pident evalue cigar qseq_translated full_qseq full_qseq_mate').split()
        if len(o.stdout.strip())>0:
            return pd.read_csv(BytesIO(o.stdout),sep=r'\s+',header=None,
                names=names)
        else:
            return pd.DataFrame([],columns=names)
        
# def hhblits_search(seg):
#      with TemporaryDirectory() as tdir:
#         SeqIO.write(SeqRecord(Seq(seg),id='idholder',
#                 name='nameholder',description='desholder'),
#             f'{tdir}/in.fa','fasta')
#         annotations=hhblits_annotation(f'{tdir}/in.fa',4)

class SegIterDiamond(IterableDataset):
    def __init__(self,seed:int=42,
                 db_pkl:Optional[str]=None,
                 db_url:Optional[str]=None,
                 target_csv:Optional[str]=None,
                 target_dmnd:Optional[str]=None,
                 **kargs
                 ) -> None:
        super().__init__()
        # self.rand=LocalRandomGenerator(seed)
        self.rootseed=seed
        #
        self.db=Database()
        if db_url is not None:
            self.db.set_connection(db_url)    
        if db_pkl is None:
            self._fetch_fastas()
        else:
            self.data:pd.DataFrame=pd.read_pickle(db_pkl)    
        self.data=self.data[~self.data['taxonomy'].isna()]
        #
        if target_csv is None:
            self._fetch_hits()
        else:
            self.target=pd.read_csv(target_csv)
            
        self.domain_list:List[str]=self.target['family_name'].unique().tolist()
        if target_dmnd is not None and os.path.isfile(target_dmnd):
            self.target_dmnd=Path(target_dmnd).absolute()
        else:
            o='target'
            self._gen_dmnd(o)
            self.target_dmnd=Path(o+'.dmnd').absolute()
        
        #
        self.hypers=HyperParamManager(**kargs)
        
    def _fetch_fastas(self):
        q='''
        MATCH (fasta:Fasta)--(r:HitRegion)
        RETURN fasta.name as name, fasta.taxonomy as taxonomy, fasta.seq as sequence,
            collect(r.begin) as begins, collect(r.end) as ends
        '''
        self.data=to_dataframe(self.db.cypher_query(q,resolve_objects=True))

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            rand=LocalRandomGenerator(self.rootseed)
        else:
            rand=LocalRandomGenerator(self.rootseed+worker_info.id)
        def check_sseqid(s:str):
            if len(s.split(':'))!=5:
                return False
            else:
                return True
        
        def sample_seg_begin(full_len:int,seg_len:int,
                begins:List[int],ends:List[int])->int:
            full=np.zeros(full_len)
            for b,e in zip(begins,ends):
                full[b:e]=1
            valid = np.convolve(full, np.ones(seg_len, dtype=bool), mode='valid') > 0
            return rand.random_generator.sample(np.where(valid)[0].tolist(),1)[0]
        
        def _generator():
            logger.info(f'seed: {rand.seed}')
            
            while True:
                r=self.data.iloc[rand.random_generator.sample(
                    range(len(self.data)),1)[0]]
                name:str=r['name']
                taxos:List[str]=r['taxonomy'].split(' ;')
                full_nt:str=r['sequence']
                begins:List[int]=r['begins'][0]
                ends:List[int]=r['ends'][0]
                full_len=len(full_nt)
                seg_len=int(10**rand.numpy_generator.normal(
                    loc=self.hypers.log_seglen_loc,scale=self.hypers.log_seglen_scal,size=1)[0])
                seg_len=min([full_len,self.hypers.seglen_cap,seg_len])
                # seg_begin=rand.random_generator.sample(range(full_len-seg_len+1),1)[0]
                seg_begin=sample_seg_begin(full_len,seg_len,begins,ends)
                seg=full_nt[seg_begin:seg_begin+seg_len]
                search_result=diamond_search(seg,db=self.target_dmnd,thread=self.hypers.diamond_thread)
                # yield (seg,search_result)
                # import pdb;pdb.set_trace()
                
                
                # print(len(search_result))
                max_hit=self.hypers.max_diamond_hit
                if len(search_result)==0:
                    o=dict(
                        name=name,
                        b_e=(seg_begin,seg_begin+seg_len),
                        taxo=taxos[2:9:2],
                        seg=seg,
                        hit=['']*max_hit,
                        hit_family=[-1]*max_hit,
                        hit_family_name=['']*max_hit,
                        #mask definition: 1 = padding to be masked
                        hit_mask=[1]*max_hit,
                    )
                else:
                    search_result['family']=search_result['sseqid'].apply(lambda x:x.split(':')[1])
                    search_result=search_result.loc[search_result.groupby('family')['evalue'].idxmin()]
                    if len(search_result)>max_hit:
                        sel=rand.random_generator.sample(
                            range(len(search_result)),max_hit)
                        o=dict(
                            name=name,
                            b_e=(seg_begin,seg_begin+seg_len),
                            taxo=taxos[2:9:2],
                            seg=seg,
                            hit=search_result.iloc[sel]['full_qseq'].to_list(),
                            hit_family=search_result.iloc[sel]['family'].apply(
                                lambda x:self.domain_list.index(x)).to_list(),
                            hit_family_name=search_result.iloc[sel]['family'].to_list(),
                            hit_mask=[0]*max_hit,
                        )
                    else:
                        len_pad=max_hit-len(search_result)
                        o=dict(
                            name=name,
                            b_e=(seg_begin,seg_begin+seg_len),
                            taxo=taxos[2:9:2],
                            seg=seg,
                            hit=search_result['full_qseq'].to_list()+['']*len_pad,
                            hit_family=search_result['family'].apply(
                                lambda x:self.domain_list.index(x)).to_list()+[-1]*len_pad,
                            hit_family_name=search_result['family'].to_list()+['']*len_pad,
                            hit_mask=[0]*len(search_result)+[1]*len_pad,
                        )
                        
                yield o
            
        return iter(_generator())

    def _fetch_hits(self,cover_threshold=0.8):
        target_hits=to_dataframe(self.db.cypher_query(
        '''
        MATCH (hf:HitFamily)-[r:hasMember]->(hit:Hit)
        WHERE ((hit.hit_end-hit.hit_begin+1)/toFloat(hf.std_length))>$cover_threshold
        RETURN hf.name as family_name, hf.accession as family_accession,
            hf.std_length as family_length, hit.hit_end-hit.hit_begin+1 as hit_length,
            r.probab as probab,((hit.hit_end-hit.hit_begin+1)/toFloat(hf.std_length)) as coverage,
            hit.name as hit_name, hit.aligned_seq as hit_seq
        ''',params={"cover_threshold":cover_threshold},resolve_objects=True))
        self.target=target_hits
        # return all_seq
        
    def _gen_dmnd(self,o:str,
                  diamond='/home/rnalab/zfdeng/pgg/taxo_sandbox/diamond',
                  thread:int=4):
        
        def s_to_record(s:pd.Series):
            def s_to_id(s:pd.Series):
                return f"{s['hit_name'].replace(' ','_')}:p{int(s['probab']*100)}:c{int(s['coverage']*100)}"

            def s_to_seq(s:pd.Series):
                return s['hit_seq'].replace('-','').upper()

            return SeqRecord(seq=Seq(s_to_seq(s)),
                            id=s_to_id(s),description='')
        
        if not hasattr(self,'target'):
            self._fetch_hits()
        target_hits=self.target
        records:pd.Series=target_hits.apply(s_to_record,axis=1)
        with TemporaryDirectory() as tdir:
            SeqIO.write(records.to_list(),f'{tdir}/all_seq.fa','fasta')
            o=run([Path(diamond).absolute(),'makedb','--in','all_seq.fa',
           '-d',Path(o).absolute(),'-p',str(thread)],capture_output=True,cwd=tdir)
            return o

        
# %%
if __name__=='__main__':
    logging.basicConfig(filename='data/hhblits_annotation.log', 
            filemode='a',           
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(module)s - %(filename)s - %(message)s')

    segiter=SegIterDiamond(db_url='bolt://neo4j:WBrtpKCUW28e@52.4.3.233:7687',
            db_pkl='taxo_data/ori_fasta.pkl',
            target_csv='taxo_data/ori_hit.csv',
            target_dmnd='taxo_data/target_hit.dmnd',
            log_seglen_loc=2.5,
            log_seglen_scal=0.2,
            seglen_cap=2000,
            max_diamond_hit=3,
            seed=43
            )
# _=iter(segiter)
# # %%
# next(_)
# # %%
# from misc.hhblits_annotation import hhblits_annotation
# o=hhblits_annotation('tggaaccaccccggaacccccccgttcagccggtccagagggtgttgttggagctgcttccggacggggcagctgtgtatgagggggatctcttcagctcagattgtttctggcttgttaatgcggccaatgctcagcacagccctggggggggcgtgtgcggtgctttttatggacgtttcccggaggctttcgaccgtacccagtttgtccatccaaatgggtcggtggctgcctatacagtgacacccaggcccataatacatgctgtggctcctgattataggcggcggagggaccctgccgccctgcaagtagcctaccaagagtgcttgtatagacaggagacggccgcctattgtttgcttggtagtggaatatatcagatccccccggacgagtccatgcaagcttggctagataaccaccttccaggtgatgagatgtatttgttgccggctatgtcatcttggtatcgagcctggcgagcccaggccggcgggacggggcgaggcggatcctcgggcgctaccccgccggctgccccggcgccatcacctccagggccccctccacccggccccgcaggggttggtggtgagggatcagcgtcgagaccttcgtcccatatattggtggtcaccccgggcctggctaataccgccaacctagccctccagcaggagtcagaaggcccgtttgggaagtttgtgggcaacgcccatgtgcccccagggccggtgcattaccggttcgttgcgggggttccagggtcgggaaagtccgtcggtgtccggcgcgaagattgtgatctggtgatcgtcccgaccaaccagctgaaagcacagtggagggctcggggtttccctgtcatgaccccgcatgtcgggctacaacattcccagggtaagaggctggttatcgacgaggcccccaccatcgcgccgcacctcctgctctgttatatgaccgccgctgccgaagtggtcttactcggggacccaagacagatccctgccatagattttgagagcaagggccttatcccagctatgcagctcaatttagaaccgacggaatggcggctccagtcacaccggtgccccagagatgtgtgttacc',
#                    minlen=50)

# %%
