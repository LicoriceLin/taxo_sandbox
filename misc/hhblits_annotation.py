'''
main pipeline for:
1) run hhblits annotation
2) group hits by their overlaps
3) fix false double hit by check their theoretical span(span of whole hhmodel)'s interval ratio
4) generate neomodel instance (nodes/relationships) and commit them to neo4j instance.

Note: 
1) ~ 3) are robust for parallelling, 
while 4) is prone to internet instability and deadlock if executed parallelly. 
That's why we have the tmp scripts to run them as separated stages
'''
# %%
# %load_ext autoreload
# %autoreload 2
from typing import Dict,Tuple,List,Union,Optional
import pandas as pd
import numpy as np
from subprocess import run
from tempfile import TemporaryDirectory
from pathlib import Path
from Bio.SeqIO import parse,write,read
from Bio.SeqRecord import SeqRecord
import re
import logging
import pickle as pkl
from neomodel import config, db,clear_neo4j_database
from neomodel.integration.pandas import to_dataframe
import networkx as nx
from itertools import combinations,tee
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
# %%
from .hhblits_models import Fasta,Hit,HitFamily,HitRegion
from .hhblits_models import (hasHit,hasRegion,
    hasDownstream,hasAnalog,hasMember,hasAffiliate)
from functools import partial
from neomodel.sync_.relationship_manager import RelationshipManager,_rel_helper
from neomodel import StructuredNode

# %% Constants
Path(__file__).parent
GENID_TAXONOMY_DICT:Dict[str,str]=pkl.load(open(Path(__file__).parent/'genid_taxonomy_dict.pkl','rb'))
PFAM_TYPE_DICT:Dict[str,str]=pkl.load(open(Path(__file__).parent/'pfam_type_dict.pkl','rb'))
logger=logging.getLogger()

# %% run orfipy
def run_orfipy(infile:str,minlen:int=900,
           cpu:int=1,table:int=9,
           conda_env:str='/home/rnalab/zfdeng/miniconda3/envs/n4jn',
           wdir:Optional[str]=None
           )->List[SeqRecord]:
    '''
    orfipy: ORF finder
    
    table:
        check: orfipy.translation_tables.translation_tables_dict
        ref: https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?chapter=cgencodes
    
    '''
    if wdir is None:
        t=TemporaryDirectory()
        wdir=t.name
    else:
        t=None
        
    o_=run([
        'conda','run',
        '-p',conda_env,
        'orfipy',Path(infile).absolute(),
        '--pep','pep.fa',
        '--min', f'{minlen}',
        '--max', '1000000',
        '--procs', f'{cpu}','--table',
        f'{table}','--outdir','orfs_out',
        '--ignore-case','--between-stops'
    ],capture_output=True,cwd=wdir)
    o=list(parse(f'{wdir}/orfs_out/pep.fa','fasta'))
    if t is not None:
        t.cleanup()
    return o


# TODO misc
split_hits_line=lambda line:[i.strip() for i in [line[:4],line[4:34],line[34:40],
    line[40:48],line[48:56],line[56:63],line[63:69],line[69:74],line[74:85],line[85:94],line[94:]]]


def parse_orfipy_description(description:str):
    '''
    turn original orfipy annotation str into formatted dict
    '''
    def match_bec(bec:str):
        '''
        accessory for `parse_orfipy_description` to process original orfipy annotation str.
        bes: begin/end/strand in the given string
        in the shape of [b-e](s)
        '''
        pattern = re.compile(r'\[(\d+)-(\d+)\]\((.+)\)')
        match = pattern.match(bec)
        if match:
            b,e,s = match.groups()
            s='s' if s=='+' else 'a'
            return {'begin':int(b),'end':int(e),'strand':s}
        else:
            logger.error(f'invalid bec string: {bec}')
            return {'begin':-1,'end':-1,'strand':'o'} 
    o=dict()
    descriptions=description.split()
    o['orf_id']=descriptions[0].split('.')[1]
    o.update(match_bec(descriptions[1]))
    o.update({i.split(':')[0]:i.split(':')[1] for i in descriptions[2:]})
    # transannot=str(int(o['frame'][-1])-1)+o['strand']
    # o['transannot']=transannot
    # o.pop('frame')
    o['frame']=str(int(o['frame'][-1])-1)
    # o.pop('strand')
    return o


# %% run hhblits 
def run_hhblits(infile:Union[Path,str,SeqRecord],cpu:int=4,
            blitsbin:str='/home/rnalab/zfdeng/pgg/hh-suite/build/src/hhblits',
            blitsdb:str='/home/rnalab/zfdeng/pgg/hh-pfam/pfam',
            wdir:Optional[str]=None
            ):
    '''
    run annotation with hhblits
    return 
    meta: dict, file head of metadata
    hits: pd.DataFrame, hit summary scores
    aligns_list: dict, hit alignments
    '''
    # infile='tmp/AAbV||GBBW01007738#2s#11629#11886.fasta'
    if wdir is None:
        t=TemporaryDirectory()
        wdir=t.name
    else:
        t=None
        
    if isinstance(infile,SeqRecord):
        infile_path=f'{wdir}/infile.fasta'
        write(infile,infile_path,'fasta')
    else:
        infile_path=Path(infile).absolute()
    o_=run([blitsbin,'-i',infile_path,'-d',blitsdb,'-cpu',str(cpu),'-o','out.hhr','-hide_pred','-hide_dssp'],
        capture_output=True,cwd=wdir)
    o=parse_hhr(f'{wdir}/out.hhr')
    if t is not None:
        t.cleanup()
    return o

def run_hhalign(qfile:Union[Path,str,SeqRecord],
            tfile:Union[Path,str,SeqRecord],
            alignbin:str='/home/rnalab/zfdeng/pgg/hh-suite/build/src/hhalign',
            cpu:int=4,
            wdir:Optional[str]=None
            ):
    '''
    run annotation with hhblits
    return 
    meta: dict, file head of metadata
    hits: pd.DataFrame, hit summary scores
    aligns_list: dict, hit alignments
    '''
    # infile='tmp/AAbV||GBBW01007738#2s#11629#11886.fasta'
    if wdir is None:
        t=TemporaryDirectory()
        wdir=t.name
    else:
        t=None
        
    if isinstance(qfile,SeqRecord):
        qfile_path=f'{wdir}/qfile.fasta'
        write(qfile,qfile_path,'fasta')
    else:
        qfile_path=Path(qfile).absolute()
        
    if isinstance(tfile,SeqRecord):
        tfile_path=f'{wdir}/tfile.fasta'
        write(tfile,tfile_path,'fasta')
    else:
        tfile_path=Path(tfile).absolute()
        
    o_=run([alignbin,'-i',qfile_path,'-t',tfile_path,'-cpu',str(cpu),'-o','out.hhr','-hide_pred','-hide_dssp'],
        capture_output=True,cwd=wdir)
    o=parse_hhr(f'{wdir}/out.hhr')
    if t is not None:
        t.cleanup()
    return o

def orfindice_to_fastaindice(
    trans_begin:int,trans_end:int,strand:str,orf_begin:int,orf_end:int):
    '''
    turn indice on ORF into indice on genome
    #TODO: useful miscs, should then be moved to a standalone script
    '''
    if strand=='s':
        return trans_begin+orf_begin*3-3,trans_begin+orf_end*3
    elif strand=='a':
        return trans_end-orf_end*3,trans_end-orf_begin*3+3


def parse_hhr(hhr:str)->Tuple[Dict[str,str],pd.DataFrame,List[dict]]:
    '''
    hhr: output for hhblits
    
    for the hits_list:
    deletion: denoted with '-';
    insertion: denoted with lower case;
    consensus: =:<-1.5 ; -:-1.5~-0.5; .:-0.5~0.5; 
               +:0.5~1.5; |:>1.5; ' ':mismatch
    '''
    def robust_full(length:int,char:str):
        '''
        accessory for `parse_hhr`
        '''
        if length>0:
            return np.full(length,char)
        elif length==0:
            return np.array([],dtype=type(char))
        else:
            logger.error(f'negative length: {length} {char}')
            # return np.array([],dtype=type(char))
            raise ValueError
    with open(hhr,'r') as f:
        line='holder'
        current_section='meta'
        meta:Dict[str,str]={}
        # parsed_hhr={'head':{},'summary':[],'alignment':[]}
        while line != '':
            line=f.readline()
            if not line.strip():
                continue
            
            if current_section=='meta' and line.startswith(' No Hit'):
                current_section='hits'
                hits_cols=split_hits_line(line)
                hits_list=[]
                continue
            elif current_section=='hits' and line.startswith('No 1'):
                current_section='aligns'
                hits=pd.DataFrame(hits_list,columns=hits_cols)
                # hits.set_index('No',inplace=True)
                aligns_list=[]
                current_hit=0
            
            if current_section=='meta':
                meta[line[:14].strip()]=line[14:].strip()
            elif current_section=='hits':
                hits_list.append(split_hits_line(line))
            elif current_section=='aligns':
                def add_align():
                    left,right=tb-1,model_length-te
                    q_align_array=np.array(list(q_align),dtype=str)
                    t_align_array=np.array(list(t_align),dtype=str)
                    ins=np.where(t_align_array=='-')[0]
                    ins_q=np.where(q_align_array=='-')[0]
                    if len(ins)>0:
                        q_align_array[ins]=np.vectorize(lambda x:x.lower())(q_align_array[ins])
                    q_align_array=''.join(np.hstack([robust_full(left,'-'),q_align_array,robust_full(right,'-')]))
                    
                    if len(ins_q)>0:
                        t_align_array[ins_q]=np.vectorize(lambda x:x.lower())(t_align_array[ins_q])
                    t_align_array=''.join(np.hstack([robust_full(left,'-'),t_align_array,robust_full(right,'-')]))
                    
                    consensus_array=''.join(np.hstack([robust_full(left,' '),
                        np.array(list(consensus),dtype=str),robust_full(right,' ')]))
                    #
                    align_meta['P-value']=hits['P-value'][current_hit-1]
                    # DON'T Delete it. It's the original parse result
                    # aligns_list.append(dict(
                    #     align_meta=align_meta,
                    #     q_align=q_align,
                    #     t_align=t_align,
                    #     consens=consensus,
                    #     qb=qb,qe=qe,tb=tb,te=te,
                    #     model_length=model_length
                    # ))
                    aligns_list.append(dict(
                        align_meta=align_meta,
                        q_align=q_align_array,
                        t_align=t_align_array,
                        consens=consensus_array,
                        query_begin=qb,query_end=qe,
                        target_begin=tb,target_end=te,
                        model_length=model_length
                    ))
                if line.startswith('No '):
                    #wrap last hit
                    if current_hit>0:
                        #
                        add_align()
                        
                    #init this hit
                    current_hit=int(line.strip().replace('No ',''))
                    align_meta={}
                    q_align=''
                    t_align=''
                    consensus=''
                    consensus_flag=0
                    qb=qe=tb=te=model_length=0
                    continue
                elif line.startswith('>'):
                    align_meta['annotation']=line.replace('>','').strip()
                
                elif line.startswith('Probab'):
                    align_meta.update({i.split('=')[0]:i.split('=')[1] 
                            for i in line.strip().split()})
                elif line.startswith('Q ') and not line.startswith('Q Consensus'):
                    s0,s1,s2,s3,s4,s5=line.split()
                    if qb==0:
                        qb=int(s2)
                    qe=int(s4)
                    q_align+=s3
                elif line.startswith('Q Consensus'):
                    seq_len=len(line.split()[3])
                    consensus_flag=1
                elif consensus_flag==1:
                    consensus+=line[22:22+seq_len]
                    consensus_flag=0
                elif line.startswith('T ') and not line.startswith('T Consensus'):
                    s0,s1,s2,s3,s4,s5=line.split()
                    if tb==0:
                        tb=int(s2)
                    te=int(s4)
                    t_align+=s3
                    if model_length==0:
                        model_length=int(s5[1:-1])
        add_align()
    return meta,hits,aligns_list


def hhblits_annotation(seg:str,cpu:int=4,stem:str='holder',minlen:int=900):
    '''
    integrated pipeline to: 
        1) identify ORF; 
        2) run hhblits and parse the output;
        3) merge results of hhblits, especially, convert orfindice to fastaindice
    this function is safe to run parallelly: safe thread & safe file-system 
    '''
    def merge_align_info(align:dict,description_dict:dict):
        '''
        accessory
        merge orf info and align info
        '''
        merged_dict={}
        merged_dict['hitannot']=align['align_meta'].pop('annotation')

        #query
        # merged_dict['transannot']=description_dict['transannot']
        merged_dict['frame'],merged_dict['strand']=description_dict['frame'],description_dict['strand']
        merged_dict['begin'],merged_dict['end']=orfindice_to_fastaindice(
            int(description_dict['begin']),int(description_dict['end']),
            description_dict['strand'],
            align['query_begin'],align['query_end']
        )
        merged_dict['align_seq']=align['q_align']
        merged_dict['align_consensus']=align['consens']

        #target info
        merged_dict['hit_begin']=align['target_begin']
        merged_dict['hit_end']=align['target_end']
        merged_dict['model_length']=align['model_length']
        merged_dict['template_neff']=float(align['align_meta']['Template_Neff'])
        #hitscores
        for k in ['Probab','Identities','Similarity', 
                'E-value','P-value',
                'Aligned_cols','Score','Sum_probs']:
            v=align['align_meta'][k]
            v=v.replace('%','') if k == 'Identities' else v
            v=int(v) if k=='Aligned_cols' else float(v)
            v=v/100 if k in ['Probab','Identities','Sum_probs'] else v
            merged_dict[k.lower()]=v
        return merged_dict
    # stem=Path(infasta).stem.replace(':genome','')
    with TemporaryDirectory() as tdir:
        SeqIO.write(SeqRecord(Seq(seg),id='idholder',
                name='nameholder',description='desholder'),
            f'{tdir}/in.fa','fasta')
        
        orfs=run_orfipy('in.fa',minlen=minlen,cpu=cpu,wdir=tdir)
        if len(orfs)<1:
            logger.error(f'no valid segments: {stem}')
            return pd.DataFrame()
        o=[]
        for orf in orfs:
            orf.id=orf.name=stem+orf.id
            description_dict=parse_orfipy_description(orf.description)
            orf.description=';'.join([f'{k}:{v}' for k,v in description_dict.items()])
            aligns=run_hhblits(orf,cpu=cpu,wdir=tdir)[2]
            # return description_dict,aligns
            for align in aligns:
                o.append(merge_align_info(align,description_dict))
        return pd.DataFrame(o).sort_values(by='begin').reset_index(drop=True)


# %%
def initial_group_hit(annotations:pd.DataFrame,threshold:float=0.8,stem:str=''):
    ''''
    first discard hit whose `probab` is lower than `threshold`
    then group them by their overlaps
    '''
    o=[]
    top_hits=annotations[annotations['probab']>threshold].copy()
    # top_hits['strand']=top_hits['transannot'].apply(lambda x:x[-1])
    for strand,subg in top_hits.groupby('strand'):
        # subg=hits_df[hits_df['transannot']=='1a'].copy(deep=True)
        # subg.sort_values(by='gen_b',inplace=True)
        break_point=np.append(np.where(subg.iloc[1:]['begin'].to_numpy()
                    -np.maximum.accumulate(subg.iloc[:-1]['end'].to_numpy()) > 0),len(subg)-1)
        differences = np.diff(break_point, prepend=-1).reshape(-1)
        subg['group_id'] = np.repeat(np.arange(differences.shape[-1]), differences)
        # subg['group_id'] = subg['transannot']+'-'+np.repeat(np.arange(differences.shape[-1]), differences).astype(str)
        subg['group_begin'] = subg.groupby('group_id')['begin'].transform('min')
        subg['group_end'] = subg.groupby('group_id')['end'].transform('max')
        if not ((subg.groupby('group_id')['group_end'].first().to_numpy()[1:
            ]-subg.groupby('group_id')['group_begin'].first().to_numpy()[:-1])>0).all():
            logger.error(f'{stem}:abnormal grouping: \n{subg[['begin','end','group_begin','group_end']]}')
        o.append(subg)
    # print(o)
    # import pdb; pdb.set_trace()
    if len(o)>1:
        grouped_hits_df=pd.concat(o,axis=0,ignore_index=True)
    else:
        grouped_hits_df=o[0]
    return grouped_hits_df


def interval_ratio(a:pd.Series,b:pd.Series,prefix:str='theory_')->float:
    '''
    calculate the ratio of Intersect/Union
    '''
    begin,end=f'{prefix}begin',f'{prefix}end'
    intersect_min = max(a[begin], b[begin])
    intersect_max = min(a[end], b[end])
    if intersect_max<=intersect_min:
        return 0.
    union_min=min(a[begin], b[begin])
    union_max=max(a[end], b[end])
    if union_max<=union_min:
        return 0.
    else:
        return (intersect_max-intersect_min)/(union_max-union_min)


def merge_pair(pairs:List[tuple])->List[set]:
    g= nx.Graph()
    for pair in pairs:
        g.add_edges_from(combinations(pair,2))
    return list(nx.connected_components(g))


def mend_pesudo_double_hit(annotations:pd.DataFrame,stem:str='',
        overlap_threshold:float=0.8,group_id:str='group_id'):
    '''
    temperary
    '''
    mend_pairs=[]
    for hitannot,subg in annotations.groupby('hitannot'):
        if len(subg)>1 and len(subg[group_id].unique())>1:
            logger.warning(f'{stem}:`{hitannot}` has same hit across groups: {subg[group_id].unique()}')
            subg['theory_begin']=subg['begin']-(subg['hit_begin']-1)*3
            subg['theory_end']=subg['end']+(subg['model_length']-subg['hit_end'])*3
            for i in range(len(subg)):
                for j in range(1,len(subg)-i):
                    inter_rat=interval_ratio(subg.iloc[i],subg.iloc[j])
                    if (subg.iloc[i][group_id]!=subg.iloc[j][group_id]
                        ):
                        group_to_merge=(subg.iloc[i][group_id],subg.iloc[j][group_id])
                        if inter_rat>overlap_threshold:
                            logger.warning((f'{stem}:`{hitannot}` indicates group {group_to_merge} should be merged '
                                            f'with interval_ratio of {inter_rat:.2f}'))
                            mend_pairs.append(group_to_merge)
                        else:
                            logger.warning((f"{stem}:`{hitannot}` doesn't indicate group {group_to_merge} should be merged though double-hit,"
                                            f'with interval_ratio of {inter_rat:.2f}'))
                        
    return mend_pairs


def merge_group(annotations:pd.DataFrame,pairs:List[set],stem:str=''):
    if len(pairs)==0:
        return annotations
    else:
        group_b=annotations.groupby('group_id')['group_begin'].first()
        group_e=annotations.groupby('group_id')['group_end'].first()
        
    if len(pairs)>1:
        pairs=merge_pair(pairs)
        pairs=[set(range(min(i),max(i)+1)) for i in pairs]
        pairs=merge_pair(pairs)
        pairs.sort(key=lambda x:min(x))
    
    new_group_map={}
    new_group_be={}
    last_merge_end=-1
    last_group_id=0
    for pair in pairs:
        merge_begin,merge_end=min(pair),max(pair)
        logger.warning(f'{stem}:group {merge_begin}-{merge_end} would be merged.')
        new_group_map.update({i:e+last_group_id for e,i in enumerate(range(last_merge_end+1,merge_begin))})
        new_group_be.update({e+last_group_id:(group_b[i],group_e[i]) for e,i in enumerate(range(last_merge_end+1,merge_begin))})
        new_group_map.update({i:last_group_id+merge_begin-last_merge_end-1 for i in pair})
        new_group_be[last_group_id+merge_begin-last_merge_end-1]=(group_b[merge_begin],group_e[merge_end])
        last_group_id=last_group_id+merge_begin-last_merge_end
        last_merge_end=merge_end
    merge_begin=annotations['group_id'].max()+1
    new_group_map.update({i:e+last_group_id for e,i in enumerate(range(last_merge_end+1,merge_begin))})
    new_group_be.update({e+last_group_id:(group_b[i],group_e[i]) for e,i in enumerate(range(last_merge_end+1,merge_begin))})
    
    annotations=annotations.copy()
    annotations['group_id']=annotations['group_id'].map(new_group_map)
    annotations['group_begin']=annotations['group_id'].map(lambda x: new_group_be[x][0])
    annotations['group_end']=annotations['group_id'].map(lambda x: new_group_be[x][1])
    
    return annotations
    # new_group_map
        
        
    # return mend_pair


def group_hit(annotations:pd.DataFrame,threshold:float=0.8,stem:str=''):
    '''
    annotations: from `hhblits_annotation`
    threshold: prob filter to keep only high-confident hits
    stem: optional, fasta's name for logging 
    '''
    grouped_annotations=initial_group_hit(annotations,threshold,stem)
    pairs=mend_pesudo_double_hit(grouped_annotations,stem)
    grouped_annotations=merge_group(grouped_annotations,pairs,stem)
    return grouped_annotations.reset_index(drop=True)


#%% commit to neo4j
def generate_neomodels(infasta:str,annotations:pd.DataFrame):
    def to_hitfamily_dict(hitrow:pd.Series):
        hitannot=hitrow['hitannot']
        model_length=hitrow['model_length']
        _=[i.strip() for i in hitannot.split(';')]
        pfam_accession,pfam_name,pfam_annotation=_[0],_[1],_[2]
        pfam_accession=pfam_accession.split('.')[0]
        pfam_hit_family_dict=dict(
        name=pfam_name,
        source='Pfam-HHSuites',
        accession=pfam_accession,
        subtype=PFAM_TYPE_DICT.get(pfam_accession),
        annotation=pfam_annotation,
        std_length=model_length
        )
        return pfam_hit_family_dict

    def to_hit_dict(hitrow:pd.Series,fasta_name:str):
        hit_dict=hitrow[['frame', 'strand', 'begin', 'end', 'align_seq',
        'align_consensus', 'hit_begin', 'hit_end']].to_dict()
        hit_dict['aligned_seq']=hit_dict.pop('align_seq')
        hit_dict['aligned_consensus']=hit_dict.pop('align_consensus')
        
        pfam_name=hitrow['hitannot'].split(';')[1].strip()
        name=f'{fasta_name}:{pfam_name}:{hit_dict['begin']}~{hit_dict['end']}'
        hit_dict['name']=name
        return hit_dict

    def to_hitregion_dict(hitrow:pd.Series,fasta_name:str):
        # hitregion_dict=hitrow[['group_id','group_begin', 'group_end']]
        hitregion_dict={}
        hitregion_dict['begin']=hitrow['group_begin']
        hitregion_dict['end']=hitrow['group_end']
        hitregion_dict['name']=f'{fasta_name}:{hitrow['group_id']}:{hitrow['group_begin']}'
        return hitregion_dict

    name=Path(infasta).stem.split(':')[0]
    accession=name.split('|')[-1]
    taxonomy=GENID_TAXONOMY_DICT.get(accession,None)
    if taxonomy is None:
        logger.error(f'{name} has no taxonomy')
    fasta_dict=dict(
        name=name,
        source='GenBank',
        seq=str(read(infasta,'fasta').seq),
        accession=accession,
        taxonomy=taxonomy
        )
    annotations['fasta']=Fasta.create_or_update(
        *[fasta_dict]*len(annotations))
    hitfamily_dicts=annotations.apply(to_hitfamily_dict,axis=1)
    annotations['hitfamily']=HitFamily.create_or_update(*hitfamily_dicts)
    hit_dicts=annotations.apply(partial(to_hit_dict,fasta_name=name),axis=1)
    annotations['hit']=Hit.create_or_update(*hit_dicts)
    hitregion_dicts=annotations.apply(partial(to_hitregion_dict,fasta_name=name),axis=1)
    annotations['hitregion']=HitRegion.create_or_update(*hitregion_dicts)


def reconnect(relationship:RelationshipManager,node:StructuredNode,props:dict=None):
    '''
    TODO misc
    reimplementation of `connect` in `RelationshipManager`,
    make sure the same relationship won't be created redundantly
    '''
    if relationship.is_connected(node):
        # relationship.disconnect(target)
        rel = _rel_helper(lhs="a", rhs="b", ident="r", **relationship.definition)
        q = f"""
                MATCH (a), (b) WHERE {db.get_id_method()}(a)=$self and {db.get_id_method()}(b)=$them
                MATCH {rel} DELETE r
            """
        relationship.source.cypher(q, {"them": node.element_id})
    relationship.connect(node,props)


def connect_hit(annotations:pd.DataFrame):
    def connect_hit_row(s:pd.Series):
        fasta:Fasta=s['fasta']
        hit:Hit=s['hit']
        hitfamily:HitFamily=s['hitfamily']
        reconnect(fasta.hits,hit)
        hf_dict=s[['template_neff','probab','identities',
                'similarity', 'e-value','p-value', 
                'aligned_cols', 'score', 'sum_probs']].to_dict()
        hf_dict['e_value']=hf_dict.pop('e-value')
        hf_dict['p_value']=hf_dict.pop('p-value')
        reconnect(hitfamily.members,hit,hf_dict)
    annotations.apply(connect_hit_row,axis=1)


def connect_hitregion(annotations:pd.DataFrame):
    def connect_hitregion_hit(s:pd.Series,rep:bool=False):
        def gen_hr_dict(s:pd.Series,rep:bool=False):
            coverage=(s['begin']-s['end'])/(
                s['group_begin']-s['group_end'])
            return dict(template_neff=s['template_neff'],
                sum_probs=s['sum_probs'],
                representation=rep,
                coverage=coverage)
        hit:Hit=s['hit']
        hitregion:HitRegion=s['hitregion']
        hr_dict=gen_hr_dict(s,rep)
        reconnect(hitregion.affiliates,hit,hr_dict)
    def connect_fasta_hitregion(s:pd.Series):
        fasta:Fasta=s['fasta']
        hitregion:HitRegion=s['hitregion']
        reconnect(fasta.regions,hitregion,{'regid':s['group_id']})
    def pairwise_iter(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..., (sn-1, sn)"
        a, b = tee(iterable)
        next(b, None) 
        return zip(a, b)
    
    rep_idx=annotations.groupby(by='group_id')['sum_probs'].idxmax().sort_index()
    rep_row=annotations.loc[rep_idx]
    rep_row.apply(partial(connect_hitregion_hit,rep=True),axis=1)
    rep_row.apply(connect_fasta_hitregion,axis=1)
    
    for u,d in pairwise_iter(rep_row['hitregion']):
        u:HitRegion
        reconnect(u.downstream,d)
        
    nonrep_row=annotations.loc[~annotations.index.isin(rep_idx)]
    nonrep_row.apply(partial(connect_hitregion_hit,rep=False),axis=1)


# %%
def main(infasta:str,cpu:int=8,threshold=0.8):
    annotations=hhblits_annotation(infasta,cpu)
    # annotations:pd.DataFrame=pd.read_pickle('../../data/annotations.pkl')
    stem=Path(infasta).stem
    if len(annotations)<1:
        logger.warning(f'{stem}: BREAK! no hit for')
        return
    if len(annotations[annotations['probab']>threshold])<1:
        max_probab=annotations['probab'].max()
        logger.warning(f'{stem}: no valid hit, max_probab: {max_probab:.2f}')
        return
    grouped_annotations=group_hit(annotations,threshold)
    generate_neomodels(infasta,grouped_annotations)
    connect_hit(grouped_annotations)
    connect_hitregion(grouped_annotations)
    return grouped_annotations


# %%
if __name__=='__main__':
    infasta='/home/hugheslab1/zfdeng/pangengraph_2/_data/genome_fasta/ScVLA||J04692:genome.fasta'
    # infasta='/home/hugheslab1/zfdeng/pangengraph_2/_data/genome_fasta/EBOV||AF086833:genome.fasta'
    main(infasta)
    families=['Flaviviridae','Filoviridae','Paramyxoviridae','Orthocoronavirinae']
