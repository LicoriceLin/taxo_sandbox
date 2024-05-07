# %%
'''
archive only
not for real run
'''
from neomodel import db,config
from neomodel.integration.pandas import to_dataframe
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import pandas as pd
import seaborn as sns
config.DATABASE_URL = 'bolt://neo4j:WBrtpKCUW28e@52.4.3.233:7687'
from matplotlib.axes import Axes

def pandas_query(q:str,**kwargs):
    params=kwargs
    return to_dataframe(db.cypher_query(q,params,resolve_objects=True))

def distribution_analysis():
    fasta_hits=pandas_query(
        '''
        MATCH (f:Fasta)--(r:HitRegion),(f:Fasta)--(h:Hit)
        RETURN f.name as name,size(f.seq) as nt_length, count(DISTINCT r) as region_count, count(h) as hit_count
        ORDER BY hit_count
        ''')
    family_hits=pandas_query(
        '''
        MATCH (hf:HitFamily)--(m:Hit)
        RETURN hf.name as name,hf.subtype as subtype,hf.std_length as length,count(m) as familysize
        ORDER BY length
        ''')
    # more_fasta_hits=pandas_query(
    #     '''
    #     MATCH (fasta:Fasta)-[hasreg:hasRegion]->(region:HitRegion)-[hasaff:hasAffiliate {representation:TRUE}]-(hit:Hit)
    #     RETURN fasta.name as name,fasta.taxonomy as taxonomy,hasreg.regid as regid,hit.begin as begin, hit.end as end, hit.aligned_seq as seq
    #     ''')
    sns.jointplot(x=np.log10(family_hits['length']),y=np.log10(family_hits['familysize']),s=2,alpha=0.4,kind="scatter")
    plt.xlabel('lg model_length')
    plt.ylabel('lg member_count')
    plt.savefig('scatter_modellength_memercount_hit.svg')
    plt.close()
    
    sns.jointplot(x=fasta_hits['region_count']+(np.random.randn(len(fasta_hits))/40),y=fasta_hits['hit_count'],s=2,alpha=0.4)
    plt.xlabel('region count')
    plt.ylabel('hit count')
    plt.savefig('scatter_reg_hit.svg')
    plt.close()
    
    plt.scatter(x=family_hits['model_length'],y=family_hits['avg_full_length'],s=2,alpha=0.4,color='indianred',label='indelled_length')
    plt.scatter(x=family_hits['model_length'],y=family_hits['avg_ori_length'],s=2,alpha=0.4,color='steelblue',label='original_length')
    plt.plot([0,max(family_hits['model_length'])],[0,max(family_hits['model_length'])],linestyle='--',color='grey',alpha=0.4)
    plt.legend()
    plt.xlabel('model_length')
    # plt.ylabel('lg member_count')
    plt.savefig('scatter_modellength_hitlength.svg')
    plt.close()
    
    fig, ax1 = plt.subplots()
    ax1:Axes
    ax2:Axes = ax1.twinx()
    l1=ax1.scatter(x=np.log2(fasta_hits['nt_length']),y=fasta_hits['region_count'],s=2,alpha=0.4,color='indianred',label='region_count')
    l2=ax2.scatter(x=np.log2(fasta_hits['nt_length']),y=fasta_hits['hit_count'],s=2,alpha=0.4,color='steelblue',label='hit_count')
    # plt.plot([0,max(fasta_hits['nt_length'])],[0,max(fasta_hits['nt_length'])],linestyle='--',color='grey',alpha=0.4)
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left') 
    ax1.set_xticks(ax1.get_xticks(),[str(int(2**i)) for i in ax1.get_xticks()])
    plt.xlabel('nt_length')
    ax1.set_ylabel('region_count')
    ax2.set_ylabel('hit_count')
    # plt.ylabel('lg member_count')
    plt.savefig('scatter_ntlength_reghitlength.svg')
    plt.close()

