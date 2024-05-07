# %%
'''
archive only
not for real run
'''
from neomodel import db,config
from neomodel.integration.pandas import to_dataframe
from typing import Optional
import pandas as pd
config.DATABASE_URL = 'bolt://neo4j:WBrtpKCUW28e@52.4.3.233:7687'
def pandas_query(q:str,**kwargs):
    params=kwargs
    return to_dataframe(db.cypher_query(q,params,resolve_objects=True))

def fetching_hits():
    '''
    generate the pkl used for first version's model
    only name, seq(concatenated by #), 
    '''

    
    fasta_hits=pandas_query(
        '''
        MATCH (fasta:Fasta)-[hasreg:hasRegion]->(region:HitRegion)-[hasaff:hasAffiliate {representation:TRUE}]-(hit:Hit)
        RETURN fasta.name as name,hasreg.regid as regid,hit.aligned_seq as seq,fasta.taxonomy as taxonomy
        ''')
    seqs={}
    taxo={}
    for name,subg in fasta_hits.groupby(by='name'):
        seqs[name]='#'.join(subg.sort_values(by='regid')['seq'])
        taxo[name]=subg['taxonomy'].iloc[0]
    o=pd.DataFrame([seqs,taxo]).T
    o.columns=['seq','taxo']
    # o.to_pickle('proseq_taxo.pkl')
    return o

