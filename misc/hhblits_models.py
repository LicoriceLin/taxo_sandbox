'''
check docstring and `help_text`for description of nodes, relationships and properties 
'''
# %%
from __future__ import annotations
from neomodel import (StructuredNode, StructuredRel,
                    OneOrMore, ZeroOrOne,ZeroOrMore,One,
                    RelationshipTo,RelationshipFrom,
                    StringProperty, FloatProperty,
                    IntegerProperty,BooleanProperty)
from neomodel import config, db
from neomodel import clear_neo4j_database
import pandas as pd
from typing import Optional

# %% 
TAXO_ORDER=['Realm', 'Subrealm', 'Kingdom', 'Subkingdom',
    'Phylum', 'Subphylum', 'Class', 'Subclass', 'Order', 'Suborder',
    'Family', 'Subfamily', 'Genus', 'Subgenus', 'Species']

choices=lambda x:{i:'' for i in set(x)}

# %% 

class hasHit(StructuredRel):
    '''
    Between Fasta and Hit
    '''
    pass

class hasRegion(StructuredRel):
    '''
    Between Fasta and HitRegion
    '''
    regid=IntegerProperty(required=True)
    # pass

class hasDownstream(StructuredRel):
    '''
    Between HitRegion and HitRegion
    '''
    pass
     
class hasMember(StructuredRel):
    '''
    Between HitFamily and Hit
    Check here for detailed description:
    github.com/soedinglab/hh-suite/wiki#hhsearchhhblits-output-hit-list-and-pairwise-alignments
    '''
    template_neff=FloatProperty(required=True,
        help_text='The number of effective sequences of the full alignment')
    probab=FloatProperty(required=True,
        help_text=('the Probability of template to be a true positive. '
                   'This is the weighted, final score for a hit ranging from 0~1.'))
    identities=FloatProperty(required=True,
        help_text=('give the percentage of aligned residue pairs '
                   'of the query and the template master sequences that are identical.'
                   'Note! Seems to be useless for our multi-domain searching task.'))
    similarity=FloatProperty(required=True,
        help_text=('the arithmetic mean of the substitution scores '
                   'between the aligned residue pairs from the query and template master sequences. '
                   'Note! Seems to be useless as it is about the master sequence instead of profile'))
    e_value=FloatProperty(required=True,
        help_text=('the random background noise.'
                   'in a database of the same size one expects to see this number of match '
                   'with a similar score, or higher, simply by chance'))
    p_value=FloatProperty(required=True,
        help_text=('the E-value divided by the number of sequences in the database. '
                   'It is the probability that in a pairwise comparison a wrong hit will score at least this good.'))
    aligned_cols=FloatProperty(required=True,
        help_text=('the number of aligned Match columns in the HMM-HMM alignment.'))
    score=FloatProperty(required=True,
        help_text=('the raw score, computed by the Viterbi HMM-HMM alignment excluding the secondary structure score.'))
    sum_probs=FloatProperty(required=True,
        help_text='the raw score is computed by the Viterbi HMM-HMM alignment excluding the secondary structure score.')
    

class hasAffiliate(StructuredRel):
    '''
    Between HitFamily and Hit
    '''
    representation=BooleanProperty(required=True)
    sum_probs=FloatProperty(required=True)
    coverage=FloatProperty(required=True)

# class hasSynteny(StructuredRel):
#     pass

class hasAnalog(StructuredRel):
    '''
    Between Hit and Hit. 
    if two Hits in one HitFamily has alignment score beyond a certain threshold,
    there will be a hasAnalog rel between them.
    
    to be implemented.
    '''
    pass

class hasFasta(StructuredRel):
    '''
    Link Fasta of the same species to the species' genome.
    '''
    pass
# %%
class Fasta(StructuredNode):
    '''
    Root node of one nt sequence 
    '''
    name = StringProperty(required=True,unique_index=True)
    source = StringProperty(required=True)
    seq = StringProperty(required=True)
    accession = StringProperty(required=True)
    annotation = StringProperty()
    circular = BooleanProperty(default=False,help_text='if the fasta is a circular fasta')
    taxonomy = StringProperty(help_text=f'semicolon separated taxonomy, order:{TAXO_ORDER}')
    hits = RelationshipTo("Hit", "hasHit", cardinality=ZeroOrMore, model=hasHit)
    regions = RelationshipTo("HitRegion", "hasRegion", cardinality=OneOrMore, model=hasRegion)


class Genome(StructuredNode):
    '''
    To be implemented
    a set of fasta to define the genome of a species
    Not Impletemented
    '''
    __abstract_node__ = True
    #hasFasta


class Segment(StructuredNode):
    '''
    abstract, not shown in neo4j graph
    a certain part on the genome
    '''
    __abstract_node__ = True
    name = StringProperty(required=True,unique_index=True)
    begin = IntegerProperty(required=True)
    end = IntegerProperty(required=True)
    

class HitRegion(Segment):
    '''
    unoverlapped annotation region
    each HitRegion might have one or more Hit,
    ideally these hits have similar function, but not guaranteed
    '''
    downstream=RelationshipTo("HitRegion", "hasDownstream", cardinality=ZeroOrOne, model=hasDownstream)
    upstream=RelationshipFrom("HitRegion", "hasDownstream", cardinality=ZeroOrOne, model=hasDownstream)
    #temperary
    affiliates=RelationshipTo('Hit','hasAffiliate',cardinality=OneOrMore,model=hasAffiliate)


class Hit(Segment):
    '''
    original annotation entries.
    Filtered by hhblits' probab, but no further procession.
    could be overlapped with each other.
    '''
    strand = StringProperty(required=True,choices=choices(['s','a']))
    frame = StringProperty(required=True,choices=choices(['0','1','2']))
    
    hit_begin = IntegerProperty(required=True)
    hit_end = IntegerProperty(required=True)
    
    aligned_seq =  StringProperty(required=True,help_text='how Hit is aligned to its HitFamily')
    aligned_consensus = StringProperty(required=True,help_text='digit-wise hit score for Hit-HitFamily alignment')
    
    analog = RelationshipTo("Hit", "hasAnalog", cardinality=ZeroOrMore, model=hasAnalog)
    analog_of = RelationshipFrom("Hit", "hasAnalog", cardinality=ZeroOrMore, model=hasAnalog)

    family = RelationshipFrom("HitFamily", "hasAlignment", cardinality=ZeroOrMore, model=hasMember)


class Family(StructuredNode):
    '''
    abstract, not shown in neo4j graph
    a group of Segments
    '''
    __abstract_node__ = True
    name = StringProperty(required=True)


class HitFamily(Family):
    '''
    
    '''
    source = StringProperty(required=True)
    accession = StringProperty(unique_index=True)
    subtype = StringProperty(help_text=" hmm model's pfam_type (family, domain, motif,...)")
    annotation = StringProperty()
    std_length = IntegerProperty(help_text=" hmm model's length1")
    members =RelationshipTo("Hit", "hasMember", cardinality=OneOrMore, model=hasMember)
    