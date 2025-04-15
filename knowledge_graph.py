import torch

import rdflib
from rdflib import Graph


class KnowledgeGraph():

    def __init__(self, g_path, excl_classes=None):
        self.graph = Graph().parse(g_path)
        if excl_classes is not None:
            self._delete_classes(excl_classes)
        self.emb, self.ent_to_index = self._subclass_embedding()
        self.name_to_index, self.name_to_embedding = self._node_name_mapping()

    def _delete_classes(self, excl_classes):
        '''
        Delete classes from KG.

        Args:
            excl_classes: List of classes to be excluded from KG
        '''
        for c in excl_classes:
            c_uri = rdflib.term.URIRef(f"http://condition.org/{c}")
            # Remove all triples with class as subject or object
            for triple in self.graph.triples((c_uri, None, None)):
                self.graph.remove(triple)
            for triple in self.graph.triples((None, None, c_uri)):
                self.graph.remove(triple)

    def _subclass_embedding(self):
        '''
        Create KG embedding based on rdfs:subClassOf relationships.

        Args:
            g_in: KG

        Returns:
            g_emb: Manual subclass embedding of knowledge graph
            ent_to_index: Dictionary for mapping entities to indeces of
                sub_class_matrix
        '''
        # Query transitive subClassOf relationship
        query = list(self.graph.query("""
            SELECT ?s ?o
            WHERE{
                ?s rdfs:subClassOf+ ?o
            }
        """))

        # Exclude dummy entity
        entities = [
            i
            for i in sorted(list(self.graph.all_nodes()))
            if i != rdflib.term.URIRef('http://condition.org/o_0')
        ]
        n_ent = len(entities)
        ent_to_index = {node: i for i, node in enumerate(entities)}

        # Create embedding matrix
        g_emb = torch.zeros((n_ent, n_ent))
        for s, o in query:
            i = ent_to_index[s]
            j = ent_to_index[o]
            g_emb[i, j] = 1

        # All classes are subclasses of themselves
        g_emb = g_emb + torch.diag(torch.ones(n_ent))

        return g_emb, ent_to_index

    def _node_name_mapping(self):
        '''
        Create mapping dictionaries for node names to indeces and embeddings.

        Returns:
            name_to_index: Dictionary for mapping node names to indeces
            name_to_embedding: Dictionary for mapping node names to embeddings
        '''
        name_to_index = {
            str(k).rsplit('/', maxsplit=1)[-1]: v
            for k, v in self.ent_to_index.items()
        }
        name_to_embedding = {
            k: self.emb[v]
            for k, v in name_to_index.items()
        }
        return name_to_index, name_to_embedding

    def map_embedding(self, y, label_mapping):
        '''
        Map a target vector to KG embedding.

        Args:
            y: Target vector

        Returns:
            y_emb: Target vector mapped to KG embedding
        '''
        y_emb = []
        for y_i in y:
            emb = None
            for k_l, v_l in label_mapping.items():
                if y_i == v_l:
                    for k_e, v_e in self.name_to_index.items():
                        if k_e == k_l:
                            emb = self.emb[v_e]
            if emb is None:
                # Zero embedding for unknown classes
                emb = torch.zeros(self.emb.size(1))
            y_emb.append(emb)

        return torch.stack(y_emb)
