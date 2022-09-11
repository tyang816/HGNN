import torch
import re
# for `torchtext-0.11`, `Field` in the `torchtext.legacy`
from torchtext.legacy.data import Field
from torch_geometric.data import InMemoryDataset,Data, HeteroData
# data prepocess tools
from src.Utils import load, save
from tqdm import tqdm
import numpy as np

subtoken = True
text_minlen = 2
text_maxlen = 25
node_seq_maxlen = 10
node_num_maxlen = 200


# special tokens
bos = '<s>'
eos = '</s>'
pad = '<pad>'
unk = '<unk>'


def id2word(word_ids, field, source=None, remove_eos=True, remove_unk=False, replace_unk=False):
    if replace_unk:
        assert type(source) is tuple and not remove_unk
        raw_src, alignments = source
    eos_id = field.vocab.stoi[eos]
    unk_id = field.vocab.stoi[unk]

    if remove_eos:
        word_ids = [s[:-1] if s[-1] == eos_id else s for s in word_ids]
    if remove_unk:
        word_ids = [filter(lambda x: x != unk_id, s) for s in word_ids]
    if not replace_unk:
        return [[field.vocab.itos[w] for w in s] for s in word_ids]
    else:
        return [[field.vocab.itos[w] if w != unk_id else rs[a[i].argmax()] \
                 for i, w in enumerate(s)] for s, rs, a in zip(word_ids, raw_src, alignments)]


def get_vocab(texts, is_tgt=True, max_vocab_size=None, save_path=None):
    if is_tgt:
        field = Field(init_token=bos, eos_token=eos, batch_first=True, pad_token=pad, unk_token=unk)
    else:
        field = Field(batch_first=True, pad_token=pad, unk_token=unk)
    if max_vocab_size:
        field.build_vocab(texts, max_size=max_vocab_size)
    else:
        field.build_vocab(texts)
    if save_path:
        print(f'Saving field to {save_path}...')
        torch.save(field, save_path)
    return field


def nl_filter(s, subtoken=subtoken, min_tokens=text_minlen, max_tokens=text_maxlen):
    s = re.sub(r"\([^\)]*\)|(([eE]\.[gG])|([iI]\.[eE]))\..+|<\S[^>]*>", " ", s)
    #    brackets; html labels; e.g.; i.e.
    s = re.sub(r"\d+\.\d+\S*|0[box]\w*|\b\d+[lLfF]\b", " num ", s)
    first_p = re.search(r"[\.\?\!]+(\s|$)", s)
    if first_p is not None:
        s = s[:first_p.start()]
    s = re.sub(r"https:\S*|http:\S*|www\.\S*", " url ", s)
    s = re.sub(r"\b(todo|TODO)\b.*|[^A-Za-z0-9\.,\s]|\.{2,}", " ", s)
    s = re.sub(r"\b\d+\b", " num ", s)
    s = re.sub(r"([\.,]\s*)+", lambda x: " " + x.group()[0] + " ", s)

    if subtoken:
        s = re.sub(r"[a-z][A-Z]", lambda x: x.group()[0] + " " + x.group()[1], s)
        s = re.sub(r"[A-Z]{2}[a-z]", lambda x: x.group()[0] + " " + x.group()[1:], s)
        s = re.sub(r"\w{32,}", " ", s)  # MD5
        s = re.sub(r"[A-Za-z]\d+", lambda x: x.group()[0] + " ", s)
    s = re.sub(r"\s(num\s+){2,}", " num ", s)
    s = s.lower().split()
    return 0 if len(s) < min_tokens else s[:max_tokens]

def tokenize_nl(texts, subtoken=subtoken, min_tokens=text_minlen,
                max_tokens=text_maxlen, save_path=None):
    '''
    :param texts: [[text],[text],...]
    :param subtoken:
    :param min_tokens:
    :param max_tokens:
    :param save_path:
    :return:
    '''
    drop_list = set()
    results = []
    for idx, text in enumerate(tqdm(texts, desc='Tokenizing texts...')):
        tok_text = nl_filter(text, subtoken, min_tokens, max_tokens)
        if tok_text:
            results.append(tok_text)
        else:
            results.append(['method'])
            drop_list.add(idx)

    if save_path is not None:
        save(results, save_path, is_json=True)
    print('number of dropped texts:', len(drop_list))
    return results, drop_list

def get_vocab_field(lang):
    assert isinstance(lang, list or tuple), "language should be ['java', 'python2']"
    for l in lang:
        codes = load(f'./data/{l}/graph/node_seqs.train.json', is_json=True)
        nodes = []
        for node_seq in tqdm(codes):
            nodes_ = []
            for node in node_seq:
                nodes_.append(node['type'])
                if node.get('value'):
                    nodes_.append(node['value'])
            nodes.append(nodes_)
        get_vocab(nodes, is_tgt=False, save_path=f'./data/field/{l}_node_field.pkl')
        a = torch.load(f'./data/{lang}/processed/node_field.pkl')
        print(a.process([['MethodDeclaration', 'FormalParameter'], ['a', 'b']]))
        nls = load(f'./data/{l}/raw/train.json', is_json=True, key='comment')
        a = torch.load(f'./data/{lang}/processed/nl_field.pkl')
        print(a.process([['MethodDeclaration', 'FormalParameter'], ['a', 'b']]))
        get_vocab(nls, is_tgt=True, save_path=f'./data/{lang}/processed/nl_field.pkl')


class ASTGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ASTGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['train.python2_graph.pt']

    def process(self):
        lang = 'python2'
        category = 'train'
        node_seqs = load(f'./data/{lang}/graph/node_seqs_value/node_seqs_value.{category}.json', is_json=True)
        dfg_seqs = load(f'./data/{lang}/graph/dfg/dfg.{category}.json', is_json=False)
        ncs_seqs = load(f'./data/{lang}/graph/ncs/ncs.{category}.json', is_json=False)
        nl_seqs = load(f'./data/{lang}/raw/{category}.json', is_json=True, key='comment')
        node_type_field = torch.load(f'./data/{lang}/processed/node_field.pkl')
        nl_field = torch.load(f'./data/{lang}/processed/nl_field.pkl')

        data_list = []
        for node_seq, dfg_seq, ncs_seq, nl_seq in tqdm(zip(node_seqs, dfg_seqs, ncs_seqs, nl_seqs)):
            dfg_seq = eval(dfg_seq)
            ncs_seq = eval(ncs_seq)
            edge_index = []
            nodes = []
            edge_attr = []
            for node in node_seq:
                # non-leaf node
                if node.get('type'):
                    if node.get('children'):
                        for child in node['children']:
                            curr_edge1 = [node['id'], child]
                            curr_edge2 = [child, node['id']]
                            if curr_edge1 not in edge_index:
                                edge_index.append(curr_edge1)
                                edge_attr.append([1, 0, 0])
                            elif curr_edge1 in edge_index:
                                edge_attr[edge_index.index(curr_edge1)][0] = 1
                            if curr_edge2 not in edge_index:
                                edge_index.append(curr_edge2)
                                edge_attr.append([1, 0, 0])
                            elif curr_edge2 in edge_index:
                                edge_attr[edge_index.index(curr_edge2)][0] = 1
                    nodes.append([node['type']])
                # leaf node
                else:
                    nodes.append([node['value']])
            if dfg_seq:
                for dfg in dfg_seq:
                    if dfg not in edge_index:
                        edge_index.append(dfg)
                        edge_attr.append([0, 1, 0])
                    elif dfg in edge_index:
                        edge_attr[edge_index.index(dfg)][1] = 1
            if ncs_seq:
                for ncs in ncs_seq:
                    if ncs not in edge_index:
                        edge_index.append(ncs)
                        edge_attr.append([0, 0, 1])
                    elif ncs in edge_index:
                        edge_attr[edge_index.index(ncs)][2] = 1
            try:
                data = Data(x=node_type_field.process(nodes),
                            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                            edge_attr=torch.IntTensor(edge_attr),
                            y=nl_field.process([nl_seq]).squeeze(0))
                data_list.append(data)
            except:
                continue
        print(data_list[0], data_list[1])
        # save data
        torch.save(data_list, self.processed_paths[0])

if __name__ == '__main__':
    # get_vocab_field(['java', 'python2'])
    # a = torch.load(f'./data/field/java_nl_field.pkl')
    # print(a.process([['MethodDeclaration', 'FormalParameter'], ['create', 'b']]))
    # a = torch.load(f'./data/field/java_node_field.pkl')
    # print(a.process([['MethodDeclaration', 'FormalParameter'], ['create', 'b']]))
    # ast_graph = ASTGraphDataset('data')
    graph = torch.load('./data/processed/train.python2_graph.pt')
    print(len(graph))
    print(graph[0], graph[1])

