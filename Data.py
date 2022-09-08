import torch
import yaml
import json
import re
from collections import Counter, OrderedDict
# for `torchtext-0.11`, `Field` in the `torchtext.legacy`
from torchtext.legacy.data import Field
from torch_geometric.data import InMemoryDataset,Data, HeteroData
from torch_geometric.loader import DataLoader
import pandas as pd
# data prepocess tools
from src.Utils import load, save
from tqdm import tqdm

JAVA_VALUE_NODE = {'FormalParameter',
                     'MemberReference',
                     'VariableDeclarator',
                     'Literal',
                   'ClassCreator',
                   'ReferenceType'}

PYTHON2_LITERAL_NODE = {'NameParam',
                        'NameLoad',
                        'NameStore',
                        'Num',
                        'Str'
                        }
subtoken = True
text_minlen = 2
text_maxlen = 20
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

def get_raw_data(path, dataset_name):
    assert dataset_name == 'code-docstring-corpus' or dataset_name == 'TL-CodeSum'
    data = {}
    for cat in ['train', 'valid', 'test']:
        codstring = []
        for type in ['raw.code', 'nl.json']:
            with open(path+dataset_name+'/{}.{}'.format(cat, type), 'r') as f:
                lines = f.readlines()
                print(f"{type} {cat} {len(lines)}")
                if type == 'raw.code':
                    for i in range(len(lines)):
                        codstring.append({'code': eval(lines[i])})
                else:
                    for i in range(len(lines)):
                        codstring[i]['comment'] = eval(lines[i])
        data[cat] = codstring
    # print(codstring)
    for cat in ['train', 'valid', 'test']:
        with open(path+dataset_name+f'.{cat}.json', 'w', encoding='utf-8') as f:
            for line in data[cat]:
                f.write(json.dumps(line) + '\n')
    return data

def add_value_node(node_seqs):
    # node_seqs: [[node_seq1],[node_seq2]]
    new_node_seqs = []
    for node_seq in node_seqs:
        # node_seq: [{node1},{node2},..]
        # print(node_seq)
        if isinstance(node_seq, str):
            node_seq = eval(node_seq)
        # print(node_seq)
        new_node_seq = node_seq
        for i in range(len(node_seq)):
            # print(node_seq[i])
            # separate the nodes with values into leaf nodes
            if node_seq[i].get('value'):
                new_node = {}
                new_node['id'] = len(new_node_seq)
                new_node['value'] = node_seq[i]['value']
                if new_node_seq[i].get('children'):
                    new_node_seq[i]['children'].append(new_node['id'])
                else:
                    new_node_seq[i]['children'] = [new_node['id']]
                new_node_seq.append(new_node)
        new_node_seqs.append(new_node_seq)
    return new_node_seqs

def ncs_pre_traverse(tree, idx):
    # shape of return: [value_node_idx1, value_node_idx2, ...]
    node = tree[idx]
    result = []
    if node.get('children'):
        value_idx = node.get('children')[-1]
        result.append(value_idx)
        print(value_idx)
        children = node.get('children')[:-1]
        for child in children:
            result.extend(ncs_pre_traverse(tree, child))
    return result


def type_pre_traverse(tree, idx, lang='java'):
    # shape of return: [(type, value, idx, children),...]
    node = tree[idx]
    result = []
    if node.get('type') in (['Assignment', 'MethodInvocation', 'BinaryOperation', 'ClassCreator'] if lang == 'java'
                    else ['Assign', 'BinOpMod', 'BinOpSub', 'BinOpAdd', 'BinOpMult', 'AttributeLoad', 'Call']):
        result.append((node['type'], [], idx, (node['children'] if node.get('children') else [])))
    elif node.get('value') and node.get('type') in (JAVA_VALUE_NODE if lang == 'java'
                                                else PYTHON2_LITERAL_NODE):
        result.append((node['type'], node['value'].lower(), idx, (node['children'] if node.get('children') else [])))
    elif node.get('value') and not node.get('type'):
        result.append(('value', node['value'].lower(), idx, []))
    if node.get('children'):
        for child in node['children']:
            result.extend(type_pre_traverse(tree, child, lang=lang))
    return result

def get_child_value_idx(nodes, idx, lang='java'):
    # nodes: [(type, value, idx, children),...]
    child_idx = []
    # print(nodes)
    for node in nodes:
        # if find target node
        if node[2] == idx:
            # print(node)
            if node[0] in (['MemberReference', 'Literal', 'ReferenceType', 'ClassCreator'] if lang == 'java'
                            else ['NameLoad', 'Str', 'Num', 'ListLoad', 'TupleLoad']):
                child_idx.append(idx)
            for child in node[3]:
                child_idx.extend(get_child_value_idx(nodes, child))
            break
    return child_idx

def get_node_by_idx(nodes, idx):
    # nodes: [(type, value, idx, children),...]
    trg_node = None
    for node in nodes:
        if node[2] == idx:
            trg_node = node
            break
    return trg_node

def get_java_DFG(node_seqs):
    edge_index_seq = []
    # print(node_seqs)
    for seq in tqdm(node_seqs):
        if isinstance(seq, str):
            seq = eval(seq)
        # nodes: [(type, value, idx, children),...]
        nodes = type_pre_traverse(seq, 0)
        # print(nodes)
        formal_parameters = []
        variable_declarators = []
        local_variable_declarations = []
        assignments = []

        edge_index = []
        for node in nodes:
            if node[0] == 'FormalParameter':
                formal_parameters.append(node)
            elif node[0] == 'VariableDeclarator':
                variable_declarators.append(node)
            elif node[0] == 'LocalVariableDeclaration':
                local_variable_declarations.append(node)
            elif node[0] == 'Assignment':
                assignments.append(node)
        # extract the flow of `FormalParameter`
        for param in formal_parameters:
            # extract variable flow for direct use
            skip_next_node = 0
            for node in nodes:
                # there has `[]` in `node_name` (node[1])
                if not isinstance(node[1], str):
                    continue
                # if assigment, next target node just uses the definition, there is no data flow
                if skip_next_node != 0:
                    skip_next_node = skip_next_node - 1
                    continue
                if node[0] == 'Assignment':
                    skip_next_node = skip_next_node + 1
                # example: `x` in `x.length`
                if (param[1] == node[1] or param[1] in node[1].split('.')) and node[0] == 'MemberReference':
                    # node[3][-1] is the value node index of the current node
                    edge_index.append([param[3][-1], node[3][-1]])
        # extract the flow of `VariableDeclarator`
        for declr in variable_declarators:
            # extract variable flow for direct use
            skip_next_node = 0
            for node in nodes:
                # there has `[]` in `node_name` (node[1])
                if not isinstance(node[1], str):
                    continue
                # if assigment, next target node just uses the definition, there is no data flow
                if skip_next_node != 0:
                    skip_next_node = skip_next_node - 1
                    continue
                if node[0] == 'Assignment':
                    skip_next_node = skip_next_node + 1
                # example: `x` in `x.length`
                if (declr[1] == node[1] or declr[1] in node[1].split('.')) and node[0] == 'MemberReference':
                    # node[3][-1] is the value node index of the current node
                    edge_index.append([declr[3][-1], node[3][-1]])
            # extract variable flow when declaration
            child_value_idx = get_child_value_idx(nodes, declr[2])
            for idx in child_value_idx:
                edge_index.append([idx, declr[3][-1]])
        # extract the flow of `Assignment`
        for assign in assignments:
            # assignment has two variables
            # there has `('Assignment', [], idx, [])` in `assignments`
            try:
                trg_var_idx = assign[3][0]
                src_var_idx = assign[3][1]
            except:
                continue
            # print(src_var_idx)
            src_child_value_idx = get_child_value_idx(nodes, src_var_idx)
            # print(src_child_value_idx)
            for idx in src_child_value_idx:
                # print(idx)
                try:
                    edge_index.append([get_node_by_idx(nodes, idx)[3][-1], get_node_by_idx(nodes, trg_var_idx)[3][-1]])
                except:
                    continue
        edge_index_seq.append(edge_index)
    return edge_index_seq

def get_python2_DFG(node_seqs):
    edge_index_seq = []
    for seq in tqdm(node_seqs):
        if isinstance(seq, str):
            seq = eval(seq)
        # nodes: [(type, value, idx, children),...]
        nodes = type_pre_traverse(seq, 0, lang='python2')
        # print(nodes)
        name_params = []
        name_stores = []
        assignments = []
        tuple_stores = []
        list_stores = []

        edge_index = []
        for node in nodes:
            if node[0] == 'NameParam':
                name_params.append(node)
            elif node[0] == 'NameStore':
                name_stores.append(node)
            elif node[0] == 'Assign':
                assignments.append(node)
            elif node[0] == 'TupleStore':
                tuple_stores.append(node)
            elif node[0] == 'ListStore':
                list_stores.append(node)

        # extract the flow of `NameParam`
        for param in name_params:
            # extract variable flow for direct use
            skip_next_node = 0
            for node in nodes:
                # there has `[]` in `node_name` (node[1])
                if not isinstance(node[1], str):
                    continue
                # if assigment, next target node just uses the definition, there is no data flow
                if skip_next_node != 0:
                    skip_next_node = skip_next_node - 1
                    continue
                if node[0] == 'Assign':
                    skip_next_node = skip_next_node + 1
                # example: `x` in `x.length`
                if (param[1] == node[1] or param[1] in node[1].split('.')) and node[0] == 'NameLoad':
                    edge_index.append([param[3][-1], node[3][-1]])
        # extract the flow of `NameStore`
        for store in name_stores:
            # extract variable flow for direct use
            skip_next_node = 0
            for node in nodes:
                # there has `[]` in `node_name` (node[1])
                if not isinstance(node[1], str):
                    continue
                # if assigment, next target node just uses the definition, there is no data flow
                if skip_next_node != 0:
                    skip_next_node = skip_next_node - 1
                    continue
                if node[0] == 'Assign':
                    skip_next_node = skip_next_node + 1
                if (store[1] == node[1] or store[1] in node[1].split('.')) and node[0] == 'NameLoad':
                    edge_index.append([store[3][-1], node[3][-1]])
            # extract variable flow when declaration
            child_value_idx = get_child_value_idx(nodes, store[2], lang='python2')
            for idx in child_value_idx:
                edge_index.append([idx, store[3][-1]])
        # extract the flow of `Assignment`
        for assign in assignments:
            # assignment has two variables
            trg_var_idx = assign[3][0]
            src_var_idx = assign[3][1]
            # print(trg_var_idx, src_var_idx)
            # find the target node
            # example: (x, y) = ..., `(x, y)` is the `TupleStore` target node, it has two children
            for node in nodes:
                if node[2] == trg_var_idx:
                    trg_node = node
                    break
            # handle multiple assignments
            if trg_node[0] in ['ListStore', 'TupleStore']:
                for (trg, src) in zip(nodes[trg_var_idx][3], nodes[src_var_idx][3]):
                    src_child_value_idx = get_child_value_idx(nodes, src, lang='python2')
                    for idx in src_child_value_idx:
                        try:
                            edge_index.append([get_node_by_idx(nodes, idx)[3][-1], get_node_by_idx(nodes, trg)[3][-1]])
                        except:
                            continue
            # handle single assignment
            else:
                src_child_value_idx = get_child_value_idx(nodes, src_var_idx, lang='python2')
                for idx in src_child_value_idx:
                    try:
                        edge_index.append([get_node_by_idx(nodes, idx)[3][-1], get_node_by_idx(nodes, trg_var_idx)[3][-1]])
                    except:
                        continue
        edge_index_seq.append(edge_index)
    return edge_index_seq

def get_vocab(texts, is_tgt=True, max_vocab_size=None, save_path=None):
    if is_tgt:
        field = Field(init_token=bos, eos_token=eos, batch_first=True, pad_token=pad, unk_token=unk, fix_length=text_maxlen)
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

class ASTGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ASTGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_dataset_java.pt']

    def process(self):
        lang = 'java'
        category = 'train'
        if lang == 'java':
            node_seqs = load(f'./data/processed/node_seqs_value/node_seqs_value_{lang}.{category}.json', is_json=False)
            edge_seqs = load(f'./data/processed/dfg_edge/edge_{lang}.{category}.json', is_json=False)
            com_seqs = load(f'./data/raw/TL-CodeSum.{category}.json', is_json=True, key='comment')
            node_type_field = torch.load('./data/field/java_node_field.pkl')
            nl_field = torch.load('./data/field/java_nl_field.pkl')
        else:
            node_seqs = load(f'./data/processed/node_seqs_value/node_seqs_value_{lang}.{category}.json', is_json=False)
            edge_seqs = load(f'./data/processed/dfg_edge/edge_{lang}.{category}.json', is_json=False)
            com_seqs = load(f'./data/raw/code-docstring-corpus.{category}.json', is_json=True, key='comment')
            node_type_field = torch.load('./data/field/python2_node_field.pkl')
            nl_field = torch.load('./data/field/python2_nl_field.pkl')
        data_list = []

        for node_seq, dfg_edge, com in tqdm(zip(node_seqs, edge_seqs, com_seqs)):
            node_seq = eval(node_seq)
            dfg_edge = eval(dfg_edge.replace('\n', ''))
            ast_edge_index = []
            ast_node = []
            ast_edge_attr = []
            for node in node_seq:
                # non-leaf node
                if node.get('type'):
                    if node.get('children'):
                        for child in node['children']:
                            curr_edge1 = [node['id'], child]
                            curr_edge2 = [child, node['id']]
                            if curr_edge1 not in ast_edge_index:
                                ast_edge_index.append(curr_edge1)
                                ast_edge_attr.append([1, 0, 0])
                            elif curr_edge1 in ast_edge_index:
                                ast_edge_attr[ast_edge_index.index(curr_edge1)][0] = 1
                            if curr_edge2 not in ast_edge_index:
                                ast_edge_index.append(curr_edge2)
                                ast_edge_attr.append([1, 0, 0])
                            elif curr_edge2 in ast_edge_index:
                                ast_edge_attr[ast_edge_index.index(curr_edge2)][0] = 1
                    ast_node.append([node['type']])
                # leaf node
                else:
                    ast_node.append([node['value']])
            if dfg_edge != []:
                for dfg in dfg_edge:
                    if dfg not in ast_edge_index:
                        ast_edge_index.append(dfg)
                        ast_edge_attr.append([0, 1, 0])
                    elif dfg in ast_edge_index:
                        ast_edge_attr[ast_edge_index.index(dfg)][1] = 1
            try:
                data = Data(x=node_type_field.process(ast_node),
                            edge_index=torch.tensor(ast_edge_index, dtype=torch.long).t().contiguous(),
                            edge_attr=torch.IntTensor(ast_edge_attr),
                            y=nl_field.process([com]).squeeze(0))
                data_list.append(data)
                # print(data.x.shape, data.edge_index.shape, data.edge_attr.shape, data.y.shape)
            except:
                continue
            # print(data['AST'].x.shape, data['AST'].edge_index.shape, data['AST'].edge_attr.shape, data['AST'].y.shape)
        print(data_list[0], data_list[1])
        # save data
        data_save, data_slices = self.collate(data_list)
        torch.save((data_save, data_slices), self.processed_paths[0])



# see src.Tree.py to get AST sequence
def process_java():
    # process java dataset
    return None

# use python2 env, see src.Tree_py2.py to get AST sequence
def process_python2():
    return None


def get_vocab_field(lang):
    assert isinstance(lang, list or tuple), "language should be ['java', 'python2']"
    if 'java' in lang:
        codes = load(f'./data/processed/node_seqs_java.train.json', is_json=False)
        nodes = []
        for node_seq in tqdm(codes):
            n = []
            node_seq = eval(node_seq)
            for node in node_seq:
                n.append(node['type'])
                if node.get('value'):
                    n.append(node['value'])
            nodes.append(n)
        get_vocab(nodes, is_tgt=False, save_path=f'./data/field/java_node_field.pkl')
        a = torch.load(f'./data/field/java_node_field.pkl')
        print(a.process([['MethodDeclaration', 'FormalParameter'], ['a', 'b']]))
        coms = load('./data/raw/TL-CodeSum.train.json', is_json=True, key='comment')
        texts = []
        for com in coms:
            texts.append((com))
        get_vocab(texts, save_path='./data/field/java_nl_field.pkl')
    if 'python2' in lang:
        codes = load(f'./data/processed/node_seqs_python2.train.json', is_json=False)
        nodes = []
        for node_seq in tqdm(codes):
            n = []
            node_seq = eval(node_seq)
            for node in node_seq:
                n.append(node['type'])
                if node.get('value'):
                    n.append(node['value'])
            nodes.append(n)
        get_vocab(nodes, is_tgt=False, save_path=f'./data/field/python2_node_field.pkl')
        a = torch.load(f'./data/field/python2_node_field.pkl')
        print(a.process([['MethodDeclaration', 'FormalParameter'], ['a', 'b']]))
        coms = load('./data/raw/code-docstring-corpus.train.json', is_json=True, key='comment')
        texts = []
        for com in coms:
            texts.append((com))
        get_vocab(coms, save_path='./data/field/python2_nl_field.pkl')

# get data flow control
def get_DFG(lang):
    assert isinstance(lang, list or tuple), "language should be ['java', 'python2']"
    if 'java' in lang:
        for cat in ['train', 'valid', 'test']:
            c = load(f'./data/processed/node_seqs_java.{cat}.json', is_json=False)
            c = add_value_node(c)
            save(c, f'./data/processed/node_seqs_value/node_seqs_value_java.{cat}.json')
            d = get_java_DFG(c)
            save(d, f'./data/processed/dfg_edge/edge_java.{cat}.json')

    if 'python2' in lang:
        for cat in ['train', 'valid', 'test']:
            c_ = load(f'./data/processed/node_seqs_python2.{cat}.json', is_json=False)
            c_ = add_value_node(c_)
            save(c_, f'./data/processed/node_seqs_value/node_seqs_value_python2.{cat}.json')
            d_ = get_python2_DFG(c_)
            save(d_, f'./data/processed/dfg_edge/edge_python2.{cat}.json')

# get natural code sequence
# first run `get_DFG()` to get node sequences with value leaves
def get_NCS(lang):
    assert isinstance(lang, list or tuple), "language should be ['java', 'python2']"
    if 'java' in lang:
        for cat in ['train', 'valid', 'test']:
            c = load(f'./data/processed/node_seqs_value/node_seqs_value_java.{cat}.json', is_json=False)
            print(eval(c[0])[0])
            break

if __name__ == '__main__':
    # get_raw_data('./data/raw/', 'TL-CodeSum')
    # get_DFG(['java', 'python2'])
    # get_NCS(['java'])
    # get_vocab_field(['java'])
    # ast_graph = ASTGraphDataset('data')
    # print(len(ast_graph))
    # print(ast_graph[0], ast_graph[1])
    x = [{'id': 0, 'type': 'MethodDeclaration', 'children': [1, 2, 3, 4, 14], 'value': 'isDoubleEqual'}, {'id': 1, 'type': 'BasicType', 'children': [15], 'value': 'boolean'}, {'id': 2, 'type': 'FormalParameter', 'children': [6, 16], 'value': 'value'}, {'id': 3, 'type': 'FormalParameter', 'children': [7, 17], 'value': 'valueToCompare'}, {'id': 4, 'type': 'body', 'children': [5]}, {'id': 5, 'type': 'ReturnStatement', 'children': [8]}, {'id': 6, 'type': 'BasicType', 'children': [18], 'value': 'double'}, {'id': 7, 'type': 'BasicType', 'children': [19], 'value': 'double'}, {'id': 8, 'type': 'BinaryOperation', 'children': [9, 10, 20], 'value': '<'}, {'id': 9, 'type': 'MethodInvocation', 'children': [11, 21], 'value': 'Math.abs'}, {'id': 10, 'type': 'Literal', 'children': [22], 'value': '0.001'}, {'id': 11, 'type': 'BinaryOperation', 'children': [12, 13, 23], 'value': '-'}, {'id': 12, 'type': 'MemberReference', 'children': [24], 'value': 'value'}, {'id': 13, 'type': 'MemberReference', 'children': [25], 'value': 'valueToCompare'}, {'id': 14, 'value': 'isDoubleEqual'}, {'id': 15, 'value': 'boolean'}, {'id': 16, 'value': 'value'}, {'id': 17, 'value': 'valueToCompare'}, {'id': 18, 'value': 'double'}, {'id': 19, 'value': 'double'}, {'id': 20, 'value': '<'}, {'id': 21, 'value': 'Math.abs'}, {'id': 22, 'value': '0.001'}, {'id': 23, 'value': '-'}, {'id': 24, 'value': 'value'}, {'id': 25, 'value': 'valueToCompare'}]

    print(ncs_pre_traverse(x, 0))
