from src.Utils import load, save
from tqdm import tqdm
import json

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

JAVA_ERROR_ORDER_NODE = {
    'MethodDeclaration',
'ReferenceType',
'VariableDeclarator',
'BinaryOperation',
'CatchClauseParameter',
'BasicType',
'FormalParameter'
}

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

# see src.Tree.py to get AST sequence
def process_java():
    # process java dataset
    return None

# use python2 env, see src.Tree_py2.py to get AST sequence
def process_python2():
    return None


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

# extract the leaf nodes of the node sequence and add them to the end
def add_value_node(node_seqs):
    # node_seqs: [[node_seq1],[node_seq2]]
    new_node_seqs = []
    for node_seq in node_seqs:
        # node_seq: [{node1},{node2},..]
        new_node_seq = node_seq
        for i in range(len(node_seq)):
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

def get_node_seqs_with_value(lang):
    assert isinstance(lang, list or tuple), "language should be ['java', 'python2']"
    for l in lang:
        for cat in ['train', 'valid', 'test']:
            c = load(f'./data/{l}/processed/node_seqs.{cat}.json', is_json=True)
            c = add_value_node(c)
            save(c, f'./data/{l}/processed/node_seqs_value/node_seqs_value.{cat}.json', is_json=True)


# get data flow control
def get_java_DFG(node_seqs):
    edge_index_seq = []
    # print(node_seqs)
    for seq in tqdm(node_seqs):
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

# first run `get_node_seqs_with_value()` to get node sequences with value leaves
def get_DFG(lang):
    assert isinstance(lang, list or tuple), "language should be ['java', 'python2']"
    if 'java' in lang:
        for cat in ['train', 'valid', 'test']:
            c = load(f'./data/java/graph/node_seqs_value/node_seqs_value.{cat}.json', is_json=True)
            d = get_java_DFG(c)
            save(d, f'./data/java/graph/dfg/dfg.{cat}.json', is_json=False)

    if 'python2' in lang:
        for cat in ['train', 'valid', 'test']:
            c_ = load(f'./data/python2/graph/node_seqs_value/node_seqs_value.{cat}.json', is_json=True)
            d_ = get_python2_DFG(c_)
            save(d_, f'./data/python2/graph/dfg/dfg.{cat}.json', is_json=False)

# get natural code sequence
def ncs_pre_traverse(tree, idx):
    # shape of return: [(type, value_node_idx), ...]
    node = tree[idx]
    result = []
    if not node.get('value') and node.get('children'):
        children = node.get('children')
        if children:
            for child in children:
                result.extend(ncs_pre_traverse(tree, child))
    if node.get('value') and node.get('children'):
        value_id = node.get('children')[-1]
        type = node.get('type')
        children = node.get('children')[:-1]
        result.append((type, value_id, node.get('value')))
        if children:
            for child in children:
                result.extend(ncs_pre_traverse(tree, child))
    return result

# first run `get_node_seqs_with_value()` to get node sequences with value leaves
def get_NCS(lang):
    assert isinstance(lang, list or tuple), "language should be ['java', 'python2']"
    if 'java' in lang:
        for cat in ['test', 'train', 'valid']:
            node_seqs = load(f'./data/java/graph/node_seqs_value/node_seqs_value.{cat}.json', is_json=True)
            nature_code_seqs = []
            for node_seq in node_seqs:
                # ncs: [(type, id), ...]
                ncs = ncs_pre_traverse(node_seq, 0)
                nature_code_seq = []
                i = 0
                while i< len(ncs)-2:
                    cur_node = ncs[i]
                    cur_type = cur_node[0]
                    if cur_type in ['MethodDeclaration', 'FormalParameter']:
                        # print(i, cur_node)
                        ncs[i], ncs[i + 1] = ncs[i + 1], ncs[i]
                        # skip the next node
                        i = i + 1
                    if cur_type in ['BinaryOperation']:
                        if ncs[i + 1][0] != 'BinaryOperation':
                            ncs[i], ncs[i + 1] = ncs[i + 1], ncs[i]
                        else:
                            ncs[i], ncs[i + 2] = ncs[i + 2], ncs[i]
                        i = i + 1
                    i = i + 1
                for i in range(len(ncs) - 1):
                    nature_code_seq.append([ncs[i][1], ncs[i + 1][1]])
                nature_code_seqs.append(nature_code_seq)
            save(nature_code_seqs, f'data/java/graph/ncs/ncs.{cat}.json', is_json=False)
    elif 'python2' in lang:
        for cat in ['test', 'train', 'valid']:
            node_seqs = load(f'./data/python2/graph/node_seqs_value/node_seqs_value.{cat}.json', is_json=True)
            nature_code_seqs = []
            for node_seq in node_seqs:
                # ncs: [(type, id), ...]
                ncs = ncs_pre_traverse(node_seq, 0)
                nature_code_seq = []
                for i in range(len(ncs) - 1):
                    nature_code_seq.append([ncs[i][1], ncs[i + 1][1]])
                nature_code_seqs.append(nature_code_seq)
            save(nature_code_seqs, f'data/python2/graph/ncs/ncs.{cat}.json', is_json=False)


if __name__ == '__main__':
    # get_node_seqs_with_value(['python2'])
    # get_DFG(['python2', 'java'])
    get_NCS(['python2'])
