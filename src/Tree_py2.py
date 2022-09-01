# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:37:17 2019

@author: Zhou
"""

import javalang
import ast
import re
import json
from tqdm import tqdm
from collections import deque
from functools import partial

node_maxlen = 15
max_nodes = 400
max_tokens = 450
max_statms = 60
min_statms = 2
subtoken = False
workers = 4

_REF = {javalang.tree.MemberReference,
        javalang.tree.ClassReference,
        javalang.tree.MethodInvocation}

_BLOCK = {'body',
          'block',
          'then_statement',
          'else_statement',
          'catches',
          'finally_block'}

_IGNORE = {'throws',
           'dimensions',
           'prefix_operators',
           'postfix_operators',
           'selectors',
           'types',
           'case'}

_LITERAL_NODE = {'Annotation',
                 'MethodDeclaration',
                 'ConstructorDeclaration',
                 'FormalParameter',
                 'ReferenceType',
                 'MemberReference',
                 'VariableDeclarator',
                 'MethodInvocation',
                 'Literal'}

def load(path, is_json=False, key=None, drop_list=()):
    print('Loading...')
    with open(path, 'r') as f:
        lines = f.readlines()
    if not is_json:
        if not drop_list:
            return lines
        else:
            return [line for i, line in enumerate(lines) if not i in drop_list]
    
    if key is None:
        return [json.loads(line) for i, line in enumerate(lines) if not i in drop_list]
    else:
        return [json.loads(line)[key] for i, line in enumerate(lines) if not i in drop_list]

def save(data, path, is_json=False):
    print('Saving...')
    with open(path, 'w') as f:
        for line in data:
            if is_json:
                line = '' if not line else json.dumps(line)
            f.write(line + '\n')

def get_value(node, token_list):
    value = None
    length = len(token_list)
    if hasattr(node, 'name'):
        value = node.name
    elif hasattr(node, 'value'):
        value = node.value
    elif type(node) in _REF and node.position:
        for i, token in enumerate(token_list):
            if node.position == token.position:
                pos = i + 1
                value = str(token.value)
                while pos < length and token_list[pos].value == '.':
                    value = value + '.' + token_list[pos + 1].value
                    pos += 2
                break
    elif type(node) is javalang.tree.TypeArgument:
        value = str(node.pattern_type)
    elif type(node) is javalang.tree.SuperMethodInvocation \
            or type(node) is javalang.tree.SuperMemberReference:
        value = str(node.member)
    elif type(node) is javalang.tree.BinaryOperation:
        value = node.operator
    return value

def parse_java(code, max_nodes=max_nodes):
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except:
        return []
    
    result = []
    q = deque([tree])
    idx = 1 # index of the next child node (level traversal)
    while len(q) > 0 and len(result) <= max_nodes:
        node = q.popleft()
        if type(node) is dict:
            result.append(node)
            continue
        node_d = {'id': len(result), 'type': node.__class__.__name__, 'children': []}
        value = get_value(node, token_list)
        if value is not None and type(value) is str:
            node_d['value'] = value
        result.append(node_d)
        
        for attr, child in zip(node.attrs, node.children):
            if idx >= max_nodes:
                break
            if attr in _BLOCK and child:
                if type(child) is javalang.tree.BlockStatement:
                    child = child.statements
                block_d = {'id': idx, 'type': attr, 'children': []}
                node_d['children'].append(idx)
                idx += 1
                q.append(block_d)
                node_d = block_d
            if isinstance(child, javalang.ast.Node):
                node_d['children'].append(idx)
                idx += 1
                q.append(child)
            elif type(child) is list and child and attr not in _IGNORE:
                child = [c[0] if type(c) is list else c for c in child[:max_nodes - idx]]
                child_idx = [idx + i for i in range(len(child))]
                node_d['children'].extend(child_idx)
                idx += len(child)
                q.extend(child)
    return result

def parse_python(code, max_nodes=None): # only for python2
    global c, d
    try:
        tree = ast.parse(code)
    except:
        return []
    
    json_tree = []
    def gen_identifier(identifier, node_type = 'identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        return pos
    
    def traverse_list(l, node_type = 'list'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        children = []
        for item in l:
            children.append(traverse(item))
        if (len(children) != 0):
            json_node['children'] = children
        return pos
        
    def traverse(node):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = 'num'
        elif isinstance(node, ast.Str):
            json_node['value'] = 'str'
        elif isinstance(node, ast.alias):
            json_node['value'] = unicode(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname))
        elif isinstance(node, ast.FunctionDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = unicode(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = unicode(node.module)
        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n))
        elif isinstance(node, ast.keyword):
            json_node['value'] = unicode(node.arg)
        
        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target))
            children.append(traverse(node.iter))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test))
            children.append(traverse_list(node.body, 'body'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.With):
            children.append(traverse(node.context_expr))
            if node.optional_vars:
                children.append(traverse(node.optional_vars))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.TryExcept):
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.handlers, 'handlers'))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse'))
        elif isinstance(node, ast.TryFinally):
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.finalbody, 'finalbody'))
        elif isinstance(node, ast.arguments):
            children.append(traverse_list(node.args, 'args'))
            children.append(traverse_list(node.defaults, 'defaults'))
            if node.vararg:
                children.append(gen_identifier(node.vararg, 'vararg'))
            if node.kwarg:
                children.append(gen_identifier(node.kwarg, 'kwarg'))
        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], 'type'))
            if node.name:
                children.append(traverse_list([node.name], 'name'))
            children.append(traverse_list(node.body, 'body'))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, 'bases'))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args))
            children.append(traverse_list(node.body, 'body'))
            children.append(traverse_list(node.decorator_list, 'decorator_list'))
        else:
            # Default handling: iterate over children.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child, ast.boolop) or isinstance(child, ast.unaryop) or isinstance(child, ast.cmpop):
                    # Directly include expr_context, and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + type(child).__name__
                else:
                    children.append(traverse(child))
                
        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, 'attr'))
                
        if (len(children) != 0):
            json_node['children'] = children
        return pos
    
    traverse(tree)
    return json_tree

def get_ast(codes, lang='java', max_nodes=max_nodes, workers=workers, save_path=None):
    desc = 'Building ASTs...'
    parse = parse_java if lang == 'java' else parse_python
    results = []
    for code in tqdm(codes, desc=desc):
        results.append(parse(code, max_nodes))
    final_results = []
    for result in results:
        res = []
        for idx in range(len(result)):
            res.append(result[idx])
            res[idx]['id'] = idx
        final_results.append(res)
    dropped = set(i for i, tree in enumerate(final_results) if len(tree) == 0)
    print('Number of parse failures:', len(dropped))
    if save_path is not None:
        save(final_results, save_path, is_json=True)
    return final_results, dropped

def node_filter(s, subtoken=subtoken):
    s = re.sub(r"\d+\.\d+\S*|0[box]\w*|\b\d+[lLfF]\b", " num ", s)
    s = re.sub(r"%\S*|[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\b\d+\b", " num ", s)
    if subtoken:
        s = re.sub(r"[a-z][A-Z]", lambda x: x.group()[0] + " " + x.group()[1], s)
        s = re.sub(r"[A-Z]{2}[a-z]", lambda x: x.group()[0] + " " + x.group()[1:], s)
        s = re.sub(r"\w{32,}", " ", s) # MD5, hash
        s = re.sub(r"[A-Za-z]\d+", lambda x: x.group()[0] + " ", s)
    s = re.sub(r"\s(num\s+){2,}", " num ", s)
    return s.lower().split()


def pre_traverse(tree, idx, node_maxlen, subtoken):
    node = tree[idx]
    result = []
    result.append(node['type'])
    if node['type'] in _LITERAL_NODE:
        value = node_filter(node['value'], subtoken)
        result.extend(value[:node_maxlen])
    elif node.get('value'):
        result.append(node['value'].lower())
    
    if node['children']:
        for child in node['children']:
            result.extend(pre_traverse(tree, child, node_maxlen, subtoken))
    return result

def get_node_seq(trees, node_maxlen=node_maxlen, subtoken=subtoken,
                 max_tokens=max_tokens, save_path=None):
    results = []
    for tree in tqdm(trees, desc='Obtaining node seqs...'):
        results.append(pre_traverse(tree, 0, node_maxlen, subtoken)[:max_tokens])
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results

def split_ast(trees, node_maxlen=node_maxlen, max_statms=max_statms,
              min_statms=min_statms, subtoken=subtoken, save_path=None):
    def traverse(tree, idx):
        node = tree[idx]
        subTrees = []
        blocks = []
        for i, child in enumerate(node['children']):
            if tree[child]['type'] in _BLOCK:
                blocks.append(child)
                del node['children'][i]
        subTrees.append(pre_traverse(tree, idx, node_maxlen, subtoken)[:max_tokens])
        for block in blocks:
            for child in tree[block]['children']:
                subTrees.extend(traverse(tree, child))
        return subTrees
    
    results = []
    dropped = set()
    for idx, tree in enumerate(tqdm(trees, desc='Splitting ASTs...')):
        result = traverse(tree, 0)[:max_statms]
        results.append(result)
        if len(result) < min_statms:
            dropped.add(idx)
    if save_path is not None:
        save(results, save_path, is_json=True)
    return results, dropped


if __name__ == '__main__':
    for t in ['train', 'test', 'valid']:
        codes = load('/home/tyang/paper_recurr/HGNN/data/raw/code-docstring-corpus.{}.json'.format(t), is_json=True, key='code')
        trees, dropped = get_ast(codes, 'python', save_path='/home/tyang/paper_recurr/HGNN/data/processed/node_seqs_python2.{}.json'.format(t), workers=5)
        # node_seqs = get_node_seq(trees, save_path='/home/tyang/paper_recurr/HGNN/data/processed/nodes_python2.json')
#    subTree_seqs, dropped_s = split_ast(trees, save_path='data/split_ast.json')
#    torch.save(dropped.union(dropped_s), 'data/dropped.pkl')
