from src.prepare.dfg import DFG_java, DFG_python
from src.utils.dfg_utils import (remove_comments_and_docstrings,
                                 tree_to_token_index,
                                 index_to_code_token,
                                 tree_to_variable_index)
from tree_sitter import Language, Parser

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
}


#load parsers
parsers={}
for lang in dfg_function:
    LANGUAGE = Language('./parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser,dfg_function[lang]]
    parsers[lang]= parser

# extract dataflow
def extract_dataflow(code, parser):
    # obtain dataflow
    try:
        tree = parser[0].parse(bytes(code,'utf8'))
        root_node = tree.root_node
        tokens_index=tree_to_token_index(root_node)
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)
        try:
            DFG,_=parser[1](root_node,index_to_code,{})
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens, dfg



txt = "public static int unionSize(long[] x,long[] y){\n\tfinal int lx=x.length, ly=y.length;\n\tfinal int min=(lx < ly) ? lx : ly;\n\tint i=0, res=0;\n\tfor (; i < min; i++) {\n\t\tres+=Long.bitCount(x[i] | y[i]);\n\t}\n\tfor (; i < lx; i++) {\n\t\tres+=Long.bitCount(x[i]);\n\t}\n\tfor (; i < ly; i++) {\n\t\tres+=Long.bitCount(y[i]);\n\t}\n\treturn res;\n}\n"
txt2 = "def _pprint_styles(_styles, leadingspace=2):\n\tif leadingspace:\n\t\tpad = ('\t' * leadingspace)\n\telse:\n\t\tpad = ''\n\t(names, attrss, clss) = ([], [], [])\n\timport inspect\n\t_table = [['Class', 'Name', 'Attrs']]\n\tfor (name, cls) in sorted(_styles.items()):\n\t\t(args, varargs, varkw, defaults) = inspect.getargspec(cls.__init__)\n\t\tif defaults:\n\t\t\targs = [(argname, argdefault) for (argname, argdefault) in zip(args[1:], defaults)]\n\t\telse:\n\t\t\targs = None\n\t\tif (args is None):\n\t\t\targstr = 'None'\n\t\telse:\n\t\t\targstr = ','.join([('%s=%s' % (an, av)) for (an, av) in args])\n\t\t_table.append([cls.__name__, (\"'%s'\" % name), argstr])\n\treturn _pprint_table(_table)\n"
print(extract_dataflow(txt, parsers['java'])[0])
