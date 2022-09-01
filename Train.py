import torch
from torch_geometric.loader import DataLoader
import yaml
from Modules import Encoder, Decoder
from Data import ASTGraphDataset
from Model import Model

# load config
config_path = './config/config.yml'
config = yaml.load(open(config_path), Loader=yaml.FullLoader)

code_embed_dim = config['model']['code_embed_dim']
com_embed_dim = config['model']['com_embed_dim']
edge_embed_dim = config['model']['edge_embed_dim']
hidden_dim = config['model']['hidden_dim']
hops = config['model']['hops']
num_layers = config['model']['num_layers']
batch_size = config['model']['batch_size']
dropout = 0.3
glob_attn = 'mul'

if __name__ == '__main__':
    ast_graph = ASTGraphDataset('data')
    loader = DataLoader(ast_graph, batch_size=batch_size, shuffle=True)
    src = next(iter(loader))
    rev = next(iter(loader))
    code_field = torch.load(f'./data/field/java_node_field.pkl')
    com_field = torch.load(f'./data/field/java_nl_field.pkl')

    encoder = Encoder(code_embed_dim=code_embed_dim, com_embed_dim=com_embed_dim, edge_embed_dim=edge_embed_dim,
                      hidden_dim=hidden_dim, hops=hops, code_field=code_field, com_field=com_field)

    # Net = Model(encoder=encoder)
    encoder(src, rev, [0.2, 0.3])