import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import yaml
from Modules import Encoder, Decoder, Attention
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
max_nl_len = config['model']['max_nl_len']
dec_dropout = 0.3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ast_graph = ASTGraphDataset('data')
loader = DataLoader(ast_graph, batch_size=batch_size, shuffle=True)
code_field = torch.load(f'./data/field/java_node_field.pkl')
com_field = torch.load(f'./data/field/java_nl_field.pkl')

attn = Attention(hidden_dim, hidden_dim)
encoder = Encoder(code_embed_dim=code_embed_dim, com_embed_dim=com_embed_dim, edge_embed_dim=edge_embed_dim,
                  hidden_dim=hidden_dim, hops=hops, code_field=code_field, com_field=com_field)
decoder = Decoder(com_field=com_field, embed_dim=com_embed_dim, hidden_dim=hidden_dim, attention=attn,
                  dropout=dec_dropout)
model = Model(encoder=encoder, decoder=decoder, out_dim=len(com_field.vocab), device=device)


def train(model, iterator, optimizer, loss):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch
        trg = batch.y
        pred = model(src, trg)

        loss = loss(pred, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

if __name__ == '__main__':
    src = next(iter(loader))
    rev = next(iter(loader))
    model(src, rev, [0.2, 0.3], src.y.reshape((batch_size, max_nl_len)), teacher_forcing=True)