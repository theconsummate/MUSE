import argparse
from src.utils import load_embeddings, normalize_embeddings
from src.utils import bool_flag
import random
import torch

parser = argparse.ArgumentParser(description='checking linearity')

parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

parser.add_argument("--num_samples", type=int, default=10, help="Number of times samples should be taken.")
parser.add_argument("--model_path", type=str, default="dumped/piecewise/fourrelu/best_mapping.pth", help="path for mapping file")

# parse parameters
params = parser.parse_args()

src_dico, _src_emb = load_embeddings(params, source=True)

mapping = torch.load(params.model_path)
error = 0

for i in range(params.num_samples):
    ids = random.sample(range(1, params.max_vocab), 2)
    x1 = _src_emb[ids[0]]
    x2 = _src_emb[ids[1]]
    mx1 = mapping(x1)
    zeros = torch.zeros(mx1.shape)
    if params.cuda:
        zeros = zeros.cuda()
    error += torch.dist(mapping(x1 + x2) - mx1 - mapping(x2), zeros)

error = error/params.num_samples
print(error)
