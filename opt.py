import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="DBLP")
parser.add_argument('--model', type=str, default="gcn+sem+tsf+ExAtt")

parser.add_argument("--d_model", type=int, default=128, help="transformer d_model.")
parser.add_argument("--sem_k", type=int, default=5, help="semantic adj nums.")
parser.add_argument("--cpu", type=str, default='False')
parser.add_argument("--gpuid", type=str, default='0')
parser.add_argument("--n_epoch", type=int, default=3000)
parser.add_argument("--early_stopping", type=int, default=300)
parser.add_argument("--eAtt_d", type=int, default=128)
parser.add_argument("--tsf_layer", type=int, default=1)
parser.add_argument("--tsf_head", type=int, default=8)
parser.add_argument("--tsf_drop", type=float, default=0.5)
parser.add_argument("--ffn_dim", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--gcn_layer", type=int, default=2)
parser.add_argument("--train_percent", type=float, default=0.6)
parser.add_argument("--val_percent", type=float, default=0.2)
parser.add_argument('--param', type=str, default='local:dblp.json')

args = parser.parse_args()