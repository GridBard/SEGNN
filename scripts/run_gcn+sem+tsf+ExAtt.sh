python main.py --gpuid 0 --name Cora --param local:cora.json --model gcn+sem+tsf+ExAtt --train_percent 0.6 --val_percent 0.2 --eAtt_d 512 --tsf_layer 1 --tsf_head 4 --tsf_drop 0.2 --gcn_layer 2 --ffn_dim 512 --hidden_dim 512 --sem_k 6

python main.py --gpuid 0 --name DBLP --param local:dblp.json --model gcn+sem+tsf+ExAtt --train_percent 0.6 --val_percent 0.2 --eAtt_d 128 --tsf_layer 1 --tsf_head 8 --tsf_drop 0.5 --gcn_layer 2 --ffn_dim 128 --hidden_dim 128 --sem_k 4


