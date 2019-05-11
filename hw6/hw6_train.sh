python3 hw6redo.py --batch 128 --epoch 50 --word_dim 200 --seq_len 40 $1 $2 $3 $4
python3 hw6redo_gru.py --batch 128 --epoch 50 --word_dim 200 --seq_len 40 $1 $2 $3 $4
python3 hw6redo_new.py --batch 128 --epoch 50 --word_dim 200 --seq_len 40 --num_layers 2 -hidden_dim 250 $1 $2 $3 $4
python3 hw6redo_new_gru.py --batch 128 --epoch 100 --word_dim 200 --seq_len 40 --num_layers 2 -hidden_dim 250 --lr 0.0005 --patience 15 $1 $2 $3 $4
