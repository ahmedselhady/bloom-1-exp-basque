CKPT=25000
# M="/home/zhengxinyong/outputs/bloom-3b_ru_continual-pretrain_100000samples_-1vocab_original/"
M="bigscience/bloom-3b"

python3 /home/zhengxinyong/lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer=$M,pretrained=$M \
--device cuda:3 \
--tasks xnli_ru \
--no_cache

# --model_args tokenizer="bs-la/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original_bsz4",pretrained="bs-la/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original_bsz4" \
# --model_args tokenizer="bigscience/bloom-7b1",pretrained="bigscience/bloom-7b1" \