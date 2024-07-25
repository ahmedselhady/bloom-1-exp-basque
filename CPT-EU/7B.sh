LANG="EUS" # language
DATA_SAMPLES="MAX" # training sample size
VOCAB_SIZE=250680 # vocab size of newly trained tokenizer
BIGS_MODEL="bigscience/bloom-7b1"
ADPT_STRATEGY="continual-pretrain"  # language adaptation strategy (train only embedding for now)
# either "replace", "overlap-replace", or "extend"
# either "replace" (for embedding strategy of "replace" and "overlap-replace") or "extend"

tokenizer_dir= # as above
tokenizer_dir="${tokenizer_dir}/tok_${BIGS_MODEL##*/}_${LANG}_oscar_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${TOK_STRATEGY}"
cache_dir="/gscratch5/users/asalem/transformers_cache/"

output_dir=... # directory to save adapted model
output_dir="${output_dir}/${BIGS_MODEL##*/}_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"
logging_dir=... # directory to log loss curves to tensorboard
logging_dir="${logging_dir}/${BIGS_MODEL##*/}_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"

mkdir -p $output_dir
mkdir -p $logging_dir

MAX_STEPS=10000
EVAL_STEPS=1000
SAVE_STEPS=1000

python ./scripts/lang_adapt/madx_run_clm.py \
    --seed 0 \
    --fp16 \
    --model_name_or_path $BIGS_MODEL \
    --cache_dir $cache_dir \


    --dataset_name HiTZ/euscrawl \
    --logging_dir $logging_dir \
    --report_to "wandb" \
    --learning_rate 1e-04 \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 4 \
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy "steps" \

    --save_steps $SAVE_STEPS \
    --save_strategy "steps" \

    --max_steps $MAX_STEPS \
    --logging_steps 100 \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --load_best_model_at_end \
    --gradient_checkpointing \
    --fp16