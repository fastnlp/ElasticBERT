export GLUE_DIR=../glue_data
export TASK_NAME=RTE
export CUDA_VISIBLE_DEVICES=0

python ./run_glue.py \
  --model_name_or_path fnlp/elasticbert-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_infer \
  --do_lower_case \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --log_dir logs/elasticbert-base/glue/$TASK_NAME \
  --output_dir ./ckpts/elasticbert-base/glue/$TASK_NAME \
  --num_hidden_layers 12 \
  --num_output_layers 1 \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --weight_decay 0.1 \
  --logging_steps 50\
  --early_stop_steps 10 \
  --num_train_epochs 10  \
  --warmup_rate 0.06 \
  --evaluate_during_training \
  --overwrite_output_dir \
