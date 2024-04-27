export CUDA_VISIBLE_DEVICES="$1"

export Epoch="$2"
export Dataset="$3"

base_dir=

python -m open_instruct.eval.ace.run_eval \
    --data_dir ${base_dir}/data/${Dataset}_v2_textEE \
    --valid_file ${Dataset}_valid_v2_inference.json \
    --test_file ${Dataset}_test_v2_inference.json \
    --save_dir ${base_dir}/results/GENEVA_n10_s2000_e96_6d_v2_GenData_e200_10d_5s_v2_epoch${Epoch}_${Dataset}/ \
    --model ${base_dir}/output/GENEVA_n10_s2000_e96_6d_v2_GenData_e200_10d_5s_v2/epoch_${Epoch} \
    --tokenizer models/Llama-2-7b-hf/ \
    --eval_batch_size 64

