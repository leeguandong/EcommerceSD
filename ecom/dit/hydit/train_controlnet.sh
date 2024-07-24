task_flag="canny_controlnet"                                # the task flag is used to identify folders.
control_type=canny
resume=./ckpts/t2i/model/                                    # checkpoint root for resume
index_file=/path/to/your/indexfile          # index file for dataloader
results_dir=./log_EXP                                        # save root for results
batch_size=1                                                 # training batch size
image_size=1024                                              # training image resolution
grad_accu_steps=2                                            # gradient accumulation
warmup_num_steps=0                                           # warm-up steps
lr=0.0001                                                    # learning rate
ckpt_every=10000                                             # create a ckpt every a few steps.
ckpt_latest_every=5000                                       # create a ckpt named `latest.pt` every a few steps.


sh $(dirname "$0")/run_g_controlnet.sh \
    --task-flag ${task_flag} \
    --control-type ${control_type} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.03 \
    --predict-type v_prediction \
    --multireso \
    --reso-step 64 \
    --ema-to-module \
    --uncond-p 0.44 \
    --uncond-p-t5 0.44 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-fp16 \
    --use-ema \
    --ema-dtype fp32 \
    --results-dir ${results_dir} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 10 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    "$@"
    #--use-flash-attn \
#    --resume-split \
#    --resume ${resume} \