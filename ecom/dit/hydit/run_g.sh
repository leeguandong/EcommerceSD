model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "
deepspeed "/home/image_team/image_team_docker_home/lgd/EcommerceSD/ecom/dit/hydit/train_deepspeed.py" ${params}  "$@"