
gpu=1

######################
# loss weight params #
######################
lr=1e-5
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
max_iter=100000
crop=768
snapshot=5000
batch=8

weights_init='/home/liuxina/Downloads/cycada_release/results/cityscapes/cityscapes_drn42-iter20000.pth'
weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='ct'
tgt='mr'
datadir='/home/liuxina/Downloads/cycada_release/data/'
dataset='/home/liuxina/Downloads/cycada_release/data/'

resdir="results/${src}_to_${tgt}/adda_sgd/${weight_share}_nolsgan_${discrim}"

# init with pre-trained cyclegta5 model
model='drn26'
baseiter=115000
#model='fcn8s'
#baseiter=100000


base_model="base_models/${model}-iter${baseiter}.pth"
outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"

# Run python script #
CUDA_VISIBLE_DEVICES=${gpu} python train_fcn_adda.py \
    ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} --gpu 0 \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --weights_init ${base_model} --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot $snapshot --num_cls 5
