
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
batch=16

weights_init="/home/chenxi/cycada_release_0717/results/cityscapes/cityscapes_drn42-iter20000.pth"
#weights_init="/home/chenxi/cycada_release_0717/base_models/drn26_cycada_cyclegta2cityscapes.pth"
weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='ct'
tgt='mr'
datadir='/home/chenxi/cycada_release_0717/data/'
dataset='/home/chenxi/cycada_release_0717/data/'

resdir="results/${src}_to_${tgt}/adda_sgd/${weight_share}_nolsgan_${discrim}"

# init with pre-trained cyclegta5 model
model='drn42'
baseiter=iter20000
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
    --weights_init ${weights_init} --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot $snapshot --num_cls 5
