if [ ! -d "../backbone_models" ]
then
    mkdir ../backbone_models
fi
if [ ! -d "../backbone_models/bart-base" ]
then
    git lfs install
    git clone https://huggingface.co/facebook/bart-base ../backbone_models/bart-base
fi
cd textRec_datasets
python newsrec_tokenize.py
python quoterec_tokenize.py
python engagerec_tokenize.py

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 newsrec.py
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 quoterec.py
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 engagerec.py
