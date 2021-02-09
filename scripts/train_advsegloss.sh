set -ex
# Unpaired Training
# train using binary loss
python train.py --name ade20k_hed_advsegloss_binary --dataroot ./datasets/ade20k_hed --model cycle_gan --segmentation --segmentation_output "binary" --direction "AtoB" --dataset_mode "unaligned"
# train using multi class loss
python train.py --name ade20k_hed_advsegloss_binary --dataroot ./datasets/ade20k_hed --model cycle_gan --segmentation --segmentation_output "multi_class" --direction "AtoB" --dataset_mode "unaligned"
# train using both
python train.py --name ade20k_hed_advsegloss_both --dataroot ./datasets/ade20k_hed --model cycle_gan --segmentation --segmentation_output "both" --direction "AtoB" --dataset_mode "unaligned"

# Paired Training
# train using binary loss
python train.py --name ade20k_hed_pix2pix_advsegloss_binary --dataroot ./datasets/ade20k_hed_pair --model pix2pix --segmentation --segmentation_output "binary" --direction "AtoB" --dataset_mode "aligned"
# train using multi class loss
python train.py --name ade20k_hed_pix2pix_advsegloss_binary --dataroot ./datasets/ade20k_hed_pair --model pix2pix --segmentation --segmentation_output "multi_class" --direction "AtoB" --dataset_mode "aligned"
# train using both
python train.py --name ade20k_hed_pix2pix_advsegloss_both --dataroot ./datasets/ade20k_hed_pair --model pix2pix --segmentation --segmentation_output "both" --direction "AtoB" --dataset_mode "aligned"
