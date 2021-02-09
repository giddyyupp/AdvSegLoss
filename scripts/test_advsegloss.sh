set -ex
# Unpaired Testing
# test binary model
python test.py --name ade20k_hed_advsegloss_binary --dataroot ./datasets/ade20k_hed --model test --segmentation --segmentation_output "binary" --direction "AtoB" --dataset_mode "unaligned"
# test multi class model
python test.py --name ade20k_hed_advsegloss_binary --dataroot ./datasets/ade20k_hed --model test --segmentation --segmentation_output "multi_class" --direction "AtoB" --dataset_mode "unaligned"
# test both model
python test.py --name ade20k_hed_advsegloss_both --dataroot ./datasets/ade20k_hed --model test --segmentation --segmentation_output "both" --direction "AtoB" --dataset_mode "unaligned"

# Paired Testing
# test binary model
python train.py --name ade20k_hed_pix2pix_advsegloss_binary --dataroot ./datasets/ade20k_hed_pair --model pix2pix --segmentation --segmentation_output "binary" --direction "AtoB" --dataset_mode "aligned"
# test multi class model
python train.py --name ade20k_hed_pix2pix_advsegloss_binary --dataroot ./datasets/ade20k_hed_pair --model pix2pix --segmentation --segmentation_output "multi_class" --direction "AtoB" --dataset_mode "aligned"
# test both model
python train.py --name ade20k_hed_pix2pix_advsegloss_both --dataroot ./datasets/ade20k_hed_pair --model pix2pix --segmentation --segmentation_output "both" --direction "AtoB" --dataset_mode "aligned"
