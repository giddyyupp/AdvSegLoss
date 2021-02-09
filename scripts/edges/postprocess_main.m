%%% parameters
% hed_mat_dir: the hed mat file directory (the output of 'batch_hed.py')
% edge_dir: the output HED edges directory
% image_width: resize the edge map to [image_width, image_width]
% threshold: threshold for image binarization (default 25.0/255.0)
% small_edge: remove small edges (default 5)

hed_mat_dir = ''
edge_dir = ''
image_width = 256
threshold = 25.0/255.0
small_edge = 5

PostprocessHED(hed_mat_dir, edge_dir, image_width, threshold, small_edge)