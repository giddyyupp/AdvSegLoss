source_dir=$1
fake_dir=$2

python -m pytorch_fid ${source_dir} ${fake_dir}