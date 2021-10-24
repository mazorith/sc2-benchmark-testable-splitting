BASE_NAME=${1}
MAX_QUALITY=${2}

if [ $# -ne 2 ]; then
  echo "Illegal number of arguments"
  exit 2
fi

for quality in $(seq 1 1 ${MAX_QUALITY});
do
  json_str='{"models": {"compression": {"params": {"quality": '
  json_str+=${quality}
  json_str+='}}}}'
  pipenv run python src/image_classification/input_compression.py \
  --config configs/ilsvrc2012/input_compression/${BASE_NAME}.yaml \
  --log log/input_compression/${BASE_NAME}-quality${quality}.txt \
  --json "${json_str}"
done
