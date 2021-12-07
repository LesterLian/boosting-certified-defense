export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)

for (( i = 1; i < 5; i++ )); do
  for (( T = 1; T < 9; T++ )); do
    python ../eval_certified_error.py \
    --config config/mnist_crown_large.json \
    -e 0.$i \
    -T $T \
    -m "mnist_crown_large/0.${i}" \
    -l "../ada_train_mnist_crown_large_eps0.${i}_T8"
  done
done
