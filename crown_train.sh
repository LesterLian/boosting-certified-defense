export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd)

for (( i = 1; i < 5; i++ )); do
   python ../CROWN_train.py \
   --config config/mnist_crown_large.json \
   -T 20 \
   -e 0.$i

#  for (( T = 1; T < 21; T++ )); do
#    python ../CRWON_select_certified_error.py \
#    --config config/mnist_crown_large.json \
#    -e 0.$i \
#    -T $T \
#    -m "../../828_data/crown-ibp_models/mnist_0.1_mnist_crown_large/mnist_crown_large" \
#    -l "../ada_mnist_crown_large_eps0.${i}_T20"
#  done
done
