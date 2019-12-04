sbatch -N 1 -p gpu --gres=gpu:v100-32gb:01 --mem=32GB --wrap "python test_train.py"
