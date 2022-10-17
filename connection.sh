salloc --time=3:0:0 --ntasks=1 --cpus-per-task=4 --mem=64G --gres=gpu:1 --account=rrg-emilios srun $VIRTUAL_ENV/bin/notebook.sh
