# Neural Causal Inference Net (Generalization Experiment on Synthetic Dataset)


#### Training and testing the Model

1. Set the causal function idx and adversarial as Example in `run.sh`:

        python main.py --args args/NN.txt --idx=0 --w=1
    
2. Run `run.sh`

3. For other parameters and settings, check `args/NN.txt` and `config.py`.

4. For visualization, run:
        
        tensorboard --logdir=runs
