# causal-relations-between-representations
```
@inproceedings{
  wang2022ncinet,
  title={Do learned representations respect causal relationships?},
  author={Lan Wang and Vishnu Naresh Boddeti},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
## Overview

<img width="892" alt="image" src="https://user-images.githubusercontent.com/16111637/163293826-2edf2613-54c1-414b-b30c-c4070b580dfc.png">

NCINet is an approach for observational causal discovery from high-dimensional data. It is trained purely on synthetically generated representations
and can be applied to real representations. It's also be applied to analyze the effect on the underlying causal relation between learned representations induced by various design choices in representation learning.
## Dataset
<img width="500" alt="image" src="https://user-images.githubusercontent.com/16111637/177241541-6bfe567c-6212-4111-a92b-fe55ff74e8a8.png">
CASIA-Webface facial attribute annotations: color of hair, eyes, eye wear, facial hair, forehead, mouth, smiling, gender.

## How to evalute NCInet (3Dshape)
Causal consistency on 6 causal pair graphs.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/16111637/177241405-b93e41de-11f8-4166-98d5-3711077bf7c5.png">


## How to evalute NCInet (CASIA-Webface)
Causal consistency on 6 causal pair graphs.

<img width="700" alt="image" src="https://user-images.githubusercontent.com/16111637/177241459-a1ab8793-c067-4271-8499-52155d6c0d99.png">

# Neural Causal Inference Net (Generalization Experiment on Synthetic Dataset)


#### Training and testing the Model

1. `cd ./NCINet`, set the causal function idx and adversarial as Example in `run.sh`:

        python main.py --args args/NN.txt --idx=0 --w=1
    
2. Run `run.sh`

3. For other parameters and settings, check `args/NN.txt` and `config.py`.

4. For visualization, run:
        
        tensorboard --logdir=runs
