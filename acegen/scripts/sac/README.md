The intended use of this code is the following:

1. After pretraining the actor/policy, the critic can be pretrained too using the following command:
    
   ```python
    python pretrain_sac.py

2. Once both the actor and the critic are pretrained, the following command can be used to generate a library of molecules:

   ```python
    python sac.py

## Note:
These scripts have not been sufficiently tested and are not guaranteed to be functional.
While the code runs, the results obtained are not comparable in  performance to the results
obtained with on-policy methods. Nonetheless, the code is provided for reference.
