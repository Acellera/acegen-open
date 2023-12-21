# AceGen-Open 

## RL environments

We provide 2 types of environments for reinforcement learning:
    
    single step: SingleStepDeNovoEnv
        - each episode lasts for a single step
        - reset provides a initial token as observation
        - step expects a sequence of tokens anding with a final token as action

    multi step: MultiStepDeNovoEnv
        - each episode lasts for a sequence of steps
        - reset provides a initial token as observation
        - each step expects a single token as action, and returns a single token as observation (the previous action)

## Transforms
    
    reward transform: SMILESReward
        - 
        - 
        - 

    penalised repeated transform: PenaliseRepeatedSMILES
        - 
        - 
        -
    
    burn in transform: BurnInTransform
        - 
        - 
        -

## Vocabularies

    Vocabulary: Vocabulary
        - is a protocol for a vocabulary, that defines the interface for a any vocabulary class
        - essentially a vocabulary has 2 methods: encode and decode
        - encode takes a string and returns a sequence of integers
        - decode takes a sequence of integers and returns a string
