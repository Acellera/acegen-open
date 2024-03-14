# Tutorial: Using PromptSMILES with AceGen

---

This tutorial will walk through the application of PromptSMILES in AceGen.
For further information about PromptSMILES, please refer to the [publication](https://chemrxiv.org/engage/chemrxiv/article-details/65e718eee9ebbb4db9f21886) and the [GitHub](https://github.com/compsciencelab/PromptSMILES).

## What is PromptSMILES?

PromptSMILES is a simple solution to conducting constrained molecule generation i.e., conduct de novo molecule generation given some chemical sub-structure(s). Firstly, SMILES-based chemical language models can complete a given prompt, for example, given the **prompt** in bold the model will likely predict the next token to be "1" to form a ring closure **"c1ccccc**1". This can be repeated at any "attachment point" we desire, the SMILES string just needs to be rearranged such that this attachment point comes last in the SMILES representation and can then be auto-completed by our model. PromptSMILES is a tool to automate this re-arrangement.

PromptSMILES requires a chemical structure(s) with labelled attachment points which it then rearranges into the different prompts required by a chemical language model. PromptSMILES interprets an * (or dummy atom) as this label, which should be inserted as a branch point i.e., (*). 

**Scaffold decoration example** Given we want to elaborate a benzene "c1ccccc1" at two meta-orientated substitution points, we would supply PromptSMILES with "c1(\*)cc(\*)ccc1", which would in turn provide two prompts for completion by a model.

**c1ccccc1**Cl  
**c1ccc(Cl)cc1**Br

**Superstructure generation example** Sometimes we don't know, or don't have desired attachement points. Then, you can simply specify every atom with available valence as an attachment point, for example, "c1(\*)c(\*)c(\*)c(\*)c(\*)c1(\*)". Some sampled tokens may simply be the stop token e.g., \<EOS\>.

**c1ccccc1**Cl  
**c1cc(Cl)ccc1**\<EOS\>  
**c1c(Cl)cccc1**F  
**c1(Cl)cc(F)ccc1**\<EOS\>  
...  
**c1(Cl)cc(F)ccc1**  

**Fragment linking example** For fragment linking, different chemical substructures with one attachment point each must be supplied, this is notated by seperation with a "." in the SMILES string. For e.g., cyclo-propyl and pyridine could be "C1CC1(\*).c1(\*)ccncc1". One fragment point is selected, and completed, then the other is concatenated. As the model can't observe both fragments, this approach is not as principled as scaffold decoration, but works well with RL.

**C1CC1**CCOCC  
C1CC1CCOCC + c1ccncc1  
C1CC1CCOCCc1ccncc1

Alternatively, we could insert the second fragment at every point in the generated linker, and evaluate the likelihood (more specifically the negative log-likelihood, where lower is better) of the model generating this de novo. Moreover, this approach can now be extended to more than two fragments.

**C1CC1**CCOCC  
**C1CC1C(c1ccncc1)COCC**  NLL=24.3  
**C1CC1CC(c1ccncc1)OCC**  NLL=25.6  
**C1CC1CCO(c1ccncc1)CC**  NLL=60.8  
**C1CC1CCOC(c1ccncc1)C**  NLL=23.2  
**C1CC1CCOCC(c1ccncc1)**  NLL=20.2 (selected insertion point)  

## Using PromptSMILES via configuration file

Several PromptSMILES can be accepted in the config file. 

See the following example for the meta-substituted benzene.

```YAML
promptsmiles: "c1c(*)cc(*)cc1"
promptsmiles_optimize: True
promptsmiles_shuffle: True
promptsmiles_multi: False
```

Or the following example for linking two fragments seperated by ".".

```YAML
promptsmiles: "C1CC1(\*).c1(\*)ccncc1"
promptsmiles_optimize: True
promptsmiles_shuffle: True
promptsmiles_multi: False
```

**optimize**: There is usually more than one potential rearrangement of a SMILES string, setting this to `True` parses each rearrangement to the chemical language model to find most likely generated arrangement (i.e., preferred by the model).  
**shuffle**: Setting shuffle to `True` means that either the selected attachment point or selected fragment is randomly selected, such that a batch contains different many different prompts.  
**multi**: This is a parameter specific to the combination with RL. If set to `True`, an RL update will be run for every intermediate prompt completion, otherwise, only the last fully complete molecule will be used (for fragment linking, this is actually the first prompt completion).  

For further understanding of performance with these different parameters see the [publication](https://chemrxiv.org/engage/chemrxiv/article-details/65e718eee9ebbb4db9f21886).