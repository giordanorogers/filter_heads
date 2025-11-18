# LLMs Process Lists with General Filter Heads

## Weakness 4

### Claim

The compared baselines (e.g., Function Vectors, Induction Heads): how did the authors implement them? The details are not clear. The reviewer thinks some approaches (e.g., Function Vectors) mentioned here are not fit for the tasks studied in the paper inherently.

### Rebuttal

#### Function Vector Heads

##### Finding the Function Vector Heads

We computed the function vector heads following Todd et al. (2024). For each attention head in the model, we extracted mean activations from correct in-context learning (ICL) prompts. We computed the average activation for each attention head across all prompts where the model successfully performs the task. We then create corrupt prompts by shuffling the ICL examples to create prompts where inputs are paired with random outputs. While running the model on these corrupted prompts, we replace each attention head's activation with the mean activation from correct prompts. We then calculate how much this intervention increases the probability of the correct answer. We then compute the average indirect effect for each head by averaging the causal indirect effect across all tasks in the dataset, as well as all corrupted prompts for each task. We then select the top 79 heads with the highest average indirect effect head scores. We chose the 79 because this corresponded to the number of filter heads found via our DCM method. These are the top 79 heads that most consistently help recover correct task performance when their activations are patched in. The corresponding function vector is then the sum of the mean outputs from these top causal heads.

##### Validating the Function Vector Heads

We validated the function vector heads by testing them on a set of tasks provided by Todd et al.

Our validation showed that we are able to use our found function vector heads to compute function vectors which successfully enforce the behavior of the in-context learning tasks from which they are extracted.

###### Function Vector Computed with Mean Activations from Capitalize Task and Our Function Vector Heads

**Example 1:**

```python
# Baseline Clean In-Context Learning
Input Sentence: '<|endoftext|>Q: cow\nA: Cow\n\nQ: him\nA: Him\n\nQ: generous\nA: Generous\n\nQ: wolf\nA: Wolf\n\nQ: lemur\nA: Lemur\n\nQ: jog\nA:' 

Input Query: 'jog', Target: 'Jog'

ICL Prompt Top K Vocab Probs:
 [(' Jog', 0.97461), (' Run', 0.00768), (' jog', 0.00398), (' I', 0.00111), (' JO', 0.00095)] 

# Zero-Shot Test
Input Sentence: '<|endoftext|>Q: jog\nA:' 

Input Query: 'jog', Target: 'Jog'

Zero-Shot Top K Vocab Probs:
 [(' run', 0.61475), (' Jog', 0.05811), (' Run', 0.04184), (' exercise', 0.03085), (' running', 0.02638)] 

Zero-Shot+FV Vocab Top K Vocab Probs:
 [(' Jog', 0.94043), (' jog', 0.04004), (' Run', 0.00419), (' JO', 0.00214), (' J', 0.00128)]

# Natural Language Test
Input Sentence:  'The word "jog" means'
Llama: '<|begin_of_text|>The word "jog" means to run at a slow pace. It is a'
Llama+FV: '<|begin_of_text|>The word "jog" means Jog Jog Jog Jog Jog Jog Jog Jog Jog Jog'
```

**Example 2:**
```python
# Baseline Clean In-Context Learning
Input Sentence: '<|endoftext|>Q: cow\nA: Cow\n\nQ: him\nA: Him\n\nQ: generous\nA: Generous\n\nQ: wolf\nA: Wolf\n\nQ: lemur\nA: Lemur\n\nQ: lizard\nA:' 

Input Query: 'lizard', Target: 'Lizard'

ICL Prompt Top K Vocab Probs:
 [(' L', 0.99707), (' Re', 0.00083), (' lizard', 0.00069), (' \n\n', 8e-05), (' ', 8e-05)]

# Zero-Shot Test
Input Sentence: '<|endoftext|>Q: lizard\nA:' 

Input Query: 'lizard', Target: 'Lizard'

Zero-Shot Top K Vocab Probs:
 [(' rept', 0.20935), (' scale', 0.15198), (' A', 0.1012), (' Re', 0.09509), (' L', 0.08728)] 

Zero-Shot+FV Vocab Top K Vocab Probs:
 [(' L', 0.93506), (' lizard', 0.02695), (' Re', 0.00875), (' A', 0.00391), (' I', 0.00263)]

# Natural Language Test
Input Sentence:  'The word "lizard" means'
Llama: '<|begin_of_text|>The word "lizard" means "little thief" in Spanish, and the green'
Llama+FV: '<|begin_of_text|>The word "lizard" means Lizard Lizard Lizard Lizard Lizard'
```

###### Function Vector Computed with Mean Activations from Present-Past Task and Our Function Vector Heads

**Example 1:**

```python
# Baseline Clean In-Context Learning
Input Sentence: '<|endoftext|>Q: capture\nA: captured\n\nQ: merge\nA: merged\n\nQ: set\nA: set\n\nQ: jump\nA: jumped\n\nQ: hit\nA: hit\n\nQ: transform\nA:' 

Input Query: 'transform', Target: 'transformed'

ICL Prompt Top K Vocab Probs:
 [(' transformed', 0.99658), (' transform', 0.00239), (' tran', 0.00016), (' transforms', 0.00015), (' (', 0.00011)] 

# Zero-Shot Test
Input Sentence: '<|endoftext|>Q: transform\nA:' 

Input Query: 'transform', Target: 'transformed'

Zero-Shot Top K Vocab Probs:
 [(' Transform', 0.17627), (' The', 0.12402), (' Transformation', 0.06046), (' change', 0.04819), (' Change', 0.03784)] 

Zero-Shot+FV Vocab Top K Vocab Probs:
 [(' transformed', 0.82471), (' transform', 0.03513), (' transformation', 0.01985), (' Transform', 0.01608), (' changed', 0.01038)]

# Natural Language Test
Input Sentence:  'The word "transform" means'
Llama: '<|begin_of_text|>The word "transform" means to change the form or appearance of something. In'
Llama+FV: '<|begin_of_text|>The word "transform" means changed transformed. transformed\ntransformed transformed transformed transformed' 
```

**Example 2:**

```python
# Baseline Clean In-Context Learning
Input Sentence: '<|endoftext|>Q: capture\nA: captured\n\nQ: merge\nA: merged\n\nQ: set\nA: set\n\nQ: jump\nA: jumped\n\nQ: hit\nA: hit\n\nQ: operate\nA:' 

Input Query: 'operate', Target: 'operated'

ICL Prompt Top K Vocab Probs:
 [(' operated', 0.99316), (' operate', 0.00475), (' operates', 0.00055), (' operating', 0.0004), (' oper', 0.00019)] 

# Zero-Shot Test
Input Sentence: '<|endoftext|>Q: operate\nA:' 

Input Query: 'operate', Target: 'operated'

Zero-Shot Top K Vocab Probs:
 [(' to', 0.14233), (' Oper', 0.10089), (' To', 0.09186), (' An', 0.06168), (' operate', 0.05316)] 

Zero-Shot+FV Vocab Top K Vocab Probs:
 [(' operated', 0.67529), (' operate', 0.14148), (' operating', 0.03809), (' Oper', 0.03662), (' operates', 0.03284)]

# Natural Language Test
Input Sentence:  'The word "operate" means'
Llama: '<|begin_of_text|>The word "operate" means to perform a function or to manage something. In'
Llama+FV: '<|begin_of_text|>The word "operate" means to operated or operated on. operated on operated on' 
```

##### Testing the Function Vector Heads on the Filter Tasks

TODO: We tested the function vector heads on the filter tasks by...

#### Concept Heads

##### Finding the Concept Heads

We followed Feucht et al. (2025) and computed the concept heads via a causal intervention experiment.
We created special prompts with a repeating structure.
The first half of each prompt included random tokens, plus a multi-token concept (e.g., "waxwing").
The second half of each prompt included the same random tokens repeated, plus the first token of the concept.
We created two versions of each prompt.
The clean prompt has the real concept in the first half.
The corrupted prompt has different random tokens in the first half.
Then, for each attention head in the model, we patched its output from the clean prompt into the corrupted prompt, specifically at the position of the last random token in the second half (the position just before where the concept starts).
We then measured whether this increased the probability of the second token of the multi-token word.
If patching a head increased the probability for the second token of the concept, like "ax" in "waxwing" (which is tokenized as "w", "ax", "wing"), this suggests the head is carrying information about the entire concept, not just individual tokens.
We call this the concept copying score.
Heads with high concept copying scores are concept induction heads.
They are distinct from regular token induction heads, which only increase the probability for the immediate next token.

##### Validating the Concept Heads

Concept induction heads have a characteristic attention pattern wherein they attend to the final token in previous occurrences of a multitoken word which follows previous occurrences of the current word in a context.
The canonical example given in Feucht et al. (2025) uses the context "By the false azure in the windowpane By the false azure in the".
In this case of Llama-2-7b, the model on which they performed their experiment, because "windowpane" is tokenized into three parts ("window", "p" and "ane"), the concept induction heads attend to the "ane" token, as visualized below.

![Attention pattern showing strong attention toward the "ane" token.](llama_2_7b_concept_attention.png)

We performed the same experiment, this time on Llama-3.3-70B-Instruct using the concept induction heads we found for Llama-3.3-70B-Instruct.
In this model, the term "windowpane" is divided into two tokens: "window" and "pane".
Below is the attention pattern we see, where the concept heads attend most heavily to the "pane" token.

![Attention pattern showing strong attention toward the "pane" token.](llama_3_70b_instruct_concept_attention.png)

##### Testing the Concept Heads on the Filter Tasks

TODO: We tested the concept heads on the filter tasks by...

