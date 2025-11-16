# LLMs Process Lists with General Filter Heads

## Weakness 4

### Claim

The compared baselines (e.g., Function Vectors, Induction Heads): how did the authors implement them? The details are not clear. The reviewer thinks some approaches (e.g., Function Vectors) mentioned here are not fit for the tasks studied in the paper inherently.

### Rebuttal

#### Function Vector Heads

##### Finding the Function Vector Heads

TODO: We followed Todd et al. (2024) and computed the function vector heads by...

##### Validating the Function Vector Heads

We validated the function vector heads by testing them on a suite of tasks provided by Todd et al.

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
GPT-J: '<|begin_of_text|>The word "jog" means to run at a slow pace. It is a'
GPT-J+FV: '<|begin_of_text|>The word "jog" means Jog Jog Jog Jog Jog Jog Jog Jog Jog Jog'
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
GPT-J: '<|begin_of_text|>The word "lizard" means "little thief" in Spanish, and the green'
GPT-J+FV: '<|begin_of_text|>The word "lizard" means Lizard Lizard Lizard Lizard Lizard'
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
GPT-J: '<|begin_of_text|>The word "transform" means to change the form or appearance of something. In'
GPT-J+FV: '<|begin_of_text|>The word "transform" means changed transformed. transformed\ntransformed transformed transformed transformed' 
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
GPT-J: '<|begin_of_text|>The word "operate" means to perform a function or to manage something. In'
GPT-J+FV: '<|begin_of_text|>The word "operate" means to operated or operated on. operated on operated on' 
```

##### Testing the Function Vector Heads on the Filter Tasks

TODO: We tested the function vector heads on the filter tasks by...

#### Concept Heads

##### Finding the Concept Heads

TODO: We followed Feucht et al. (2025) and computed the concept heads by...

##### Validating the Concept Heads

##### Testing the Concept Heads on the Filter Tasks

TODO: We tested the concept heads on the filter tasks by...

#### Induction Heads

TODO: Explain how we implemented the induction head baseline.

##### Finding the Induction Heads

TODO: We followed Feucht et al. (2025) and computed the induction heads by...

##### Validating the Induction Heads

##### Testing the Induction Heads on the Filter Tasks

TODO: We tested the induction heads on the filter tasks by...


