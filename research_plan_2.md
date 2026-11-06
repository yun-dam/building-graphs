# Study 2. Ontology-Aware GNN-Based Link Prediction for Building Knowledge Graphs

## 1. Research Objective

Develop and evaluate a method that predicts missing relations by leveraging the structure and ontology information of building knowledge graphs.

Natural language questions and question–SPARQL pairs are not used.

- Input: observed building knowledge graph + ontology
- Output: candidate missing relations and their prediction scores
- Application: building knowledge graph completion and recommendation of candidates for data review

Example:
- Candidate relation: (Sensor_01, locatedIn, Room_203)
- Output: prediction score for that relation

The relation names above are for illustration only; actual experiments use the relations defined in the data's ontology.

## 2. Research Questions

- RQ1. Does an ontology-aware GNN outperform existing link prediction models?
- RQ2. What are the individual effects of type information, class hierarchy, and relation domain/range?
- RQ3. Does ontology information mitigate performance degradation as observed links decrease?
- RQ4. Does ontology information reduce semantically inappropriate relation predictions?

## 3. Dataset

### 3.1 Data Source

Where possible, use the original knowledge graphs and ontologies corresponding to the same 80 buildings from Build2SPARQL used in Study 1.

Before starting the study, verify the following:

1. Can the original building graphs be obtained?
2. Are class and relation definitions consistent across buildings?
3. Are the target relations observed in sufficient quantity?
4. Does the ontology provide hierarchy and domain/range information?

Question–SPARQL data alone is not used as a substitute for the original graphs.

### 3.2 Data Usage

| Data | Usage |
|---|---|
| Observed graph triples | Relation learning and evaluation |
| Node class information | Type-based features |
| Class hierarchy | Semantic hierarchy representation |
| Relation domain/range | Reflecting relation compatibility |
| Natural language questions | Not used |
| Question–SPARQL pairs | Not used |

Node features may consist of non-linguistic information such as type and structure. A natural language text encoder is not a required component.

### 3.3 Prediction Targets

After examining the actual graphs, select relations with a sufficient number of instances.

Possible relation categories:
- Sensor–Space
- Sensor–Equipment
- Equipment–Space
- Equipment–Equipment connections

Specific relations and their directionality will be finalized based on the actual ontology.

## 4. Experimental Design

### 4.1 Main Experiment: Missing Link Recovery Within the Same Building

Split each building's target links in the following ratios.

| Split | Ratio | Purpose |
|---|---:|---|
| Train | 80% | Model training |
| Validation | 10% | Hyperparameter selection |
| Test | 10% | Missing link recovery evaluation |

- Preserve the per-relation distribution where possible.
- Train the model using the training graph.
- Validation and test target links are also removed from the message-passing graph.
- If node types are retained as auxiliary information, distinguish this from type-relation prediction.
- This setting is link recovery within known buildings, not generalization to unseen buildings.

### 4.2 Extended Experiment: Generalization to Unseen Buildings

| Split | Number of Buildings |
|---|---:|
| Train | 56 |
| Validation | 8 |
| Test | 16 |

- Train the model on training buildings and evaluate on new buildings.
- For test buildings, provide part of the observed graph and predict the hidden target links.
- Specify the scope of observed information and the link hiding ratio.
- The model must handle new nodes through shared type, relation, and structural features.
- Models relying solely on fixed node ID embeddings cannot be applied as-is, so the applicability conditions of baseline models are distinguished.

### 4.3 Sparsity Experiment

Fix the test target links and progressively reduce the proportion of observed links used for training.

This evaluates the effect of using the ontology when data is scarce.

## 5. Proposed Method

Pipeline:

Observed graph + ontology
→ Construct type and hierarchy features
→ Relation-specific GNN message passing
→ Compute candidate triple scores
→ Incorporate ontology compatibility
→ Rank missing relation candidates

### 5.1 Graph Encoder

Use a relational GNN such as R-GCN to learn node representations that reflect different relation types.

### 5.2 Ontology Information Integration

- Type information: incorporate classes such as sensor, space, and equipment into node features.
- Hierarchy information: incorporate superclass–subclass relations into representation learning.
- Relation compatibility: incorporate domain/range information into the score or training loss.

Since domain/range carry inferential semantics, they are not treated as unconditional data validation rules.
Considering missing types and incompleteness, soft constraints are examined first.

### 5.3 Link Scoring and Training

Compute a score for each candidate triple (head, relation, tail).

- Positive examples: observed training triples
- Negative examples: candidate triples with the head or tail replaced
- Loss function: negative-sampling-based classification or ranking loss

Since a relation absent from the graph is not necessarily false, ensure that known positive triples are not selected as negative examples.

## 6. Comparative Experiments

| Condition | Purpose |
|---|---|
| DistMult or ComplEx | Knowledge graph embedding baseline |
| Plain relational GNN | Effect of graph structure learning |
| Ontology-aware GNN | Effect of ontology integration |
| Plain model + ontology candidate filter | Effect of simple candidate reduction |
| Ablation studies | Individual contributions of types, hierarchy, and constraints |

Main ablations:
- Remove type features
- Remove class hierarchy information
- Remove ontology compatibility loss

Control the candidate set and evaluation protocol to distinguish whether performance gains come from representation learning or from candidate filtering.

## 7. Evaluation Metrics

### Primary Metrics

- Filtered MRR
- Hits@1
- Hits@3
- Hits@10

### Additional Analyses

- Performance by relation type
- Per-building macro-averaged performance
- Violation rate against the defined ontology compatibility criteria
- Performance by observed link ratio
- Mean and variance across multiple random seeds

A predicted link is not judged to be actually true merely because it is ontology-compatible.

## 8. Data Leakage and Evaluation Controls

1. Remove validation and test links from the training and message-passing graphs.
2. Manage inverse and equivalent relations jointly so that answers are not directly revealed through them.
3. Do not use hidden links when generating graph-based features or inference results.
4. Separately analyze links derivable solely from ontology rules and those that are not.
5. Do not rely only on easy negatives that are impossible by type.
6. Apply the same evaluation candidate set and filtering criteria across all models.
7. Predictions for naturally occurring missing relations are not treated as confirmed facts without reference answers or separate review.

## 9. Expected Contributions

1. Validation of the effectiveness of ontology-integrated GNNs for building relation recovery
2. Analysis of the individual contributions of types, hierarchy, and relation constraints
3. Evaluation of performance and semantic compatibility on sparse graphs
4. A relation recommendation method for building knowledge graph completion

## 10. Relationship to Study 1

- Study 1: accesses existing building knowledge through natural language.
- Study 2: predicts missing building knowledge through the graph and ontology.

The two studies may share the same building data, but differ in input, training signal, output, and evaluation metrics.

They are first evaluated independently. Afterwards, an integrated experiment may be conducted in which relations predicted by Study 2 are verified and then provided to Study 1.
