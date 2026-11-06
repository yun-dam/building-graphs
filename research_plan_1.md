# Study 1. Ontology- and Instance-Grounded Question Answering over Building Knowledge Graphs

## 1. Objective

Develop and evaluate a method that links natural-language questions to the ontology and actual instances of a building knowledge graph, generates SPARQL, and returns accurate answers.

The core challenge is to correctly map the expressions in a question to the classes, relations, and instances used in the target building.

- Input: natural-language question + building knowledge graph + ontology
- Output: SPARQL query and an answer based on its execution results
- Application: natural-language building information retrieval and question answering

## 2. Research Questions

- RQ1. Does the proposed method produce more accurate answers than a baseline NL-to-SPARQL method?
- RQ2. How much do ontology retrieval and instance grounding each contribute to the performance gain?
- RQ3. Does the method generalize to buildings not seen during development?
- RQ4. How does performance vary with query pattern and BGP complexity?

## 3. Dataset

### 3.1 Data Source

We use 80 buildings selected from Build2SPARQL, together with the natural-language questions and reference SPARQL queries associated with them.

According to the provided Table 12, the full Build2SPARQL dataset has the following scale:

| Item | Full dataset |
|---|---:|
| Buildings | 201 |
| SPARQL queries | 6,136 |
| Natural-language questions | 30,680 |
| Pattern types | 6 |
| Avg. BGP triples | 3.9 |
| Unique ontology terms | 232 |
| Q4 instance-grounding rate | 79% |

These figures are statistics for the full dataset. Statistics for the selected 80 buildings will be computed separately.

### 3.2 Data Used

| Data | Purpose |
|---|---|
| Building knowledge graphs | SPARQL execution and information retrieval |
| Ontology | Interpreting classes, relations, and hierarchy |
| Natural-language questions | System input |
| Reference SPARQL | Ground truth for query and answer evaluation |
| Reference SPARQL execution results | Gold answers for comparison |

The existing natural-language questions serve as the primary evaluation input; generating new questions is not a required task.

### 3.3 Selection and Quality Checks

1. Verify that the source graphs are accessible.
2. Verify the links between questions, SPARQL queries, and target buildings.
3. Verify that the reference queries execute successfully.
4. Select 80 buildings considering graph size, pattern distribution, ontology coverage, etc.
5. Record excluded data and the reasons for exclusion.

### 3.4 Building-Level Split

| Split | Buildings | Purpose |
|---|---:|---|
| Train/Dev | 56 | Model training or few-shot example construction |
| Validation | 8 | Prompt and configuration selection |
| Test | 16 | Evaluation on unseen buildings |

- If no model is trained directly, the train/dev data is used to build examples and retrieval resources.
- Questions and paraphrases derived from the same reference query must not be mixed across splits.
- Test questions, reference SPARQL, and gold answers are not used for development or configuration selection.
- Access to the test buildings' graphs and ontology at inference time is permitted.
- The building-level split evaluates generalization to unseen buildings, which is distinct from generalization to unseen query templates.

## 4. Proposed Method

Pipeline:

Natural-language question
→ Retrieve relevant ontology elements
→ Building-specific instance grounding
→ SPARQL generation
→ Query validation and execution
→ Answer return

### 4.1 Ontology Retrieval

Select the classes, relations, and hierarchical information relevant to the question.

### 4.2 Instance Grounding

Link mentions of spaces, sensors, equipment, etc. in the question to actual identifiers in the graph.

Example: "Room 203" → the corresponding space instance in that building's graph

### 4.3 SPARQL Generation

Generate SPARQL using the question, the relevant ontology information, and the grounded instances.

### 4.4 Validation and Execution

Check for syntax errors, use of non-existent identifiers, etc., then execute the query.
If a query repair step is used, record the number of repairs and their cost.

## 5. Experimental Comparison

All conditions use the same base language model and execution environment.

| Condition | Configuration |
|---|---|
| Baseline | Question + ontology schema |
| Few-shot | Baseline + question–query examples from the dev data |
| Proposed | Ontology retrieval + instance grounding + generation/validation |
| Ablation | Proposed method with individual components removed |

Main ablations:
- Remove ontology retrieval, or replace it with the full schema
- Remove instance grounding
- Remove query validation/repair

BuildingQA is a dataset-level point of comparison and is not treated as a system performance baseline.

## 6. Evaluation Metrics

### Primary Metrics

- Execution-result exact-match accuracy
- Answer-set Precision, Recall, and F1

### Secondary Metrics

- SPARQL syntactic validity rate
- Execution success rate
- Instance linking accuracy
- Performance by query pattern
- Performance by BGP complexity
- Response time and model usage cost

### Reporting

- Evaluate primarily on execution results rather than SPARQL string matching.
- Report both question-level (micro) averages and building-level macro averages.
- Compute confidence intervals via building-level resampling (bootstrap).
- Categorize errors into ontology selection, instance linking, query structure, execution errors, etc.

## 7. Expected Contributions

Using the existing Build2SPARQL natural-language questions and 80 buildings, this study will validate:

1. The accuracy improvement from ontology and instance grounding in question answering
2. Generalization performance on unseen buildings
3. Performance and error characteristics across query patterns and complexity levels

The focus of this study is not building a new question dataset, but accurate and generalizable question answering over existing questions for buildings.
