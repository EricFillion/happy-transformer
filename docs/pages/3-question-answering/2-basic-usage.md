---
title: Usage
parent: Question Answering
nav_order: 2
layout: page
permalink: /question-answering/usage/
---

## Question Answering Basic Usage

### answer_question()

Inputs: 
1. context (string): background information, which contains a text-span that is the answer 
2. question (string): the question that will be asked 
3. top_k (int): the number of results that will be returned (default=1)

Returns: 
 A list of a objects with fields: "answer", "score", "start" and "end." 
The list is in descending order by score

#### Example 3.1:
```python
from happytransformer import HappyQuestionAnswering
# --------------------------------------#
happy_qa = HappyQuestionAnswering()
result = happy_qa.answer_question("Today's date is January 10th, 2021", "What is the date?")
print(type(result))  # <class 'list'>
print(result)  # [QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34)]
print(type(result[0]))  # <class 'happytransformer.happy_question_answering.QuestionAnsweringResult'>
print(result[0])  # QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34)
print(result[0].answer)  # January 10th, 2021
```

#### Example 3.2:
```python
from happytransformer import HappyQuestionAnswering
# --------------------------------------#
happy_qa = HappyQuestionAnswering()
result = happy_qa.answer_question("Today's date is January 10th, 2021", "What is the date?", top_k=2)
print(type(result))  # <class 'list'>
print(result)  # [QuestionAnsweringResult(answer='January 10th, 2021', score=0.9711642265319824, start=16, end=34), QuestionAnsweringResult(answer='January 10th', score=0.017306014895439148, start=16, end=28)]
print(result[1].answer)  # January 10th

```