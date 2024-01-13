from typing import List, Tuple
from uuid import UUID, uuid4

import torch
import torch.nn.functional as F

from modeling_enc_t5 import EncT5ForSequenceClassification
from tokenization_enc_t5 import EncT5Tokenizer

import asyncio


def tart_rerank(queries: List[str], contents_list: List[List[str]],
                scores_list: List[List[float]], ids_list: List[List[UUID]],
                instruction: str = "Find passage to answer given question") -> List[Tuple[List[str]]]:
    model_name = "facebook/tart-full-flan-t5-xl"
    model = EncT5ForSequenceClassification.from_pretrained(model_name)
    tokenizer = EncT5Tokenizer.from_pretrained(model_name)
    # Run async tart_rerank_pure function
    tasks = [tart_rerank_pure(query, contents, scores, ids, model, tokenizer, instruction) \
             for query, contents, scores, ids in zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results


async def tart_rerank_pure(query: str, contents: List[str], scores: List[float], ids: List[UUID],
                           model, tokenizer, instruction: str) -> Tuple[List[str]]:
    instruction_queries: List[str] = ['{0} [SEP] {1}'.format(instruction, query) for _ in range(len(contents))]
    features = tokenizer(instruction_queries, contents, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**features).logits
        normalized_scores = [float(score[1]) for score in F.softmax(scores, dim=1)]

    contents_ids_scores = list(zip(contents, ids, normalized_scores))

    sorted_contents_ids_scores = sorted(contents_ids_scores, key=lambda x: x[2], reverse=True)

    return tuple(map(list, zip(*sorted_contents_ids_scores)))


if __name__ == '__main__':
    queries_example = ["What is the capital of France?",
                       "How many members are in Newjeans?"]
    contents_example = [["NomaDamas is Great Team", "Paris is the capital of France.", "havertz is suck at soccer"],
                        ["i am hungry", "LA is a country in the United States.", "Newjeans has 5 members."]]
    ids_example = [[uuid4() for _ in range(len(contents_example[0]))], [uuid4() for _ in range(len(contents_example[1]))]]
    scores_example = [[0.1, 0.8, 0.1],[0.1, 0.2, 0.7]]

    result = tart_rerank(queries_example, contents_example, scores_example, ids_example)

    first_contents = result[0][0]
    second_contents = result[1][0]

    # result[0]:
    # [(['Paris is the capital of France.', 'havertz is suck at soccer', 'NomaDamas is Great Team'],
    # [UUID('67975b30-a212-4a5b-9e76-89a4a3e9f37a'), UUID('ae3a1a94-a78c-4a0a-8835-960f42dc154f'),
    # UUID('51642d38-ecf6-426e-9f86-ace92a527888')], [0.5542998909950256, 0.17034107446670532, 0.1374085247516632]),

    # result[1]:
    # (['Newjeans has 5 members.', 'LA is a country in the United States.', 'i am hungry'],
    # [UUID('62911b03-98c5-4425-900c-99c86bb44121'), UUID('04585fa4-0e0e-4dc4-ba69-712cc94b5bc4'),
    # UUID('9b94c8c5-492a-4128-99e4-87881ea6b6a6')], [0.5379835367202759, 0.13214750587940216, 0.11529212445020676])]

    print(first_contents)
    # ['Paris is the capital of France.', 'NomaDamas is Great Team', 'havertz is suck at soccer']
    assert first_contents[0] == "Paris is the capital of France."

    print(second_contents)
    # ['Newjeans has 5 members.', 'LA is a country in the United States.', 'i am hungry']
    assert second_contents[0] == "Newjeans has 5 members."
