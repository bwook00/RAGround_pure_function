from typing import List, Tuple
from uuid import UUID, uuid4

import torch
import asyncio

from transformers import T5Tokenizer, T5ForConditionalGeneration


prediction_tokens = {
    'castorini/monot5-base-msmarco': ['▁false', '▁true'],
    'castorini/monot5-base-msmarco-10k': ['▁false', '▁true'],
    'castorini/monot5-large-msmarco': ['▁false', '▁true'],
    'castorini/monot5-large-msmarco-10k': ['▁false', '▁true'],
    'castorini/monot5-base-med-msmarco': ['▁false', '▁true'],
    'castorini/monot5-3b-med-msmarco': ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco-10k': ['▁false', '▁true'],
    'unicamp-dl/mt5-base-en-msmarco': ['▁no', '▁yes'],
    'unicamp-dl/ptt5-base-pt-msmarco-10k-v2': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-pt-msmarco-100k-v2': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2': ['▁não', '▁sim'],
    'unicamp-dl/mt5-base-en-pt-msmarco-v2': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v2': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-en-pt-msmarco-v1': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v1': ['▁no', '▁yes'],
    'unicamp-dl/ptt5-base-pt-msmarco-10k-v1': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-pt-msmarco-100k-v1': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não', '▁sim'],
    'unicamp-dl/mt5-3B-mmarco-en-pt': ['▁', '▁true'],
    'unicamp-dl/mt5-13b-mmarco-100k': ['▁', '▁true'],
}


def mono_t5_rerank(queries: List[str], contents_list: List[List[str]],
                   scores_list: List[List[float]], ids_list: List[List[UUID]],
                   model_name: str = 'castorini/monot5-3b-msmarco-10k') -> List[Tuple[List[str]]]:
    # Load the tokenizer and model from the pre-trained MonoT5 model
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
    # Retrieve the tokens used by the model to represent false and true predictions
    token_false, token_true = prediction_tokens[model_name]
    token_false_id = tokenizer.convert_tokens_to_ids(token_false)
    token_true_id = tokenizer.convert_tokens_to_ids(token_true)
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run async mono_t5_rerank_pure function
    tasks = [mono_t5_rerank_pure(query, contents, scores, ids, model, device, tokenizer, token_false_id, token_true_id) \
             for query, contents, scores, ids in zip(queries, contents_list, scores_list, ids_list)]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results


async def mono_t5_rerank_pure(query: str, contents: List[str], scores: List[float], ids: List[UUID],
                              model, device, tokenizer, token_false_id, token_true_id) -> Tuple[List[str]]:
    """
    Rerank a list of contents based on their relevance to a query using MonoT5.
    :param query:
    :param contents:
    :param scores:
    :param ids:
    :param model:
    :param device:
    :param tokenizer:
    :param token_false_id:
    :param token_true_id:
    :return: tuple of lists containing the reranked contents, ids, and scores
    """
    model.to(device)

    # Format the input for the model by combining each content with the query
    input_texts = [f'Query: {query} Document: {content}' for content in contents]
    # Tokenize the input texts and prepare for model input
    input_encodings = tokenizer(input_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(
        device)

    # Generate model predictions without updating model weights
    with torch.no_grad():
        outputs = model.generate(input_ids=input_encodings['input_ids'],
                                 attention_mask=input_encodings['attention_mask'],
                                 output_scores=True,
                                 return_dict_in_generate=True)

    # Extract logits for the 'false' and 'true' tokens from the model's output
    logits = outputs.scores[-1][:, [token_false_id, token_true_id]]
    # Calculate the softmax probability of the 'true' token
    probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]  # Get the probability of the 'true' token

    # Create a list of tuples pairing each content with its relevance probability
    content_probs_pairs = list(zip(contents, ids, probs.tolist()))

    # Sort the list of pairs based on the relevance score in descending order
    sorted_content_probs_pairs = sorted(content_probs_pairs, key=lambda x: x[2], reverse=True)

    return tuple(map(list, zip(*sorted_content_probs_pairs)))


if __name__ == '__main__':
    queries_example = ["What is the capital of France?",
                       "How many members are in Newjeans?"]
    contents_example = [["NomaDamas is Great Team", "Paris is the capital of France.", "havertz is suck at soccer"],
                        ["i am hungry", "LA is a country in the United States.", "Newjeans has 5 members."]]
    ids_example = [[uuid4() for _ in range(len(contents_example[0]))], [uuid4() for _ in range(len(contents_example[1]))]]
    scores_example = [[0.1, 0.8, 0.1],[0.1, 0.2, 0.7]]

    result = mono_t5_rerank(queries_example, contents_example, scores_example, ids_example)

    first_contents = result[0][0]
    second_contents = result[1][0]

    # [(['Paris is the capital of France.', 'havertz is suck at soccer', 'NomaDamas is Great Team'],
    # [UUID('67975b30-a212-4a5b-9e76-89a4a3e9f37a'), UUID('ae3a1a94-a78c-4a0a-8835-960f42dc154f'),
    # UUID('51642d38-ecf6-426e-9f86-ace92a527888')], [0.5542998909950256, 0.17034107446670532, 0.1374085247516632]),
    # (['Newjeans has 5 members.', 'LA is a country in the United States.', 'i am hungry'],
    # [UUID('62911b03-98c5-4425-900c-99c86bb44121'), UUID('04585fa4-0e0e-4dc4-ba69-712cc94b5bc4'),
    # UUID('9b94c8c5-492a-4128-99e4-87881ea6b6a6')], [0.5379835367202759, 0.13214750587940216, 0.11529212445020676])]
    print(result)
    # ['Paris is the capital of France.', 'NomaDamas is Great Team', 'havertz is suck at soccer']
    print(first_contents)
    # ['Newjeans has 5 members.', 'LA is a country in the United States.', 'i am hungry']
    print(second_contents)
