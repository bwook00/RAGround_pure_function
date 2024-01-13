from typing import List

import asyncio
from dotenv import load_dotenv

from llama_index.llms.llm import BaseLLM
from llama_index.llms.openai import OpenAI

decompose_prompt = """Decompose a question in self-contained sub-questions. Use \"The question needs no decomposition\" when no decomposition is needed.
    
    Example 1:
    
    Question: Is Hamlet more common on IMDB than Comedy of Errors?
    Decompositions: 
    1: How many listings of Hamlet are there on IMDB?
    2: How many listing of Comedy of Errors is there on IMDB?
    
    Example 2:
    
    Question: Are birds important to badminton?
    
    Decompositions:
    The question needs no decomposition
    
    Example 3:
    
    Question: Is it legal for a licensed child driving Mercedes-Benz to be employed in US?
    
    Decompositions:
    1: What is the minimum driving age in the US?
    2: What is the minimum age for someone to be employed in the US?
    
    Example 4:
    
    Question: Are all cucumbers the same texture?
    
    Decompositions:
    The question needs no decomposition
    
    Example 5:
    
    Question: Hydrogen's atomic number squared exceeds number of Spice Girls?
    
    Decompositions:
    1: What is the atomic number of hydrogen?
    2: How many Spice Girls are there?
    
    Example 6:
    
    Question: {question}
    
    Decompositions:"
    """


def query_decompose(queries: List[str], llm: BaseLLM,
                    prompt: str = decompose_prompt) -> List[List[str]]:
    """
    decompose query to little piece of questions.
    :param queries: List[str], queries to decompose.
    :param llm: BaseLLM, language model to use. llama_index's default model is gpt3.5-turbo.
    :param prompt: str, prompt to use for query decomposition.
    :return: List[List[str]], list of decomposed query. Return input query if query is not decomposable.
    """
    # Run async query_decompose_pure function
    tasks = [query_decompose_pure(query, llm, prompt) for query in queries]
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results


async def query_decompose_pure(query: str, llm: BaseLLM = OpenAI(temperature=0.2),
                               prompt: str = decompose_prompt) -> List[str]:
    """
    decompose query to little piece of questions.
    :param query: str, query to decompose.
    :param llm: BaseLLM, language model to use. llama_index's default model is gpt3.5-turbo.
    :param prompt: str, prompt to use for query decomposition.
    :return: List[str], list of decomposed query. Return input query if query is not decomposable.
    """
    full_prompt = "prompt: " + prompt + "\n\n" "question: " + query
    answer = llm.complete(full_prompt)
    if answer.text == "the question needs no decomposition.":
        return [query]
    try:
        lines = [line.strip() for line in answer.text.splitlines() if line.strip()]
        if lines[0].startswith("Decompositions:"):
            lines.pop(0)
        questions = [line.split(':', 1)[1].strip() for line in lines if ':' in line]
        if not questions:
            return [query]
        return questions
    except:
        return [query]


if __name__ == '__main__':
    load_dotenv()
    llm = OpenAI(temperature=0.2)
    sample_query = "Which group has more members, Newjeans or Espa?"
    result = query_decompose([sample_query], llm)

    print(result)
    # result:
    # [['How many members are in the Newjeans group?', 'How many members are in the Espa group?']]

    # test code
    assert len(result[0]) > 1
