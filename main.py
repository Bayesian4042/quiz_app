from langchain import PromptTemplate

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import streamlit as st


# Define your desired data structure.
class ObjectiveQuestion(BaseModel):
    question: str
    A: str = None
    B: str = None
    C: str = None
    D: str = None

class Quiz(BaseModel):
    questions: List[ObjectiveQuestion]
    answers: List[str]

def create_quiz_template():
    template = """
    You are an expert quiz maker for {technical_field}.
    Create a quiz with {number_of_questions} {quiz_type} about following concept/context: {quiz_content}.
    Let's think step by step. 
    {format_instructions}
    The format of each quiz type should be as such:
    - Multi-Choice: 
      - Questions:
        <Question1>: <a. Answer1>, <b. Answer2>, <c. Answer3>, <d. Answer4>
        <Question2>: <a. Answer1>, <b. Answer2>, <c. Answer3>, <d. Answer4>
        ...
      - Answers:
        <Answer1>: <a|b|c|d>
        <Answer2>: <a|b|c|d>
        ...
      - Example:
        Questions:
         - Question 1. What command is used to create a new table in SQL?
          a. CREATE TABLE
          b. BUILD TABLE
          c. GENERATE TABLE
          d. None of the above

        Answers:
          - Answer 1: a

    - True-False:
      - Questions:
        <Question1>: <true|false>
        <Question2>: <true|false>
        ...

     - Answers:
        <Answer1>: true
        <Answer2>: false
        ...
      - Example:
        Questions:
          - Question 1. SQL table can contain more than one primary key.?
            a. True
            b. False
        Answers:
          - Answer 1: b

    - Open-ended:
        - Questions:
          <Question1>: 
          <Question2>:
          ...
        - Answers:
          <Answer1>:
          <Answer2>:
          ...
        - Example:
          Questions:
            - Question 1: What is the difference between SQL and MySQL?

          Answers:
            - Answer 1: SQL is a standard language which stands for Structured Query Language based on the English language. MySQL is a database management system. 
    """
    parser = PydanticOutputParser(pydantic_object=Quiz)
    prompt = PromptTemplate.from_template(template=template,
                                          partial_variables={"format_instructions": parser.get_format_instructions()})

    return prompt


def create_quiz_chain(prompt):
    model_name = "gpt-3.5-turbo"
    temperature = 0.7
    chain = LLMChain(llm=ChatOpenAI(model_name=model_name, temperature=temperature,
                                    openai_api_key=os.environ["openai_key"]),
                     prompt=prompt)

    return chain

def main():
    st.title("QUIZ test app")
    st.write("")
    prompt_template = create_quiz_template()
    chain = create_quiz_chain(prompt_template)
    topic = st.text_area("Topic ex. SQL Optimization : <skill> <sub-skill>")
    context = st.text_area("Enter the context of topic: ex. basic sql query optimization techniques")
    num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=5)
    quiz_type = st.selectbox("Select quiz type: ", ["Multi-Choice", "True-False", "Open-Ended"])
    if st.button("Generate Quiz"):
        quiz_response = chain.run(number_of_questions=num_questions, quiz_type=quiz_type, technical_field=topic,
                                  quiz_content=context)
        st.json(quiz_response)



if __name__ == "__main__":
    main()