from operator import itemgetter
from langchain_aws import ChatBedrockConverse
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import trim_messages
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

def demo_chatbot():
    llm = ChatBedrockConverse(
        model = "deepseek.v3.2", 
        region_name="us-east-2",
        temperature=0.7,
        max_tokens=512)
    return llm

def demo_memory():
    # nMemoryChatMessageHistory() - stores a chat conversation in memory (RAM) while your application is running.
    return InMemoryChatMessageHistory()

def demo_conversation(input_text, memory):
    llm = demo_chatbot()
    # RunnablePassthrough passes the input through unchanged, but assign() lets you add or modify fields.
    base_chain = (RunnablePassthrough.assign(history=itemgetter("history") | trimmer) | CHAT_PROMPT | llm )
    # RunnableWithMessageHistory is a LangChain wrapper that automatically manages chat history for a chain.
    chat_with_history = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="history")
     # send a user message to the chatbot chain, including the conversation history associated with a specific session.
    try:
        result = chat_with_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": "bepec-session"}},
        )
        chat_reply = result.content
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory
    
# fewshot examples
examples = [
    {
        "input": "Who are you?",
        "output": "I'm the Ashwin's assistant. I help you with questions about our "
                  "AI, Data Science, and Data Engineering courses, placements, and careers.",
    },
    {
        "input": "I'm from a non-IT background. Can I learn AI?",
        "output": "Absolutely. Many Ashwin's learners come from non-IT backgrounds. "
                  "We start from fundamentals and take you to job-ready, step by step. "
                  "No prior coding needed to begin.",
    },
    {
        "input": "How long does the course take?",
        "output": "It depends on the track and your pace, but most learners become "
                  "job-ready in a few focused months. Pick a track and commit to "
                  "consistent daily practice.",
    }
]

# chat template
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

# few shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

# full chat prompt
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a helpful, friendly AI assistant built by Ashwin's Solutions. "
            "You remember the ongoing conversation and answer clearly and concisely. "
            "If you don't know something, say so honestly instead of making things up. "
            "Match the tone and style of the examples and end with Thanks for asking Ashwin",
        ),
        few_shot_prompt, 
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}")
    ]
)

# trimmer
trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=demo_chatbot(),
    include_system=False,
    start_on="human",
)