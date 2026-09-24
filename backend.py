from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import trim_messages
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
# api keys
from config import ApiKeys
keys = ApiKeys()
# llm
def demo_chatbot():
    api_key = keys.openai_key
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Create a .env file with: "
            "OPENAI_API_KEY=your_key_here"
        )

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1024,)
    return llm
#
def demo_memory():
    """
    Return a fresh in-memory chat history store.
    Frontend holds one of these per session and passes it back each turn.
    """
    return InMemoryChatMessageHistory()
#
def demo_conversation(input_text, memory):
    """
    Run one turn of the conversation.
    `memory` is an InMemoryChatMessageHistory instance from demo_memory().
    Returns: (assistant_reply, updated_memory)
    """
    llm = demo_chatbot()

    base_chain = (
        RunnablePassthrough.assign(
            history=itemgetter("history") | trimmer
        )
        | CHAT_PROMPT
        | llm
    )


    chat_with_history = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: memory,        # always return this session's history
        input_messages_key="input",
        history_messages_key="history",
    )

    try:
        result = chat_with_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": "bepec-session"}},
        )
        chat_reply = result.content
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory

# example for ai
examples = [
    {
        "input": "Who are you?",
        "output": "I'm the BEPEC assistant. I help you with questions about our "
                  "AI, Data Science, and Data Engineering courses, placements, and careers.",
    },
    {
        "input": "I'm from a non-IT background. Can I learn AI?",
        "output": "Absolutely. Many BEPEC learners come from non-IT backgrounds. "
                  "We start from fundamentals and take you to job-ready, step by step. "
                  "No prior coding needed to begin.",
    },
    {
        "input": "How long does the course take?",
        "output": "It depends on the track and your pace, but most learners become "
                  "job-ready in a few focused months. Pick a track and commit to "
                  "consistent daily practice.",
    },
]

# example prompt
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# few shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# chat prompt
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful, friendly AI assistant built by BEPEC Solutions. "
            "You remember the ongoing conversation and answer clearly and concisely. "
            "If you don't know something, say so honestly instead of making things up. "
            "Match the tone and style of the examples. and end with Thanks for Asking BEPEC",
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
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


