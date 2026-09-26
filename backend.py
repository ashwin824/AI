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

# InMemoryChatMessageHistory() is a LangChain class that stores a chat conversation in memory (RAM) while your application is running.
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

    # lang chain expression language
    # RunnablePassthrough passes the input through unchanged, but assign() lets you add or modify fields.

    # RunnablePassthrough.assign(history=itemgetter("history") | trimmer)
    # Extract the history field.
    # Pass it through the trimmer.
    # Replace the original history with the trimmed version.
    # This chain:
    # Receives a question and chat history.
    # Trims the history to stay within token limits.
    # Replaces the original history with the trimmed version.
    # Builds a prompt using the trimmed history.
    # Sends the prompt to the LLM and returns the response.

    base_chain = (RunnablePassthrough.assign(history=itemgetter("history") | trimmer) | CHAT_PROMPT | llm )

    # RunnableWithMessageHistory is a LangChain wrapper that automatically manages chat history for a chain.
    # lambda session_id: memory - Given a session ID, where should I get the chat history?
    # input_messages_key="input" - The user's message is stored in the input field.
    # history_messages_key="history" - Inject conversation history into the chain using the variable named history
    # lambda session_id: memory - defines an anonymous function (lambda).
    # It's equivalent to:
    # def get_memory(session_id):
    #       return memory

    chat_with_history = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: memory,        # always return this session's history
        input_messages_key="input",
        history_messages_key="history",
    )

    # This code sends a user message to the chatbot chain, including the conversation history associated with a specific session.
    # chat_with_history.invoke() - executes the chain
    # {"input": input_text} - passes user's message into the chain
    # config={"configurable": {"session_id": "bepec-session"}} - provides the conversation identifier.
    # the code sends the user's message to the chatbot using the chat history for session "bepec-session", 
    # gets the AI response, and extracts the response text into chat_reply

    try:
        result = chat_with_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": "bepec-session"}},
        )
        chat_reply = result.content
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory

# few shot examples
# What it does: Creates a list of dictionary pairs containing sample user inputs and the exact responses you expect from the AI.
# Why it matters: This teaches the model the required tone, structure, and formatting before it answers the actual user prompt.
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

# formatting examples
# What it does: Defines a template showing LangChain how to turn each dictionary entry into a chat message pair (a human turn followed by an ai turn).
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# few shot prompt
# What it does: Iterates through your list of examples and converts them into formatted chat messages ready to be inserted into the main prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Assembling the full chat prompt
# ("system", ...): Sets the global system instruction for the AI
# few_shot_prompt: Places the few-shot examples right after the system prompt.
# MessagesPlaceholder(variable_name="history"): Acts as a dynamic placeholder where memory (past conversation turns) will be inserted.
# ("human", "{input}"): Captures the new message sent by the user.

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            "You are a helpful, friendly AI assistant built by BEPEC Solutions. "
            "You remember the ongoing conversation and answer clearly and concisely. "
            "If you don't know something, say so honestly instead of making things up. "
            "Match the tone and style of the examples and end with Thanks for Asking BEPEC",
        ),
        few_shot_prompt, 
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}"), 
    ]
)

# trimmer
# This trimmer:
# Keeps conversation history under 1000 tokens.
# Retains the most recent messages (strategy="last").
# Uses demo_chatbot() to calculate token counts.
# Does not consider the system message in the trimming process.
# Makes sure the resulting history starts with a HumanMessage.

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=demo_chatbot(),
    include_system=False,
    start_on="human",
)


