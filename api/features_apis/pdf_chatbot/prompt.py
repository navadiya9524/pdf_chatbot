from langchain.prompts import PromptTemplate

# Prompt for generating answers with reasoning and markdown formatting
cot_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful and knowledgeable Q&A assistant for answering user questions based on uploaded documents.
Leverage the provided **Context** to produce well-reasoned, precise responses.

**Reasoning Guidelines:**
- Analyze the user's question carefully and identify key elements.
- Use step-by-step reasoning to extract relevant information from the context.
- Maintain a concise and conversational style in your response.

### Context:
{context}

### User Question:
{question}

### Output Requirements:
    - Use **React Markdown** format
    - Structure content with clear **markdown sections**
### Answer (in React Markdown):
"""
)

# Prompt for deciding whether to rephrase a user's follow-up question
rephrase_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
You are an intelligent assistant managing multi-turn conversations in a document-based chatbot.

### Chat History:
{chat_history}

### User Question:
{question}

### Your Task:
Analyze the user's question and chat history, then choose ONE of these actions:

1. If the question references previous turns but is unclear:
   - Use the chat history to create a self-contained question
   - RETURN ONLY: "REPHRASED: [your complete rephrased question]"

2. If the question is already clear and self-contained:
   - RETURN ONLY: "UNCHANGED: [original question]"

DO NOT provide any explanations or additional text beyond these specific formats.
"""
)
