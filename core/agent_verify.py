from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

memory = ConversationBufferMemory(memory_key="history")
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo") 
conversation = ConversationChain(llm=llm, memory=memory)

class VeirfyAgent:
    @staticmethod
    def AgentCheck(question, useranswer, botanswer):
    
        #Prompt for our agent
        prompt = f"""You are a helpful assistant who can help with some task
                    Task: Given will be given a maths question as Question: and answers in two way UserAnswer: and BotAnswer: and you have to verify if the UserAnswer: and BotAnswer: if user one is correct or BotAnswer: is correct. If UserAnswer: is correct then you have to say 'EQUIVALENT' and if BotAnswer: is correct then you have to say 'NOT_EQUIVALENT'
                    Intructions:
                    1) Please go through the Question: and UserAnswer: and BotAnswer: first
                    2) Don't make anything of your own which is not in the conversation
                    3) Always verify your opinion once created
                    4) Give 'EQUIVALENT' or 'NOT_EQUIVALENT' directly don't add any other sentences which is not required.
                    Take a deep breath and check carefully your answer and verify your answer if it matches with the given question
                    Few Examples:
                    Question: How do I find 12% of 60?
                    UserAnswer: 5
                    BotAnswer: 7.2
                    Now having the understanding of the examples and intructions and task Question: , based on the Conversation:
                    Question: {str(question)}
                    UserAnswer: {str(useranswer)}"
                    BotAnswer: {str(botanswer)}"""
        response = conversation.predict(input=prompt)
        return response
    
