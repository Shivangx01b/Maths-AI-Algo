from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub




llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
tools = load_tools(["llm-math"], llm=llm)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

class Agent1:
    @staticmethod
    def agent_to_solve_math(question):
        output = agent_executor.invoke({"input": "{0}".format(question)})
        return output




