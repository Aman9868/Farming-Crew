from crewai import Agent,Task,Crew,Process
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
import gradio as gr
from langchain.agents import Tool
search=DuckDuckGoSearchRun()
llm=Ollama(model="mistral")

def farm_expert(crop,period,state):
    nutrition_expert=Agent(
    role="Nutrition Expert",
    goal=f"""Assess nutritional requirement required by {crop} grow during  {period} season in a {state}.Suggest some nutrition tips & strategies for that kind of crop""",
    backstory=f"""Expert at understanding the nutrition needs,crop specific requirements, and state specific considertaions.Skilled in developing a best nutrition tips & strategies""",
    verbose=True,
    llm=llm,
    allow_delegations=True,
    tools=[search]
    )

    pesticide_expert=Agent(
    role="Pesticide Expert",
    goal=f"""Evalauate the pest threat faced by {crop} grow during {period} season in a {state} and recommend appropriate pesticide products based on pest biology, crop type, and environmental considerations""",
    backstory=f"""Specializing in agricultural pest management, primary responsibility is to develop and implement effective strategies for the control and mitigation of pests that threaten crop health and yield.""",
    verbose=True,
    llm=llm,
    allow_delegations=True
    )

    fertilizer_expert=Agent(
    role="Fertlizers Expert",
    goal=f"""Expertise in management of fertilizers required by {crop} grows during {period} season in a {state} and recommend suitable fertilizer blends or formulations for that particular crop""",
    backstory=f"""Knowledgeable expertise in soil science, agronomy, and plant nutrition Expert aims to support sustainable agriculture practices, maximize yield potential, and promote environmental stewardship.""" ,
    verbose=True,
    llm=llm,
    allow_delegations=True
    )


    botanical_expert=Agent(
    role="Botanical Growth Expert",
    goal=f"""Analyze the growing stage of {crop} grows during {period} season in a {state}.Offered tailored advice to optimize the growth, yield, and quality of that particular crop from begining stage  till the plant grows fully.Explain the requiremnet stage by stage.""",
    backstory=f""" Expertise in plant biology, horticulture, and environmental science to optimize growing conditions, implement cultivation techniques, and promote the vitality of botanical collections.""",
    verbose=True,
    llm=llm,
    allow_delegations=True
    )

    task1=Task(
    description=f"""Assess nutritional requirement required by {crop} grow during  {period} season in a {state}.Suggest some nutrition tips & strategies for that kind of crop""",
    agent=nutrition_expert,
    llm=llm

    )

    task2=Task(
    description=f"""Evalauate the pest threat faced by {crop} grow during {period} season in a {state} and recommend appropriate pesticide products based on pest biology, crop type, and environmental considerations""",
    agent=pesticide_expert,
    llm=llm
    )
    task3=Task(
    description=f"""Expertise in management of fertilizers required by {crop} grows during {period} season in a {state} and recommend suitable fertilizer blends or formulations for that particular crop.""",
    agent=fertilizer_expert,
    llm=llm
    )

    task4=Task(
    description=f"""Analyze the growing stage of {crop} grows during {period} season in a {state}.Offered tailored advice to optimize the growth, yield, and quality of that particular crop from begining stage  till the plant grows fully.Explain the requiremnet stage by stage.""",
    agent=botanical_expert,
    llm=llm
    )

    health_crew = Crew(
            agents=[nutrition_expert, pesticide_expert,fertilizer_expert,botanical_expert],
            tasks=[task1, task2, task3,task4],
            verbose=2,
            process=Process.sequential,
        )

    result=health_crew.kickoff()
    return result
iface = gr.Interface(
    fn=farm_expert, 
    inputs=["text", "text", "text"], 
    outputs="text",
    title="CrewAI Farming Expert Analysis",
    description="Enter crop name, crop season, and state to analyze  nutrition, and all other"
)

iface.launch()