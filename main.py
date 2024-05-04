from fastapi import FastAPI, File, Request
from fastapi.responses import JSONResponse
from core import agent_solve_math
from core import agent_convert_to_question
from parser import getinteraction
from core import agent_verify
import pickle
from aimodel import utils
from aimodel import data_load
from aimodel import math_bot


#For self trained model
VOCAB_SIZE = 958
#Loading VOCAB
with open('aimodel\\vocab_math_4.pkl', 'rb') as f:
    VOCAB = pickle.load(f)
device = utils.get_device()

#Load the model
math_bot_trained = math_bot.init_chatbot()

app = FastAPI()




#Using LLMS only for math questions
@app.post("/query-llm")
async def query_index(query: Request):
    req_info = await query.json()


    conv = req_info['conversation']
    useranswer = req_info['useranswer']

    #Do pre flight checks to check for image and get image convert to text
    data_parsed, is_image = getinteraction.usebotalk.getimage(str(conv))
    if is_image == True:


        original_question_image = agent_convert_to_question.ConvertJsonConvtoQuestion.parse_conv_for_image(data_parsed)
        output = agent_solve_math.Agent1.agent_to_solve_math(original_question_image)
        output_final = output.get("output", "")

        verify_data = agent_verify.VeirfyAgent.AgentCheck(original_question_image, useranswer, output_final)
        return JSONResponse(content={"Evaluation": str(verify_data), "original_question_got": str(original_question_image), "botanswer": str(output_final), "useranswer": str(useranswer)}, status_code=200)
    
    else:

        original_question = agent_convert_to_question.ConvertJsonConvtoQuestion.parse_conv(conv)

        output = agent_solve_math.Agent1.agent_to_solve_math(original_question)
        output_final = output.get("output", "")


        verify_data = agent_verify.VeirfyAgent.AgentCheck(original_question, useranswer, output_final)
        return JSONResponse(content={"Evaluation": str(verify_data), "original_question_got": str(original_question), "botanswer": str(output_final), "useranswer": str(useranswer)}, status_code=200)
    

@app.post("/query-ai-bot")
async def query_index(query: Request):
    req_info = await query.json()
    conv = req_info['conversation']
    useranswer = req_info['useranswer']
    
    #Do pre flight checks to check for image and get image convert to text
    data_parsed, is_image = getinteraction.usebotalk.getimage(str(conv))
    if is_image == True:


        original_question_image = agent_convert_to_question.ConvertJsonConvtoQuestion.parse_conv_for_image(data_parsed)
        print(original_question_image)
        data = math_bot.main_chatbot(math_bot_trained, str(original_question_image))
        if useranswer == data:
            return JSONResponse(content={"Evaluation": "EQUIVALENT", "original_question_got": str(original_question_image), "botanswer": str(data), "useranswer": str(useranswer)}, status_code=200)
        else:
            return JSONResponse(content={"Evaluation": "NOT_EQUIVALENT", "original_question_got": str(original_question_image), "botanswer": str(data), "useranswer": str(useranswer)}, status_code=200)
    
    else:
            
            original_question = agent_convert_to_question.ConvertJsonConvtoQuestion.parse_conv(conv)
            print(original_question)
            data = math_bot.main_chatbot(math_bot_trained, str(original_question))
            if useranswer == data:
                return JSONResponse(content={"Evaluation": "EQUIVALENT", "original_question_got": str(original_question), "botanswer": str(data), "useranswer": str(useranswer)}, status_code=200)
            else:
                return JSONResponse(content={"Evaluation": "NOT_EQUIVALENT", "original_question_got": str(original_question), "botanswer": str(data), "useranswer": str(useranswer)}, status_code=200)