from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os
import re
import requests

memory = ConversationBufferMemory(memory_key="history")
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo") 
conversation = ConversationChain(llm=llm, memory=memory)



class usebotalk:
    @staticmethod
    def getimage(conversation_to_read):

        conv1 = "'[{'user': [{'type': 'text', 'text': ''}, {'type': 'image_url', 'image_url': {'url': 'https://zoe-images.s3.amazonaws.com/file_471_1707266353.jpg'}}], 'bot': 'Hi there! I'm Zoe, your AI-Tutor. 😄 To start, could you let me know what grade you're in? This will help me explain better! 📚', 'date': '2024-02-07T00:39:16.516Z'}, {'user': '7th', 'bot': 'Great, thanks for sharing! 👍 Let's look at your math problem. Ready to solve it together? 🧮', 'date': '2024-02-07T00:39:36.491Z'}, {'user': 'Yep', 'bot': 'Perfect! When you see a minus sign in front of a parenthesis, like -(-20), what operation do you do? 🤔', 'date': '2024-02-07T00:39:47.724Z'}, {'user': 'Change it into addition', 'bot': 'That's right! 🎉 So what does -(-20) become?', 'date': '2024-02-07T00:40:31.219Z'}, {'user': 'Positive', 'bot': 'Awesome! Now, let's add the numbers together. What's -12 + 10 + 20? 🧐', 'date': '2024-02-07T00:40:49.373Z'}]'"

        conv2 = "'[{'user': 'hi', 'bot': 'Hello there! I'm Zoe, your AI-Tutor, and I'm super excited to help you out! 😊 What grade are you in, and what state are you studying in?', 'date': '2023-12-21T20:15:02.106Z'}, {'user': [{'type': 'text', 'text': ''}, {'type': 'image_url', 'image_url': {'url': 'https://zoe-images.s3.amazonaws.com/file_223_1703189810.jpg'}}], 'bot': 'Thanks for sharing the image with me! 😄 It looks like we have a math question about finding the area of a circle when given the circumference.\n\nFirst things first, do you remember the formula for the circumference of a circle? 🤔', 'date': '2023-12-21T20:16:54.854Z'}, {'user': 'yes', 'bot': 'Great! Since you remember the formula for the circumference of a circle, could you remind me what it is? This will help us find the radius, which we need to calculate the area. 🌟', 'date': '2023-12-21T20:18:15.380Z'}, {'user': 'C = 2 PI r \nor\nC = 2 x 3.14 x r', 'bot': 'Absolutely, you got it! The circumference (C) is equal to 2 times π times the radius (r). Now, we know the circumference is 18.84 units. Can you use this information to solve for the radius? 🧮 Remember, we're using 3.14 for π.', 'date': '2023-12-21T20:19:08.787Z'}, {'user': '2 x 3.14 x r = 6.28r \n6.28r x 18.84 = 118.3152', 'bot': 'It looks like there's been a little mix-up in the calculation. We have the equation \\( 2 \\times 3.14 \\times r = 18.84 \\). To find \\( r \\), we should divide both sides of the equation by \\( 2 \\times 3.14 \\) (which is \\( 6.28 \\)). \n\nSo, if we take \\( 18.84 \\) and divide it by \\( 6.28 \\), what does \\( r \\) equal? 🤓', 'date': '2023-12-21T20:20:54.311Z'}, {'user': '3.', 'bot': 'Perfect, you've got it! The radius \\( r \\) is 3 units. 🎉 Now, knowing the radius, can you recall the formula for the area of a circle? 🤔', 'date': '2023-12-21T20:21:27.454Z'}, {'user': 'A = pi r squared\nor\nA = 3.14 x r squared', 'bot': 'That's the one! Area (A) equals π times the radius squared. Since we've found the radius to be 3 units, can you plug that into the formula and calculate the area? 📐✨ Remember we're using 3.14 for π.', 'date': '2023-12-21T20:22:19.501Z'}, {'user': '28.26', 'bot': 'You got it! When you square the radius (3 units) and multiply by π (3.14), you indeed get an area of approximately 28.26 square units. 🌟 Nice work!\n\nNow you've got the area of the circle. Do you have any other questions or another topic you'd like to explore? 😊', 'date': '2023-12-21T20:23:12.968Z'}, {'user': [{'type': 'text', 'text': ''}, {'type': 'image_url', 'image_url': {'url': 'https://zoe-images.s3.amazonaws.com/file_224_1703190520.jpg'}}], 'bot': 'Thanks for the new image! It's a question about finding the perimeter and area of a semicircle.\n\nLet's start with the perimeter. Do you remember how to calculate the perimeter of a semicircle, which includes the straight edge (the diameter) plus the curved part (half the circle's circumference)? 📏🤔', 'date': '2023-12-21T20:28:47.362Z'}, {'user': 'no', 'bot': 'No worries at all! To calculate the perimeter of a semicircle, we add the length of the diameter to the length of the half-circumference of the circle. Since the diameter is just twice the radius, and we have a radius of 5 feet, what would the diameter be? 📐✨', 'date': '2023-12-21T20:29:10.806Z'}, {'user': '10', 'bot': 'That's right, the diameter is 10 feet. Now, to find the half-circumference, we use the formula for the circumference of a full circle which is \\( C = 2 \\times \\pi \\times r \\), but since we need only half of that for a semicircle, we divide it by 2.\n\nSo, half the circumference is \\( \\pi \\times r \\). With \\( r \\) being 5 feet and using 3.14 for \\( \\pi \\), can you calculate the half-circumference? 🤔🧮', 'date': '2023-12-21T20:29:42.769Z'}]'"
        #Prompt for our agent
        prompt = f"""You are a helpful assistant who can help with some task
                    Task: Given a list of Conversation: between user and bot  . Sperate image_url if present and give image url back as response only which should start with https:// or http:// don't give any other words back.
                    Intructions:
                    1) Please go through the whole conversation first between user and bot
                    2) Don't make anything of your own which is not in the conversation
                    3) Only give image url back.
                    4) If image url is not present don't answer anything. 
                    Take a deep breath and check carefully your answer and verify your answer if it matches with the given question
                    Few Examples:
                    Conversation: {conv1}
                    Imageurl: https://zoe-images.s3.amazonaws.com/file_471_1707266353.jpg
                    Conversation: {conv2}
                    Imageurl: 
                    Now having the understanding of the examples and intructions and task Question: , based on the Conversation:
                    Converstation: {str(conversation_to_read)}
                    Imageurl:"""
        response = conversation.predict(input=prompt)
        if "zoe-images.s3.amazonaws.com" in response:
            print("[!] Feeding image to vision llm")
            url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            response = re.findall(url_pattern, response)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            }

            # Configure the model
            payload = {
                "model": "gpt-4-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What's in this image?. Transform that question into text"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": response
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            if response.status_code == 200:
                response_text = response.json()["choices"][0]["message"]["content"]
                return response_text, True
            else:
                
                return {"error": "An error occurred during image processing."}, False
        else:
            return "No  image", False
        
        