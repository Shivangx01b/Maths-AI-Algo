from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

memory = ConversationBufferMemory(memory_key="history")
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo") 
conversation = ConversationChain(llm=llm, memory=memory)

class ConvertJsonConvtoQuestion:
    @staticmethod
    def parse_conv(conv):
        

        conv1 = "'[{'user': 'I don't want to do math', 'bot': 'Hey there! 🌟 I'm Zoe, your friendly AI-Tutor, here to make math a bit more fun. 😊 What grade and state are you studying in? Let's tackle some math together! What topic are you on?', 'date': '2024-03-01T18:11:28.267Z'}, {'user': 'I hate math', 'bot': 'Oh no, hate is such a strong word! 😅 But hey, no pressure. Maybe we can find a way to make it a bit more enjoyable. What part of math is giving you trouble? Let's chat about it! 🤓', 'date': '2024-03-01T18:11:46.878Z'}, {'user': 'How do I find 12% of 60', 'bot': 'Sure, to find 12% of 60, you multiply 60 by 0.12. What do you get? 🧮', 'date': '2024-03-01T18:12:30.740Z'}, {'user': 'But why .12?', 'bot': 'Percent means per hundred. So, 12% is 12 per 100 or 12/100. When you convert a percent to a decimal, you divide by 100. That's why 12% becomes 0.12. 🎯 Now, can you try multiplying 60 by 0.12?', 'date': '2024-03-01T18:12:48.001Z'}]'"
        conv2 = "'[{'user': 'hi', 'bot': 'Hello there! I'm Zoe, your AI-Tutor, and I'm super excited to help you out! 😊 What grade are you in, and what state are you studying in?', 'date': '2023-12-21T20:15:02.106Z'}, {'user': [{'type': 'text', 'text': ''}, {'type': 'image_url', 'image_url': {'url': 'https://zoe-images.s3.amazonaws.com/file_223_1703189810.jpg'}}], 'bot': 'Thanks for sharing the image with me! 😄 It looks like we have a math question about finding the area of a circle when given the circumference.\n\nFirst things first, do you remember the formula for the circumference of a circle? 🤔', 'date': '2023-12-21T20:16:54.854Z'}, {'user': 'yes', 'bot': 'Great! Since you remember the formula for the circumference of a circle, could you remind me what it is? This will help us find the radius, which we need to calculate the area. 🌟', 'date': '2023-12-21T20:18:15.380Z'}, {'user': 'C = 2 PI r \nor\nC = 2 x 3.14 x r', 'bot': 'Absolutely, you got it! The circumference (C) is equal to 2 times π times the radius (r). Now, we know the circumference is 18.84 units. Can you use this information to solve for the radius? 🧮 Remember, we're using 3.14 for π.', 'date': '2023-12-21T20:19:08.787Z'}, {'user': '2 x 3.14 x r = 6.28r \n6.28r x 18.84 = 118.3152', 'bot': 'It looks like there's been a little mix-up in the calculation. We have the equation \\( 2 \\times 3.14 \\times r = 18.84 \\). To find \\( r \\), we should divide both sides of the equation by \\( 2 \\times 3.14 \\) (which is \\( 6.28 \\)). \n\nSo, if we take \\( 18.84 \\) and divide it by \\( 6.28 \\), what does \\( r \\) equal? 🤓', 'date': '2023-12-21T20:20:54.311Z'}, {'user': '3.', 'bot': 'Perfect, you've got it! The radius \\( r \\) is 3 units. 🎉 Now, knowing the radius, can you recall the formula for the area of a circle? 🤔', 'date': '2023-12-21T20:21:27.454Z'}, {'user': 'A = pi r squared\nor\nA = 3.14 x r squared', 'bot': 'That's the one! Area (A) equals π times the radius squared. Since we've found the radius to be 3 units, can you plug that into the formula and calculate the area? 📐✨ Remember we're using 3.14 for π.', 'date': '2023-12-21T20:22:19.501Z'}, {'user': '28.26', 'bot': 'You got it! When you square the radius (3 units) and multiply by π (3.14), you indeed get an area of approximately 28.26 square units. 🌟 Nice work!\n\nNow you've got the area of the circle. Do you have any other questions or another topic you'd like to explore? 😊', 'date': '2023-12-21T20:23:12.968Z'}, {'user': [{'type': 'text', 'text': ''}, {'type': 'image_url', 'image_url': {'url': 'https://zoe-images.s3.amazonaws.com/file_224_1703190520.jpg'}}], 'bot': 'Thanks for the new image! It's a question about finding the perimeter and area of a semicircle.\n\nLet's start with the perimeter. Do you remember how to calculate the perimeter of a semicircle, which includes the straight edge (the diameter) plus the curved part (half the circle's circumference)? 📏🤔', 'date': '2023-12-21T20:28:47.362Z'}, {'user': 'no', 'bot': 'No worries at all! To calculate the perimeter of a semicircle, we add the length of the diameter to the length of the half-circumference of the circle. Since the diameter is just twice the radius, and we have a radius of 5 feet, what would the diameter be? 📐✨', 'date': '2023-12-21T20:29:10.806Z'}, {'user': '10', 'bot': 'That's right, the diameter is 10 feet. Now, to find the half-circumference, we use the formula for the circumference of a full circle which is \\( C = 2 \\times \\pi \\times r \\), but since we need only half of that for a semicircle, we divide it by 2.\n\nSo, half the circumference is \\( \\pi \\times r \\). With \\( r \\) being 5 feet and using 3.14 for \\( \\pi \\), can you calculate the half-circumference? 🤔🧮', 'date': '2023-12-21T20:29:42.769Z'}]'"
        #Prompt for our agent
        prompt = f"""You are a helpful assistant who can help with some task
                    Task: Given a list of Conversation: between user and bot convert the whole converstation into a  original mathematically Question:
                    Intructions:
                    1) Please go through the whole conversation first between user and bot
                    2) Don't make anything of your own which is not in the conversation
                    3) Always verify your question once created, if it has the same context and the conversation
                    4) Give question directly don't add any other sentences which is not required.
                    Take a deep breath and check carefully your answer and verify your answer if it matches with the given question
                    Few Examples:
                    Conversation: {conv1}
                    Question:How do I find 12% of 60?
                    Conversation: {conv2}
                    Question: If the circumference of the circle is 18.84 units, what is the area?
                    Now having the understanding of the examples and intructions and task Question: , based on the Conversation:
                    Converstation: {str(conv)}
                    Question:"""
        response = conversation.predict(input=prompt)
        return response
    @staticmethod
    def parse_conv_for_image(conv):
        
        #Prompt for our agent
        prompt = f"""You are a helpful assistant who can help with some task
                    Task: Given a response convert that to original mathematically Question:
                    Intructions:
                    1) Please go through the whole response first
                    2) Don't make anything of your own which is not in the conversation
                    3) Always verify your question once created, if it has the same context and the response
                    4) Give question directly don't add any other sentences which is not required.
                    Take a deep breath and check carefully your answer and verify your answer if it matches with the given question
                    Response: {str(conv)}
                    Question:"""
        response = conversation.predict(input=prompt)
        return response
