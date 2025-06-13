import google.generativeai as genai

genai.configure(api_key="AIzaSyAiDaoJg5rIaxQqNfA51kGrQW_7_59V77A")

model = genai.GenerativeModel("models/gemini-1.5-flash")
response = model.generate_content("Qual é a capital do Brasil?")
print(response.text)