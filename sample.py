import google.generativeai as genai


genai.configure(api_key="")

print("Checking available models...\n")
for m in genai.list_models():
    print(m.name)

print("\n Generating test response...\n")

model = genai.GenerativeModel("gemini-flash-latest")

response = model.generate_content("Say hello from Gemini 1.5 Flash!")
print("Response:", response.text)
