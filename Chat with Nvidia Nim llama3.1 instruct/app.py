from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "your api key"
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-8b-instruct",
  messages=[{"role":"user","content":"Write a poem on SpaceX ambition for Mars."}],
  temperature=0.5,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

