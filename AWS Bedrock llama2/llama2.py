import boto3
import json

# Prompt data
prompt_data = """
Act as Shakespeare and write a poem on Machine Learning.
"""

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Payload for model invocation
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

# Convert payload to JSON
body = json.dumps(payload)
model_id = "meta.llama2-70b-chat-v1"

# Invoke the model
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Parse the response
response_body = json.loads(response['body'].read().decode('utf-8'))
response_text = response_body.get('generated_text')  # Adjust key based on API spec

# Display the response
print(response_text)
