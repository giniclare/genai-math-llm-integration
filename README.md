## Integration of a Mathematical Calulations with a Chat Completion System using LLM Function-Calling

### AIM:
To design and implement a Python function for calculating the volume of a cylinder, integrate it with a chat completion system utilizing the function-calling feature of a large language model (LLM).

### PROBLEM STATEMENT:

Large Language Models (LLMs) cannot inherently perform precise mathematical calculations like finding the volume of a geometry because they are language prediction systems rather than calculation engines. This project demonstrates how to overcome this limitation by using OpenAI's Function-Calling mechanism to securely delegate execution to a precise Python function that calculates a cylinder's volume ((V = \pi r^2 h)).

### DESIGN STEPS:

1. Define the Core Logic: Create a native Python function calculate_cylinder_volume(radius, height) that utilizes the math library to execute precise volume geometry operations.
2. Construct JSON Schema: Build a structural metadata payload defining the function name, description, parameters (radius and height as float numbers), and required parameters.
3. Initialize Chat State: Pass the tool specification schema along with the user's conversational prompt (e.g., "Find the volume of a cylinder with radius 5 and height 10") to the Chat Completions endpoint.
4. Intercept Tool Arguments: Extract the arguments payload generated programmatically by the model (args["radius"], args["height"]), parse them using json.loads(), and run the local calculation.
5. Generate Final Context: Append the mathematical execution results back to the conversation thread using the function role, and call the model once more to create a friendly response.

### DEVELOPED BY: GINI CLARE G
### REG NO: 212225240041

### PROGRAM:

```python
import os
import math
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv(find_dotenv())

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def calculate_cylinder_volume(radius, height):
    """Calculate the volume of a cylinder."""
    if radius <= 0 or height <= 0:
        return "Radius and height must be positive numbers."

    volume = math.pi * (radius ** 2) * height
    return round(volume, 2)


def chat_with_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You help calculate the volume of a cylinder."
            },
            {"role": "user", "content": prompt}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate_cylinder_volume",
                    "description": (
                        "Calculate the volume of a cylinder "
                        "given its radius and height."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "radius": {
                                "type": "number",
                                "description": "Radius of the cylinder"
                            },
                            "height": {
                                "type": "number",
                                "description": "Height of the cylinder"
                            }
                        },
                        "required": ["radius", "height"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ],
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "calculate_cylinder_volume":
                arguments = json.loads(
                    tool_call.function.arguments
                )

                radius = arguments["radius"]
                height = arguments["height"]

                return calculate_cylinder_volume(radius, height)

    return message.content


radius = float(input("Enter the radius of the cylinder: "))
height = float(input("Enter the height of the cylinder: "))

prompt = (
    f"What is the volume of a cylinder with a radius of {radius} "
    f"and a height of {height}?"
)

result = chat_with_openai(prompt)
print("Result:", result)

```

### OUTPUT:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/783633ad-d486-43e9-b940-7d2666876f48" />

### RESULT:

The system successfully integrated a precise backend calculation workflow with an LLM chat completion flow. By exploiting function calling, the model dynamically verified user inputs, structured calculation inputs correctly into an executable state, run the arithmetic local routines without hallucinations, and formatted the data into a fluent conversational output.
