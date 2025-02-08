def main_topics(text, client):

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", 
            "content": '''
            You will receive a user-provided text describing the desired code.
            Your task is to analyze the request and determine the most relevant programming modules or topics needed to fulfill it.
            Dont list general topics only topics specific to the user request. 
            Examples: 
            User: how to create a text input in streanlit?
            Return: "text input"

            User: how to build two text inputs one next to the other in streamlit?
            Return: "text input; layout arrangement"

            Stick to Streamlit modules only!
            Return a list of up to 10 of the most important modules or topics, separated by semicolons. 
            Important: return only the list of modules or topics, Noting else!
            '''
            },
            {"role": "user", "content":f"User request:{text}"}
        ],
        temperature=0,
    )

    return response.choices[0].message.content
