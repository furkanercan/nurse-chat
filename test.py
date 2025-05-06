import requests

response = requests.post("http://localhost:8000/ask", json={
    "patient_id": "patient8",
    "question": "Tell me the name, age, gender, race, ethnicity, weight, height, religion, mother, favorite animal and sexual activity history of the patient."
    # "question": "Tell me about the sensory perception of this patient."
    # "question": "Tell me about the surgical history of this patient."
    # "question": "Tell me about the religion of this patient."
})

print(response.json())
