import json
import json
import openai

openai.api_key = ""
openai.api_base = ""
openai.api_type = ""
openai.api_version = ""




with open('../../xpo_events.json', 'r') as fp:
    xpo_events = json.load(fp)
    

all_events = xpo_events
all_demonstrations = []

start_template = """
Now I will give you some ontology, and then you need to output their definitions. Your output should carefully focus on the difference between these events.

Example 1
Input:
parent event: mental_state
son event:  belief
son event:  doubt
son event:  awakening
son event:  emotional_state
son event:  intention
son event:  mental_confusion
son event:  understanding

Output:
Mental State: A mental state refers to an individual's current psychological condition or state of mind. It encompasses various emotions, thoughts, and cognitive processes that a person may experience at any given time.
Belief: Belief is a mental state characterized by the acceptance or conviction that something is true or exists, often without concrete evidence. It involves having confidence in the validity of a particular idea, concept, or proposition.
Doubt: Doubt is a mental state characterized by uncertainty or skepticism regarding the truth or validity of a belief or idea. It involves questioning or being unsure about the accuracy of a particular belief or assertion.
Awakening: Awakening refers to a profound shift in an individual's perception or understanding. It is often associated with a realization or enlightenment, leading to a significant change in one's perspective or awareness, often of a spiritual or philosophical nature.
Emotional State: An emotional state relates to a person's feelings and emotions at a specific moment. These states can include happiness, sadness, anger, fear, or any other emotional response to internal or external stimuli.
Intention: Intention is a mental state involving a purposeful and deliberate desire to perform a particular action or achieve a specific goal. It is often accompanied by a plan or strategy to accomplish the intended outcome.
Mental Confusion: Mental confusion is a mental state characterized by a lack of clarity or cognitive disarray. It involves difficulty in processing information, making decisions, or understanding one's surroundings, leading to a sense of disorientation.
Understanding: Understanding is a mental state involving comprehension and insight into a concept, idea, or situation. It signifies the ability to grasp and make sense of information or knowledge, often leading to increased knowledge or awareness.




Now I will give you some event ontology, and your output should carefully follow the previous examples. Your output should carefully focus on the event ontology.

Input:
"""

Final_Data = {}


for key in all_events.keys():
    
    temp_sons = []
    
    sons = all_events[key]
    
    Final_Data[key] = {}
    Final_Data[key]["parent"] = key
    Final_Data[key]["sons"] = sons
    Final_Data[key]["events"] = [key] + sons
    Final_Data[key]["data"] = {}
    Final_Data[key]["data"][key] = {}
    
    
    
    
    temp_sons.append(key)
    
    this_template = start_template + "parent event: " + key + "\n"
    
    for son in sons:
        
        if son != key:
            this_template = this_template + "son event: " + son + "\n"
            
            Final_Data[key]["data"][son] = {}
        
        temp_sons.append(son)
    
    this_template = this_template + "\n\n\nOutput:\n"
    
    all_demonstrations.append([this_template, temp_sons, key])
    
    
    
with open('./demonstrations.json', 'w') as fp:
    json.dump(all_demonstrations, fp)

with open('./Final_Data_step1.json', 'w') as fp:
    json.dump(Final_Data, fp)