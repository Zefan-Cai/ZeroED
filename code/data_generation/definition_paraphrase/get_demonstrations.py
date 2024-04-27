import json
import copy
import openai

openai.api_key = ""
openai.api_base = ""
openai.api_type = ""
openai.api_version = ""



with open('../generation_samples/Final_Data_step3.json', "r") as fp:
    final_data = json.load(fp)

all_demonstrations = []


for key in final_data.keys():
    
    for event in final_data[key]['events']:
        
        if "name" in final_data[key]['data'][event].keys():
    
            this_parent = final_data[key]['parent']
            sons = final_data[key]['sons']
            
            if event == this_parent:
                selected_sons = copy.deepcopy(sons)
                selected_sons = selected_sons[:5]
            else:
                sons_wo_event = [son for son in sons if son != event]
                selected_sons = sons_wo_event[:4]
                selected_sons.append(event)
            

            start_template = """
            Now I will give you an event ontology and its corresponding event definitions for each event.
            You need to output more event definitions for the target event.
            Your output should carefully focus on the difference between these events in the event ontology. The output event definitions for the target event should be different from event definitions for other events. And you need to try to give me diverse event definitions for the target event, but maintain the original semantics.




            Example 1
            Input:
            parent event:  Mental State: A mental state refers to an individual's current psychological condition or state of mind. It encompasses various emotions, thoughts, and cognitive processes that a person may experience at any given time.
            son event:  Belief: Belief is a mental state characterized by the acceptance or conviction that something is true or exists, often without concrete evidence. It involves having confidence in the validity of a particular idea, concept, or proposition.
            son event:  Doubt: Doubt is a mental state characterized by uncertainty or skepticism regarding the truth or validity of a belief or idea. It involves questioning or being unsure about the accuracy of a particular belief or assertion.
            son event:  Awakening: Awakening refers to a profound shift in an individual's perception or understanding. It is often associated with a realization or enlightenment, leading to a significant change in one's perspective or awareness, often of a spiritual or philosophical nature.
            son event:  Emotional State: An emotional state relates to a person's feelings and emotions at a specific moment. These states can include happiness, sadness, anger, fear, or any other emotional response to internal or external stimuli.
            son event:  Intention: Intention is a mental state involving a purposeful and deliberate desire to perform a particular action or achieve a specific goal. It is often accompanied by a plan or strategy to accomplish the intended outcome.


            Now please give me 10 more definitions for the event Awakening:
            Your output event definitions should be diverse, but maintain the semantics of the original definition. 
            Your output should follow the below format.


            Output: 
            1. Awakening is a transformative experience marked by a deep and sudden insight that fundamentally alters a person's way of thinking or viewing the world, frequently triggering a newfound sense of purpose or clarity.
            2. An Awakening signifies a profound shift in an individual's consciousness, usually triggered by a moment of profound understanding or revelation, which can lead to a complete reevaluation of their beliefs and values.
            3. The concept of Awakening encompasses a powerful, introspective moment where an individual experiences a heightened awareness of their existence, often bringing about a renewed sense of meaning and direction in life.
            4. Awakening is a term used to describe a radical change in an individual's perception or comprehension, often stemming from a moment of enlightenment or self-discovery, and it can lead to a profound transformation of their inner world.
            5. Awakening denotes a significant, often spiritual, awakening where an individual transcends their previous limitations, gaining a deeper understanding of the universe and their place in it, resulting in a newfound sense of inner peace and harmony.
            6. An Awakening is a pivotal event characterized by a shift in an individual's understanding of reality, often accompanied by a heightened sense of empathy, interconnectedness, and a more profound connection with the world around them.
            7. Awakening is a term that describes a momentous shift in an individual's perception and awareness, typically associated with a breakthrough in their understanding of life's mysteries, leading to a profound change in their personal philosophy and outlook.
            8. The experience of Awakening involves a profound and sudden revelation, often of a spiritual or philosophical nature, which can ignite a profound transformation in an individual's life, altering their perspective and values in a significant way.
            9. Awakening signifies a radical shift in an individual's consciousness, usually brought about by a moment of profound insight or self-realization, leading to a profound change in their beliefs, priorities, and sense of purpose.
            10. An Awakening is a momentous event in one's life where they undergo a profound shift in their perception and understanding of the world, often catalyzed by an epiphany or a deep introspective journey, resulting in a profound personal transformation.











            Now I will give you an event ontology and their corresponding event definitions.
            And your output should carefully follow the previous examples. Your output should carefully focus on the event ontology.
            You need to output more event definitions for the target event.
            Your output should carefully focus on the difference between these events in the event ontology. The output event definitions for the target event should be different from event definitions for other events. And you need to try to give me diverse event definitions for the target event, but maintain the semantics of the original definition.


            Input:
            """
            
            start_template += f"parent event: {final_data[key]['data'][this_parent]['name']}: {final_data[key]['data'][this_parent]['definition']}\n"
            
            # print(f"debug selected_sons {selected_sons}")
            
            for son in selected_sons:
                if "name" in final_data[key]['data'][son].keys():
                    start_template += f"son event: {final_data[key]['data'][son]['name']}: {final_data[key]['data'][son]['definition']}\n"

            start_template += f"""


            Now please give me 10 more definitions for the event {event}:
            Your output event definitions should be diverse, but maintain the semantics of the original definition. 
            Your output should follow the below format.


            Output:
            1.
            2.
            3.
            4.
            5.
            6.
            7.
            8.
            9.
            10.

            """


            final_data[key]['data'][event]["demonstration_sample"] = start_template


            all_demonstrations.append([start_template, final_data[key]['events'], key, event])
        
    
with open('./demonstrations.json', 'w') as fp:
    json.dump(all_demonstrations, fp)

with open('./Final_Data_step3.json', 'w') as fp:
    json.dump(final_data, fp)
    