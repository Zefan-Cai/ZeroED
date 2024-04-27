import json
import openai

openai.api_key = ""
openai.api_base = ""
openai.api_type = ""
openai.api_version = ""



with open('../generation_definitions_v2/Final_Data_step2.json', "r") as fp:
    final_data = json.load(fp)

all_demonstrations = []

print(final_data.keys())

for key in final_data.keys():
    
    for event in final_data[key]['events']:
        
        if "name" in final_data[key]['data'][event].keys():
    
            this_parent = final_data[key]['parent']
            sons = final_data[key]['sons']
            

            start_template = """
            Here we provide the definition and examples of three event types. The word between ＜KNOW＞ and ＜KNOW＞ is the trigger. The trigger must be a verb. Your output should be very diverse.


            Example 1
            Event: KNOW
            Definition: Know refers to the state of having knowledge or information about a particular subject, idea, or concept. It encompasses the understanding of facts, concepts, and principles, often obtained through learning, experience, or reasoning.
            Examples:
            1. She ＜KNOW＞realized＜KNOW＞ that she had been manipulated by her ex-boyfriend.
            2. The team's coach helped the players ＜KNOW＞understand＜KNOW＞ their weaknesses and work on improving them.
            3. After reading the article, he ＜KNOW＞learned＜KNOW＞ about the harmful effects of plastic on the environment.
            4. The company ＜KNOW＞discovered＜KNOW＞ a new market opportunity through market research.
            5. The students ＜KNOW＞recognized＜KNOW＞ the impact of their actions on the community after volunteering at a local shelter.
            6. The scientist ＜KNOW＞uncovered＜KNOW＞ a new method to reduce carbon emissions.
            7. The patient ＜KNOW＞realized＜KNOW＞ the importance of healthy lifestyle choices after being diagnosed with a chronic illness.
            8. The parents ＜KNOW＞discovered＜KNOW＞ their child's passion for music after seeing them perform in a school concert.
            9. The police officer ＜KNOW＞noticed＜KNOW＞ a suspicious individual and investigated further, leading to the prevention of a crime.


            Example 2
            Event: TRANSPORT
            Definition: A TRANSPORT Event occurs whenever an ARTIFACT (WEAPON or VEHICLE) or a PERSON is moved from one PLACE (GPE, FACILITY, LOCATION) to another.
            Examples:
            1. Zone escaped the incident with minor injuries, and Kimes was ＜TRANSPORT＞moved＜TRANSPORT＞ to the prison's disciplinary housing unit, the authorities said.
            2. The Palestinian leaders also warned that Israel must＜TRANSPORT＞remove＜TRANSPORT＞ its soldiers from the outskirts of Palestinian cities.
            3. Mr. Erekat is due to ＜TRANSPORT＞travel≤TRANSPORT＞ to Washington to meet with US Secretary of State Madeleine Albright and other US officials attempting to win a ceasefire.

            Example 3
            Event: DEMONSTRATE
            Definition: A DEMONSRATE Event occurs whenever a large number of people come a together in a public area to protest or demand some sort of official action.DEMONSTRATE Events include, but are not limited to, protests, sit-ins, strikes, and riots.
            Examples:
            1. Thousands of people ＜DEMONSTRATE＞rioted＜DEMONSTRATE＞ in Port-au-Prince, Haiti over the weekend.
            2. The union began its ＜DEMONSTRATE＞strike＜DEMONSTRATE＞ on Monday.
            3. Protesters ＜DEMONSTRATE＞rallied＜DEMONSTRATE＞ on the White House lawn.

            Please generating a new event type following the same format. 
            Note that there is an ontology, we provide event types and their definitions.




            """
            
            start_template += f"parent event: {final_data[key]['data'][this_parent]['name']}: {final_data[key]['data'][this_parent]['definition']}\n"
            start_template += f"son event: {final_data[key]['data'][event]['name']}: {final_data[key]['data'][event]['definition']}\n"
            
            count = 0
            
            for son in sons:
                if "name" in final_data[key]['data'][son].keys() and son != event and count < 5:
                    start_template += f"son event: {final_data[key]['data'][son]['name']}: {final_data[key]['data'][son]['definition']}\n"
                    count += 1

            start_template += f"""
            
            
            Please generate samples for event {final_data[key]['data'][event]['name']}.
            Your output should carefully consider the ontology. Your output examples should not be examples for other son events.


            Event: {final_data[key]['data'][event]['name']}
            Definition: {final_data[key]['data'][event]['definition']}
            Examples:
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

print(len(all_demonstrations))
    
with open('./demonstrations.json', 'w') as fp:
    json.dump(all_demonstrations, fp)

with open('./Final_Data_step3.json', 'w') as fp:
    json.dump(final_data, fp)