import openai
from dotenv import dotenv_values

'''
Give it the entire article and prompt it to generate a list of 
the people in each paragraph and their affiliation 
    - Either Punishment Bureaucracy (includes police, the mayor, police oversight bodies)
    - Civilian aligned (victim, victim's family, attorney, civil rights activists)
    - {
        "Civilian-aligned": [Brydon Whitstone, Darren Stanley, 
            Witnesses] 
        "Punishment Bureaucracy": [Maureen Levy (RCMP Chief), Drew Wilby (Justice Ministry Spokesman), ...]
    }
    - [
        [,..., "Drew Wilby", "Drew Wilby", "Drew Wilby", "Levy", "Levy", "Context"]
    ]
===========================      PROMPT      =============================================

# NOTE
## currently: testing with Dexter Reed article. Able to send and receive but Task 1 seems to miss entities, Task 2 skipped paragraphs.
## Questions:
## Naming entities - I'm not sure that my descriptions of affiliations are sufficient
## I said "dominant entity" as the person whose perspective is in that paragraph since the paragraph could mention several entities
## for part b (testing) do I just do that manually on a subset of the articles we have so far
## TLDR i dont think my prompt is precise or robust enough so results are not great

# make task 2 its own prompt and then 

# task 2 could be a list of lists 
# so one paragraph = 1 list 

'''
prompt = """
Read the provided article on an incident of police violence to complete the following two tasks. Task 1: Identify each person mentioned in the article and determine their affiliation. The same person could be mentioned many times by different referring expressions.

For instance, the terms "he", "Bob Dylan", "the singer", and "Dylan" all refer to the same entity.

Use the article context to recognize these mentions as co-references and list the person only once using their most complete name (e.g. "Bob Dylan"). Do not create separate entities for pronouns, mentions, and names (e.g. "he", "the singer", "Bob Dylan", etc). A person can either be on the side of the police, or the civilian victim.

For example, government officials including police chiefs, mayors, police oversight officials, police unions officials. Government officials should be assigned into the police-aligned category, even if their statements are ostensibly sympathetic. There may also be bystanders who take the side of the police. For these bystanders, it is important to take the quotes into account in determining whether they side with the police or not.  

People who side with the civilian victim, on the other hand, tend to be, of course, the victim's family or friends, their attorney, community members who knew the victim, witnesses, civil rights organization leaders, and more.

People mentioned from previous cases (whether officers or victims) should also be recorded, even if they are not part of the current case itself. 

{

"Victim-aligned": [Person1, Person2, Person3],

"Government officials": [Person4, Person5, Person6],

"Policing institution officials": [Person 7, Person8]

}

Here is the article:
"""


def load_client():
    config = dotenv_values(".env")
    key = config["OPENAI_API_KEY"] 
    client = openai.AzureOpenAI(
            azure_endpoint="https://ubcnlpgpt4.openai.azure.com/", 
            api_key=key,
            api_version="2023-05-15" 
        )
    return client

def main():
    with open("./data/articles/Dexter_Reed-2024_04_10-foody.json", "r") as f:
        article = f.read()
    message = prompt + article
    client = load_client()
    response = client.chat.completions.create(
            model="gpt-4o", # gpt-4o, or you can also try gpt-4o-mini. See here for more details https://platform.openai.com/docs/models. But stick to 4o or 4o-mini
            temperature=0,
            max_tokens = 500, # set this to whatever makes sense. It's the maximum number of tokens (basically, words) that you think are necessary for responding to the prompt
            messages = [
                {"role": "user", "content" : message}
            ]
    )
    print(response.choices[0].message.content)
    
if __name__ == "__main__":
    main()