"""
queries wikipedia for a list of pokemon names and generates a random code name from the list
used for bundling the slurm runs in an identifiable way
"""
import random
import pandas as pd
import re

def get_pokemon_names():
    """
    Retrieves the names of Pokémon from a Wikipedia page and returns them as a list.

    Returns:
        list: A list of Pokémon names.
    """
    url= "https://en.wikipedia.org/wiki/List_of_Pok%C3%A9mon"
    df = pd.read_html(url, attrs={"class": "wikitable"})[2]
    #df.info()
    #for each column in the dataframe, get the names
    names = []
    for column in df.columns:
        for name in df[column].values:
            if isinstance(name, str):
                if not name.isnumeric() and not name.startswith("No additional") and not name.startswith("("):
                    # strip any special characters at the end of the name
                    name = re.sub(r"\W+$", "", name)
                    name = name.replace("※", "")
                    name = name.replace("‡", "")
                    name = name.replace("†", "")
                    name = name.replace("~[d]", "")
                    name = name.replace("♭[e]", "")
                    name = name.replace("·", "")
                    name = re.sub(r"\[♭\[e\]]", "", name)
                    name = name.replace(" ", "_")
                    #print(name)

                    names.append(name)
    return names

def generate_code_name():
    """
    Generates a random code name using a list of Pokemon names.

    Returns:
        str: A randomly chosen code name.
    """
    code_names = get_pokemon_names()
    return random.choice(code_names)

code_name = generate_code_name()
print(code_name)
exit(0)