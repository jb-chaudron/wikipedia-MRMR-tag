import re


def get_year_month(date_string: str) -> tuple:

    # Extract the month name and year using regex
    match = re.search(r"\d{1,2} (\w+) (\d{4})", date_string)
    if match:
        month_name = match.group(1)
        year = match.group(2)
    else:
        month_name = None
        year = None

    # Map French month names to their corresponding month numbers
    month_mapping = {
        "janvier": 1,
        "février": 2,
        "mars": 3,
        "mar" : 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "août": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "décembre": 12
    }

    # Get the numerical equivalent of the month
    month_number = month_mapping.get(month_name)

    # Output the results
    return month_number, year

def date_check(date : str, 
               mois : str, 
               an : str) -> bool:
    m, a = get_year_month(date.lower())
    #print(date,m, a)
    return ((mois[0] <= m) and (an >= int(a)))

def articles_to_qual_vect(adq_dict : dict, 
                          ba_dict : dict, 
                          mois : int, 
                          an : int, 
                          articles : list) -> list:
    out = []
    #print(mois, an)
    for article in articles:
        
        out.append(int( 
            (
                (date_check(adq_dict[article],mois, an) if article in adq_dict.keys() else False) 
                    or 
                (date_check(ba_dict[article],mois, an) if article in ba_dict.keys() else False)
            )
                    )
                        )
    return out 

