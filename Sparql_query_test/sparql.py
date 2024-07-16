from SPARQLWrapper import SPARQLWrapper, JSON
from wikidataintegrator import wdi_core

list = ["(wd:Q499707 wd:Q217475)", "(wd:Q54946455 UNDEF)"]
value = "(wd:Q54946455 UNDEF)"
sparql_query = """
SELECT ?subjectEntity ?subjectEntityLabel ?objectEntity ?objectEntityLabel WHERE {{
  VALUES ?value {{
    "{0}"
  }}
  OPTIONAL {{
    ?subjectEntity wdt:P414 ?objectEntity .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
}}
""".format(value)

# Wikidata endpoint URL
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

sparql.setQuery(sparql_query)
sparql.setReturnFormat(JSON)

results = sparql.query().convert()

for result in results["results"]["bindings"]:
    id = result["subjectEntity"]["value"].split("/")[-1]
    label = result["subjectEntityLabel"]["value"]
    print(f"label = {label}, ID = {id}")

