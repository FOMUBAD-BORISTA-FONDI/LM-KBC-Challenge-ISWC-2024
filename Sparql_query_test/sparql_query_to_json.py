from wikidataintegrator import wdi_core

import json

# Define your SPARQL query
sparql_query = """
SELECT ?subjectEntity ?subjectEntityLabel ?objectEntity ?objectEntityLabel WHERE {
  VALUES (?subjectEntity ?objectEntity) {
    (wd:Q499707 wd:Q217475)
    (wd:Q54946455 UNDEF)
    (wd:Q682759 wd:Q13677)
    (wd:Q682759 wd:Q151139)
    (wd:Q682759 wd:Q82059)
    (wd:Q5216057 wd:Q217475)
    (wd:Q5135229 wd:Q891561)
    (wd:Q427800 UNDEF)
    (wd:Q109627057 wd:Q732670)
    (wd:Q4747070 wd:Q171240)
    (wd:Q7064390 UNDEF)
    (wd:Q56273715 wd:Q13677)
    (wd:Q1475554 wd:Q1930860)
    (wd:Q24358175 UNDEF)
    (wd:Q11510293 wd:Q217475)
    (wd:Q5987322 UNDEF)
    (wd:Q645708 wd:Q13677)
    (wd:Q645708 wd:Q936563)
    (wd:Q67810843 UNDEF)
    (wd:Q659230 UNDEF)
  }
  OPTIONAL {
    ?subjectEntity wdt:P414 ?objectEntity .
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
"""

result = wdi_core.WDItemEngine.execute_sparql_query(sparql_query)

json_filename = 'sparql_results.json'
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"Results saved to {json_filename}")
