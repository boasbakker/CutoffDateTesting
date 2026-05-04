- Detailbeeld van cutoff Gemini 3 Flash, test eerst goede views cutoff (zie jan 2022 document)
- Gemini 3 pro / gpt-5 mini / gpt-5-nano / gpt-5.2 pro / oudere modellen testen. Dan eerst prompts vinden om reasoning uit te zetten, per model een aparte prompt maken. Ook testen of reasoning helpt is interessant later. 
- Games, muziek, films, boeken met een release date is een alternatief voor overlijdensberichten. Maar wel minder goed: die dingen worden van tevoren aangekondigd
- Vragen naar geboortejaar van het persoon, en als die juist is, vragen of ie nog leeft
  - Op deze manier test je of de LLM het persoon "kent". Dit zou ook kunnen door gewoon te vragen "do you know ...", maar dan kunnen ze liegen.
  - Het is misschien handig om, zeker voor Claude modellen, die eerste vraag nog in context te houden voor de tweede vraag, om "I don't know" te voorkomen.
  - Prompt kan ook aangepast worden: "According to your most recent knowledge, is ... still alive?"
  - Dit kan ook losgekoppeld worden naar een aparte benchmark: PeopleBench, die test hoeveel personen een LLM kent. Kan uitgedrukt worden als percentage van Wikipedia. 
- publiceren? Kijken of dit nieuwe resultaten zijn. https://malihehizadi.github.io/ Nadeel: zij focust vooral op software 

GROK TESTEN

> TU Delft BSc/MSc students: TU Delft students interested in LLMs and Software Engineering are welcome to join my lab for thesis projects, internships, or research. Contact me via email to schedule a meeting.
Zij doe

- Claude toont een sterke US-bias: Kennis over de dood van Indian actors is zeer slecht, maar ook de kennis van de geboortejaren van die personen lijkt slecht te zijn. https://en.wikipedia.org/wiki/Siddique_(director) Deze is geboren in 1955, Claude 4.5 Opus beweert 1959, zelfs met nadenken. 
- Performance voor de eerste paar doden die hij fout had neemt wel flink toe met reasoning voor Claude Opus 4.5. Ik denk dat er niet veel tokens voor nodig zijn, 100 is al wel genoeg. 
`Greg Gumbel` weet ie niet zelfs met reasoning (december 2024), `Bob Bryar` (november 2024) weet ie met reasoning wel, zonder reasoning niet. 


Voor gemini flash: detail van 2024-10 tot 2025-2 nemen

Gemini lijkt alles met >10000 views te kennen dus dan zouden zelfs dagelijkse resultaten mogelijk moeten zijn 