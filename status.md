# TODO:
- [~] Computer Vision
    - [x] Use PyQT to create an overlay that allows users to select parts of the screen to capture
    - [x] Use Tesseract-OCR to process selected parts as numbers
    - [x] Successfully able to get CS from top-right.
    - [~] Should also be able to get enemy/teammate CS via Scoreboard.
    - [ ] Potentially track appeareance and and locations of players via minimap
        - Very difficult
- [ ] ML
    - [ ] Figure out what to predict. Some ideas.
        - 
- [ ] Data 
    - [~] Determine where to get historical match data. 
        - OP.gg sounds good because of lane score
        - [x] Integrated with Riot's Dev API. 
            - Succesfully managed collected data, contains position of players! Useful.
            - Lacks granularity. Match data timesteps is 60s, that is too long. 
            - [~] Investigating replay data via Live Client API to see what kind of data we can get from there.
- [ ] Live Client
    - [x] Successfully integrated for real games, but data is too limited. Nothing useful can be collected durring a live match with Live Client. ~~(failed)~~
        - So in order to collect real-game data, we implemented a Computer Vision program to collect data by capturing it on screen.
    - [~] Working on integrating with replays.
        - [~] Need to develop a system that downloads replays, using the Live Client API to collect data with smaller timesteps. 
        - Download replay `/spectator/v4/download`. Parse the .rofl file with RoflParser or League of Stats. Processed and store data. Delete replay.



# NOTES: