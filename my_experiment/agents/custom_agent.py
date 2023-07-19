from matrx.actions import Action, ActionResult, GrabObject, DropObject
from my_experiment.custom_actions import OpenDoor, EarthquakeEvent, OpenCollapsedDoor, MagicalDoorAppear, ManageRobotImg #, DragVictim
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.agents import AgentBrain
from matrx.messages import Message
from my_experiment.objects import Victim
from matrx.utils import get_distance

import pickle
import json
import os

class CustomAgent(AgentBrain):
    """ An artificial agent whose behaviour can be programmed to be, for example, (semi-)autonomous.
    
    For more extensive documentation on the functions below, see: 
    http://docs.matrx-software.com/en/master/sections/_generated_autodoc/matrx.agents.agent_brain.AgentBrain.html#matrx.agents.agent_brain.AgentBrain
    """


    def __init__(self, max_carry_objects=1, waypoints=[], move_speed=0, **kwargs):
        """ Creates an agent brain to move along a set of waypoints.
        """
        super().__init__(**kwargs)
        # self.visualize_when_busy = visualize_when_busy
        self.navigator = None
        self.move_speed = move_speed
        self.__max_carry_objects = max_carry_objects
        self.__vics_gezien = [] #lijst van tuples met alle victims. Elke tuple bevat alle victim informatie
        self.__modderlocations = [] #list of locations of all the mud tiles
        self.doors_visited = [] #list of all the door locations of all the rooms that are visited by the human
        self.waypoints_to_add = [] #als de robot iets tegenkomt wat prioriteit heeft over de huidige agent loop, dan voegt hij de nieuwe te bezoeken plekken toe aan deze lijst. Deze lijst wordt vervolgens vooraan in de waypoint lijst toegevoegd.
        self.door_loc = [] #list of all the door locations in the world
        self.human_help_needed = False # Deze parameter geeft aan of de mens de robot vraagt (via cue, assign of explain) om te helpen een slachtoffer te tillen. Die blijft op True totdat ze samen het slachtoffer weer neerleggen. Zie REF-T07.
        self.wall_locs = [] #lijst van alle locaties waarop een wall zit
        self.carry_together_vic = "" # Deze var wordt gebruikt door de robot om het slachtoffer dat samen gedrag moet worden in op te slaan.
        self.carry_door = False #zijn de robot en mens samen een deur aan het dragen
        self.add_waypoints_door = False #eenmalig moeten een aantal waypoints toegevoegd worden, deze boolean houdt bij of dit al gebeurd is of niet.
        self.waypoint_list_reset = [] #om te voorkomen dat de robot na het samen dragen van de deur nog heen en weer gaat lopen voor de deur wordt de waypoint lijst voordat de robot begint met heen en weer lopen hier opgeslagen bug sensitive code
        self.earthquake = False
        self.broken_door_timer = 0 #birgit nog uitleggen
        self.earthquake_timer = 100
        self.moving_to_door = False
        self.carry_together = False # Wordt gebruikt door de robot om bij te houden of hij samen aan het dragen is of niet.
        ## Tjeerd: onderstaande 5 variabelen kunnen weg als we willen dat de robot niet het tillen van een slachtoffer initieert nadat hij dit heeft geleerd (zie REF-T04).
        self.wait_for_human_arrival = False
        self.count_ticks = 0
        self.asked_human_help_carry = False
        self.wait_for_response_human = False
        self.wait_time_response_human = 100
        ## Tjeerd: tot hier kan dus weg
        self.learned_carry = False # houdt bij of de robot samen dragen heeft geleerd of niet. Wordt gelogd aan einde spel.
        self.learned_mud = False # houdt bij of de robot modder vermijden heeft geleerd of niet. Wordt gelogd aan einde spel.
        self.changed_navigator_mud = False #twijfelachtig of deze code nodig is. Het houdt iig bij of de navigator veranderd is doordat de robot modder geleerd heeft (maar dat wordt volgens mij nergens voor gebruikt)
        self.human_to_door_prev = None
        self.human_to_agent_prev = None  #equavalent van human_to_door_prev. Als het goed is is dit de afstand van de mens tot de agent
        self.carry_door_behavior = False #zijn (deels) van de interventies geactiveerd? #birgit nog uitleggen
        self.last_eq_cue = 'behavioral'
        self.last_bd_cue = 'behavioral' #wat is de huidige interventie die plaatsvind
        self.door_target = [] #plek waar de deur neergelegd wordt.
        self.drag_vic = False #is robot op dit moment een zwaargewond slachtoffer aan het slepen
        self.drag_target = [] #locatie waar het zwaargewonde slachtoffer dat gesleept wordt neergelegd moet worden.
        self.seismograaf_on = True
        self.einde = False # Heeft de robot al zijn taken afgerond (waypoint lijst leeg en niet met human iets aan het doen
        self.human_at_door = False #deze variabele wordt in 2 if statements gecheckt en op True gezet als nodig, is het niet handiger dit in 1x in de filter_observations te doen?

        # ----------------------------------------------------------------------------------------------------
        #In dit blok staan de variabelen die volgens mij niet gebruikt worden
        self.move_command_loc = []
        self.victim_list = []
        self.first_time = True
        # self.victim_found = None
        self.wall_locs = [] #staat er dubbel in (ook in het lijstje hierboven)
        self.waypoint_list_reset_2 = []
        self.turn_cog_invisible = False
        # ----------------------------------------------------------------------------------------------------

        # Code to retrieve info about learned situations if available
        # Als het leerbestand al is aangemaakt (zie REF-T08 in main), dan moet deze uitgelezen worden. Op deze manier
        # neemt de robot het geleerde gedrag mee over de scenarios.
        # Zie REF-T09 in co_learning_logger voor het opslaan van het geleerde gedrag in de pickle file.
        if os.path.isfile('./learned_backup.pkl'):
            with open ('learned_backup.pkl', 'rb') as pickle_file:
                pickle_contents = pickle.load(pickle_file)
            if len(pickle_contents) > 1:
                self.learned_carry = pickle_contents[0]
                self.learned_mud = pickle_contents[0]

    def initialize(self):
        """ This method is called each time a new world is created or the same world is reset. This prevents the agent to
        remember that it already moved and visited some waypoints.

        Resets the agent's to be visited waypoints.
        """
        # Initialize this agent's state tracker, needed for the navigator
        self.state_tracker = StateTracker(agent_id=self.agent_id)
        self.navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                   algorithm=Navigator.A_STAR_ALGORITHM)


    def filter_observations(self, state):
        """ Filters the world state before deciding on an action. """
        # update state tracker for navigator
        self.state_tracker.update(state)
        #een lijst met alle victims in de wereld.
        victim_list = state.get_of_type(Victim.__name__)

        closest_door = state.get_closest_room_door() #het kan misschien handig zijn om hier een self variabele van te maken gezien dit heel veel gebruikt wordt
        # elke tick wordt gekeken of er een nieuwe victim gespot is. Elke tick wordt deze leeg begonnen.
        new_victim = []
        if closest_door[0]["location"] == self.agent_properties["location"] and closest_door[0]["location"] not in self.doors_visited:
            #moet in deze if-statement niet ook "and closest_door[0]["location"] != (24,4) (CP room door)"
            room_name = closest_door[0]["room_name"]
            room_objs = state.get_room_objects(room_name)
            # kijk of er victims in de room_objs zijn:
            vics_in_room = [obj for obj in room_objs if "vic" in obj["name"]]
            #als er victims in de room zijn voeg deze toe aan de new_victim lijst en ook aan de __vics_gezien lijst
            # of het een nieuwe vic is wordt bijgehouden zodat hieronder (4 for loops verderop in deze filter_observations) de mens kan worden bericht over nieuw gevonden victims
            for vic in vics_in_room:
                if vic not in self.__vics_gezien:
                    new_victim.append(vic)
                    self.__vics_gezien.append(vic)
                    self.asked_human_help_carry = False
            #robot visited the room. Add room to the list doors_visited
            self.doors_visited.append(closest_door[0]["location"])
            # de lijst van victims wordt uitgelezen om te kijken welke victims zichtbaar zouden moeten zijn voor de robot
            # elke victim die al wel gevonden is wordt uit de lijst victim_list verwijderd
            # na deze for loops bestaat de lijst victim_list dus uit alle victims die nog NIET gevonden zijn
        for vic_g in self.__vics_gezien:
            for vic_l in victim_list:
                if vic_g['name'] == vic_l['name']:
                    victim_list.remove(vic_l)
        # als de mens een bericht stuurt van een gevonden vic wordt deze ook uit de victim_list verwijderd.
        for mssg in list(self.received_messages):
            for vic_l in victim_list:
                if vic_l['name'] == mssg.content:
                    victim_list.remove(vic_l)
        # als de robot een victim vind stuurt hij een bericht naar de mens met de naam van de victim:
        for vic in new_victim:
            self.send_message(
                Message(content=f"{vic['name']}",
                        from_id=self.agent_id,
                        to_id=None))  # None = all agents

            # REF-T04
            # Tjeerd: onderstaande code is voor de situatie waarin de robot al heeft geleerd om samen een slachtoffer te dragen.
            # Als de robot dan in het vervolg een zwaargewond slachtoffer tegenkomt, vraagt hij aan de mens om hem te komen helpen tillen.
            # Deze code heb ik echter uitgecomment om twee redenen: 1) nadat de mens heeft geleerd over het dragen van een deur, kan de mens ook
            # niet aan de robot vragen of hij komt helpen als hij een kapotte deur tegenkomt. 2) het stukje code veroorzaakte problemen in bepaalde
            # situaties met combinaties van leer-events.
            # Wat mij betreft laat je deze dus uit de nieuwe code.

            # # If robot finds heavily wounded victim and has learned that it should carry them together, ask human
            # if self.learned_carry and vic['treatment_need'] == 2 and not self.asked_human_help_carry:
            #     self.asked_human_help_carry = True
            #     self.send_message(Message(content={"chat_text": "Ik heb een zwaargewond slachtoffer gevonden! "
            #                                                     "Wil je me komen helpen tillen?", "notice": "help_robot_carry"},
            #                 from_id=self.agent_id,
            #                 to_id=None))
            #     self.carry_together_vic = vic
            #     self.wait_for_response_human = True

        # remove all the victims that the robot or human haven't found yet
        # alle victims die nog niet gevonden zijn worden uit de wereld verwijderd.
        [state.remove(vic["obj_id"]) for vic in victim_list]

        #------------------------------------- deuren al bezocht door mens------------------------------------------------
        #als de mens in een deuropening staat stuurt hij een bericht naar robot welke deuropening (locatie deur) het is en verwijderd de robot deze locatie en de locatie vlak voor de duer uit zijn WP lijst
        # ook mist hier de self.received_messages.remove(message)
        for mssg in list(self.received_messages):
            # Tjeerd: added this if statement, because agent kept looping through all messages and all points at each tick
            if mssg.from_id == self.agent_id:
                for point in self.navigator.get_all_waypoints(self.state_tracker):
                    point_s = str(point[1])
                    if mssg.content == point_s:
                        waypoints = []
                        for loc in self.door_loc:
                            if loc == point[1]:
                                # print(self.wall_locs[45])
                                # print(loc[0]-3,loc[1])
                                if (loc[0]-3) >0 and (loc[0]-3,loc[1]) in self.wall_locs:
                                    #"bottom"
                                    point_t = (loc[0],loc[1]+1)
                                if (loc[0]+3) >0 and (loc[0]+3,loc[1]) in self.wall_locs:
                                    #"top"
                                    point_t = (loc[0],loc[1]-1)
                                if (loc[1]-3) >0 and (loc[0], loc[1]-3) in self.wall_locs:
                                    #"left"
                                    point_t = (loc[0]-1,loc[1])
                                if (loc[1]+3) >0 and (loc[0], loc[1]+3) in self.wall_locs:
                                    #"right"
                                    point_t = (loc[0]+1,loc[1])

                        for point in self.navigator.get_upcoming_waypoints(self.state_tracker):
                            if mssg.content != str(point[1]) and point_t != point[1]:
                                waypoints.append(point[1])
                        self.navigator.reset_full()
                        self.navigator.add_waypoints(waypoints, is_circular=False)
        # ----------------------------------------------------------------------------------------------------------------

        # REF-T05
        #----------------------------------bepaal hoe de robot eruit ziet in het spel--------------------------------------
        if self.agent_properties['is_carrying'] and not self.drag_vic:
            #robot draagt alleen een lichtgewond slachtoffer
            self.agent_properties['img_name'] = 'robot_carry.png'
        elif self.carry_together == True:
            #robot en mens dragen samen een zwaargewond slachtoffer.
            self.agent_properties['img_name'] = 'invisible.png'
        elif self.agent_properties['is_carrying'] and self.drag_vic:
            #robot sleept een zwaargewond slachtoffer
            self.agent_properties['img_name'] = 'robot_drag_wounded.png'
        elif self.carry_door:
            #robot en mens dragen samen een deur
            self.agent_properties['img_name'] = 'carry_door.png'
        else:
            #als geen van de bovenstaande dingen geld dan is de robot het normale robot icoontje.
            self.agent_properties['img_name'] = 'robot.png'
        # ----------------------------------------------------------------------------------------------------------------

        return state

    def first_tick(self, state):
        #verstuur bij de eerste tick een verzoek of de mens bij het postkantoor wilt beginnen en dat jij bij de dorpsstraat begint
        msg = "Ik begin bij de dorpsstraat, wil jij beginnen bij het postkantoor?"
        self.send_message(
            Message(content={"chat_text": msg},
                    from_id=self.agent_id,
                    to_id=None))  # None = all agents
        #------------ bepaal alle deur locaties in de wereld-------------------------------------------------------------
        room_names = self.state.get_all_room_names()
        room_names_real = [x for x in room_names if x is not None]
        room_names_real.sort()
        for n in room_names_real:
            for doors in self.state[{'class_inheritance': 'Door'}]:
                # print(doors['location'])
                if doors['room_name'] == n:
                    self.door_loc.append(doors['location'])
        # print(self.door_loc)
        # ----------------------------------------------------------------------------------------------------------------




        #------------------------------ bepaal initiele waypoints van robot (=alle gebouwen afgaan)----------------------
        # De robot gaat langs alle gebouwen uit de wereld:
        # Indien het gebouw is ingestort, voegt de robot alleen de locatie van het vakje vóór de deur toe
        # Indien het gebouw niet is ingestort voegt de robot ook de locatie van de deur zelf toe.
        # LET OP: aanname is dat de deur altijd aan de lange kant van het gebouw zit en het gebouw een afmeting van 5,6 heeft) (aanname niet meer geldig als je custom property heeft "door_side" gebruikt)
        self.wall_locs = [] #ik vraag me af of dit een self. hoeft te zijn
        walls = self.state.get_of_type('Wall')
        new_waypoints = []
        for wall in walls:
            self.wall_locs.append( wall['location'])
        for loc in self.door_loc: #loop door alle deurlocaties
            for door in self.state["is_open"]: #loop door alle objects in de werelf die een property hebben die "is_open" heet (dit zijn alle deuren) #bug sensitive code
                if door['location'] == loc:
                    #volgens mij zijn deze if-statements niet meer zo ingewikkeld nodig omdat een deur tegenwoordig de custom property heeft "door_side". Deze kan ook uitgelezen worden.
                    if (loc[0]-3) >=0 and (loc[0]-3,loc[1]) in self.wall_locs: #als dit specifieke vakje een muur is dan zit de deurpositie aan de bottom (zelfde effect als if door["door_side"] == "bottom")
                        #bottom
                        new_waypoints.append((loc[0], loc[1]+1)) #voeg vakje voor deur toe aan waypoints
                        if door['is_open']:
                            new_waypoints.append(loc) #voeg locatie deur toe aan waypoints, alleen als het geen ingestort gebouw is
                    if (loc[0]+3) >=0 and (loc[0]+3,loc[1]) in self.wall_locs:#als dit specifieke vakje een muur is dan zit de deurpositie aan de top (zelfde effect als if door["door_side"] == "top")
                        #top
                        new_waypoints.append((loc[0], loc[1]-1)) #voeg vakje voor deur toe aan waypoints
                        if door['is_open']:
                            new_waypoints.append(loc)  #voeg locatie deur toe aan waypoints, alleen als het geen ingestort gebouw is
                    if (loc[1]-3) >=0 and (loc[0], loc[1]-3) in self.wall_locs: #als dit specifieke vakje een muur is dan zit de deurpositie aan de left (zelfde effect als if door["door_side"] == "left")
                        #left
                        new_waypoints.append((loc[0]-1, loc[1])) #voeg vakje voor deur toe aan waypoints
                        if door['is_open']:
                            new_waypoints.append(loc) #voeg locatie deur toe aan waypoints, alleen als het geen ingestort gebouw is
                    if (loc[1]+3) >=0 and (loc[0], loc[1]+3) in self.wall_locs:#als dit specifieke vakje een muur is dan zit de deurpositie aan de right (zelfde effect als if door["door_side"] == "right")
                        #right
                        new_waypoints.append((loc[0]+1, loc[1])) #voeg vakje voor deur toe aan waypoints
                        if door['is_open']:
                            new_waypoints.append(loc) #voeg locatie deur toe aan waypoints, alleen als het geen ingestort gebouw is
        self.navigator.add_waypoints(new_waypoints, is_circular=False) #voeg de waypoints toe aan de navigator
        # print(new_waypoints)
        # ----------------------------------------------------------------------------------------------------------------


        # lijst maken waarin alle locaties met modder staan
        modderlijst = state.get_with_property({"name": "modder"}, combined=False)
        if modderlijst:
            for mud in modderlijst:
                self.__modderlocations.append(mud['location'])


    def decide_on_action(self, state):
        """ Contains the decision logic of the agent. """
        #---------------------- modder vertraagd snelheid----------------------------------------------------------------
        if self.agent_properties["location"] in self.__modderlocations and not self.learned_mud:
            action_kwargs = {"action_duration": 15} #let op weer op 15 zetten voor echte experiment
            # log that robot moved over a mud tile
            # Tjeerd: hiermee logt de robot of hij over een modderpad loopt. Dit wordt opgeslagen in een variabele, die op de
            # laatste tick van het spel wordt toegevoegd aan het log-bestandje (zie co_learning_logger.py).
            if self.previous_action and 'move' in str.lower(self.previous_action):
                self.agent_properties['log_robot_mud'] += 1
        else:
            action_kwargs = {"action_duration": 5}
        # ---------------------------------------------------------------------------------------------------------------

        # alle acties die gedaan eenmalig moeten worden, zoals het toevoegen van begin waypoints en defineren van modder
        if state['World']['nr_ticks'] == 0:
            self.first_tick(state)
        action_s = False #variabele die een tijd lang gebruikt is om te bepalen of de robot al een actie gekozen had. Is denk ik overbodig in nieuwe code.

        # closest door
        closest_door = state.get_closest_room_door() #wordt in filter_observations ook al elke tick aangevraagd

        #onderstaande variabele (in comment) wordt niet meer gebruikt.
        # all_vics = self.state.get_closest_with_property("treatment_need")

        human = state.get_with_property({"name": "rescue_worker"})
        loc_human = human[0]['location']  # locatie van de human

        #------------------------ bepaal welke van de zichtbare victims er zwaargewond en lichtgewond zijn -------
        # zwaargewonde slachtoffers staat iets verder op. Zie #REF birgit 001
        #list of all heavily wounded victims and all lightly wounded victims in the world
        vics_lightly = self.state[{"treatment_need": 1}]
        vics_heavily = self.state[{"treatment_need": 2}]
        #every tick vics_need_rescue and vics_to_drag are made empty
        vics_need_rescue = []
        vics_to_drag = []

        #kijk voor elke lichtgewonde victim of deze niet al in de commando post ligt
        #indien niet in commandopost, voeg toe aan vics_need_rescue
        # hier wordt pas veel later iets mee gedaan: # REF birgit 003
        if vics_lightly and isinstance(vics_lightly, dict):
            for obj in self.state.get_room_objects(room_name='command_post'):
                if vics_lightly["obj_id"] != obj["obj_id"] and vics_lightly not in vics_need_rescue:
                    vics_need_rescue.append(vics_lightly)
                    #code klopt niet!!!!!
        if vics_lightly and isinstance(vics_lightly, list):
            for vic in vics_lightly:
                if vic not in self.state.get_room_objects(room_name='command_post'):
                    vics_need_rescue.append(vic)

        # print("locs_vics_lightly: ")
        # print(vics_lightly)
        # if vics_heavily:
        #     for n in self.state.get_all_room_names():
        #         for obj in self.state.get_room_objects(room_name=n):
        #             if obj in vics_heavily: #and not self.state[{"room_name": n}]["door_open"]:
        #                 a = self.state[{"room_name": n}][-1]
                        # b = self.state.get_room(room_name=n)["doors_open"]
                        # vics_to_drag.append(obj)
                        # print("vics_to_drag")
        # ---------------------------------------------------------------------------------------------------------------

        # ------------------------Bepaal aan de hand van de waypoint lijst of er waypoints weggegooid mogen worden--------
        # Als de robot een slachtoffer wilt redden die al door de mens opgepakt is
        # Als de robot op weg is naar een deur die al door de mens is open gemaakt

        # maak een lijst van de upcoming waypoints
        if len(self.navigator.get_upcoming_waypoints(self.state_tracker))>1:
            tj, upcoming_way = list(zip(*self.navigator.get_upcoming_waypoints(self.state_tracker)))#[1]
        else:
            upcoming_way = None

        #als het punt (24,2) (dit is de drop locatie voor victims voor de robot)voorkomt in de upcoming waypoints en het waypoint daarvoor is geen lightly wounded victim: verwijder dit waypoint
        # bug sensitive code
        if upcoming_way and (24,2) in upcoming_way:
            index = upcoming_way.index((24,2))
            # index = self.navigator.get_upcoming_waypoints(self.state_tracker).index((24,2))
            if index != 0:
                vic_at_location = False #initiate with false
                for vic in vics_need_rescue:
                    if upcoming_way[index-1] == vic["location"]:
                        vic_at_location = True #als het waypoint voor 24,2 een victim is
                if not vic_at_location and not self.human_help_needed: #als er geen victim ligt op waypoint voor 24,2 verwijder dan het waypoint
                    list_upcoming_way = list(upcoming_way)
                    list_upcoming_way.pop(index) #remove waypoints
                    list_upcoming_way.pop(index - 1)
                    self.navigator.reset_full()
                    self.waypoints_to_add = self.waypoints_to_add + list_upcoming_way #voeg aan het einde van de decide_on_action de waypoints toe aan de navigator
                    #Bovenstaande kan misschien wel beter door gelijk de navigator waypoints te vullen ipv alles aan de self.waypoints_to_add toe te voegen
                    # bijvoorbeeld door: self.navigator.add_waypoints(self.waypoints_to_add, is_circular=False)
                    # bug sensitive code
        #als de agent op weg is naar een deur die al open gemaakt is, verwijder de deuropening uit de waypoints
        #dit zou wel eens kunnen conflicteren met de aardbeving schuilen, gezien de agent dan juist in een open deur moet gaan staan
        # bug sensitive code
        #deze code is erbij gekomen zodat de agent niet naar achter de mens aan loopt of een leeg gebouw dat de mens al heeft gecheckt alsnog gaat checken.
        if upcoming_way:
            if upcoming_way[0] in self.door_loc:
                door = self.state[{"location": upcoming_way[0]}]
                if isinstance(door, dict):
                    if door["door_opened"] and door["room_name"]!= "command_post":
                        list_upcoming_way = list(upcoming_way)
                        list_upcoming_way.pop(0)
                        self.navigator.reset_full()
                        self.waypoints_to_add = self.waypoints_to_add + list_upcoming_way
                        # Bovenstaande kan misschien wel beter door gelijk de navigator waypoints te vullen ipv alles aan de self.waypoints_to_add toe te voegen
                        # bijvoorbeeld door: self.navigator.add_waypoints(self.waypoints_to_add, is_circular=False)
                        # bug sensitive code
                elif isinstance(door, list):
                    if door[0]["door_opened"] and door[0]["room_name"] != "command_post":
                        list_upcoming_way = list(upcoming_way)
                        list_upcoming_way.pop(0)
                        self.navigator.reset_full()
                        self.waypoints_to_add = self.waypoints_to_add + list_upcoming_way
                        # Bovenstaande kan misschien wel beter door gelijk de navigator waypoints te vullen ipv alles aan de self.waypoints_to_add toe te voegen
                        # bijvoorbeeld door: self.navigator.add_waypoints(self.waypoints_to_add, is_circular=False)
                        # bug sensitive code
            #indien het vakje voor de deur ook nog in de waypoint lijst staat, verwijder ook dat vakje voor de deur.
            elif len(upcoming_way)>1 and upcoming_way[1] in self.door_loc:
                if get_distance(upcoming_way[1], upcoming_way[0])<2:
                    door = self.state[{"location": upcoming_way[1], "is_open": [True, False]}]
                    if isinstance(door, dict):
                        if door["door_opened"] and door["room_name"] != "command_post":
                            list_upcoming_way = list(upcoming_way)
                            list_upcoming_way.pop(0)
                            list_upcoming_way.pop(0)
                            self.navigator.reset_full()
                            self.waypoints_to_add = self.waypoints_to_add + list_upcoming_way
                            # Bovenstaande kan misschien wel beter door gelijk de navigator waypoints te vullen ipv alles aan de self.waypoints_to_add toe te voegen
                            # bijvoorbeeld door: self.navigator.add_waypoints(self.waypoints_to_add, is_circular=False)
                            # bug sensitive code
                    elif isinstance(door, list):
                        if door[0]["door_opened"] and door[0]["room_name"] != "command_post":
                            list_upcoming_way = list(upcoming_way)
                            list_upcoming_way.pop(0)
                            list_upcoming_way.pop(0)
                            self.navigator.reset_full()
                            self.waypoints_to_add = self.waypoints_to_add + list_upcoming_way
                            # Bovenstaande kan misschien wel beter door gelijk de navigator waypoints te vullen ipv alles aan de self.waypoints_to_add toe te voegen
                            # bijvoorbeeld door: self.navigator.add_waypoints(self.waypoints_to_add, is_circular=False)
                            # bug sensitive code
        # ---------------------------------------------------------------------------------------------------------------


        # ------------------------------als er zwaargewonde slachtoffers zichtbaar zijn voor de robot---------------------------------------------------------------------------------
        #REF birgit 001
        #hier wordt pas verderop iets mee gedaan (ik weet niet waarom deze code zo verspreid staat) #REF birgit 002
        if vics_heavily: #als er zwaargewonde slachtoffers in de wereld aanwezig zijn.
            if closest_door[0]["img_name"] == "invisible.png": #indien het om een zwaargewond slachtoffer gaat in een ingestort gebouw moet er gesleept worden.
                for n in self.state.get_all_room_names():
                    for obj in self.state.get_room_objects(room_name=n):
                        if isinstance(vics_heavily, dict): #vics_heavily is soms een dict van lijsten, maar als er maar 1 zwaargewond slachtoffer is dan is het een list
                            if obj == vics_heavily:
                                vics_to_drag.append(obj) #deze victims moeten gesleept worden
                        if isinstance(vics_heavily, list):
                            if obj in vics_heavily:
                                vics_to_drag.append(obj)
                                # print("vics_to_drag")

        # -------------------- Robot wacht op reactie van de mens op verzoek om samen te dragen -------------------
        # Tjeerd: Alle code onder de "if self.wait_for_response_human" wordt nooit aangeroepen, omdat deze variabele
        # pas op True gaat zodra: de robot heeft geleerd om samen te tillen, de robot een nog-niet-gered zwaargewond slachtoffer
        # tegenkomt, en de mens het verzoek van de robot om samen te tillen heeft geaccepteerd. (zie REF-T04)
        # Onderstaande code kan dus weg als we besluiten dat de robot dit verzoek nooit zal sturen, en de mens dus altijd initiatiefnemer moet
        # zijn bij het tillen van een zwaargewond slachtoffer (net als de robot altijd initiatiefnemer is bij het tillen van een kapotte deur).
        human_id = self.state.get_agents_with_property({"obj_id": "rescue_worker"})[0]["obj_id"]
        if self.wait_for_response_human:
            #de 2 regels hieronder zien er heel dubbelop uit. Waarom niet if self.count_ticks == self.wait_time_response_human?
            #Tjeerd --> dit mag weg
            tick_threshold = self.wait_time_response_human
            if self.count_ticks == tick_threshold:
                self.send_message(
                    Message(content={'chat_text': "Ik ga door met mijn taken", 'notice': "got_no_response"},
                            from_id=self.agent_id, to_id=human_id))
                self.wait_for_response_human = False
                self.count_ticks = 0

            else:
                self.count_ticks += 1
                for message in self.received_messages:
                    # if human responds to request from robot to help carry
                    if message.from_id == human_id and 'notice' in message.content and message.content['notice'] == \
                            "response_to_carry_request":
                        # human is going to help the robot, so go near the victim and wait
                        self.received_messages.remove(message)
                        self.send_message(Message(content={'chat_text': "Oké, ik wacht bij het slachtoffer."},
                                                  from_id=self.agent_id, to_id=None))
                        self.wait_for_response_human = False
                        self.wait_for_human_arrival = True
                        self.waypoints_to_add = [self.carry_together_vic['location']]
                        break
                action = None
                action_kwargs = None
                return action, action_kwargs

        # Tjeerd: onderstaande 5 regels moeten weg als we bovenstaande ook weghalen. #dit mag allemaal weg
        # # als robot moet wachten tot mens arriveert bij slachtoffer, wacht
        # if self.wait_for_human_arrival:
        #     action = self.navigator.get_move_action(self.state_tracker)
        #     action_kwargs = {'action_duration': 5}
        #     return action, action_kwargs

        # --------------- START LOOPING THROUGH MESSAGES ------------------

        for message in self.received_messages:
            # REF-T02
            # als de robot een bericht krijgt met 'end help' in de Key "action", dan is hij klaar met het samen tillen van een zwaargewond slachtoffer.
            # De mens stuurt dit bericht zodra hij/zij een zwaargewond slachtoffer neerlegt.
            if message.from_id == human_id and isinstance(message.content, dict) and "action" in message.content and message.content["action"] == "end help" and self.agent_properties["location"] == (24, 2):
                self.carry_together = False
                self.received_messages.remove(message)

                # log that we carried a victim together
                self.agent_properties['log_carry_together'] += 1

                # REF-T01
                # after bringing 1 heavily wounded victim to the CP together, the robot directly learns that
                # all heavily wounded victims should be carried together from now on
                if not self.learned_carry:
                    self.learned_carry = True
                    self.agent_properties["robot_learned_carry"] = True
                    # Tjeerd: stuur bericht aan de mens, zodat de mens geen cues meer gaat sturen als de robot het al geleerd heeft
                    # (zie "if self.robot_learned_carry" in custom_human_agent)
                    self.send_message(Message(content="robot_learned_carry",
                                from_id=self.agent_id,
                                to_id=None))

                # Tjeerd: als robot klaar is met samen tillen, dan moet het tandwieltje van de robot ook weer zichtbaar worden.
                # de robot wordt zelf ook weer zichtbaar, maar dat gebeurt vanaf REF-T05.
                # De reden dat deze code hier staat is dat het (on)zichtbaar maken van het tandwieltje een Action vereist.
                # turn off visibility of the busy-icon of the robot
                action = ManageRobotImg.__name__
                action_kwargs['cog_visible'] = True
                return action, action_kwargs

            # Tjeerd: de robot krijgt via een message een verzoek van de mens binnen om samen een slachtoffer te tillen.
            # In dat geval wordt de locatie van het slachtoffer, en de hard-coded drop-locatie in de commandopost toegevoegd #bug sensitive code
            # aan de waypoints_to_add lijst van de robot.
            # Ook wordt de parameter human_help_needed op True gezet, zodat de robot weet dat hij moet komen helpen (zie later in code).
            if isinstance(message.content, dict) and 'action' in message.content and 'help carry' in message.content["action"]:
                self.location_victim = message.content["victim"]['location']
                self.carry_together_vic = message.content["victim"]
                self.waypoints_to_add.append(self.location_victim)
                # REF-T06
                self.waypoints_to_add.append((24,2))
                self.human_help_needed = True
                self.received_messages.remove(message)

                # inform the human that the robot is coming to help
                self.send_message(
                    Message(content={"chat_text": "Oké. Ik kom eraan om je te helpen."},
                            from_id=self.agent_id,
                            to_id=None))

                # inform the HQ agent that he should not send a message to the human about carrying (Reflection)
                hq_agent_id = self.state.get_agents_with_property({"obj_id": "headquarters"})[0]["obj_id"]
                self.send_message(
                    Message(content="do_not_reflect_carry",
                            from_id=self.agent_id,
                            to_id=hq_agent_id))

            # Tjeerd: in deze if-statement krijgt de robot een andere Navigator (weighted_a_star in plaats van a_star) mee,
            # die ervoor zorgt dat hij modderpaden kan ontwijken. Trigger hiervoor is een bericht van de mens.
            if message.from_id == 'rescue_worker' and 'modder geleerd' in message.content or \
                    (self.learned_mud and not self.changed_navigator_mud):
                self.changed_navigator_mud = True
                waypoints_over = self.navigator.get_upcoming_waypoints(self.state_tracker)
                waypoints_overzetten = [] #ziet er een betje dubbelop uit om en waypoints_over en waypoints_overzetten te gebruiken
                for waypoint in waypoints_over:
                    waypoints_overzetten.append(waypoint[1])
                # print(waypoints_overzetten)
                self.navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                           algorithm=Navigator.WEIGHTED_A_STAR_ALGORITHM)
                self.navigator.add_waypoints(waypoints_overzetten, is_circular=False)
                # print("Ik heb modder geleerd te ontwijken!")
                self.received_messages.remove(message)
                self.send_message(Message(content="navigator modder",
                                          from_id=self.agent_id,
                                          to_id=None))

        # --------------- END LOOPING THROUGH MESSAGES ------------------

        # --------------------------------------carry together initiatief robot-------------------------------------------------------------------------

        # if the human requested help, check whether the robot has arrived near the victim
        # or if the robot requested help, wait for the human to arrive (Tjeerd: dit kan weg als we niet willen
        # dat de robot het samen tillen initieert nadat hij dit heeft geleerd)
        # de onderstaande code lijkt niet te werken, ik heb iig nog niet een keer gezien dat de robot initiatief neemt in samen dragen
        if (self.human_help_needed and self.agent_properties['location']==self.location_victim) \
                or (self.wait_for_human_arrival and self.agent_properties['location'] == self.carry_together_vic['location']):

            # robot has arrived at victim, so check if human is also there. If yes, we are going to carry together.
            top_left_of_robot = (self.agent_properties["location"][0]-1, self.agent_properties["location"][1]-1)
            for obj in state.get_objects_in_area(top_left_of_robot, 2, 2):
                # if human is next to robot
                if obj["obj_id"] == human_id:
                    # stuur bericht aan mens dat we samen kunnen gaan dragen
                    self.send_message(
                        Message(content={"action": "carry together", "loc_robot": self.agent_properties["location"],
                                         "victim_id": self.carry_together_vic['obj_id']},
                                from_id=self.agent_id,
                                to_id=human_id))
                    self.carry_together = True # self.carry_together houdt bij of we samen aan het dragen zijn.
                    self.human_help_needed = False # robot hoeft niet meer naar de mens te gaan om samen te dragen. REF-T07.
                    self.wait_for_human_arrival = False # deze var. kan weg (alleen relevant als robot samen dragen ook mag initiëren).
                    self.waypoints_to_add.append((24, 2)) # voeg drop-locatie in commandopost toe (wordt nu volgens mij dubbelop gedaan! Namelijk ook bij REF-T06

                    # Tjeerd: als robot samen gaat tillen, dan moet het tandwieltje van de robot onzichtbaar worden.
                    # de robot wordt zelf ook onzichtbaar, maar dat gebeurt vanaf REF-T05.
                    # De reden dat deze code hier staat is dat het (on)zichtbaar maken van het tandwieltje een Action vereist.
                    action = ManageRobotImg.__name__
                    action_kwargs['cog_visible'] = False
                    return action, action_kwargs


                # if the human requested help but is not there, the carry task is skipped
                # if the robot requested help but human is not there yet, wait
                elif self.wait_for_human_arrival:
                    action = None
                    action_kwargs = None
                    return action, action_kwargs

        # ---------------------------------------------------------------------------------------------------------------

        # -----------------------------SEISMOGRAAF----------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # Check if Seismograaf conditions are met (find how many victims are lost)
        room_objs = state.get_room_objects('command_post')
        saved_vics = [obj for obj in room_objs if "vic" in obj["name"]]

        intervention_timer = 50

        # Function to get list of doors sorted on the distance to a given point
        def closest_door_location(start_location):
            room_doors = []
            for n in self.state.get_all_room_names():
                room_door = self.state.get_room_doors(room_name=n)
                distance_to_door = get_distance(start_location, room_door[0]['location'])
                room_doors.append([room_door[0], distance_to_door])
            room_doors_sorted = sorted(room_doors, key=lambda x:x[1])
            return room_doors_sorted

        # Set seismograaf variable back to True between earthquakes
        if not self.seismograaf_on and len(saved_vics) == 3:
            self.seismograaf_on = True

        if self.seismograaf_on and not self.agent_properties['is_carrying'] and not human[0]['is_carrying'] \
                and (len(saved_vics) == 2 or len(saved_vics) == 4) and \
                (self.state['setting']['scenario'] == 2 or self.state['setting']['scenario'] == 4):
            self.earthquake = True  # extra boolean voor of open deur al toegevoegd is

        if self.earthquake:
            # Check if the earthquake event holds, and if the robot is already at a door or not
            # print(self.earthquake_timer)
            if self.agent_properties['location'] == closest_door[0]["location"]: # Leerinterventies starten hieronder
                self.moving_to_door = False
                self.earthquake_timer = self.earthquake_timer - 1
                # Check timer
                if self.earthquake_timer > 6:
                    # Hold actions and check whether to initiate a learning intervention
                    action = None
                    if self.human_to_door_prev:
                        door_human = None
                        if closest_door_location(loc_human)[0][0]['collapsed']:
                            door_human = closest_door_location(loc_human)[1][0]
                        else:
                            door_human = closest_door_location(loc_human)[0][0]

                        human_to_door = get_distance(loc_human, door_human['location'])


                        if human_to_door < 1 or self.human_to_door_prev > human_to_door:
                            print("Human is at door or moving to door!")
                            self.human_to_door_prev = human_to_door
                            # Add variable for if human is at door, to be used later
                            if human_to_door < 1:
                                self.human_at_door = True
                            else:
                                self.human_at_door = False
                        # only do the explicit interventions for the experimental group, and in the training scenarios
                        if self.state['setting']['exp_condition'] == 'exp' and \
                                (self.state['setting']['scenario'] == 2 or self.state['setting']['scenario'] == 3):
                            if self.earthquake_timer < intervention_timer:
                                if self.last_eq_cue == 'behavioral':
                                    self.send_message(
                                        Message(content={"chat_text": "Ik ga schuilen voor een aankomende aardbeving."},
                                                from_id=self.agent_id,
                                                to_id=None))
                                    self.last_eq_cue = 'explanation'
                                    self.earthquake_timer = 100
                                elif self.last_eq_cue == 'explanation':
                                    self.send_message(
                                        Message(content={"chat_text": "Wil je naar de dichtstbijzijnde deuropening lopen?"},
                                                from_id=self.agent_id,
                                                to_id=None))
                                    self.last_eq_cue = 'assignment'
                                    self.earthquake_timer = 100
                                elif self.last_eq_cue == 'assignment':
                                    self.send_message(
                                        Message(content='earthquake_trigger',
                                                from_id=self.agent_id,
                                                to_id=self.state.get_agents_with_property({"obj_id": "headquarters"})[0]["obj_id"]))
                                    self.last_eq_cue = 'hq'
                                    self.earthquake_timer = 50
                    else:
                        self.human_to_door_prev = get_distance(loc_human, closest_door_location(loc_human)[0][0]['location']) # If variable doesn't exist yet, create value here
                    return action, action_kwargs  # Return zodat de loop verder wordt afgebroken
                elif self.earthquake_timer > 0:
                    # Check if human is currently carrying, break loop here when that is the case and add to timer
                    if human[0]['is_carrying'] and not self.human_at_door:
                        self.earthquake_timer = self.earthquake_timer + 1
                        action = None
                        return action, action_kwargs

                    # Check again if human is at door and update variable
                    if closest_door_location(loc_human)[0][0]['collapsed']:
                        door_human = closest_door_location(loc_human)[1][0]
                    else:
                        door_human = closest_door_location(loc_human)[0][0]
                    human_to_door = get_distance(loc_human, door_human['location'])
                    if human_to_door < 1:
                        self.human_at_door = True
                    else:
                        self.human_at_door = False

                    # Activate Earthquake
                    action = EarthquakeEvent.__name__
                    action_kwargs['earthquake_happening'] = True
                    # Send earthquake message, to notify human to be slow and change their image
                    if not self.human_at_door:
                        self.send_message(
                            Message(content='hit_earthquake',
                                    from_id=self.agent_id,
                                    to_id=None))
                        
                    return action, action_kwargs  # Return zodat de loop verder wordt afgebroken
                else:
                    # End of earthquake event, reset and resume
                    self.seismograaf_on = False
                    self.earthquake_timer = 100
                    self.earthquake = False
                    self.human_at_door = False
                    self.last_eq_cue = 'behavioral'
                    action = EarthquakeEvent.__name__
                    action_kwargs['earthquake_happening'] = False
                    return action, action_kwargs
            else:
                #print("NOT AT DOOR")
                # Check if the closest door is already the first waypoint
                if not self.moving_to_door:
                    #print("Let's add a door to the waypoints")
                    # Add door waypoint
                    if closest_door[0]['is_open']:
                        self.waypoints_to_add.append(closest_door[0]["location"])
                        self.moving_to_door = True
                    else:
                        self.waypoints_to_add.append(closest_door_location(self.agent_properties['location'])[0][0]["location"])
                        self.moving_to_door = True
        # --------------------------------------------------------------------------------------------------------------

        # ------------------------------------------------leg slachtoffer neer---------------------------------------------------------------
        #hieronder een paar situaties waarin de robot het slachtoffer dat hij aan het dragen is moet neerleggen.

        # de "if not self.wait_for_human_arrival" kan weg als we niet willen dat de robot het tillen mag initiëren.
        # if the robot is not waiting for the human to arrive at the victim
        if not self.wait_for_human_arrival: #alleen deze regel mag weg (dus deze if) #Variabele gemaakt door Tjeerd om te checken of robot aan het wachten is op hulp. het lijkt mij dat er veel meer in deze if moet dan alleen de onderstaande dingen
            if self.agent_properties['is_carrying'] and self.human_help_needed: #als de mens de robot om hulp vraagt, maar de robot draagt nog iets
                action = DropObject.__name__
                action_s = True
                self.waypoints_to_add.append(self.agent_properties["location"]) #voeg locatie waar je slachtoffer gedropt hebt om later weer op te pikken
                self.waypoints_to_add.append((24, 2)) #voeg locatie in CP toe om slachtoffer neer te leggen
            if self.agent_properties['is_carrying'] and self.agent_properties["location"] == (24, 2): #als robot aan het dragen is en in de CP is, leg slachtoffer neer
                action = DropObject.__name__
                action_s = True
            if self.agent_properties['is_carrying'] and self.agent_properties["location"] == self.drag_target: #als robot aan het dragen/slepen is en op de locatie net naast de deur van ingestort gebouw is (=drag_target), leg slachtoffer neer
                action = DropObject.__name__
                action_s = True
                self.drag_target = [] #reset drag_target locatie
                self.drag_vic = False
        # ---------------------------------------Open door-----------------------------------------------------------------------
        #Als robot in deuropening staat en de deur is nog niet geopend (image deur kwart slag draaien) dan moet de robor dit als actie doen
            if self.agent_properties["location"] == closest_door[0]["location"] and not closest_door[0]["door_opened"]:
                    if closest_door[0]["door_side"] == "right":
                        new_img = "door_bottom.png"
                    if closest_door[0]["door_side"] == "left":
                        new_img = "door_top.png"
                    if closest_door[0]["door_side"] == "bottom":
                        new_img = "door_left.png"
                    if closest_door[0]["door_side"] == "top":
                        new_img = "door_right.png"
                    action = OpenDoor.__name__
                    action_kwargs = {**action_kwargs, "opacity": 1.0, "obj_id": closest_door[0]["obj_id"], "img_name": new_img}
                    action_s = True
                    #hier zou misschien goed een return action kunnen
        # --------------------------------------------------------------------------------------------------------------

        # -------------------------------------------------zwaargewond slachtoffer slepen-------------------------------------------------------------
        # REF birgit 002
        # als een zwaargewond slachtoffer gevonden is in een
        if vics_to_drag:
            for vic in vics_to_drag:
                # print(vic['name'])
                #als de agent op de victim staat die niet in de CP ligt en hij niet al aan het dragen is, sleep dan victim
                if self.agent_properties["location"] == vic["location"] and not \
                        self.agent_properties['is_carrying'] and \
                        closest_door[0]["room_name"] != 'command_post' and \
                        not action_s and \
                        not self.human_help_needed and \
                        not self.carry_together and \
                        vic["treatment_need"] == 2:
                    #mogelijk mist hier nog de check of de robot (en vic) zich in een ingestord gebouw bevinden)
                    action = GrabObject.__name__ #pak victim op
                    action_s = True
                    self.drag_vic = True
                    action_kwargs['object_id'] = vic['obj_id']
                    action_kwargs['max_objects'] = self.__max_carry_objects
                    action_kwargs['grab_range'] = 1
                    #misschien zou het goed zijn hier een return action te doen
                #als de agent in een deuropening staat die niet CP is en een victim ziet die zwaargewond is: ga naar locatie van zwaargewonde victim.
                # mogelijk mist hier nog de check of de robot (en vic) zich in een ingestord gebouw bevinden)
                if self.agent_properties["location"] == closest_door[0]["location"] and \
                        closest_door[0]["room_name"] != 'command_post' and \
                        vic["treatment_need"] == 2 and \
                        not self.agent_properties['is_carrying'] \
                        and vic["location"] not in self.navigator.get_upcoming_waypoints(self.state_tracker) and \
                        not self.human_help_needed and \
                        not self.carry_together and get_distance(self.agent_properties["location"], vic["location"]) < 6:
                    # self.navigator._Navigator__update_waypoints(self.state_tracker)
                    self.waypoints_to_add.append(vic['location'])
                    print("zwaargewond slachtoffer aan waypoints toegevoegd")
                    # print("upcomming waypoints:")
                    # for point in self.navigator.get_upcoming_waypoints(self.state_tracker):
                    #     print(point[1])

                    #bepaal de locatie waar gesleepte victim neergelegd moet worden en voeg deze toe aan drag_target en aan waypoints.
                    if closest_door[0]["door_side"] == "left":
                        self.drag_target = (closest_door[0]["location"][0] - 1, closest_door[0]["location"][1])
                        self.waypoints_to_add.append(self.drag_target)
                        # print("upcomming waypoints:")
                        # for point in self.navigator.get_upcoming_waypoints(self.state_tracker):
                        #     print(point[1])
                    elif closest_door[0]["door_side"] == "right":
                        self.drag_target = (closest_door[0]["location"][0] + 1, closest_door[0]["location"][1])
                        self.waypoints_to_add.append(self.drag_target)
                    elif closest_door[0]["door_side"] == "bottom":
                        self.drag_target = (closest_door[0]["location"][0], closest_door[0]["location"][1] + 1)
                        self.waypoints_to_add.append(self.drag_target)
                    elif closest_door[0]["door_side"] == "top":
                        self.drag_target = (closest_door[0]["location"][0], closest_door[0]["location"][1] - 1)
                        self.waypoints_to_add.append(self.drag_target)

        # -------------------------------------------------lichtgewond slachtoffer redden-------------------------------------------------------------
        # REF birgit 003
        if vics_need_rescue:
            # print(vics_need_rescue)
            for vic in vics_need_rescue:
                # print(vic['name'])
                if self.agent_properties["location"] == vic["location"] and not \
                        self.agent_properties['is_carrying'] and \
                        closest_door[0]["room_name"] != 'command_post' and \
                        not action_s and \
                        not self.human_help_needed and \
                        not self.carry_together and \
                        vic["treatment_need"] == 1:
                    upcomming_waypoints = self.navigator.get_upcoming_waypoints(self.state_tracker)
                    action = GrabObject.__name__
                    action_s = True
                    action_kwargs['object_id'] = vic['obj_id']
                    action_kwargs['max_objects'] = self.__max_carry_objects
                    action_kwargs['grab_range'] = 1
                if self.agent_properties["location"] == closest_door[0]["location"] and \
                        closest_door[0]["room_name"] != 'command_post' and \
                        vic["treatment_need"] == 1 and \
                        not self.agent_properties['is_carrying'] and \
                        not self.human_help_needed and \
                        not self.carry_together and \
                        (not closest_door[0]["door_opened"] or (closest_door[0]["door_opened"] and closest_door[0]["img_name"] == "invisible.png")):
                    # and \
                    #         not closest_door[0]["door_opened"]
                    old_waypoints = self.waypoints_to_add
                    self.waypoints_to_add = [vic['location']]
                    self.waypoints_to_add.append((24, 2))
                    self.waypoints_to_add = self.waypoints_to_add + old_waypoints
                    # print(self.waypoints_to_add)
                    # for point in self.navigator.get_upcoming_waypoints(self.state_tracker):
                    #     print(point[1])

        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------- START carry door together block---------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        #Zodra mens en robot beide bij de gebroken deur in de buurt staan wordt add_waypoints_door True en wordt het volgende gedaan:
        #het loop gedrag van mens en robot die samen de deur tillen wordt toegevoegd aan waypoints
        #locatie waypoints afhankelijk van orientatie deur.
        #deze waypoints worden eenmalig toegevoegd. (hierna wordt add_waypoints_door weer False)
        #door_target is de plek waar de deur weer neergelegd moet worden zodra de carry_door together afgelopen is.
        # REF birgit 004
        if self.carry_door and self.add_waypoints_door and not closest_door[0]["door_opened"]:
            if closest_door[0]["door_side"] == "left":
                earlier_waypoints_to_add = self.waypoints_to_add
                self.waypoints_to_add = [(closest_door[0]["location"][0] - 1, closest_door[0]["location"][1])]
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 2, closest_door[0]["location"][1]))
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 2, closest_door[0]["location"][1] - 1))
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 1, closest_door[0]["location"][1] - 1))
                self.waypoints_to_add = self.waypoints_to_add + earlier_waypoints_to_add
                self.door_target = (closest_door[0]["location"][0] - 1, closest_door[0]["location"][1] - 1)
                #print("waypoints worden toegevoegd")
                self.add_waypoints_door = False
            elif closest_door[0]["door_side"] == "right":
                earlier_waypoints_to_add = self.waypoints_to_add
                self.waypoints_to_add = [(closest_door[0]["location"][0] + 1, closest_door[0]["location"][1])]
                self.waypoints_to_add.append((closest_door[0]["location"][0] + 2, closest_door[0]["location"][1]))
                self.waypoints_to_add.append((closest_door[0]["location"][0] + 2, closest_door[0]["location"][1] - 1))
                self.waypoints_to_add.append((closest_door[0]["location"][0] + 1, closest_door[0]["location"][1] - 1))
                self.waypoints_to_add = self.waypoints_to_add + earlier_waypoints_to_add
                self.door_target = (closest_door[0]["location"][0] + 1, closest_door[0]["location"][1] - 1)
                #print("waypoints worden toegevoegd")
                self.add_waypoints_door = False
            elif closest_door[0]["door_side"] == "bottom":
                earlier_waypoints_to_add = self.waypoints_to_add
                self.waypoints_to_add = [(closest_door[0]["location"][0], closest_door[0]["location"][1] + 1)]
                self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] + 2))
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 1, closest_door[0]["location"][1] + 2))
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 1, closest_door[0]["location"][1] + 1))
                self.waypoints_to_add = self.waypoints_to_add + earlier_waypoints_to_add
                self.door_target = (closest_door[0]["location"][0] - 1, closest_door[0]["location"][1] + 1)
                #print("waypoints worden toegevoegd")
                self.add_waypoints_door = False
            elif closest_door[0]["door_side"] == "top":
                earlier_waypoints_to_add = self.waypoints_to_add
                self.waypoints_to_add = [(closest_door[0]["location"][0], closest_door[0]["location"][1] - 1)]
                self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] - 2))
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 1, closest_door[0]["location"][1] - 2))
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 1, closest_door[0]["location"][1] - 1))
                self.waypoints_to_add = self.waypoints_to_add + earlier_waypoints_to_add
                self.door_target = (closest_door[0]["location"][0] - 1, closest_door[0]["location"][1] - 1)
                #print("waypoints worden toegevoegd")
                self.add_waypoints_door = False

        # ------------------------------------------start samen deur dragen--------------------------------------------------------------------
        #als mens en robot vlakbij de dichte deur staan en niet aan het tillen zijn dan start de carry door together hierboven
        # REF birgit 004
        if get_distance(self.agent_properties["location"], closest_door[0]["location"]) < 2 and \
                get_distance(loc_human,closest_door[0]["location"]) < 2 \
                and not closest_door[0]["is_open"] and not self.carry_door and not closest_door[0]["door_opened"] and not human[0]['is_carrying']:
            print("start carry door together")
            self.carry_door = True
            self.carry_door_behavior = False #reset the carry door behavior
            self.add_waypoints_door = True #initiate the route walked with the door
            #onderstaande variabele wordt niet gebruikt en is dus uitgecommend
            # id_door = closest_door[0]['obj_id']

            #stuur bericht naar mens zodat deze onzichtbaar wordt en geen acties kan uitvoeren tot de carry door klaar is
            self.send_message(
                Message(content="start carry door",
                        from_id=self.agent_id,
                        to_id=None))
            action = OpenCollapsedDoor.__name__
            action_kwargs = {**action_kwargs, "opacity": 1.0, "obj_id": closest_door[0]["obj_id"],
                             "img_name": 'invisible.png'}
            return action, action_kwargs

        # ----------------------------------------------einde carry door together----------------------------------------------------------------
        #zodra de robot op de door_target locatie staat en carry_door nog gaande is wordt de carry door actie beeindigd door de deur neer te leggen en de variabele te resetten
        if self.carry_door and not self.add_waypoints_door and self.agent_properties["location"] == self.door_target:
            # print(self.agent_properties["location"])
            # print(closest_door[0]["location"])

            #stuur mens dat de deur samen dragen klaar is (en dus weer zichtbaar mag worden en acties mag uitvoeren)
            self.send_message(
                Message(content="end carry door",
                        from_id=self.agent_id,
                        to_id=None))

            # log that we have carried a door together
            self.agent_properties['log_carry_door'] += 1
            #print("end carry door")

            #eindig de carry_door actie
            self.carry_door = False

            self.waypoints_to_add.append(closest_door[0]["location"])

            # op plek # REF birgit 005 wordt de lijst met waypoints opgeslagen die de agent had voordat hij begon met heen en weer lopen naar de dichte deur.
            # ik verwacht dat dit bug sensitive code is omdat als er tussen het begin van heen en weer lopen naar de deur en het einde van de carrydoor together dingen toegevoegd worden aan de waypoints dit niet meegenomen wordt
            if self.waypoint_list_reset:
                self.navigator.reset_full()
                self.waypoints_to_add = self.waypoints_to_add + self.waypoint_list_reset
                self.waypoint_list_reset = []
            action = MagicalDoorAppear.__name__
            action_kwargs = {**action_kwargs, "location_door": self.agent_properties["location"], "opacity": 1.0, "obj_id": closest_door[0]["obj_id"],
                                              "img_name": 'invisible.png'}

            return action, action_kwargs



        # ----------------------------------------leerinterventies carry door together----------------------------------------------------------------------
        human_to_broken_door = get_distance(loc_human, closest_door[0]["location"])
        if self.carry_door_behavior:
            self.broken_door_timer +=1
            #print(self.broken_door_timer)
            #timer om te bepalen na hoeveel ticks er overgegaan wordt op de volgende interventie.
            intervention_timer = 10
            #elke 3 ticks, indien de mens nog niet bij de deur aangekomen is, voeg het heen en weer loop gedrag toe.
            if self.human_to_agent_prev and self.broken_door_timer%3 == 0: #elke 3 ticks
                closest_door_location = closest_door[0]["location"]
                if closest_door[0]["door_side"] == "left" and "broken" in closest_door[0]["img_name"]:
                    self.waypoints_to_add.append(
                        (closest_door_location[0] - 3, closest_door_location[1]))
                    self.waypoints_to_add.append((closest_door[0]["location"][0] - 1, closest_door[0]["location"][1]))
                if closest_door[0]["door_side"] == "right" and "broken" in closest_door[0]["img_name"]:
                    self.waypoints_to_add.append(
                        (closest_door_location[0] + 3, closest_door_location[1]))
                    self.waypoints_to_add.append((closest_door[0]["location"][0] + 1, closest_door[0]["location"][1]))
                if closest_door[0]["door_side"] == "top" and "broken" in closest_door[0]["img_name"]:
                    self.waypoints_to_add.append(
                        (closest_door_location[0], closest_door_location[1] - 3))
                    self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] - 1))
                if closest_door[0]["door_side"] == "bottom" and "broken" in closest_door[0]["img_name"]:
                    self.waypoints_to_add.append(
                        (closest_door_location[0], closest_door_location[1] + 3))
                    self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] + 1))
                #check of mens naar de deur loopt
                if human_to_broken_door < 1 or self.human_to_agent_prev > human_to_broken_door:
                    print("Human is at agent loc or moving towards agent!")

            #check of er een nieuwe interventie geactiveerd moet worden.
            if self.broken_door_timer == intervention_timer + 1 and \
                    self.state['setting']['exp_condition'] == 'exp' and \
                                (self.state['setting']['scenario'] == 2 or self.state['setting']['scenario'] == 3):
                # print(self.last_bd_cue)
                if self.last_bd_cue == 'behavioral':
                    self.send_message(
                        Message(content={"chat_text": "Ik kan niet door de deur, ik kan hem niet zelf open maken."},
                                from_id=self.agent_id,
                                to_id=None))
                    self.last_bd_cue = 'explanation'
                    self.broken_door_timer = 0
                elif self.last_bd_cue == 'explanation':
                    self.send_message(
                        Message(content={"chat_text": "Wil je mij helpen de deur open te maken?"},
                                from_id=self.agent_id,
                                to_id=None))
                    self.last_bd_cue = 'assignment'
                    self.broken_door_timer = 0
                elif self.last_bd_cue == 'assignment':
                    self.send_message(
                        Message(content='broken_door_hq',
                                from_id=self.agent_id,
                                to_id='headquarters'))
                    self.last_bd_cue = 'hq'
                    self.broken_door_timer = 0
            #als er nog geen nieuwe interventie moet komen, vernieuw de human_to_agent_prev met de nieuwe afstand
            else:
                self.human_to_agent_prev = human_to_broken_door
            #als alle leerinterventies geweest zijn en de mens is nog steeds niet gekomen, stop met carry_door proberen te initieren
            if self.broken_door_timer >10 and self.last_bd_cue == 'hq':
                    self.carry_door_behavior = False #stop met carry door
                    #print(self.waypoints_to_add)
                    #print(self.waypoint_list_reset)

                    # op plek # REF birgit 005 wordt de lijst met waypoints opgeslagen die de agent had voordat hij begon met heen en weer lopen naar de dichte deur.
                    # ik verwacht dat dit bug sensitive code is omdat als er tussen het begin van heen en weer lopen naar de deur en het einde van de carrydoor together dingen toegevoegd worden aan de waypoints dit niet meegenomen wordt
                    if self.waypoint_list_reset:
                        self.navigator.reset_full()
                        self.waypoints_to_add = self.waypoints_to_add + self.waypoint_list_reset
                        self.waypoint_list_reset = []
                        self.broken_door_timer = 0
                    elif self.waypoints_to_add ==[]:
                        self.waypoints_to_add.append((24,4))




        # -------------------------------------Robot komt dichte deur tegen--------------------------------------------------

        if not closest_door[0]["is_open"] and not self.carry_door and self.agent_properties[
            'is_carrying'] == [] and not self.carry_together and not self.carry_door_behavior and not action_s and \
                (closest_door[0]["location"] == (
                self.agent_properties["location"][0] - 1, self.agent_properties["location"][1]) #closest door location is links, rechts, boven of onder de agent location, dit zou misschien ook afgevangen kunnen worden met distance(closest door, agent) = 1
                 or closest_door[0]["location"] == (
                 self.agent_properties["location"][0] + 1, self.agent_properties["location"][1])
                 or closest_door[0]["location"] == (
                 self.agent_properties["location"][0], self.agent_properties["location"][1] + 1)
                 or closest_door[0]["location"] == (
                 self.agent_properties["location"][0], self.agent_properties["location"][1] - 1)):
            self.carry_door_behavior = True
            self.last_bd_cue = "behavioral"
            self.waypoint_list_reset = []
            # REF birgit 005
            for point in self.navigator.get_upcoming_waypoints(self.state_tracker):
                # print(point[1])
                self.waypoint_list_reset.append(point[1])
            # self.first_time = False
            if closest_door[0]["door_side"] == "left":
                self.waypoints_to_add.append(
                    (self.agent_properties["location"][0] - 3, self.agent_properties["location"][1]))
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 1, closest_door[0]["location"][1]))
                self.waypoints_to_add.append(
                    (self.agent_properties["location"][0] - 3, self.agent_properties["location"][1]))
                self.waypoints_to_add.append((closest_door[0]["location"][0] - 1, closest_door[0]["location"][1]))
            if closest_door[0]["door_side"] == "right":
                self.waypoints_to_add.append(
                    (self.agent_properties["location"][0] + 3, self.agent_properties["location"][1]))
                self.waypoints_to_add.append((closest_door[0]["location"][0] + 1, closest_door[0]["location"][1]))
                self.waypoints_to_add.append(
                    (self.agent_properties["location"][0] + 3, self.agent_properties["location"][1]))
                self.waypoints_to_add.append((closest_door[0]["location"][0] + 1, closest_door[0]["location"][1]))
            if closest_door[0]["door_side"] == "top":
                self.waypoints_to_add.append(
                    (self.agent_properties["location"][0], self.agent_properties["location"][1] - 3))
                self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] - 1))
                self.waypoints_to_add.append(
                    (self.agent_properties["location"][0], self.agent_properties["location"][1] - 3))
                self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] - 1))
            if closest_door[0]["door_side"] == "bottom":
                self.waypoints_to_add.append(
                    (self.agent_properties["location"][0], self.agent_properties["location"][1] + 3))
                self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] + 1))
                self.waypoints_to_add.append(
                    (self.agent_properties["location"][0], self.agent_properties["location"][1] + 3))
                self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] + 1))

        # --------------------------------------------------------------------------------------------------------------
        #--------------------------------- end carry door together block---------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # ---------------------------------------Waypoints toevoegen aan voorkant -----------------------------------------------------------------------
        if self.waypoints_to_add:
            for point in self.navigator.get_upcoming_waypoints(self.state_tracker):
                # print(point[1])
                self.waypoints_to_add.append(point[1])
            self.navigator.reset_full()
            #print(self.waypoints_to_add)
            self.navigator.add_waypoints(self.waypoints_to_add, is_circular=False)
            self.waypoints_to_add = []
        if not action_s:
            action = self.navigator.get_move_action(self.state_tracker)
        if self.carry_together and self.agent_properties["location"] == (24, 2):
            # print("Arrived at CP. Waiting for human to drop the victim in the CP")
            action = None
        elif self.wait_for_human_arrival:
            action = self.navigator.get_move_action(self.state_tracker)
            action_kwargs = {'action_duration': 5}
        # print(action)
        ### onderstaande nodig voor instructie video
        # if state['World']['nr_ticks'] !=0 and state['World']['nr_ticks'] < 290:
        #     action = None

        # ---------------------------------------------Nadat robot alle gebouwen 1x bekeken heeft-----------------------------------------------------------------
        if self.navigator.get_upcoming_waypoints(self.state_tracker) == [] and not self.human_help_needed and not self.carry_together:
            #print("Robot einde taken")
            self.einde = True
            if vics_to_drag:
                for vic in vics_to_drag:
                    self.waypoints_to_add.append(vic["location"])
                    if closest_door[0]["door_side"] == "left":
                        self.drag_target = (closest_door[0]["location"][0] - 1, closest_door[0]["location"][1])
                        self.waypoints_to_add.append(self.drag_target)
                        # print("upcomming waypoints:")
                        # for point in self.navigator.get_upcoming_waypoints(self.state_tracker):
                        #     print(point[1])
                    elif closest_door[0]["door_side"] == "right":
                        self.waypoints_to_add.append((closest_door[0]["location"][0] + 1, closest_door[0]["location"][1]))
                        self.drag_target = (closest_door[0]["location"][0] + 1, closest_door[0]["location"][1])
                    elif closest_door[0]["door_side"] == "bottom":
                        self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] + 1))
                        self.drag_target = (closest_door[0]["location"][0], closest_door[0]["location"][1] + 1)
                    elif closest_door[0]["door_side"] == "top":
                        self.waypoints_to_add.append((closest_door[0]["location"][0], closest_door[0]["location"][1] - 1))
                        self.drag_target = (closest_door[0]["location"][0], closest_door[0]["location"][1] - 1)

            if vics_need_rescue:
                for vic in vics_need_rescue:
                    self.waypoints_to_add.append(vic["location"])
                    self.waypoints_to_add.append((24, 2))
            for door in self.state[{'class_inheritance': 'Door'}]:
                if "broken" in door["img_name"]:
                    if door["door_side"] == "bottom":
                        self.waypoints_to_add.append((door["location"][0], door["location"][1]+1))
                    if door["door_side"] == "top":
                        #top
                        self.waypoints_to_add.append((door["location"][0], door["location"][1]-1))
                    if door["door_side"] == "left":
                        #left
                        self.waypoints_to_add.append((door["location"][0]-1, door["location"][1]))
                    if door["door_side"] == "right":
                        #right
                        self.waypoints_to_add.append((door["location"][0]+1, door["location"][1]))
            # robot is klaar met alles en blijft staan in de command post
            self.waypoints_to_add.append((24,2))
        return action, action_kwargs


    def create_context_menu_for_other(self, agent_id_who_clicked, clicked_object_id, click_location):
        """ Generate options for a context menu for a specific object/location that a user NOT controlling this
        human agent opened.
        """
        #print("Context menu other")
        context_menu = []

        # Generate a context menu option for every action
        # for action in self.action_set:
        #     context_menu.append({
        #         "OptionText": f"Do action: {action}",
        #         "Message": Message(content=action, from_id=clicked_object_id, to_id=self.agent_id)
        #     })
        return context_menu


    def _set_messages(self, messages=None):
        # make sure we save the entire message and not only the content
        for mssg in messages:
            received_message = mssg
            self.received_messages.append(received_message)


