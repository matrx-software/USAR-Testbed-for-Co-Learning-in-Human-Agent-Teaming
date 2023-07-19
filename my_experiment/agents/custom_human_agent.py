from matrx.actions.object_actions import GrabObject, DropObject
from my_experiment.custom_actions import EarthquakeEvent, OpenDoor
from matrx.actions.door_actions import OpenDoorAction, CloseDoorAction
from matrx.objects import Door
from matrx.agents import HumanAgentBrain
import numpy as np
from matrx.api.api import get_latest_state
from matrx.messages import Message

import pickle
import os

from my_experiment.objects import Victim
from my_experiment.custom_actions import MoveEastHuman, MoveNorthHuman, MoveSouthHuman, MoveWestHuman

class CustomHumanAgent(HumanAgentBrain):
    """ Creates an Human Agent which is an agent that can be controlled by a
    human.

    For more extensive documentation on the functions below, see: 
    http://docs.matrx-software.com/en/master/sections/_generated_autodoc/matrx.agents.agent_types.human_agent.HumanAgentBrain.html
    """

    def __init__(self, memorize_for_ticks=None, max_carry_objects=1,
                 grab_range=2, drop_range=1, door_range=1):
        """ Creates an Human Agent which is an agent that can be controlled by
        a human.
        """
        # self.memoryA = []
        # self.memoryB = []
        super().__init__(memorize_for_ticks=memorize_for_ticks)
        self.__max_carry_objects = max_carry_objects
        self.__grab_range = grab_range
        self.__drop_range = drop_range
        self.__door_range = door_range
        self.carrying_together = False #boolean that keeps track of whether robot and human are carrying a heavily wounded victim together
        self.__vics_gezien = [] #list of all victims that are visible for the human
        self.__modderlocations = [] #list of locations of all the mud tiles
        self.carry_door = False #boolean that keeps track of whether robot and human are carrying a door together
        self.hit_earthquake = False
        self.eq_active = False
        self.hurt_duration = 150
        self.can_respond_to_robot = False # Deze variabele was er om een optie in het contextmenu toe te voegen waarmee de mens in staat is om op een verzoek van de robot om een zwaarg. slachtoffer te tillen te reageren. ("Ik kom je helpen"). Kan er echter uit als we besluiten dat de robot dit verzoek niet kan sturen.
        self.memory_of_victims = [] #dit is volgens mij dezelfde variabele als self.__vics_gezien (Tjeerd: weet ik niet, volgens mij is dit een left-over variabele omdat we de locatie van gevonden victims eerst wilden opslaan in een memory)
        self.sent_carry_cue = False # Deze houdt bij of de mens een carry cue heeft gestuurd. Bijvoorbeeld om te zorgen dat dit na X ticks niet een tweede keer gestuurd wordt (zie in de buurt van r.339)
        self.sent_carry_request = False # Deze houdt bij of de mens een expliciet carry-verzoek heeft gestuurd aan de robot via explanation of assignment. Dan wordt er o.a. een berichtje naar de robot gestuurd. (zie in de buurt van r.320)
        self.robot_learned_carry = False # Deze houdt aan de mens-kant bij of de robot heeft geleerd om samen te tillen. Dat leert hij op het moment dat de mens voor het eerst een zwaargewond slachtoffer in de CP neerlegt (zie r.499 in de robot code)
        self.carry_vic_cue_time = 0 # Deze houdt bij hoe lang de mens naast een zwaargewond slachtoffer dat samen gedragen kan worden staat te wachten op de robot (tenminste, we nemen aan dat de mens dat doet). - zie rond r.344
        self.carry_vic_cue_threshold = 60 # Hier wordt ingesteld hoe lang de mens naast een z.g. slachtoffer moet staan voordat een cue wordt gestuurd.
        self.wait_for_carry_time = 0 # Zelfde als self.carry_vic_cue_time, alleen deze variabele wordt gebruikt als de robot al heeft geleerd om samen te tillen (dan is de cue niet meer nodig, maar moet de robot wel na X tijd naar de mens komen om te helpen tillen.
        self.wait_for_carry_threshold = 25  # Zelfde als self.carry_vic_cue_threshold, alleen deze wordt gebruikt als de robot al heeft geleerd om samen te tillen.
        self.mud_cue_tick_threshold = 300 # Dit bepaalt hoeveel ticks vanaf de start van het spel er geteld moet worden over hoeveel moddertegels de mens heeft gelopen. Dat wordt gebruikt als "cue" aan de robot om modder te vermijden. (zie r. 291)
        self.mud_cue_count_threshold = 10 # Dit bepaalt over hoeveel modder-tegels de mens maximaal mag hebben gelopen om de cue te mogen zenden aan de robot dat hij modder moet vermijden (zie r. 291)
        self.injured_vic_nearby = False

        #onderstaande variables zijn volgens mij dubbelop. __visited_doorlocs wordt volgens mij niet gebruikt
        # ----------------------------------------------------------------------------------------------------
        self.doors_visited = [] #list of all the door locations of all the rooms that are visited by the human
        self.__visited_doorlocs = []  # list of all the roomnames of all the rooms that are visited
        # ----------------------------------------------------------------------------------------------------

        # Code to retrieve info about learned situations if available
        # Als het leerbestand al is aangemaakt (zie REF-T08 in main), dan moet deze uitgelezen worden. Op deze manier
        # neemt de robot het geleerde gedrag mee over de scenarios.
        # Zie REF-T09 in co_learning_logger voor het opslaan van het geleerde gedrag in de pickle file.
        if os.path.isfile('./learned_backup.pkl'):
            with open('learned_backup.pkl', 'rb') as pickle_file:
                pickle_contents = pickle.load(pickle_file)
            if len(pickle_contents) > 1:
                self.robot_learned_carry = bool(int(pickle_contents[0]))

    def initialize(self):
        """ This method is called each time a new world is created or the same world is reset. This prevents the agent to
        remember that it already moved and visited some waypoints."""

        self.memory_of_victims = []


    def filter_observations(self, state):
        """ Filters the world state before deciding on an action. """

        #een lijst met alle victims in de wereld.
        victim_list = state.get_of_type(Victim.__name__)

        closest_door = state.get_closest_room_door() #het kan misschien handig zijn om hier een self variabele van te maken gezien dit heel veel gebruikt wordt
        #elke tick wordt gekeken of er een nieuwe victim gespot is. Elke tick wordt deze leeg begonnen.
        new_victim = []
        location_door = closest_door[0]["location"]
        if location_door == self.agent_properties["location"] and location_door not in self.doors_visited:
            room_name = closest_door[0]["room_name"]
            #location 24,4 is the location of the CP door. Bug sensitive code
            #location_door not in self.__visited_doorlocs lijkt een hele rare vergelijking. de __visited_doorlocs is namelijke een lijst met room_names (zie hieronder bij de append)
            #ik denk dat self.__visited_doorlocs vervangen moet worden door self.doors_visited
            if location_door not in self.__visited_doorlocs:
                #stuur de robot een bericht van de deur die je voor het eerst bezocht hebt. (zodat hij deze uit zijn waypoint lijst haalt).
                self.send_message(
                    Message(content={"info": "door", "location": location_door},
                            from_id=self.agent_id,
                            to_id=None))  # None = all agents
                self.__visited_doorlocs.append(room_name)
            room_objs = state.get_room_objects(room_name)
            #kijk of er victims in de room_objs zijn:
            vics_in_room = [obj for obj in room_objs if "vic" in obj["name"]]
            #als er victims in de room zijn voeg deze toe aan de new_victim lijst en ook aan de __vics_gezien lijst
            #of het een nieuwe vic is wordt bijgehouden zodat hieronder (laatste for loop in deze filter_observations) de robot kan worden bericht over nieuw gevonden victims
            for vic in vics_in_room:
                if vic not in self.__vics_gezien:
                    new_victim.append(vic)
                    self.__vics_gezien.append(vic)
            #human visited the room. Add room to the list doors_visited
            self.doors_visited.append(closest_door[0]["location"])
        #de lijst van victims wordt uitgelezen om te kijken welke victims zichtbaar zouden moeten zijn voor de mens
        #elke victim die al wel gevonden is wordt uit de lijst victim_list verwijderd
        #na deze for loops bestaat de lijst victim_list dus uit alle victims die nog NIET gevonden zijn
        for vic_g in self.__vics_gezien:
            for vic_l in victim_list:
                if vic_g['name'] == vic_l['name']:
                    victim_list.remove(vic_l)
        #als de robot een bericht stuurt van een gevonden vic wordt deze ook uit de victim_list verwijderd.
        for mssg in list(self.received_messages):
            for vic_l in victim_list:
                if vic_l['name'] ==mssg.content:
                    victim_list.remove(vic_l)
        #als de mens een victim vind stuurt de mens een bericht naar de robot met de naam van de victim:
        for vic in new_victim:
            self.send_message(
                Message(content=f"{vic['name']}",
                        from_id=self.agent_id,
                        to_id=None))  # None = all agents
        #alle victims die nog niet gevonden zijn worden uit de wereld verwijderd.
        [state.remove(vic["obj_id"]) for vic in victim_list]

        return state 

    def filter_user_input(self, user_input):
        if user_input is None:
            return []
        possible_key_presses = list(self.key_action_map.keys())
        return list(set(possible_key_presses) & set(user_input))

    def first_tick(self, state):
        # lijst maken waarin alle locaties met modder staan
        modderlijst = state.get_with_property({"name": "modder"}, combined=False)
        if modderlijst:
            for mud in modderlijst:
                self.__modderlocations.append(mud['location'])

        # Initialize log_human_mud property correctly to make sure mud can be learned even if no mudtiles are crossed
        self.agent_properties['log_human_mud'] = 0

        return state

    def decide_on_action(self, state, user_input):
        """ Contains the decision logic of the agent. """
        action = None
        action_from_msg = False
        action_kwargs = {"action_duration": 1}
        robot_id = self.state.get_agents_with_property({"obj_id": "explorer"})[0]["obj_id"]

        # change our icon
        if self.agent_properties['is_carrying']:
            if self.carrying_together:
                # set image als de mens samen met de robot een zwaargewond slachtoffer draagt
                self.agent_properties['img_name'] = "carry_together.png"
            else:
                # of als de mens een lichtgewond slachtoffer draagt
                self.agent_properties['img_name'] = "human_carry.png"
        elif self.carry_door:
            #als de robot en mens samen een deur dragen wordt de mens onzichtbaar
            self.agent_properties['img_name'] = "invisible.png"
        elif self.hit_earthquake:
            # override the image if human is hit by earthquake
            self.agent_properties['img_name'] = "human_hurt.png"
        else:
            #als er geen van de bovenstaande dingen gebeurd is de mens de gewone human afbeelding.
            self.agent_properties['img_name'] = "human.png"

        # alle acties die gedaan eenmalig moeten worden aan het begin van het scenario, zoals modderlocaties vaststellen
        if state['World']['nr_ticks'] == 0:
            self.first_tick(state)
        #------------------hieronder staat een stuk waarin de agent alleen hallo stuurt. Ik heb dit uitgecomment.-----------------
        # # send hello on the first tick to all agents
        # if state['World']['nr_ticks'] == 1:
        #     self.send_message(Message(content=f"Hello, my name is (human agent) {self.agent_name} and I sent this message at "
        #                                         f"tick {state['World']['nr_ticks']}",
        #                                 from_id=self.agent_id,
        #                                 to_id=None)) # None = all agents
        #------------------------------------------------------------------------------------------------------------------------


        #het stukje hieronder wordt volgens mij dubbelop gedaan. In filter observations wordt namelijk ook een lijst met slachtoffers gemaakt in de variabele self.__vics_gezien
        #bovendien wordt self.memory_of_victims verder niet gebruikt Bug sensitive code dubbelop
        # ----------------------------------------------------------------------------------------------------
        # Figure out where I am, am I in a room?
        room = self.state.get_closest_with_property("room_name")
        obs_in_closest_room = self.state.get_room_objects(room_name=room[0]['room_name'])
        if self.agent_id in obs_in_closest_room:
            print('inside!!')
            if not room['visited']:
                # Blijkbaar ben je binnen. Nu kijken naar slachtoffers!
                for o in obs_in_closest_room:
                    if 'victim' in o['name']:
                        # Voeg slachtoffers toe aan geheugen.
                        self.memory_of_victims.append(o)
                        # print(self.memory_of_victims)
        # ----------------------------------------------------------------------------------------------------

        #als de mens op een moddertegel staat vertraagd zijn snelheid (=action_duration)
        if self.agent_properties["location"] in self.__modderlocations:
            action_kwargs = {"action_duration": 5}
            # T: log that human moved over a mud tile
            # T: als de mens net een stapje heeft gezet en nu op een moddertegel staat, dan telt dat als over modder heenlopen.
            if self.previous_action and 'move' in str.lower(self.previous_action):
                self.agent_properties['log_human_mud'] += 1
        else:
            #als een mens niet op een moddertegel staat is zijn snelheid normaal
            action_kwargs = {"action_duration": 1}

        for message in list(self.received_messages):
            # T: check if the robot already learned to carry victims together
            # T: de robot stuurt aan de mens dat hij heeft geleerd samen te dragen (zie REF-T01 in robot-code)
            if message.content == "robot_learned_carry":
                self.robot_learned_carry = True
                self.received_messages.remove(message)

            # here, the human keeps track whether the eq_event started (from the moment that the robot takes shelter)
            if message.content == "eq_started":
                self.eq_active = True
                self.received_messages.remove(message)
            if message.content == "eq_stopped":
                self.eq_active = False
                self.received_messages.remove(message)

            if message.content == 'hit_earthquake':
                self.hit_earthquake = True
                self.received_messages.remove(message)
                # log of mens in deuropening staat
                self.agent_properties['log_hit_by_earthquake'] += 1

            if message.content == "carry impossible, human not there":
                self.received_messages.remove(message)
                # if the human was not there when the robot arrived, reset self.sent_carry_request
                self.sent_carry_cue = False
                self.sent_carry_request = False

            # T: if I sent only myself a message, it means that I want to perform an action
            if isinstance(message.content, dict) and message.from_id == self.agent_id and message.to_id == self.agent_id:
                # T: if I want to pick up a lightly wounded victim, do it
                # T: volgens mij mogen onderstaande acties gelijk gereturned worden
                if message.content["action"] == "carry light":
                    action_from_msg = True
                    action = GrabObject.__name__
                    action_kwargs['object_id'] = message.content["object_id"]
                    # set grab range
                    action_kwargs['grab_range'] = self.__grab_range
                    # Set max amount of objects
                    action_kwargs['max_objects'] = self.__max_carry_objects
                    self.received_messages.remove(message)

                # T: if I want to drop a victim that I am carrying, do it
                # T: ook deze actie mag volgens mij gelijk gereturned worden (wel pas na if self.carrying_together!)
                if message.content["action"] == "drop victim":
                    action_from_msg = True
                    action = DropObject.__name__
                    action_kwargs['object_id'] = message.content["object_id"]
                    # set grab range
                    action_kwargs['drop_range'] = self.__drop_range
                    self.received_messages.remove(message)

                    # Check if this drop action is happening in command post;
                    # if so, let the robot know that the victim was saved
                    for obj in self.state.get_room_objects(room_name="command_post"):
                        if obj['obj_id'] == self.agent_properties['obj_id']:
                            self.send_message(
                                Message(content={"action": "saved", "object_id": self.agent_properties['is_carrying'][0]['obj_id']}, from_id=self.agent_id, to_id=robot_id))

                    # T: check if I was carrying this victim together with the robot
                    # T: mens informeert de robot dat samen dragen nu klaar is. Zie REF-T02 in robot code.
                    if self.carrying_together:
                        self.send_message(Message(content={"action": "end help"}, from_id=self.agent_id, to_id=robot_id))
                        self.carrying_together = False
                        self.carry_vic_cue_time = 0 # Reset this timer here!!
            # B: robot laat weten wanneer het carry door begint en eindigd. De mens houdt bij of dit gaande is door boalean self.carry_door
            # B: MISSEND: hier mist denk ik 2x self.received_messages.remove(message) Bug sensitive code
            if message.from_id == "explorer" and "start carry door" in message.content:
                self.carry_door = True
            if message.from_id == "explorer" and "end carry door" in message.content:
                self.carry_door = False
                # print("human helpt deur openen")


            ######## MESSAGES ABOUT LEARNING SITUATIONS ########

            #het stukje vlak hierboven (if message.from_id == "explorer" and "start/end carry door" in message.content:) kan denk ik ook in dit elif stuk, gezien hier ook alle messages van de robot bekeken worden.
            # if I receive message from robot
            elif message.from_id == robot_id and isinstance(message.content, dict):
                # T: deze 'notice' komt maar 1x voor en was alleen maar om aan te geven dat dit een verzoek is van de robot aan de mens om een zwaargewond slachtoffer samen te tillen nadat hij dat heeft geleerd.
                # Deze hele "elif" kan dus weg als we besluiten dat de robot dat niet mag doen.
                # Ook de self.can_respond_to_robot kan dan weg (zie ook comment bovenaan bij initialiseren van deze variabele)
                if "notice" in message.content:
                    # B: -------------------------- dit stukje "help_robot_carry" wordt nooit verstuurd en kan dus denk ik weg-----------------------------
                    # B: let op dat dus de self.can_respond_to_robot nergens op True gezet wordt!!! Bug sensitive code
                    # Tjeerd --> dit mag weg
                    if message.content['notice'] == "help_robot_carry":
                        self.can_respond_to_robot = True
                    # ----------------------------------------------------------------------------------------------------
                    # Tjeerd --> dit mag weg
                    elif message.content['notice'] == "got_no_response":
                        self.can_respond_to_robot = False
                    self.received_messages.remove(message)

                # T: Onderstaande "action" is het verzoek van de robot aan de mens om een bepaald slachtoffer samen te tillen.
                # De robot geeft aan om welk victim het gaat, en de mens voert de carry actie uit.
                if "action" in message.content:
                    action_from_msg = True
                    action = GrabObject.__name__
                    action_kwargs['object_id'] = message.content["victim_id"]
                    # set grab range
                    action_kwargs['grab_range'] = self.__grab_range
                    # Set max amount of objects
                    action_kwargs['max_objects'] = self.__max_carry_objects
                    self.received_messages.remove(message)
                    self.carrying_together = True

        # If rescue worker is hit by earthquake, make them slower
        if self.hit_earthquake and self.hurt_duration > 0:
            action_kwargs["action_duration"] = 5
            self.hurt_duration = self.hurt_duration - 1
            # print(self.hurt_duration)
        else:
            # Reset parameters
            self.hurt_duration = 150
            self.hit_earthquake = False

        ### Behaviour cue from human about mud path ###
        # T: de mens stuurt hier een cue (message) over modder vermijden. Dat doet hij als hij na X ticks over max. Y moddertegels heeft gelopen.
        # Als de mens dus zelf veel over modder heeft gelopen, dan telt dat niet als cue en leert de robot het dus niet via deze cue.
        if state['World']['nr_ticks'] == self.mud_cue_tick_threshold:
            # print("log_human_mud:")
            # print(self.agent_properties['log_human_mud'])
            # print("threshold:")
            # print(self.mud_cue_count_threshold)
            # print(self.agent_properties['log_human_mud'] <= self.mud_cue_count_threshold and \
            #     (state['setting']['scenario'] == 3 or state['setting']['scenario'] == 4))
            if self.agent_properties['log_human_mud'] <= self.mud_cue_count_threshold and \
                (state['setting']['scenario'] == 3 or state['setting']['scenario'] == 4):
                msg = Message(content={"modder geleerd": True}, from_id=self.agent_id, to_id=None)
                self.send_message(msg)
                # print("message send")
                self.agent_properties['log_mud_cue'] += 1
        # print(f"Moddertegels bewandeld: {self.agent_properties['log_human_mud']}")

        # if no keys were pressed, do nothing
        if not action_from_msg:
            if user_input is None or user_input == []:
                # T: als de mens geen toets indrukt, dan telt dat als idle time (dit is trouwens niet helemaal eerlijk, want de mens kan ook aan het "muizen" zijn), of gewoon even denken over een aanpak, maar goed.).
                # De totale idle time wordt op de laatste tick opgeslagen in de log (zie co_learning_logger file)
                self.agent_properties['idle_time'] = self.agent_properties['idle_time'] + 1

                # T: Hier wordt vast bepaald welk vakje linksboven van de mens zit. En de variabele injured_vic_nearby wordt in ieder tick weer (terug)gezet op False.
                # Straks gaan we dit vakje gebruiken om rondom de mens te kunnen kijken of daar gewwonde slachtoffers zitten. Dat is nodig om de cue te sturen (zie hieronder)
                #Waarom top_left en niet locatie human? --> matrix vraagt coordinaat van de top left en niet van het midden
                top_left_of_human = (self.agent_properties["location"][0] - 1, self.agent_properties["location"][1] - 1)
                #injured_vic_nearby = False

                # T: If the robot already learned to carry together, no cue needs to be sent.
                # Robot should then just go help the human after a short amount of time.
                #Tjeerd --> enige verschil tussen deze en de behavioral cue van mens is dat de robot nu eerder komt aangesnelt
                if self.robot_learned_carry:
                    human_next_to_injured_vic = False
                    victim = None
                    # Hier wordt gekeken of er een nog niet gered, zwaargewond slachtoffer 1 vakje rondom de mens is.
                    for obj in state.get_objects_in_area(top_left_of_human, 2, 2):
                        # if a not-already-saved heavily wounded victim is next to the human
                        if Victim.__name__ in obj["class_inheritance"] and obj["treatment_need"] == 2 and not obj["saved"]:
                            human_next_to_injured_vic = True
                            victim = obj
                            self.wait_for_carry_time += 1
                            break # als we er een gevonden hebben, dan hoeven we niet verder te zoeken door de objecten rondom de mens.
                    # als er een niet-gered zwaargewond slachtoffer naast de mens staat en de mens niet al een carry request (expl. of assignment, dus niet de cue want dat is carry_cue) heeft gedaan
                    if human_next_to_injured_vic and not self.sent_carry_request and not self.eq_active:
                        # als de threshold verstreken is, stuur een bericht aan de robot om te helpen tillen.
                        if self.wait_for_carry_time >= self.wait_for_carry_threshold and victim:
                            self.send_message(Message(
                                content={"action": "help carry", "victim": victim},
                                from_id=self.agent_id, to_id=None))
                            self.sent_carry_request = True
                            self.wait_for_carry_time = 0
                    # als er geen zwaargewond slachtoffer rondom de mens ligt, of als de mens al een request heeft gestuurd, reset de timer.
                    else:
                        self.wait_for_carry_time = 0

                    # we assume that the human carries-together the victim that it just was standing next to
                    # reset self.carry_request
                    if (self.carrying_together and self.sent_carry_request):
                        self.sent_carry_request = False


                # T: robot did not learn to carry together yet, so a cue needs to be sent after some time
                # Als de robot nog niet heeft geleerd om samen te tillen en de mens nog geen cue heeft gestuurd,
                # dan moet er een cue gestuurd gaan worden nadat de carry_vic_cue_threshold is verstreken
                # dit mag alleen als het earthquake event niet is gestart (anders wordt de cue niet goed verwerkt door de robot)
                elif not self.sent_carry_cue and not self.eq_active:
                    for obj in state.get_objects_in_area(top_left_of_human, 2, 2):
                        # if a not-already-saved heavily wounded victim is next to the human
                        if Victim.__name__ in obj["class_inheritance"] and obj["treatment_need"] == 2 and not obj["saved"]:
                            self.injured_vic_nearby = True # we gebruiken hier injured_vic_nearby in plaats van human_next_to_injured_vic, zodat die twee elkaar niet conflicteren. De functie van beide variabelen is echter precies hetzelfde.
                            self.carry_vic_cue_time += 1
                            # if we reach the cue threshold, send the cue to the agent
                            if self.carry_vic_cue_time >= self.carry_vic_cue_threshold:
                                msg = Message(
                                    content={"action": "cue help carry", "victim": obj},
                                    from_id=self.agent_id, to_id=None)
                                self.send_message(msg)
                                self.sent_carry_cue = True
                                #self.carry_vic_cue_time = 0
                                self.agent_properties['log_carry_cue'] += 1
                            break # we hoeven niet door te zoeken in objecten rondom de mens, als een zwaargewond slachtoffer is gevonden.
                        else:
                            self.injured_vic_nearby = False
                # als er geen zwaargewond slachtoffer (meer) rondom de mens is, reset de cue counter en sent_carry_cue
                if not self.injured_vic_nearby:
                    self.carry_vic_cue_time = 0
                    self.sent_carry_cue = False

                return None, {}
            # take the latest pressed key and fetch the action
            # associated with that key
            # T: helemaal hier pas wordt gekeken of de mens een toets heeft ingedrukt om een beweeg-actie te doen.
            # De andere acties worden gestart vanuit berichten aan de mens (al dan van de mens zelf, via het context_menu).
            pressed_keys = user_input[-1]
            action = self.key_action_map[pressed_keys]

        # B: if the user chose to do an open or close door action, find a door to
        # open/close within range
        elif action == OpenDoorAction.__name__ or action == CloseDoorAction.__name__:
            action_kwargs['door_range'] = self.__door_range
            action_kwargs['object_id'] = None

            # Get all doors from the perceived objects
            doors = state[{"class_inheritance": Door.__name__}]
            if doors is not None and not isinstance(doors, list):
                doors = [doors]

            # get all doors within range
            doors_in_range = []
            for door in doors:
                object_id = door['obj_id']
                # Select range as just enough to grab that object
                # Dit weet Tjeerd? -> T: Nee, ik heb deze code niet geschreven. Misschien Tjalling?
                dist = int(np.ceil(np.linalg.norm( np.array(state[object_id]['location']) - np.array( state[self.agent_id]['location']))))
                if dist <= action_kwargs['door_range']:
                    doors_in_range.append(object_id)

            # choose a random door within range
            if len(doors_in_range) > 0:
                action_kwargs['object_id'] = self.rnd_gen.choice(doors_in_range)

        #B: als de mens samen met de robot een deur aan het dragen is --> doe als mens niks
        if self.carry_door:
            action = None

        #B: als mens een nieuw gebouw in loopt, open dan de deur. Nieuwe door image is een kwart slag gedraaide deur image als de dichte deur.
        if self.agent_properties["location"] == state.get_closest_room_door()[0]["location"] and not state.get_closest_room_door()[0]["door_opened"]:
            #in deze if'jes wordt de nieuwe door image bepaald aan de hand van welke zijde van het gebouw de deur zit
            if state.get_closest_room_door()[0]["door_side"] == "right":
                new_img = "door_bottom.png"
            if state.get_closest_room_door()[0]["door_side"] == "left":
                new_img = "door_top.png"
            if state.get_closest_room_door()[0]["door_side"] == "bottom":
                new_img = "door_left.png"
            if state.get_closest_room_door()[0]["door_side"] == "top":
                new_img = "door_right.png"
            action = OpenDoor.__name__ #veranderd de image van de deur en zet "door_opened" property op True
            action_kwargs = {**action_kwargs, "opacity": 1.0, "obj_id": state.get_closest_room_door()[0]["obj_id"], "img_name": new_img}

        if action == None: 
            self.agent_properties['idle_time'] = self.agent_properties['idle_time'] + 1

        return action, action_kwargs


    def create_context_menu_for_self(self, clicked_object_id, click_location,
                                     self_selected):
        """ Generate options for a context menu for a specific object/location
        which the user controlling this human agent opened.
        """
        context_menu = []
        obj = self.state[clicked_object_id] # opslaan op welk object de mens heeft geklikt.
        robot_id = self.state.get_agents_with_property({"obj_id": "explorer"})[0]["obj_id"]
        settings = self.state['setting'] # lees de setting parameter uit. Zie REF-T03 in scenario_builder file.

        if obj is None:
            return context_menu

        else:
            if obj['obj_id'] == robot_id and self.can_respond_to_robot:
                # T: Dit is dus het berichtje dat de mens kan sturen aan de robot om te reageren op zijn verzoek om een zwaargewond slachtoffer te helpen tillen, nadat de robot dit heeft geleerd. Kunnen we er uitlaten.
                context_menu.append({
                    "OptionText": "Ik kom je nu helpen!",
                    "Message": Message(content={"notice": "response_to_carry_request", "chat_text": "Ik kom je nu helpen."},
                                       from_id=self.agent_id,
                                       to_id=robot_id)
                })

            if 'vic_' in clicked_object_id:
                # if human is not already carrying a victim & victim is slightly injured
                # add option to pick up victim
                victim = self.state[clicked_object_id]
                if len(self.state[self.agent_id]['is_carrying']) == 0 and victim["treatment_need"]<=1:
                    top_left_of_human = (self.agent_properties["location"][0] - 1, self.agent_properties["location"][1] - 1)
                    for obj in self.state.get_objects_in_area(top_left_of_human, 2, 2):
                        if obj['obj_id'] == victim['obj_id']:
                            context_menu.append({
                                "OptionText": "Draag lichtgewond slachtoffer",
                                "Message": Message(content={"action": "carry light", "object_id": clicked_object_id},
                                                   from_id=self.agent_id, to_id=self.agent_id)
                            })
                            break

                # if victim is heavily wounded, the carry together interventions can be selected (only in exp-condition and in scenario 2 (contains heavily-wounded vics) and 3 (testscenario)
                elif victim["treatment_need"] == 2 and settings['exp_condition'] == 'exp' and (settings['scenario'] == 2 or settings['scenario'] == 3):
                    # add context menu item: Assignment
                    context_menu.append({
                        "OptionText": f"Vraag de robot om te helpen met tillen van dit zwaargewonde slachtoffer",
                        "Message": Message(content={"action": "assignment help carry", "victim":victim, "chat_text":
                                                        "Wil je dit slachtoffer samen met mij tillen?"},
                                           from_id=self.agent_id, to_id=None)
                    })
                    # add context menu item: Explanation
                    context_menu.append({
                        "OptionText": f"Vertel de robot dat zwaargewonde slachtoffers met z'n tweeën gedragen moeten worden",
                        "Message": Message(content={"action": "explanation help carry", "victim":victim, "chat_text":
                                                        "Zwaargewonde slachtoffers moeten we met z'n tweeën dragen!"},
                                           from_id=self.agent_id, to_id=None)
                    })


            # als de mens op zichzelf klik en een lichtgewond slachtoffer draagt, moet hij deze overal kunnen neerleggen.
            elif self.agent_id in clicked_object_id and self.agent_properties['is_carrying'] and self.agent_properties['is_carrying'][0]["treatment_need"] == 1:
                context_menu.append({
                    "OptionText": f"Leg slachtoffer neer",
                    "Message": Message(content={"action": "drop victim", "object_id": self.agent_properties['is_carrying'][0]['obj_id']},
                                       from_id=self.agent_id, to_id=self.agent_id)
                })

            # als de mens op de zichzelf klikt en draagt samen met de robot een zwaargewond slachtoffer
            elif self.agent_id in clicked_object_id and self.agent_properties['is_carrying'] and self.agent_properties['is_carrying'][0]["treatment_need"] == 2:
                # Kijk of de mens in de CP staat. Zo ja, dan mag hij het slachtoffer neerleggen.
                obj_in_cp = self.state.get_room_objects('command_post')
                for obj in obj_in_cp:
                    if self.agent_id == obj["obj_id"]:
                        context_menu.append({
                            "OptionText": f"Leg slachtoffer samen neer",
                            "Message": Message(content={"action": "drop victim", "object_id": self.agent_properties['is_carrying'][0]['obj_id']},
                                               from_id=self.agent_id, to_id=self.agent_id)
                        })
                        break

            # als de mens op de robot klikt, en de mens heeft geen cue gestuurd aan de robot, dan mag de mens expliciete interventies kiezen
            # (alleen bij experimentele groep tijdens de training scenarios
            elif "explorer" in clicked_object_id and self.agent_properties['log_mud_cue'] == 0:
                if settings['exp_condition'] == 'exp' and \
                                (settings['scenario'] == 3):
                    context_menu.append({
                        "OptionText": f"Vraag de robot om modderpaden te vermijden",
                        "Message": Message(content={"modder geleerd": True, "chat_text": "Wil je vanaf nu modderpaden vermijden?"},
                                           from_id=self.agent_id, to_id=None)
                    })
                    context_menu.append({
                        "OptionText": f"Vertel de robot dat modder dat modder zijn snelheid vertraagt",
                        "Message": Message(content={"modder geleerd": True, "chat_text": "Modderpaden vertraagt je snelheid"},
                                           from_id=self.agent_id, to_id=None)
                    })
            return context_menu

    def create_context_menu_for_other(self, agent_id_who_clicked,
                                      clicked_object_id, click_location):
        """ Generate options for a context menu for a specific object/location
        that a user NOT controlling this human agent opened.
        """
        context_menu = []

        # # Generate a context menu option for every action
        # for action in self.action_set:
        #     context_menu.append({
        #         "OptionText": f"Do action: {action}",
        #         "Message": Message(content=action, from_id=clicked_object_id,
        #                            to_id=self.agent_id)
        #     })
        # return context_menu

    def _set_messages(self, messages=None):
        # make sure we save the entire message and not only the content
        for mssg in messages:
            received_message = mssg
            self.received_messages.append(received_message)