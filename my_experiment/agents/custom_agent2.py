from matrx.actions import Action, ActionResult, GrabObject, DropObject
from my_experiment.custom_actions import OpenDoor, EarthquakeEvent, OpenCollapsedDoor, MagicalDoorAppear, ManageRobotImg #, DragVictim
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.agents import AgentBrain
from matrx.messages import Message
from my_experiment.objects import Victim
from matrx.utils import get_distance

import pickle
import os

class CustomAgent2(AgentBrain):
    """ An artificial agent whose behaviour can be programmed to be, for example, (semi-)autonomous.

    For more extensive documentation on the functions below, see:
    http://docs.matrx-software.com/en/master/sections/_generated_autodoc/matrx.agents.agent_brain.AgentBrain.html#matrx.agents.agent_brain.AgentBrain
    """

    def __init__(self, max_carry_objects=1, waypoints=[], move_speed=0, **kwargs):
        """ Creates an agent brain to move along a set of waypoints.
        """
        super().__init__(**kwargs)

        self.navigator = None
        self.cp_location = (24, 2)
        self.human = None

        # Environment variables: victims
        self.victim_list = []
        self.__vics_gezien = [] #lijst van tuples met alle victims. Elke tuple bevat alle victim informatie
        self.saved_vics = []

        # Environment variables: doors
        self.closest_door = None
        self.doors_visited = [] #list of all the door locations of all the rooms that have been visited
        self.door_loc = []  # list of all the door locations in the world
        self.doors = []
        self.unexplored_doors = []

        # Agent related variables
        self.action_state = None
        self.__max_carry_objects = max_carry_objects
        self.human_to_agent_prev = None  # equavalent van human_to_door_prev. Als het goed is is dit de afstand van de mens tot de agent

        # Mud functionality related variables
        self.__modderlocations = []  # list of locations of all the mud tiles

        # Carry heavily wounded victims related variables
        self.carry_together_vic = None # The to-be carried victim
        self.carry_together = False # This variable tracks if we're currently carrying together
        self.recent_drop_together = False

        # Carry door together related variables
        self.carry_door = False
        self.drag_vic = False
        self.broken_door_timer = 0
        self.last_bd_cue = 'behavioral'  # wat is de huidige interventie die plaatsvind
        self.skiplist = [] # Create a list
        self.opened_doors = []
        self.recent_door_drop = False
        self.collapsed_buildings = []

        # Earthquake related variables
        self.earthquake_timer = 100
        self.human_at_door = False
        self.seismograaf_on = True
        self.earthquake = False
        self.last_eq_cue = 'behavioral'
        self.human_to_door_prev = None
        self.informed_human_about_eq = False
        self.nr_earthquakes = 0

        # Variables that track learning
        self.learned_mud = False
        self.learned_carry = False

        # Code to retrieve info about learned situations if available
        # Als het leerbestand al is aangemaakt (zie REF-T08 in main), dan moet deze uitgelezen worden. Op deze manier
        # neemt de robot het geleerde gedrag mee over de scenarios.
        # Zie REF-T09 in co_learning_logger voor het opslaan van het geleerde gedrag in de pickle file.
        if os.path.isfile('./learned_backup.pkl'):
            with open('learned_backup.pkl', 'rb') as pickle_file:
                pickle_contents = pickle.load(pickle_file)
                print("PICKLE STUFF")
                print(pickle_contents)
            if len(pickle_contents) > 1:
                self.learned_carry = bool(int(pickle_contents[0]))
                self.agent_properties['robot_learned_carry'] = self.learned_carry
                self.learned_mud = bool(int(pickle_contents[1]))
                self.agent_properties['robot_learned_mud'] = self.learned_mud

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
        """ Filters the world state before deciding on an action.
            For our work, this consists of filtering out victims that have not yet been found."""

        # Update state tracker for navigator
        self.state_tracker.update(state)

        # Create a list of all victims in the world
        self.victim_list = state.get_of_type(Victim.__name__)
        # Find the nearest door
        self.closest_door = state.get_closest_room_door()

        # ------------------------------------ Check if new victims were found---------------------------------------
        # Variable to save new victims
        new_victim = []

        # Check if we're at the closest door, and if it's a new door
        if self.closest_door[0]["location"] == self.agent_properties["location"] and self.closest_door[0][
            "location"] not in self.doors_visited:
            # Find the objects in the room that the closest door belongs to
            room_name = self.closest_door[0]["room_name"]
            room_objs = state.get_room_objects(room_name)
            # Check if any of those objects is a victim
            vics_in_room = [obj for obj in room_objs if "vic" in obj["name"]]
            # For all victims in a room, add them to the list of found victims, and to new_victim
            # (for communicating to human)
            for vic in vics_in_room:
                if vic not in self.__vics_gezien:
                    new_victim.append(vic)
                    self.__vics_gezien.append(vic)

        # Remove all found victims from victim_list, to be able to filter out unseen victims later
        for vic_g in self.__vics_gezien:
            for vic_l in self.victim_list:
                if vic_g['name'] == vic_l['name']:
                    self.victim_list.remove(vic_l)

        # Send all newly found victims to the human
        for vic in new_victim:
            self.send_message(
                Message(content=f"{vic['name']}",
                        from_id=self.agent_id,
                        to_id=None))  # None = all agents

        # --------------------------Filter unseen victims out of the observations----------------------------------
        [state.remove(vic["obj_id"]) for vic in self.victim_list]
        # ---------------------------------------------------------------------------------------------------------

        # ----------------------------------bepaal hoe de robot eruit ziet in het spel--------------------------------------
        if self.agent_properties['is_carrying'] and not self.drag_vic:
            # robot draagt alleen een lichtgewond slachtoffer
            self.agent_properties['img_name'] = 'robot_carry.png'
        elif self.carry_together == True:
            # robot en mens dragen samen een zwaargewond slachtoffer.
            self.agent_properties['img_name'] = 'invisible.png'
        elif self.agent_properties['is_carrying'] and self.drag_vic:
            # robot sleept een zwaargewond slachtoffer
            self.agent_properties['img_name'] = 'robot_drag_wounded.png'
        elif self.carry_door:
            # robot en mens dragen samen een deur
            self.agent_properties['img_name'] = 'carry_door.png'
        else:
            # als geen van de bovenstaande dingen geld dan is de robot het normale robot icoontje.
            self.agent_properties['img_name'] = 'robot.png'
        # ----------------------------------------------------------------------------------------------------------------

        return state

    def first_tick(self, state):
        """ Contains code that is run once during the first tick in each world"""

        # If mud has been learned, change navigator
        if self.learned_mud:
            self.navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                       algorithm=Navigator.WEIGHTED_A_STAR_ALGORITHM)

        # Send starting message for initial task division
        msg = "Ik begin bij het Postkantoor, wil jij beginnen bij Dorpsstraat 1?"
        self.send_message(
            Message(content={"chat_text": msg},
                    from_id=self.agent_id,
                    to_id=None))  # None = all agents

        # ------------ Collect all locations of doors in the world ----------------------------------------------------
        room_names = self.state.get_all_room_names()
        room_names_real = [x for x in room_names if x is not None]
        room_names_real.sort()
        for n in room_names_real:
            for doors in self.state[{'class_inheritance': 'Door'}]:
                if doors['room_name'] == n:
                    self.door_loc.append(doors['location'])
                    self.doors.append(doors)

        # Check whether there are unexplored doors, sort list on distance
        for door in self.doors:
            if door['location'] not in self.doors_visited:
                # Calculate the distance to these doors
                distance = get_distance(self.agent_properties['location'], door['location'])
                self.unexplored_doors.append([door, distance])

        self.unexplored_doors.sort(key=lambda x: x[1])

        for door in self.unexplored_doors:
            if door[0]['collapsed']:
                self.collapsed_buildings.append(door[0]['room_name'])

        # -------------Collect all locations with mud------------------------------------------------------------------
        modderlijst = state.get_with_property({"name": "modder"}, combined=False)
        if modderlijst:
            for mud in modderlijst:
                self.__modderlocations.append(mud['location'])
        # -------------------------------------------------------------------------------------------------------------

    def decide_on_action(self, state):
        """ Contains the decision logic of the agent. """
        action = None
        action_kwargs = {}

        self.agent_properties['robot_learned_carry'] = self.learned_carry
        self.agent_properties['robot_learned_mud'] = self.learned_mud

        # Execute first tick actions
        if state['World']['nr_ticks'] == 0:
            self.first_tick(state)
        # print(state['World']['nr_ticks'])
        # Collect data of human
        self.human = self.state.get_agents_with_property({"obj_id": "rescue_worker"})[0]

        # Mud slows down action duration
        if self.agent_properties["location"] in self.__modderlocations:
            action_kwargs = {"action_duration": 15}
            if self.previous_action and 'move' in str.lower(self.previous_action): #TODO check of het een standaard MATRX variabele is
                self.agent_properties['log_robot_mud'] += 1
        else:
            action_kwargs = {"action_duration": 5}

        # Find out whether there are victims to save and/or to drag
        vics_need_rescue, vics_to_drag = self.determine_victim_status()

        # Check whether the earthquake conditions hold
        self.determine_earthquake_conditions()

        # Deal with incoming messages
        self.message_handling()

        # Earthquake goes above all, exceptions are handled in the earthquake determining conditions function
        if self.earthquake:
            self.action_state = 'earthquake'

        # ----------------------------GENERAL ACTION PLANNING LOOP------------------------------------------------
        # We first check if any of the learning situations hold. If not, we follow a general action planning loop,
        # with the following action priorities:
        # 1. If there are any heavily wounded victims in collapsed buildings: drag them outside
        # 2. Else, if there are any lightly wounded victims, save the one nearest to you
        # 3. Else, explore the nearest unexplored building
        print(f"Action state: {self.action_state}")

        if self.action_state == 'earthquake':
            # Run earthquake event
            self.navigator.reset_full()
            action, action_kwargs = self.earthquake_event(action_kwargs)
        elif self.action_state == 'carry_vic_together':
            # Carry victim together
            self.navigator.reset_full()
            action, action_kwargs = self.carry_wounded_victims(state, action_kwargs)
        elif self.action_state == 'carry_door_together':
            # Carry door togeter
            self.navigator.reset_full()
            action, action_kwargs = self.carry_door_together(action_kwargs)
        else:
            if (vics_to_drag and self.action_state is None) or self.action_state == 'drag':
                # Drag heavily wounded victims in collapsed buildings outside
                action, action_kwargs = self.drag_victim(action_kwargs, vics_to_drag)
            elif (vics_need_rescue and self.action_state is None) or self.action_state == 'rescue':
                # Rescue lightly wounded victims
                self.navigator.reset_full()
                action, action_kwargs = self.rescue_victim(action_kwargs, vics_need_rescue)
            elif (len(self.unexplored_doors) > 0 and self.action_state is None) or self.action_state == 'explore' or len(self.skiplist) > 0:
                # Explore new buildings
                self.navigator.reset_full()
                action, action_kwargs = self.explore_new_buildings(action_kwargs)
            else:
                # Wait until human is done
                action = None

        return action, action_kwargs

    def explore_new_buildings(self, action_kwargs):
        # Code that describes how the agent explores new buildings
        action = None
        chosen_building = None
        new_waypoints = []

        # If a door is not yet chosen, make list of doors that have not yet been visited
        self.unexplored_doors = []
        if chosen_building is None:
            for door in self.doors:
                if door['location'] not in self.doors_visited:
                    # Calculate the distance to these doors
                    distance = get_distance(self.agent_properties['location'], door['location'])
                    # Add to list when the door is not in the skiplist
                    if door['obj_id'] not in self.skiplist:
                        self.unexplored_doors.append([door, distance])

            self.unexplored_doors.sort(key=lambda x: x[1])

            # copy unexplored doors list to agent properties, so it is accessible in vics_saved_goals
            # (where it is used to determine whether the scenario should end)
            self.agent_properties['unexplored_doors'] = self.unexplored_doors

            # Choose the door with the smallest distance.
            # If there are no more doors, check if there are doors left in the skiplist. If not, return.
            if len(self.unexplored_doors) > 0:
                chosen_building = self.unexplored_doors[0][0]
            elif len(self.skiplist) > 0:
                # If there are no more doors but there is a skiplist:
                for door in self.doors:
                    if door['location'] not in self.doors_visited:
                        if door['obj_id'] == self.skiplist[0]:
                            chosen_building = door
                self.skiplist.pop(0)
            else:
                self.action_state = None
                return action, action_kwargs
            self.action_state = 'explore'

        # Open door and finish, add to visited doors
        new_img = None
        if self.agent_properties["location"] == self.closest_door[0]["location"] and not self.closest_door[0][
                "door_opened"]:
            if self.closest_door[0]["door_side"] == "right":
                new_img = "door_bottom.png"
            if self.closest_door[0]["door_side"] == "left":
                new_img = "door_top.png"
            if self.closest_door[0]["door_side"] == "bottom":
                new_img = "door_left.png"
            if self.closest_door[0]["door_side"] == "top":
                new_img = "door_right.png"
            action = OpenDoor.__name__
            action_kwargs = {**action_kwargs, "opacity": 1.0, "obj_id": self.closest_door[0]["obj_id"],
                                 "img_name": new_img}
            self.doors_visited.append(self.closest_door[0]["location"])
            self.unexplored_doors.pop(0)
            self.action_state = None

        # If not yet arrived at door, decide whether to go to the door or just before the door (if collapsed)
        elif self.agent_properties['location'] != chosen_building['location']:
            # If chosen door is not collapsed, add door location to waypoints
            if chosen_building['is_open'] or chosen_building['door_opened']:
                new_waypoints.append(chosen_building['location'])  # voeg locatie deur toe aan waypoints, alleen als het geen ingestort gebouw is
            # This means the door is collapsed, add location before door to waypoints
            else:
                # If the distance between the robot and the door is 1 or less, go to new state, unless already opened
                if get_distance(self.agent_properties['location'], chosen_building['location']) == 1:
                    # If already opened, just enter the building
                    if chosen_building['obj_id'] in self.opened_doors:
                        self.navigator.add_waypoints([chosen_building['location']], is_circular=False)
                        action = self.navigator.get_move_action(self.state_tracker)
                    else:
                        self.action_state = 'carry_door_together'
                # Else keep moving towards the door
                else:
                    if chosen_building["door_side"] == "bottom":
                        new_waypoints.append((chosen_building['location'][0], chosen_building['location'][1]+1)) #voeg vakje voor deur toe aan waypoints
                    elif chosen_building["door_side"] == "top":
                        new_waypoints.append((chosen_building['location'][0], chosen_building['location'][1]-1)) #voeg vakje voor deur toe aan waypoints
                    elif chosen_building["door_side"] == "left":
                        new_waypoints.append((chosen_building['location'][0]-1, chosen_building['location'][1])) #voeg vakje voor deur toe aan waypoints
                    elif chosen_building["door_side"] == "right":
                        new_waypoints.append((chosen_building['location'][0]+1, chosen_building['location'][1])) #voeg vakje voor deur toe aan waypoints

            # Add waypoints to navigator
            self.navigator.add_waypoints(new_waypoints, is_circular=False)  # voeg de waypoints toe aan de navigator
            action = self.navigator.get_move_action(self.state_tracker)

        return action, action_kwargs

    def rescue_victim(self, action_kwargs, vics_need_rescue):
        # Code that describes how the agent saves a victim
        action = None

        # If carrying a victim
        new_waypoints = []
        if self.agent_properties['is_carrying']:
            # If at command post, drop
            if self.agent_properties['location'] == self.cp_location:
                self.action_state = None
                victim_id = self.agent_properties['is_carrying'][0]['obj_id']
                self.saved_vics.append(victim_id)
                action = DropObject.__name__
            # Else, add command post to waypoints
            else:
                self.navigator.add_waypoints([self.cp_location], is_circular=False)
                action = self.navigator.get_move_action(self.state_tracker)
        # If not carrying, calculate distances to the different victims
        else:
            victim_distlist = []
            for vic in vics_need_rescue:
                distance = get_distance(self.agent_properties['location'], vic['location'])
                victim_distlist.append([vic, distance])

            victim_distlist.sort(key=lambda x: x[1])

            # Choose nearest victim to save
            if len(victim_distlist) > 0:
                chosen_victim = victim_distlist[0][0]
                self.action_state = 'rescue'
            else:
                self.action_state = None
                return action, action_kwargs

            # Check if we're at that victim
            if self.agent_properties['location'] == chosen_victim['location']:
                # Pick up the victim
                action = GrabObject.__name__
                action_kwargs['object_id'] = chosen_victim['obj_id']
                action_kwargs['max_objects'] = self.__max_carry_objects
                action_kwargs['grab_range'] = 1
            else:
                # Add victim location to waypoints
                self.navigator.add_waypoints([chosen_victim['location']], is_circular=False)  # voeg de waypoints toe aan de navigator
                action = self.navigator.get_move_action(self.state_tracker)

        return action, action_kwargs

    def drag_victim(self, action_kwargs, vics_to_drag):
        # Code that describes how the agent drags a heavily wounded victim in a collapsed building outside
        action = None
        drag_target = None

        # If carrying a victim
        new_waypoints = []
        if self.agent_properties['is_carrying']:
            # bepaal de locatie waar gesleepte victim neergelegd moet worden en voeg deze toe aan drag_target
            if self.closest_door[0]["door_side"] == "left":
                drag_target = (self.closest_door[0]["location"][0] - 1, self.closest_door[0]["location"][1])
            elif self.closest_door[0]["door_side"] == "right":
                drag_target = (self.closest_door[0]["location"][0] + 1, self.closest_door[0]["location"][1])
            elif self.closest_door[0]["door_side"] == "bottom":
                drag_target = (self.closest_door[0]["location"][0], self.closest_door[0]["location"][1] + 1)
            elif self.closest_door[0]["door_side"] == "top":
                drag_target = (self.closest_door[0]["location"][0], self.closest_door[0]["location"][1] - 1)

            # If at drag target, drop
            if self.agent_properties['location'] == drag_target:
                self.drag_vic = False
                self.action_state = None
                action = DropObject.__name__
            # Else, add drag target to waypoints
            else:
                self.navigator.add_waypoints([drag_target], is_circular=False)
                action = self.navigator.get_move_action(self.state_tracker)
        # If not carrying, calculate distances to the different victims
        else:
            victim_distlist = []
            for vic in vics_to_drag:
                distance = get_distance(self.agent_properties['location'], vic['location'])
                victim_distlist.append([vic, distance])

            victim_distlist.sort(key=lambda x: x[1])

            # Choose nearest victim to save
            if len(victim_distlist) > 0:
                chosen_victim = victim_distlist[0][0]
                self.action_state = 'drag'
            else:
                self.action_state = None
                return action, action_kwargs

            # Check if we're at that victim
            if self.agent_properties['location'] == chosen_victim['location']:
                # Pick up the victim
                action = GrabObject.__name__
                self.drag_vic = True
                action_kwargs['object_id'] = chosen_victim['obj_id']
                action_kwargs['max_objects'] = self.__max_carry_objects
                action_kwargs['grab_range'] = 1
            else:
                # Add victim location to waypoints
                self.navigator.add_waypoints([chosen_victim['location']], is_circular=False)  # voeg de waypoints toe aan de navigator
                action = self.navigator.get_move_action(self.state_tracker)

        return action, action_kwargs

    def carry_door_together(self, action_kwargs):
        # Code for what happens when an agent encounters a collapsed door
        action = None
        target_location = None
        human_to_broken_door = get_distance(self.human['location'], self.closest_door[0]["location"])
        intervention_timer = 30

        # New piece of code that only serves to go to the door opening and open that door after the drop together
        # Returns directly to make sure the code below is not accessed anymore; action state is also reset here
        if self.recent_door_drop:
            # If the robot is at the door opening, 'open the door' and reset
            if self.agent_properties["location"] == self.closest_door[0]["location"]:
                self.doors_visited.append(self.closest_door[0]["location"])
                if self.unexplored_doors:
                    self.unexplored_doors.pop(0)
                self.action_state = None
                self.recent_door_drop = False
                return action, action_kwargs
            # Else, walk to the door opening
            else:
                self.navigator.add_waypoints([self.closest_door[0]['location']], is_circular=False)
                action = self.navigator.get_move_action(self.state_tracker)
                return action, action_kwargs

        # If carrying door
        if self.carry_door:
            # If at target location, drop door
            target_location = self.door_target_location()
            if self.agent_properties['location'] == target_location:
                # Drop door
                # stuur mens dat de deur samen dragen klaar is (en dus weer zichtbaar mag worden en acties mag uitvoeren)
                self.send_message(
                    Message(content="end carry door",
                            from_id=self.agent_id,
                            to_id=None))

                # log that we have carried a door together
                self.agent_properties['log_carry_door'] += 1

                # eindig de carry_door actie
                self.carry_door = False

                # DO NOT reset action state here, the robot still needs to walk to the door opening to look for victims
                # Instead, record that there was a recent doordrop
                #self.action_state = None
                self.recent_door_drop = True

                self.opened_doors.append(self.closest_door[0]['obj_id'])
                action = MagicalDoorAppear.__name__
                action_kwargs = {**action_kwargs, "location_door": self.agent_properties["location"], "opacity": 1.0,
                                 "obj_id": self.closest_door[0]["obj_id"],
                                 "img_name": 'invisible.png'}
            # If not, walk to target location
            else:
                self.navigator.add_waypoints([target_location], is_circular=False)
                action = self.navigator.get_move_action(self.state_tracker)

        # If not carrying door
        else:
            self.broken_door_timer += 1
            # Check if there has previously been a distance 'human to agent'
            if self.human_to_agent_prev:
                # If human is near, start carrying door
                if human_to_broken_door < 2 and not self.human['is_carrying'] \
                        and get_distance(self.agent_properties['location'], self.closest_door[0]['location']) < 2:
                    self.carry_door = True
                    self.last_bd_cue = 'behavioral'
                    # stuur bericht naar mens zodat deze onzichtbaar wordt en geen acties kan uitvoeren tot de carry door klaar is
                    self.send_message(
                        Message(content="start carry door",
                                from_id=self.agent_id,
                                to_id=None))
                    self.broken_door_timer = 0 # Reset timer when we start carrying the door
                    action = OpenCollapsedDoor.__name__
                    action_kwargs = {**action_kwargs, "opacity": 1.0, "obj_id": self.closest_door[0]["obj_id"],
                                     "img_name": 'invisible.png'}
                # If human not near, of if they are moving towards robot, and the behavioral cue timer did not run out, do behavioral cue
                elif (human_to_broken_door > 1 or self.human_to_agent_prev > human_to_broken_door) and \
                        self.broken_door_timer < intervention_timer:

                    if self.broken_door_timer % 3 == 0:  # elke 3 ticks
                        closest_door_location = self.closest_door[0]["location"]
                        if self.closest_door[0]["door_side"] == "left" and "broken" in self.closest_door[0]["img_name"]:
                            self.navigator.add_waypoints(
                                [(closest_door_location[0] - 2, closest_door_location[1]),
                                 (self.closest_door[0]["location"][0] - 1, self.closest_door[0]["location"][1])],
                                is_circular=False)
                            action = self.navigator.get_move_action(self.state_tracker)
                        elif self.closest_door[0]["door_side"] == "right" and "broken" in self.closest_door[0]["img_name"]:
                            self.navigator.add_waypoints(
                                [(closest_door_location[0] + 2, closest_door_location[1]),
                                 (self.closest_door[0]["location"][0] + 1, self.closest_door[0]["location"][1])],
                                is_circular=False)
                            action = self.navigator.get_move_action(self.state_tracker)
                        elif self.closest_door[0]["door_side"] == "top" and "broken" in self.closest_door[0]["img_name"]:
                            self.navigator.add_waypoints(
                                [(closest_door_location[0], closest_door_location[1] - 2),
                                 (self.closest_door[0]["location"][0], self.closest_door[0]["location"][1] - 1)],
                                is_circular=False)
                            action = self.navigator.get_move_action(self.state_tracker)
                        elif self.closest_door[0]["door_side"] == "bottom" and "broken" in self.closest_door[0]["img_name"]:
                            self.navigator.add_waypoints(
                                [(closest_door_location[0], closest_door_location[1] + 2),
                                 (self.closest_door[0]["location"][0], self.closest_door[0]["location"][1] + 1)],
                                is_circular=False)
                            action = self.navigator.get_move_action(self.state_tracker)

                    # Only do the learning interventions if the human is not moving towards the door
                    if self.human_to_agent_prev > human_to_broken_door:
                        # If the human is moving towards the door, make sure the timer is reset
                        self.broken_door_timer = self.broken_door_timer - 2

                    else:
                        # Continue sequence of learning interventions until end, then break out of action state
                        if self.broken_door_timer % 10 == 0 and self.state['setting']['exp_condition'] == 'exp' and \
                                    (self.state['setting']['scenario'] == 2 or self.state['setting']['scenario'] == 3):
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

                    # Vernieuw de human_to_agent_prev met de nieuwe afstand
                    self.human_to_agent_prev = human_to_broken_door

                elif self.broken_door_timer > intervention_timer and (self.last_bd_cue == 'hq' or \
                        self.state['setting']['exp_condition'] != 'exp') or self.state['setting']['scenario'] == 4:
                    self.broken_door_timer = 0
                    self.action_state = None
                    self.last_bd_cue = 'behavioral'
                    # Some code that indicates the agent should skip this door for a while
                    self.skiplist.append(self.closest_door[0]['obj_id'])
            else:
                # If variable doesn't exist yet, create value here
                self.human_to_agent_prev = get_distance(self.human['location'], self.closest_door[0]["location"])

        return action, action_kwargs

    def carry_wounded_victims(self, state, action_kwargs):
        # Code for what happens when a human asks the robot to help with a heavily wounded victim (by behavior cue, explanation or assignment
        action = None

        # Check if the robot is still carrying a victim, if so, drop and return immediately
        if self.agent_properties['is_carrying']:
            action = DropObject.__name__
            return action, action_kwargs

        # If they're carrying
        if self.carry_together:
            # If robot is at CP
            if self.agent_properties["location"] == self.cp_location:
                # If there was a drop, become visible again, set carry_together to False
                if self.recent_drop_together:
                    # Tjeerd: als robot klaar is met samen tillen, dan moet het tandwieltje van de robot ook weer zichtbaar worden.
                    # de robot wordt zelf ook weer zichtbaar, maar dat gebeurt vanaf REF-T05.
                    # De reden dat deze code hier staat is dat het (on)zichtbaar maken van het tandwieltje een Action vereist.
                    # turn off visibility of the busy-icon of the robot
                    action = ManageRobotImg.__name__
                    action_kwargs['cog_visible'] = True
                    self.recent_drop_together = False
                    self.carry_together = False
                    self.action_state = None
                # If not, wait
                else:
                    action = None
            # If not, walk to CP
            else:
                self.navigator.add_waypoints([self.cp_location], is_circular=False)
                action = self.navigator.get_move_action(self.state_tracker)

        # If they're not carrying
        else:
            # If the robot is at the victim location, start carrying if the human is also still there
            if self.agent_properties['location'] == self.carry_together_vic['location']:
                # robot has arrived at victim, so check if human is also there. If yes, we are going to carry together.
                top_left_of_robot = (self.agent_properties["location"][0] - 1, self.agent_properties["location"][1] - 1)
                human_present = False
                for obj in state.get_objects_in_area(top_left_of_robot, 2, 2):
                    # Check if the human is next to the robot
                    if obj["obj_id"] == self.human['obj_id']:
                        human_present = True

                # If the human is present
                if human_present:
                    # stuur bericht aan mens dat we samen kunnen gaan dragen
                    self.send_message(Message(
                        content={"action": "carry together", "loc_robot": self.agent_properties["location"],
                                    "victim_id": self.carry_together_vic['obj_id']}, from_id=self.agent_id,
                        to_id=self.human['obj_id']))
                    self.carry_together = True  # self.carry_together houdt bij of we samen aan het dragen zijn.

                    # Tjeerd: als robot samen gaat tillen, dan moet het tandwieltje van de robot onzichtbaar worden.
                    # de robot wordt zelf ook onzichtbaar, maar dat gebeurt vanaf REF-T05.
                    # De reden dat deze code hier staat is dat het (on)zichtbaar maken van het tandwieltje een Action vereist.
                    action = ManageRobotImg.__name__
                    action_kwargs['cog_visible'] = False

                # if the human requested help but is not there, the carry task is skipped
                else:
                    # let the human know that we could not carry together
                    self.send_message(Message(
                        content="carry impossible, human not there",
                        from_id=self.agent_id,
                        to_id=self.human['obj_id']))
                    action = None
                    action_kwargs = {}
                    self.action_state = None
            # If not, walk to victim location
            else:
                self.navigator.add_waypoints([self.carry_together_vic['location']], is_circular=False)
                action = self.navigator.get_move_action(self.state_tracker)

        return action, action_kwargs

    def earthquake_event(self, action_kwargs):
        # Code for what happens when the earthquake event is triggered
        # notify human that earthquake started (carry cues cannot be sent)
        if not self.informed_human_about_eq:
            self.send_message(Message(content="eq_started",
                                      from_id=self.agent_id,
                                      to_id=None))
            self.informed_human_about_eq = True
        action = None
        intervention_timer = 50
        # Check if the earthquake event holds, and if the robot is already at a door or not
        if self.agent_properties['location'] == self.closest_door[0]["location"]:  # Leerinterventies starten hieronder
            # print('Robot at door')
            self.earthquake_timer = self.earthquake_timer - 1
            # Check timer
            if self.earthquake_timer > 6:
                # Hold actions and check whether to initiate a learning intervention
                action = None
                if self.human_to_door_prev:
                    door_human = None
                    if self.closest_door_location(self.human['location'])[0][0]['collapsed']:
                        door_human = self.closest_door_location(self.human['location'])[1][0]
                    else:
                        door_human = self.closest_door_location(self.human['location'])[0][0]

                    human_to_door = get_distance(self.human['location'], door_human['location'])

                    if human_to_door < 1 or self.human_to_door_prev > human_to_door:
                        # print("Human is at door or moving to door!")
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
                                            to_id=self.state.get_agents_with_property({"obj_id": "headquarters"})[0][
                                                "obj_id"]))
                                self.last_eq_cue = 'hq'
                                self.earthquake_timer = 50
                else:
                    # If variable doesn't exist yet, create value here
                    self.human_to_door_prev = get_distance(
                        self.human['location'], self.closest_door_location(self.human['location'])[0][0]['location'])
                return action, action_kwargs  # Return zodat de loop verder wordt afgebroken
            elif self.earthquake_timer > 0:
                # Check if human is currently carrying, break loop here when that is the case and add to timer
                if self.human['is_carrying'] and not self.human_at_door:
                    self.earthquake_timer = self.earthquake_timer + 1
                    action = None
                    return action, action_kwargs

                # Check again if human is at door and update variable
                if self.closest_door_location(self.human['location'])[0][0]['collapsed']:
                    door_human = self.closest_door_location(self.human['location'])[1][0]
                else:
                    door_human = self.closest_door_location(self.human['location'])[0][0]
                human_to_door = get_distance(self.human['location'], door_human['location'])
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
                self.action_state = None
                action = EarthquakeEvent.__name__
                action_kwargs['earthquake_happening'] = False
                # notify human that earthquake event stopped (carry cues can be sent again)
                self.send_message(Message(content="eq_stopped",
                                          from_id=self.agent_id,
                                          to_id=None))
                self.informed_human_about_eq = False
                self.nr_earthquakes = self.nr_earthquakes +1 # Count that there was an earthquake
                return action, action_kwargs
        else:
            print('robot not at door')
            if self.closest_door[0]['is_open']:
                self.navigator.add_waypoints([self.closest_door[0]["location"]], is_circular=False)
                action = self.navigator.get_move_action(self.state_tracker)
            else:
                self.navigator.add_waypoints([self.closest_door_location(
                    self.agent_properties['location'])[1][0]["location"]], is_circular=False)
                action = self.navigator.get_move_action(self.state_tracker)

        return action, action_kwargs

    def message_handling(self):
        # Code for dealing with messages

        for message in self.received_messages:
            # -------------------------------------- Victim found -----------------------------------------------------
            # Add if-statement that checks if it is this type of message
            for vic_l in self.victim_list:
                # If a message is received that contains information about a victim
                if vic_l['name'] == message.content:
                    # Put that victim on the found list
                    self.__vics_gezien.append(vic_l)

            # -------------------------------------- Door explored ----------------------------------------------------
            # If a message is received that contains information about an explored door location
            if isinstance(message.content, dict) and "info" in message.content and message.content["info"] == "door":
                door_location = message.content['location']
                if door_location not in self.doors_visited:
                    self.doors_visited.append(door_location)
                    for door in self.unexplored_doors:
                        if str(door[0]['location']) == door_location:
                            self.unexplored_doors.remove(door)

            # ------------------------------- Victim saved by human --------------------------------------------------
            if isinstance(message.content, dict) and "action" in message.content and message.content["action"] == "saved":
                saved_victim = message.content['object_id']
                self.saved_vics.append(saved_victim)

            # -------------------------------------- Finish carrying together------------------------------------------
            # REF-T02
            # als de robot een bericht krijgt met 'end help' in de Key "action", dan is hij klaar met het samen tillen van een zwaargewond slachtoffer.
            # De mens stuurt dit bericht zodra hij/zij een zwaargewond slachtoffer neerlegt.
            if message.from_id == self.human['obj_id'] and isinstance(message.content, dict) and "action" in message.content and \
                    message.content["action"] == "end help":

                # Set recent drop variable to False
                self.recent_drop_together = True

                # log that we carried a victim together
                self.agent_properties['log_carry_together'] += 1

                # REF-T01
                # after bringing 1 heavily wounded victim to the CP together, the robot directly learns that
                # all heavily wounded victims should be carried together from now on
                if not self.learned_carry:
                    self.learned_carry = True
                    self.agent_properties['robot_learned_carry'] = self.learned_carry
                    # Tjeerd: stuur bericht aan de mens, zodat de mens geen cues meer gaat sturen als de robot het al geleerd heeft
                    # (zie "if self.robot_learned_carry" in custom_human_agent)
                    self.send_message(Message(content="robot_learned_carry",
                                              from_id=self.agent_id,
                                              to_id=None))

            # ------------------------------ Request for carrying together---------------------------------------------
            # Tjeerd: de robot krijgt via een message een verzoek van de mens binnen om samen een slachtoffer te tillen.
            # Ook wordt de parameter human_help_needed op True gezet, zodat de robot weet dat hij moet komen helpen (zie later in code).
            if isinstance(message.content, dict) and 'action' in message.content and 'help carry' in message.content[
                "action"]:

                # Extract the victim from the message
                self.carry_together_vic = message.content["victim"]

                # Inform the human that the robot is coming to help
                if not self.action_state == 'earthquake':
                    self.send_message(
                        Message(content={"chat_text": "Ik kom eraan om je te helpen."},
                                from_id=self.agent_id,
                                to_id=None))
                    self.action_state = 'carry_vic_together'

                # inform the HQ agent that he should not send a message to the human about carrying (Reflection)
                hq_agent_id = self.state.get_agents_with_property({"obj_id": "headquarters"})[0]["obj_id"]
                self.send_message(
                    Message(content="do_not_reflect_carry",
                            from_id=self.agent_id,
                            to_id=hq_agent_id))

            # --------------------------------- Mud is learned --------------------------------------------------------
            # Tjeerd: in deze if-statement krijgt de robot een andere Navigator (weighted_a_star in plaats van a_star) mee,
            # die ervoor zorgt dat hij modderpaden kan ontwijken. Trigger hiervoor is een bericht van de mens.
            if (message.from_id == 'rescue_worker' and 'modder geleerd' in message.content):

                self.learned_mud = True
                self.agent_properties['robot_learned_mud'] = self.learned_mud
                # Change Navigator to weighted A*
                self.navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                           algorithm=Navigator.WEIGHTED_A_STAR_ALGORITHM)
                # respond if the human explicitly asked the robot to avoid mud (do not respond when this is done by cueing)
                if "chat_text" in message.content:
                    self.send_message(
                        Message(content={"chat_text": "Oke."},
                                from_id=self.agent_id,
                                to_id=None))

            # After dealing with each message, remove it
            self.received_messages.remove(message)

        return

    def determine_victim_status(self):
        # Code for determining which victims are heavily and lightly wounded
        # List of all heavily wounded victims and all lightly wounded victims in the world
        vics_lightly = self.state[{"treatment_need": 1}]
        vics_heavily = self.state[{"treatment_need": 2}]

        # Every tick vics_need_rescue and vics_to_drag are made empty
        vics_need_rescue = []
        vics_to_drag = []

        # Check whether lightly wounded victims have already been saved (location = CP), add to rescue list if not saved
        if vics_lightly and isinstance(vics_lightly, dict):
            if vics_lightly['obj_id'] not in self.saved_vics:
                vics_need_rescue.append(vics_lightly)
        if vics_lightly and isinstance(vics_lightly, list):
            for vic in vics_lightly:
                if vic['obj_id'] not in self.saved_vics:
                    vics_need_rescue.append(vic)

        # Check whether there are any heavily wounded victims in collapsed buildings, add to draglist
        if vics_heavily:
            # Look at the room objects of the collapsed buildings
            for room_name in self.collapsed_buildings:
                room_objects = self.state.get_room_objects(room_name)

                # Then check if the heavily wounded victims are in these objects. If yes, record
                if isinstance(vics_heavily, dict):
                    if vics_heavily in room_objects:
                        vics_to_drag.append(vics_heavily)
                elif isinstance(vics_heavily, list):
                    for vic in vics_heavily:
                        if vic in room_objects:
                            vics_to_drag.append(vic)

        return vics_need_rescue, vics_to_drag

    def door_target_location(self):
        # Code that calculates the target location of where the door should be dropped
        door_target = ()

        if self.closest_door[0]["door_side"] == "left":
            door_target = (self.closest_door[0]["location"][0] - 1, self.closest_door[0]["location"][1] - 1)
        elif self.closest_door[0]["door_side"] == "right":
            door_target = (self.closest_door[0]["location"][0] + 1, self.closest_door[0]["location"][1] - 1)
        elif self.closest_door[0]["door_side"] == "bottom":
            door_target = (self.closest_door[0]["location"][0] - 1, self.closest_door[0]["location"][1] + 1)
        elif self.closest_door[0]["door_side"] == "top":
            door_target = (self.closest_door[0]["location"][0] - 1, self.closest_door[0]["location"][1] - 1)

        return door_target

    def determine_earthquake_conditions(self):
        # Code die checkt of aan de aardbeving condities wordt voldaan

        # Set seismograaf variable back to True between earthquakes
        if not self.seismograaf_on and len(self.saved_vics) >= 3:
            self.seismograaf_on = True

        if self.seismograaf_on and not self.agent_properties['is_carrying'] and not self.human['is_carrying'] and \
                (self.state['setting']['scenario'] == 2 or self.state['setting']['scenario'] == 4):
            # Check that the agent does not happen to be in a carry together related action state
            if self.action_state != 'carry_vic_together' and self.action_state != 'carry_door_together':
                # If 2 or more victims were saved, and there was less than 1 earthquake, start the eq event
                if len(self.saved_vics) >= 2 and self.nr_earthquakes < 1:
                    self.earthquake = True
                # If 4 or more victims were saved, and there was 1 earthquake, start the eq event
                elif len(self.saved_vics) >= 4 and self.nr_earthquakes == 1:
                    self.earthquake = True
        return

    # Function to get list of doors sorted on the distance to a given point
    def closest_door_location(self, start_location):
        room_doors = []
        for n in self.state.get_all_room_names():
            room_door = self.state.get_room_doors(room_name=n)
            distance_to_door = get_distance(start_location, room_door[0]['location'])
            room_doors.append([room_door[0], distance_to_door])
        room_doors_sorted = sorted(room_doors, key=lambda x: x[1])
        return room_doors_sorted

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
