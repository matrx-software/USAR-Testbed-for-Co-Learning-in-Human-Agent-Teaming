import json
import pickle
import os
from datetime import datetime
from my_experiment.custom_actions import MoveEastHuman, MoveNorthHuman, MoveSouthHuman, MoveWestHuman
from my_experiment.objects import Victim
from my_experiment.agents.custom_agent import CustomAgent
from my_experiment.agents.custom_agent2 import CustomAgent2
from my_experiment.agents.custom_human_agent import CustomHumanAgent
from my_experiment.goals.vics_saved_goal import VictimsSavedGoal

from matrx.objects import EnvObject

from matrx.world_builder import WorldBuilder
from matrx.actions import MoveNorth, OpenDoorAction, CloseDoorAction
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest
from matrx.grid_world import DropObject, GrabObject
from my_experiment.agents.hq_agent import HQAgent
from my_experiment.loggers.co_learning_logger import CoLearningLogger

# add victims based on scenario file
def generate_victims_from_file(factory, config):
    # config = json.load(open("usar/experiment_oefen_scenario.json"))
    victims_config = config["victims"]
    for j, victim_info in enumerate(victims_config):
        name = "vic_"+str(j)
        vic_location = victim_info["location"]
        factory.add_object(location=[vic_location[0], vic_location[1]], name=name, callable_class=Victim,
                           alive=victim_info["alive"], treatment_need=victim_info["treatment_need"], need_increase_time=victim_info["need_increase_time"])

    # , visualize_depth=110
    return(len(victims_config))


# Score
class Score(EnvObject):
    def __init__(self, location, name="scores", victims_total=0, score_vis_placement='right'):
        super().__init__(name=name, location=location, visualize_opacity=0,
                         is_traversable=True, class_callable=Score)

        self.victims_total = victims_total
        self.add_property('victims_total', self.victims_total)

        self.victims_alive = 0
        self.add_property('victims_alive', self.victims_alive)

        self.victims_retrieved = 0
        self.add_property('victims_retrieved', self.victims_retrieved)

        self.ticks_elapsed = 0
        self.add_property('ticks_elapsed', self.ticks_elapsed)

        self.batteries_replaced = 0
        self.add_property('batteries_replaced', self.batteries_replaced)

        self.add_property('score_vis_placement', score_vis_placement)

        # shortcut so we can access it as such: state['scores']
        self.obj_id = self.obj_name


# Settings (this object is there so that the json config file is only read in one place)
class Settings(EnvObject):
    def __init__(self, location, name="settings", config={}, **kwargs):
        super().__init__(name=name, location=location, visualize_opacity=0,
                         is_traversable=True, class_callable=Settings, **kwargs)

        # # Probability that a room collapses after an automatic earthquake
        # self.p_collapse_after_earthquake = config['p_collapse_after_earthquake']
        # self.add_property('p_collapse_after_earthquake', self.p_collapse_after_earthquake)
        #
        # # Probability that a victim in a room gets hurt if the room collapses after an automatic earthquake
        # self.p_victim_gets_hurt_after_room_collapse = config['p_victim_gets_hurt_after_room_collapse']
        # self.add_property('p_victim_gets_hurt_after_room_collapse', self.p_victim_gets_hurt_after_room_collapse)

        # Epicenter of manual earthquake
        self.epicenter = config['manual_earthquake']['epicenter']
        self.add_property('epicenter', self.epicenter)

        # radius of the earthquake
        self.earthquake_radius = config['manual_earthquake']['radius']
        self.add_property('earthquake_radius', self.earthquake_radius)

        # List of affected buildings affected by manual earthquake
        self.affected_buildings = config['manual_earthquake']['affected_buildings']
        self.add_property('affected_buildings', self.affected_buildings)

        # List of affected victims affected by manual earthquake
        self.affected_victims = config['manual_earthquake']['affected_victims']
        self.add_property('affected_victims', self.affected_victims)

        self.explorer_victim_pickup_ticks = config['explorer_victim_pickup_ticks']
        self.add_property('explorer_victim_pickup_ticks', self.explorer_victim_pickup_ticks)

        self.explorer_victim_carry_move_ticks = config['explorer_victim_carry_move_ticks']
        self.add_property('explorer_victim_carry_move_ticks', self.explorer_victim_carry_move_ticks)

        self.task_completed_message = config['task_completed_message']
        self.add_property('task_completed_message', self.task_completed_message)

        # shortcut so we can access it as such: state['setting']
        self.obj_id = self.obj_name

        self.add_property('trial_completed', False)


def create_builder(scenario, mirror_scenario=False, seed=None, tick_duration=0.1, use_mental_model=True, run_vis=True, params=None,
                   agent_properties_json="usar/scenarios/usar_agent_properties.json", logger_subfolder=None, next_experiment_url=None, 
                   ppn=-1, conditie=None, questionnaire_link=None):
    #current_exp_folder = f"exp_testsubjectid={test_subject_id}_scenario="
    config = json.load(open(f"my_experiment/cases/exp_scenario-{scenario}.json"))
    #current_exp_folder += f"scenario{scenario}_"
    #current_exp_folder += f"mm={use_mental_model}_explanation={params['explanation']}_mirrored={mirror_scenario}_"

    # Create world
    world = config['world']

    world_builder = create_scenario(shape=world['shape'], seed=seed, tick_duration=tick_duration, scenario=scenario,
                                    mirror_scenario=mirror_scenario,
                                    use_mental_model=use_mental_model, run_vis=run_vis, params=params,
                                    agent_properties_json=agent_properties_json, next_experiment_url=next_experiment_url)

    # add agent
    agentProperties = config['explorer']
    brain = CustomAgent2(waypoints=[(10,2),(15,10)], max_carry_objects=1)
    world_builder.add_agent(agentProperties['location'], agent_brain=brain,
                            name=agentProperties['name'], is_human_agent=False, use_mental_model=use_mental_model,
                            params=params, visualize_colour=agentProperties['color'],
                            img_name=agentProperties['image'], memory={}, unexplored_doors=[],
                            visualize_when_busy=True, is_traversable=True,
                            log_robot_mud=0, log_carry_together=0, log_carry_door=0, robot_learned_carry=False,
                            robot_learned_mud=False)

    # add hq agent
    agentProperties = config['hq_agent']
    hq_brain = HQAgent()
    world_builder.add_agent(agentProperties['location'], agent_brain=hq_brain,
                            name=agentProperties['name'], is_human_agent=False, use_mental_model=use_mental_model,
                            params=params, visualize_size=2,
                            img_name=agentProperties['image'], memory={},
                            visualize_when_busy=False, is_traversable=True)

    # Add rescue_worker
    rescue_worker = config['rescue_worker']
    action_map = {
        'ArrowUp': MoveNorthHuman.__name__,
        'ArrowRight': MoveEastHuman.__name__,
        'ArrowDown': MoveSouthHuman.__name__,
        'ArrowLeft': MoveWestHuman.__name__,
        'g': GrabObject.__name__,
        'p': DropObject.__name__,
        'o': OpenDoorAction.__name__,
        'c': CloseDoorAction.__name__,
    }
    brain = CustomHumanAgent(max_carry_objects=1)
    world_builder.add_human_agent(rescue_worker['location'], agent=brain,
                                  name=rescue_worker['name'], key_action_map=action_map,
                                  img_name=rescue_worker['image'], memory={}, visualize_when_busy=True,
                                  is_traversable=True,
                                  test_subject_id=999999, scenario=scenario,
                                  log_human_mud=0, log_hit_by_earthquake=0, idle_time=0, log_carry_cue=0, log_mud_cue=0)

    world_builder.add_object([0, 0], "vesuvius", erupting_and_quaking=False, is_traversable=True, is_movable=False,
                             visualize_opacity=0)
                             
    # REF-T03
    # setting parameter
    world_builder.add_object([0,0], name="setting", is_traversable=True, visualize_opacity=False, scenario=scenario, ppn=ppn, exp_condition=conditie,
                            n_victims=len(config["victims"]), vics_saved_list=[], vics_not_saved_and_dead_list=[], vics_alive_list=[], goal_reached=False,
                            questionnaire_link=questionnaire_link)


    ####################################################
    # Logging 
    ####################################################
    # make folder if does not exist for logging 
    current_exp_folder = f"scenario={scenario}_mirrored={mirror_scenario}_seed={seed}" + "_" + datetime.now().strftime("time=%Hh-%Mm-%Ss_date=%dd-%mm-%Yy")
    logger_save_folder = os.path.join("experiment_logs", current_exp_folder)
    if not os.path.exists(logger_save_folder):
        os.makedirs(logger_save_folder)

    # add our logger 
    world_builder.add_logger(logger_class=CoLearningLogger, save_path=logger_save_folder, file_name_prefix="logger")

    return world_builder


def create_scenario(shape, seed, tick_duration, scenario, mirror_scenario, use_mental_model=True, run_vis=True, params=None,
                    agent_properties_json=None, next_experiment_url=None):

    config = json.load(open(f"my_experiment/cases/exp_scenario-{scenario}.json"))
    score_vis_placement = "right"
     # # Create world
    # world = config['world']
    factory = WorldBuilder(random_seed=seed,
                           # LimitedTimeGoal(50),
                           shape=shape, tick_duration=tick_duration,
                           run_matrx_api=run_vis, run_matrx_visualizer=False, verbose=False,
                           visualization_bg_img=f"world_{scenario}.png", simulation_goal=VictimsSavedGoal())
    victims = 0

    # Add rooms
    # status randomized and not many collapsed in this scenario?
    rooms = config['rooms']
    i = 1   # Room ID
    for room in rooms:
        collapsed = not room['doorOpen']
        name = 'b' + str(i)
        _, victims = init_room(factory, room_name=name,
                                   top_left=room['top_left'], dimensions=room['dimensions'],
                                   door=room['door'], collapsed=collapsed, door_open = room['doorOpen'], scenario=scenario, total_victims_in_world=victims, visited=False)
        i += 1

    # add victims based on defined scenario
    victims = generate_victims_from_file(factory, config)

    # Add command post
    cp = config['command_post']
    init_command_post(factory, mirrored=False, room_name='command_post',
                      top_left=cp['top_left'], dimensions=cp['dimensions'], door=cp['door'])
    modder_loc = []
    if scenario==3:
        modder_loc = [
            (7,2), (7,3), (7,4), (7,8), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8), (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7), (9,8), (9,9), (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7), (10,8), (11,3), (11,4), (11,5), (11,6), (11,7), (11,8), (12,6), (12,7), (12,8),
            (9,19), (10,17), (10,18), (10,19), (10,20), (11,17), (11,18), (11,19), (11,20), (11,21), (12,16), (12,17), (12,18), (12,19), (12,20), (13,17), (13,18), (13,19),
            (17,12), (18,10), (18,11), (18,12), (18,13), (18,14), (19,10), (19,11), (19,12), (19,13), (19,14), (19,15), (20,8), (20,9), (20,10), (20,11), (20,12), (20,13), (20,14), (21,9), (21,10), (21,11), (21,12), (21,13), (21,14), (22,10), (22,11), (22,12), (22,13), (22,14), (23,11),
            (27, 2), (27, 3), (27, 4), (27, 9), (28, 1), (28, 2), (28, 3), (28, 4), (28, 5), (28, 6), (28, 7), (28, 8),
            (28, 9), (28, 10), (28, 11), (28, 12), (29, 1), (29, 2), (29, 3), (29, 4),
            (29, 5), (29, 6), (29, 7), (29, 8), (29, 9), (29, 10), (29, 11), (29, 12), (30, 1), (30, 2), (30, 3),
            (30, 4), (30, 5), (30, 6), (30, 7), (30, 8), (30, 9), (30, 10), (30, 11), (30, 12), (31, 2), (31, 3),
            (31, 4), (31, 5), (31, 6), (31, 7), (31, 8), (31, 9), (31, 10), (31, 11), (31, 12), (32, 5), (32, 6),
            (32, 7), (32, 8), (32, 9), (32, 10), (32, 11), (32, 12)
        ]
    if scenario==4:
        modder_loc = [
            (12,4), (12,5), (13, 4), (13, 5), (13, 6), (13, 7), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (17, 1), (17, 2), (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (18, 0), (18, 1), (18, 2), (18, 3), (18, 4), (18, 5), (18, 6), (18, 7), (18, 8), (19, 0), (19, 1), (19, 2), (19, 3), (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (20, 1), (20, 2), (20, 3), (20, 4), (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (21, 5), (21, 6), (21, 7), (21, 8), (21, 9), (22, 6), (22, 7), (22, 8), (22, 9),
            (4,12), (4,13), (5,12), (5,13), (6,12), (6,13), (6,14), (7,12), (7,13), (7,14), (8,12), (8,13), (8,14), (8,15), (9,13), (9,14), (9,15), (9,16), (10,13), (10,14), (10,15), (10,16), (10,17), (10,18), (11,15), (11,16), (11,17), (11,18), (12,15), (12,16), (12,17), (12,18), (12,19), (13,15), (13,16), (13,17), (13,18), (13,19), (14,17),
            (22, 16), (23,13), (23,14), (23,15), (23,16), (24,14), (24,15), (24,16), (24,17), (25,14), (25,15), (25,16), (25,17), (26,14), (26,15), (26,16), (27,14),
            (29, 6), (30, 4), (30, 5), (30, 6), (31, 5), (31, 6), (31, 7), (31, 10), (31, 11), (32, 5), (32, 6), (32, 7), (32, 10), (32, 11), (32, 12), (33, 5), (33, 6), (33, 7), (33, 8), (33, 9), (33, 10), (33, 11), (33, 12), (34, 5), (34, 6), (34, 7), (34, 8), (34, 9), (34, 10), (34, 11), (34, 12), (35, 7), (35, 8), (35, 9), (35, 10), (36, 7), (36, 8), (36, 9), (36, 10), (37, 8)
        ]
    if modder_loc!=[]:
        for x,y in modder_loc:
            loc = (x,y)
            factory.add_object(location = loc, name = 'modder', visualize_opacity=0.0, is_traversable=True, clickable = False, traversability_penalty = 0.8)

    return factory


def init_room(world_builder, room_name, top_left, dimensions, door, collapsed, door_open, scenario, total_victims_in_world, visited):


    bottom_right_x = top_left[0] + dimensions[0] - 1
    bottom_right_y = top_left[1] + dimensions[1] - 1
    bottom_right = [bottom_right_x, bottom_right_y]

    #do we need an roomAgent?
    # world_builder.add_agent(top_left, agent_brain=RoomAgent(top_left, bottom_right),
    #                         name=room_name, collapsed=collapsed, victims=[], agents=[],
    #                         visualize_opacity=0)
    if door == 'left':
        door_loc_x = top_left[0]
        door_loc_y = top_left[1]+dimensions[1]- 3
        door_loc = [(door_loc_x, door_loc_y)]
        if door_open:
            img_door = "door_left.png"
        else:
            img_door = "door_broken_left.png"
            #add invisible door at drop location
            world_builder.add_object(location=(door_loc_x -1, door_loc_y -1), name='door', visualize_opacity=0.0, is_traversable=True, clickable=False, img_name="broken_door.png", dropped_door="yes")
        world_builder.add_room(top_left_location=top_left, width=dimensions[0], height=dimensions[1], name=room_name,
                              door_locations=door_loc,
                              doors_open=door_open, wall_visualize_opacity=0.0, door_custom_properties={"door_side": "left", "img_name": img_door, "door_opened": False, "collapsed":collapsed}, door_customizable_properties=["img_name"])
    elif door =='right':
        door_loc_x = top_left[0] + dimensions[0] - 1
        door_loc_y = top_left[1] + 2
        door_loc = [(door_loc_x, door_loc_y)]
        if door_open:
            img_door = "door_right.png"
        else:
            img_door = "door_broken_right.png"
            # add invisible door at drop location
            world_builder.add_object(location=(door_loc_x - 1, door_loc_y - 1), name='door',
                                     visualize_opacity=0.0, is_traversable=True, clickable=False, img_name="broken_door.png", dropped_door="yes")
        world_builder.add_room(top_left_location=top_left, width=dimensions[0], height=dimensions[1], name=room_name,
                               door_locations=door_loc,
                               doors_open=door_open, wall_visualize_opacity=0.0, door_custom_properties={"door_side": "right", "img_name": img_door, "door_opened": False, "collapsed":collapsed}, door_customizable_properties=["img_name"])
    elif door =='bottom':
        door_loc_x = top_left[0] + dimensions[0] - 3
        door_loc_y = top_left[1] + dimensions[1] - 1
        door_loc = [(door_loc_x, door_loc_y)]
        if door_open:
            img_door = "door_bottom.png"
        else:
            img_door = "door_broken_bottom.png"
            # add invisible door at drop location
            world_builder.add_object(location=(door_loc_x - 1, door_loc_y + 1), name='door',
                                     visualize_opacity=0.0, is_traversable=True, clickable=False, img_name="broken_door.png", dropped_door="yes")
        world_builder.add_room(top_left_location=top_left, width=dimensions[0], height=dimensions[1], name=room_name,
                               door_locations=door_loc,
                               doors_open=door_open, wall_visualize_opacity=0.0, door_custom_properties={"door_side": "bottom", "img_name": img_door, "door_opened": False, "collapsed":collapsed}, door_customizable_properties=["img_name"])
    elif door =='top':
        door_loc_x = top_left[0] + 2
        door_loc_y = top_left[1]
        door_loc = [(door_loc_x, door_loc_y)]
        if door_open:
            img_door = "door_top.png"
        else:
            img_door = "door_broken_top.png"
            # add invisible door at drop location
            world_builder.add_object(location=(door_loc_x - 1, door_loc_y - 1), name='door',
                                     visualize_opacity=0.0, is_traversable=True, clickable=False, img_name="broken_door.png", dropped_door='yes')
        world_builder.add_room(top_left_location=top_left, width=dimensions[0], height=dimensions[1], name=room_name,
                               door_locations=door_loc,
                               doors_open=door_open, wall_visualize_opacity=0.0, door_custom_properties={"door_side": "top", "img_name": img_door, "door_opened": False, "collapsed":collapsed}, door_customizable_properties=["img_name"])
    else:
        print("unknown door location")


    return world_builder, total_victims_in_world


def init_command_post(world_builder, mirrored, room_name, top_left, dimensions, door):
    bottom_right_x = top_left[0] + dimensions[0] - 1
    bottom_right_y = top_left[1] + dimensions[1] - 1
    bottom_right = [bottom_right_x, bottom_right_y]

    # Do we need a commandpost agent?
    # world_builder.add_agent(top_left, agent_brain=RoomAgent(top_left, bottom_right), name=room_name, victims=[],
    #                         agents=[], visualize_opacity=0)
    if door == 'left':
        door_loc_x = top_left[0]
        door_loc_y = top_left[1]+dimensions[1]- 3
        door_loc = [(door_loc_x, door_loc_y)]
        world_builder.add_room(top_left_location=top_left, width=dimensions[0], height=dimensions[1], name=room_name,
                              door_locations=door_loc,
                              doors_open=True, wall_visualize_opacity=0.0, door_custom_properties = {"door_side":"left", "img_name": "door_left.png", "door_opened": False, "collapsed":False})
    elif door =='right':
        door_loc_x = top_left[0] + dimensions[0] - 1
        door_loc_y = top_left[1] + 2
        door_loc = [(door_loc_x, door_loc_y)]
        world_builder.add_room(top_left_location=top_left, width=dimensions[0], height=dimensions[1], name=room_name,
                               door_locations=door_loc,
                               doors_open=True, wall_visualize_opacity=0.0, door_custom_properties = {"door_side":"right", "img_name": "door_right.png", "door_opened": False, "collapsed":False})
    elif door =='bottom':
        door_loc_x = top_left[0] + dimensions[0] - 3
        door_loc_y = top_left[1] + dimensions[1] - 1
        door_loc = [(door_loc_x, door_loc_y)]
        world_builder.add_room(top_left_location=top_left, width=dimensions[0], height=dimensions[1], name=room_name,
                               door_locations=door_loc,
                               doors_open=True, wall_visualize_opacity=0.0, door_custom_properties = {"door_side":"bottom", "img_name": "door_bottom.png", "door_opened": False, "collapsed":False})
    elif door =='top':
        door_loc_x = top_left[0] + 2
        door_loc_y = top_left[1]
        door_loc = [(door_loc_x, door_loc_y)]
        world_builder.add_room(top_left_location=top_left, width=dimensions[0], height=dimensions[1], name=room_name,
                               door_locations=door_loc,
                               doors_open=True, wall_visualize_opacity=0.0, door_custom_properties = {"door_side":"top", "img_name": "door_top.png", "door_opened": False, "collapsed":False})
    else:
        print("unknown door location")

    return world_builder