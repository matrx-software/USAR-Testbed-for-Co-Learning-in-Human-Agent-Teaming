import itertools
from collections import OrderedDict
from itertools import product
import os

# MATRX stuff
from matrx import WorldBuilder, utils
import numpy as np
from matrx.actions import MoveNorth, OpenDoorAction, CloseDoorAction
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest
from matrx.grid_world import GridWorld, DropObject, GrabObject, AgentBody
from matrx.objects import EnvObject, SquareBlock
from matrx.world_builder import RandomProperty
from matrx.goals import WorldGoal, CollectionGoal

# custom code
from my_experiment.agents.custom_agent import CustomAgent
from my_experiment.agents.custom_human_agent import CustomHumanAgent


# Some general settings
tick_duration = 1 / 10  # 0.1s or lower tick duration recommended, to keep the human agent responsive
random_seed = 1
verbose = False
key_action_map = {  # For the human agents
    'ArrowUp': MoveNorth.__name__,
    'ArrowRight': MoveEast.__name__,
    'ArrowDown': MoveSouth.__name__,
    'ArrowLeft': MoveWest.__name__,
    # 'g': GrabObject.__name__,
    # 'p': DropObject.__name__,
    # 'o': OpenDoorAction.__name__,
    # 'c': CloseDoorAction.__name__,
}


def create_builder():

    # Set numpy's random generator
    np.random.seed(random_seed)

    # The world size
    world_size = (40, 23)

    # Create our world builder
    builder = WorldBuilder(shape=world_size, tick_duration=tick_duration, random_seed=random_seed, run_matrx_api=True,
                           run_matrx_visualizer=False, verbose=verbose,
                           visualization_bg_img="world_test.png")
    # usar/my_experiment/images/world_1.PNG
    # , visualization_bg_clr="#FFFFFF"

    #add a room
    builder.add_room(top_left_location=(11, 9), width=5, height=6, name="Room", door_locations=[(15, 11)], doors_open=False, wall_visualize_opacity=0.0)
    #, wall_visualize_opacity=0.0
    builder.add_room(top_left_location=(21, 0), width=6, height=5, name="CP", door_locations=[(24, 4)],
                     doors_open=True, wall_visualize_opacity=0.0)
    builder.add_room(top_left_location=(35, 1), width=5, height=6, name="Room", door_locations=[(35, 4)],
                     doors_open=False, wall_visualize_opacity=0.0)
    builder.add_room(top_left_location=(33, 15), width=6, height=5, name="Room", door_locations=[(35, 15)],
                     doors_open=False, wall_visualize_opacity=0.0)
    builder.add_room(top_left_location=(14, 18), width=6, height=5, name="Room", door_locations=[(16, 18)],
                     doors_open=False, wall_visualize_opacity=0.0)
    builder.add_room(top_left_location=(3, 18), width=6, height=5, name="Room", door_locations=[(5, 18)],
                     doors_open=False, wall_visualize_opacity=0.0)
    builder.add_room(top_left_location=(0, 0), width=6, height=5, name="Room", door_locations=[(3, 4)],
                     doors_open=False, wall_visualize_opacity=0.0)
    # create objects 
    builder.add_object([38,2], name="Victim", is_traversable=True, is_movable=True)
    builder.add_object([13, 12], name="Victim", is_traversable=True, is_movable=True)
    builder.add_object([34, 17], name="Victim", is_traversable=True, is_movable=True)
    builder.add_object([37, 18], name="Victim", is_traversable=True, is_movable=True)

    # Now we add our agents as part of the same team
    team_name = "Team Awesome"

    # Custom human agent
    brain = CustomHumanAgent(max_carry_objects=1)
    builder.add_human_agent((24, 2), brain, team=team_name, name="Human", key_action_map=key_action_map, is_traversable=True, img_name="person.svg")

    # Custom artificial agent
    brain = CustomAgent(waypoints=[(1, 1), (17,18)], max_carry_objects=1)
    builder.add_agent((22,2), brain, team=team_name, name=f"Agent Smith #1", is_traversable=True, img_name="r2d2.svg")

    # Return the builder
    return builder

