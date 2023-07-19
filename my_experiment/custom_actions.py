import numpy as np
from matrx.actions.action import Action, ActionResult
from matrx.actions.object_actions import _is_drop_poss, _act_drop, _possible_drop, _find_drop_loc, GrabObject, GrabObjectResult, DropObject
from matrx.actions import Action, ActionResult
from matrx.actions.move_actions import Move
from matrx.actions.move_actions import _is_possible_movement, MoveActionResult, _act_move
from matrx.objects import AgentBody
from random import uniform

class ManageHQImg(Action):
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, **kwargs):
        # Is a check necessary??
        return ManageHQImgResult(ManageHQImgResult.RESULT_SUCCESS, True)

    def mutate(self, grid_world, agent_id, **kwargs):
        reg_ag = grid_world.registered_agents[agent_id]

        if kwargs['animation']:
            reg_ag.change_property('img_name', 'HQ_icon_red.gif')
        else:
            reg_ag.change_property('img_name', 'HQ_icon_base.png')

        return ManageHQImgResult(ManageHQImgResult.RESULT_SUCCESS, True)

class ManageHQImgResult(ActionResult):
    RESULT_SUCCESS = 'Successfully visually notified the team'

    RESULT_FAILED = 'Failed to visually notify the team'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)

class ManageRobotImg(Action):
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, **kwargs):
        return ManageRobotImgResult(ManageHQImgResult.RESULT_SUCCESS, True)

    def mutate(self, grid_world, agent_id, **kwargs):
        reg_ag = grid_world.registered_agents[agent_id]

        if kwargs['cog_visible']:
            reg_ag.change_property('visualize_when_busy', True)
        else:
            reg_ag.change_property('visualize_when_busy', False)

        return ManageRobotImgResult(ManageHQImgResult.RESULT_SUCCESS, True)

class ManageRobotImgResult(ActionResult):
    RESULT_SUCCESS = 'Successfully changed visiblity of cog-icon of robot'
    RESULT_FAILED = 'Failed to change visibility of cog-icon of robot.'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)

class PickUpVictim(GrabObject):
    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)


class DropVictim(DropObject):
    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)
#----------------voor zover ik kon vinden wordt de onderstaande actie nergens meer gebruikt. Ik heb deze dus helemaal uitgecommend.
# class FindVictim(Action):
#
#     def __init__(self):
#         super().__init__()
#
#     def mutate(self, grid_world, agent_id, world_state, name, opacity, **kwargs):
#
#         victim = world_state[{"name": name}]
#         victim_id = victim["obj_id"]
#         objs = grid_world.environment_objects
#         objs[victim_id].change_property("visualize_opacity", opacity)
#
#         return ActionResult("Hoera! Birgit's eerste actie werkt!", True)
#
#
#     def is_possible(self, grid_world, agent_id, world_state, **kwargs):
#         return ActionResult("Hoera! Birgit's eerste actie werkt!", True)

class OpenDoor(Action):
    #veranderd de afbeelding van een deur en de property door_opened.
    #gebruik: om de deur een kwartslag te draaien naar "open"
    def __init__(self):
        super().__init__()

    def mutate(self, grid_world, agent_id, world_state, obj_id, img_name, opacity, **kwargs):
        objs = grid_world.environment_objects
        door = objs[obj_id]
        door.change_property("img_name", img_name)
        door.change_property("door_opened", True)
        return ActionResult("Hoera! Birgit's eerste actie werkt!", True)

    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        return ActionResult("Hoera! Birgit's eerste actie werkt!", True)

class OpenCollapsedDoor(Action):
    #changes door image to new image and makes the door traversable.
    def __init__(self):
        super().__init__()

    def mutate(self, grid_world, agent_id, world_state, obj_id, img_name, opacity, **kwargs):
        objs = grid_world.environment_objects
        door = objs[obj_id]
        door.change_property("img_name", img_name)
        door.change_property("is_traversable", True)
        return ActionResult("Hoera! Birgit's eerste actie werkt!", True)


    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        return ActionResult("Hoera! Birgit's eerste actie werkt!", True)

class MagicalDoorAppear(Action):
    #this action makes the door become visible at the dropped_door_location
    def __init__(self):
        super().__init__()

    def mutate(self, grid_world, agent_id, world_state, **kwargs):
        objs = grid_world.environment_objects

        dropped_door = world_state[{"dropped_door": "yes", "location": kwargs["location_door"]}]
        if dropped_door:
            grid_world.environment_objects[dropped_door["obj_id"]].change_property("visualize_opacity", kwargs["opacity"])

        door = objs[kwargs["obj_id"]]
        door.change_property("door_opened", True)
        door.change_property("is_open", True)
        door.change_property("is_traversable", True)
        return ActionResult("Hoera! Birgit's eerste actie werkt!", True)


    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        return ActionResult("Hoera! Birgit's eerste actie werkt!", True)
#----------------voor zover ik kon vinden wordt de onderstaande actie nergens meer gebruikt. Ik heb deze dus helemaal uitgecommend.
# class DragVictim(Action):
#
#     def __init__(self):
#         super().__init__()
#
#     def mutate(self, grid_world, agent_id, world_state, img_name, opacity, **kwargs):
#         objs = grid_world.environment_objects
#         door = objs[obj_id]
#         door.change_property("img_name", img_name)
#         door.change_property("door_opened", True)
#         return ActionResult("Hoera! Birgit's eerste actie werkt!", True)
#
#
#     def is_possible(self, grid_world, agent_id, world_state, **kwargs):
#         return ActionResult("Hoera! Birgit's eerste actie werkt!", True)
#

# Code for showing an image of an earthquake event
class EarthquakeEvent(Action): #TODO update with Tjalling's earthquake code

    def __init__(self):
        super().__init__()

    def is_possible(self, grid_world, agent_id, **kwargs):
        # Is a check necessary??
        return EarthquakeEventResult(EarthquakeEventResult.RESULT_SUCCESS, True)

    def mutate(self, grid_world, agent_id, world_state, **kwargs):

        print("SHAAAAAAKEEEE")
        vesuvius = grid_world.environment_objects["vesuvius"]
        vesuvius.change_property('erupting_and_quaking', kwargs['earthquake_happening'])
        print("Changed earthquake happening to:", kwargs['earthquake_happening'])

        # # log of mens in deuropening staat
        # hu_ag = grid_world.registered_agents['rescue_worker']
        # doors = world_state[{"location": hu_ag.properties["location"], "is_open": True}]
        # if doors is not None:
        #     shelter_count = grid_world.registered_agents['rescue_worker'].properties['log_human_shelter']
        #     grid_world.registered_agents['rescue_worker'].change_property('log_human_shelter', shelter_count+1)

        return EarthquakeEventResult(EarthquakeEventResult.RESULT_SUCCESS, True)


class EarthquakeEventResult(ActionResult):
    RESULT_SUCCESS = 'Successfully showed the earthquake event'

    RESULT_FAILED = 'Failed to show the earthquake event'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)


class MoveHuman(Action):
    """ Move actions for the human, which is not allowed to walk into a collapsed building"""
    def __init__(self, duration_in_ticks=0):
        super().__init__(duration_in_ticks)
        self.dx = 0
        self.dy = 0

    def is_possible(self, grid_world, agent_id, world_state, **kwargs):
        # check if a normal move would succeed 
        result = _is_possible_movement(grid_world, agent_id, self.dx, self.dy)
        if not result.succeeded:
            return result 

        # also make sure we are not walking into a collapsed room 
        agent_loc = world_state[agent_id]['location']
        new_loc = (agent_loc[0] + self.dx, agent_loc[1] + self.dy)

        door = world_state[{"location": new_loc, "collapsed": True}]

        if door is not None:
            return MoveActionResult(MoveActionResult.RESULT_NOT_PASSABLE_OBJECT, succeeded=False)

        return MoveActionResult(MoveActionResult.RESULT_SUCCESS, succeeded=True)

    def mutate(self, grid_world, agent_id, **kwargs):
        return _act_move(grid_world, agent_id=agent_id, dx=self.dx, dy=self.dy)

class MoveNorthHuman(MoveHuman):
    def __init__(self):
        super().__init__()
        self.dx = 0
        self.dy = -1

class MoveEastHuman(MoveHuman):
    def __init__(self):
        super().__init__()
        self.dx = +1
        self.dy = 0

class MoveSouthHuman(MoveHuman):
    def __init__(self):
        super().__init__()
        self.dx = 0
        self.dy = +1

class MoveWestHuman(MoveHuman):
    def __init__(self):
        super().__init__()
        self.dx = -1
        self.dy = 0
