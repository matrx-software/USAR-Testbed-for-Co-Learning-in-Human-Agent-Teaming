from matrx.objects import EnvObject
from random import uniform

class Victim(EnvObject):
    def __init__(self, location, name='victim', alive=None, treatment_need=None, saved=False, need_increase_time=0, img_name = "victim_lightlywounded.png"):
        super().__init__(name=name, location=location, is_traversable=True, visualize_shape='img', is_movable=True,
                         class_callable=Victim)
        #img_name="victim_lightlywounded.png"

        # Victim is alive or not
        self.alive = alive
        self.add_property('alive', self.alive)

        # Victim is saved (in command post) or not
        self.saved = saved
        self.add_property('saved', self.saved)

        # Treatment need (1=lightly wounded, 2=heavily wounded, 3=dead)
        self.treatment_need = treatment_need
        self.add_property('treatment_need', self.treatment_need)
        self.tick = 0

        #self.img_name = img_name

        # Number of ticks until treatment_need increases
        self.need_increase_time = need_increase_time
        self.add_property('need_increase_time', int(uniform(0.85, 1.15)*need_increase_time))

        self.img_name = img_name
        self.add_property('img_name', "victim_dead.png")


    def update(self, grid_world, state):
        ticks = state['World']['nr_ticks']
        need_increase_time = self.custom_properties["need_increase_time"]
        objects_in_cp = state.get_room_objects("command_post")

        # change color at the first tick
        if ticks == 1:
            self.update_icon(self.treatment_need)

        # If I am in the command post, I am saved and cannot be harmed
        if not self.saved:
            if not self.custom_properties["alive"] and self.obj_name not in state['setting']['vics_not_saved_and_dead_list']:
                state['setting']['vics_not_saved_and_dead_list'].append(self.obj_name)
            for obj in objects_in_cp:
                if obj['obj_id'] == self.obj_id:
                    self.change_property('saved', True)
                    if self.obj_name not in state['setting']['vics_saved_list']:
                        state['setting']['vics_saved_list'].append(self.obj_name)

            # check if I (victim) am alive and how often I should worsen in health, and if that moment is now
            if self.custom_properties["alive"] and need_increase_time > 0 and ticks % need_increase_time == 0:
                # check if the victim is NOT in the CP or being carried.
                if not self.treatment_need == 0 and \
                            not self.carried_by and \
                            ticks != 0:
                    # ouch!
                    new_treatment_need = self.custom_properties["treatment_need"] + 1
                    self.change_property("treatment_need", new_treatment_need)
                    if self.custom_properties["treatment_need"] > 2:
                        self.change_property('alive', False)
                        if self.obj_name in state['setting']['vics_alive_list']:
                            state['setting']['vics_alive_list'].remove(self.obj_name)
                    self.update_icon(self.custom_properties["treatment_need"])

        if self.custom_properties["alive"] and self.obj_name not in state['setting']['vics_alive_list']:
            state['setting']['vics_alive_list'].append(self.obj_name)

    def update_icon(self, treatment_need):
            if treatment_need == 1:
                self.visualize_colour = '#70AD47'
                self.change_property("img_name", "victim_lightlywounded.png")
            elif treatment_need == 2:
                self.visualize_colour = '#FFC000'
                self.change_property("img_name", "victim_wounded.png")
            else:
                self.visualize_colour = '#3F0101'
                self.change_property("img_name", "victim_dead.png")