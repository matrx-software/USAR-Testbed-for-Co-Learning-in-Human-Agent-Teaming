from matrx.goals import WorldGoal

class VictimsSavedGoal(WorldGoal):
    """
    A world goal that tracks whether all victims have been either saved or died
    """

    def __init__(self):
        """ Initialize the goal
        """
        super().__init__()
        self.vic_ids = [] 
        self.initialized = False 

        self.countdown_to_endscreen= 5
        self.countdown_to_stop = 10
        self.counting_down = False

        self.learn_carry_door_timer = 1200 # 2 minutes, assuming a tick_duration of 0.1

    def goal_reached(self, grid_world):
        """ Checked every tick to see if the goal has been reached"""

        # continue for a few ticks after we achieve our goal 
        if self.counting_down:
            if self.countdown_to_endscreen > 0: 
                self.countdown_to_endscreen -= 1
            else:
                settings = grid_world.environment_objects['setting']
                settings.change_property("goal_reached", True)
                # settings.properties['goal_reached'] = True
                print("Showing endscreen")

                if self.countdown_to_stop > 0:
                    self.countdown_to_stop -= 1
                else:
                    self.is_done = True
                    print("Shutting down world")
            return self.is_done

        self.is_done = False

        if not self.initialized:
            self.initialized = True

        settings = grid_world.environment_objects['setting']
        n_victims_not_saved_and_dead = len(settings.properties['vics_not_saved_and_dead_list'])
        n_victims = settings.properties['n_victims']
        n_saved_victims = len(settings.properties['vics_saved_list'])
        n_alive_victims = len(settings.properties['vics_alive_list'])

        # END GOAL 1: check if all victims are saved, or all not_saved victims are dead
        if grid_world.current_nr_ticks > 1 and (n_saved_victims == n_alive_victims
                                                or (n_victims_not_saved_and_dead + n_saved_victims) == n_victims):
            self.counting_down = True 
            print("Goal 1 reached. Counting down..")

        # END GOAL 2: check if all closed doors that are left are collapsed
        # if so, start timer to enable human to learn "carry door together"
        if grid_world.current_nr_ticks > 1:
            unexplored_doors = grid_world.registered_agents['explorer'].custom_properties['unexplored_doors']
            log_carry_door = grid_world.registered_agents['explorer'].custom_properties['log_carry_door']
            collapsed_doors = []
            all_closed_and_collapsed = True
            for unexplored_door in unexplored_doors:
                if unexplored_door[0]['collapsed'] == True: # de deur is van een collapsed building
                    collapsed_doors.append(unexplored_door)

            if len(unexplored_doors) == len(collapsed_doors): # als er alleen nog collapsed doors over zijn
                for collapsed_door in collapsed_doors:
                    if collapsed_door[0]['door_opened'] == True: # check of de deur van de collapsed building open is (mens heeft dus geleerd)
                        all_closed_and_collapsed = False

                if log_carry_door == 0 and all_closed_and_collapsed: # all collapsed buildings have closed doors, so the human did not learn to carry a door together (yet)
                    self.learn_carry_door_timer -= 1 # decrease the timer
                    if self.learn_carry_door_timer < 1: # when the timer reaches 0, the simulation stops
                        self.counting_down = True
                        print("Goal 2 reached. Counting down..")
                        return self.is_done
                #print(log_carry_door)

    def get_progress(self, grid_world):
        """ Returns the progress of reaching the goal in the simulated grid world.
        This is optional, so we don't do anything with it
        """
        return 0
