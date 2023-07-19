from matrx.logger.logger import GridWorldLoggerV2
import pickle


class CoLearningLogger(GridWorldLoggerV2):
    """ Log the number of ticks the Gridworld was running on completion """

    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimeter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension,
                         delimiter=delimeter, log_strategy=self.LOG_ON_LAST_TICK)

    def log(self, world_state, agent_data, grid_world):
        human = grid_world.registered_agents['rescue_worker']
        robot = grid_world.registered_agents['explorer']
        setting = grid_world.environment_objects['setting']

        #is het niet risky om alle room objecten te pakken voor vics? # bug sensitive code
        victims_in_cp = world_state.get_room_objects("command_post")
        vic_score = {"lightly_wounded": 0, "severely_wounded": 0, "death": 0}
        for obj in victims_in_cp:
            # find victims 
            if "alive" in obj.keys(): # way to check whether the object is a victim
                if obj['treatment_need'] == 1:
                    vic_score['lightly_wounded'] += 1
                elif obj['treatment_need'] == 2:
                    vic_score['severely_wounded'] += 1
                else:
                    vic_score['death'] += 1

        # here, the logging data are read from the relevant parameters, and saved as a dictionary
        # this dictionary contains ALL the data that are logged (all is determined at the last tick;
        # nothing is logged on each tick)
        log_statement = {
            "ppn_nummer": setting.properties['ppn'], # taken from setting parameter (created in scenario_builder)
            "conditie": setting.properties['exp_condition'], # taken from setting
            "scenario": setting.properties["scenario"], # taken from setting
            "task_dur": grid_world.current_nr_ticks, # taken from grid_world
            "tot_vic_lightly_wounded": vic_score["lightly_wounded"],  # vic score is calculated above
            "tot_vic_sever_wounded": vic_score["severely_wounded"], # vic score is calculated above
            "tot_vic_death": vic_score["death"], # vic score is calculated above
            "tot_idle_time": human.properties['idle_time'], # idle time is determined for each tick in the human agent (if action == None)
            "nr_carry_cues": human.properties['log_carry_cue'], # taken from human_agent (when human sends a carry cue to the robot)
            "nr_mud_moves_human": human.properties['log_human_mud'], # calculated in human_agent (when human walks over mud tile)
            "nr_mud_moves_robot": robot.properties['log_robot_mud'], # calculated in robot agent (when robot walks over mud)
            #moet hieronder niet ook een log times_earthquake? omdat niet altijd de aardbeving getriggerd wordt? #Tjeerdvragen
            "times_hit_by_earthquake": human.properties['log_hit_by_earthquake'], # calculated in human agent (when hit by earthquake)
            "times_carry_vic_together": robot.properties['log_carry_together'], # calculated in robot agent (when carrying together)
            "times_carry_door_together": robot.properties['log_carry_door'], # calculated in robot agent (when carrying a door together)
            "robot_learned_carry": robot.properties['robot_learned_carry'], # calculated in robot agent (when robot and human drop a victim in the CP for the first time)
            "robot_learned_mud": robot.properties['robot_learned_mud'] # calculated in robot agent (when human avoids mud for some number of ticks)
        }

        # REF-T09
        # Pickle dump to make sure information is retained.
        # Hier wordt de pickle file "learned_backup" gevuld met een lijstje met 2 Booleans;
        # de eerste geeft aan of de robot heeft geleerd om samen te tillen, de tweede geeft aan of hij heeft geleerd om modder te vermijden.
        # Als de pickle file nog niet bestaat (wordt verwijderd bij het starten van de tutorial, zie REF-T08 in main), dan wordt hij hier ook automatisch aangemaakt.
        with open('learned_backup.pkl', 'wb') as f:
            print('THIS WILL BE LOGGED: ')
            print(robot.properties['robot_learned_carry'])
            print(robot.properties['robot_learned_mud'])
            pickle.dump([robot.properties['robot_learned_carry'], robot.properties['robot_learned_mud']], f, pickle.HIGHEST_PROTOCOL)

        return log_statement