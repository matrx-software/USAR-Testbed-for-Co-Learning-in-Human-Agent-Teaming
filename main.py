
from matrx import cases
import os, requests
import pickle
from my_experiment.cases import scenario_builder
from my_custom_visualizer import visualization_server
from time import sleep

def run_tdp(test_subject_id, tdp="tut", conditie=None):
    # By creating scripts that return a builder, we can define infinite number of use cases and select them (in the
    # future) through a UI.
    
    if tdp == "tut":
        txt = "Baseline"
        scenario = 1

    elif tdp == "t1":
        txt = "first training"
        scenario = 2
        
    elif tdp == "t2":
        txt = "second training"
        scenario = 3

    elif tdp == "test":
        txt = "test scenario"
        scenario = 4

    print("#" * 30)
    print(f"Running the {txt}")
    print("#" * 30)

    # questionnaire link for odd test subject IDs
    questionnaire_link = "http://tno.nl/"
    # questionnaire link for even test subject IDs
    if test_subject_id == 0 or test_subject_id % 2 == 0:
        questionnaire_link = "https://google.com"

    builder = scenario_builder.create_builder(scenario, seed=5, ppn=test_subject_id, conditie=conditie, questionnaire_link=questionnaire_link)

    # startup world-overarching MATRX scripts, such as the api and/or visualizer if requested
    media_folder = os.path.join(os.path.realpath("my_experiment"), "images")
    builder.startup()

    # start the custom visualizer
    print("Starting Custom visualizer")
    vis_thread = visualization_server.run_matrx_visualizer(verbose=False, media_folder=media_folder)

    # run each world
    for world in builder.worlds():
        world.run(builder.api_info)

    print("Keeping MATRX alive untill the questionnaire has been filled in..")
    sleep(1000000)

    # stop the custom visualizer
    print("Shutting down custom visualizer")
    r = requests.get("http://localhost:" + str(visualization_server.port) + "/shutdown_visualizer")
    vis_thread.join()

    # stop MATRX scripts such as the api and visualizer (if used)
    builder.stop()

if __name__ == "__main__":
    print("\n----Welcome to the MMT project experiment  ----")
    print("What is the ID of the test subject?")
    test_subject_id = int(input())

    # TODO: fix this
    conditie = "exp" if test_subject_id % 2 == 0 else "control"

    choice = 0
    while not choice == "exit":
        print("\n\nType one of the options shown between brackets: ")
        print("(tut): Do the tutorial")
        print("(t1): Run first training")
        # print("(t1L): Run first training with learning patterns")
        print("(t2): Run second training")
        #print("(t2L): Run second training with learning patterns")
        print("(test): Run Test scenario")
        print("(pickle): Set pickle file")
        print("(exit): Exit")
        print("What do you want to do?")
        choice = input()

        if choice == "exit":
            print("Quitting experiment")
            break
        # REF-T08
        # Bij het starten van de Tutorial wordt de learned_backup file verwijderd.
        # Bij het voltooien van een scenario (ook de tutorial) wordt deze file automatisch aangemaakt als die nog niet bestaat en gevuld met
        # info over wat de robot heeft geleerd (REF-T09 in co_learning_logger).
        if choice in ["tut", "t1", "t2", "test"]:
            print("Running experiment TDP ", choice)
            if choice == "tut":
                # remove the stored learning file if it exists
                if os.path.isfile('./learned_backup.pkl'):
                    os.remove('./learned_backup.pkl')
                    print("Cleared robot learning file")
            run_tdp(test_subject_id, tdp=choice, conditie=conditie)
        if choice == "pickle":
            if os.path.isfile('./learned_backup.pkl'):
                os.remove('./learned_backup.pkl')
                print("Cleared robot learning file")
            print("Carry (0 or 1)?")
            learn_carry = input()
            print("Mud (0 or 1)?")
            learn_mud = input()
            with open('learned_backup.pkl', 'wb') as f:
                pickle.dump([learn_carry, learn_mud], f, pickle.HIGHEST_PROTOCOL)
            print(f"Set learning file: carry = {learn_carry}, mud = {learn_mud}")
        else:
            print("Sorry, did not recognize that option")