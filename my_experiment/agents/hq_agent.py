from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.agents import AgentBrain
from matrx.messages import Message
from my_experiment.custom_actions import ManageHQImg
from my_experiment.objects import Victim


class HQAgent(AgentBrain):
    """ The code for the HQ agent that sends messages to the team
    """

    def __init__(self, **kwargs):
        """ Creates an agent brain to move along a set of waypoints.
        """
        super().__init__(**kwargs)
        self.animation_timer = 0
        self.animation = False
        self.modder_geleerd = False
        self.ticks_to_msg_about_carrying = 100
        self.count_ticks = 0
        self.counting = False
        self.reflect_carry = True
        self.earthquake = False
        self.broken_door_hq = False

    def initialize(self):
        self.state_tracker = StateTracker(agent_id=self.agent_id)
        self.animation_timer = 30


    def filter_observations(self, state): #Needs to be changed based on timing of the messages
        """ Filters the world state before deciding on an action. """
        # update state tracker for navigator
        self.state_tracker.update(state)

        # read messages and remove them
        for message in list(self.received_messages):
            print("Received message:", message.content, " form ", message.from_id)
            if message.from_id == 'rescue_worker' and 'modder geleerd' in message.content:
                self.modder_geleerd = True
            # if human requested robot to help carry, HQ should never do Reflection about this again
            if message.content == "do_not_reflect_carry":
                self.reflect_carry = False
            # print("Received message:", message.content, " from ", message.from_id)
            if message.from_id == 'explorer':
                if message.content == 'earthquake_trigger':
                    print("HQ NOTICE")
                    self.earthquake = True
                if message.content == 'broken_door_hq':
                    self.broken_door_hq = True
            self.received_messages.remove(message)

        return state

    def decide_on_action(self, state):
        """ Contains the decision logic of the agent. """
        action = None
        action_kwargs = {}

        open_door = False
        carry_victim = False
        mud_path = False

        # samen tillen
        human_id = self.state.get_agents_with_property({"obj_id": "rescue_worker"})[0]["obj_id"]
        robot_id = self.state.get_agents_with_property({"obj_id": "explorer"})[0]["obj_id"]
        top_left_human = (self.state[human_id]['location'][0] - 1, self.state[human_id]['location'][1] - 1)
        objects_around_human = state.get_objects_in_area(top_left_human, 2, 2)
        self.counting = False
        for obj in objects_around_human:
            # If a heavily wounded victim is near the human, start counting
            if Victim.__name__ in obj['class_inheritance'] and obj['treatment_need'] == 2 and not obj['saved']:
                victim = obj
                self.counting = True
                break
                
        # HQ sends reflection to human
        # if we reach the wait threshold
        if self.counting and self.count_ticks >= self.ticks_to_msg_about_carrying:
            self.count_ticks = 0
            self.counting = False
            if self.reflect_carry and self.state['setting']['exp_condition'] == 'exp':
                carry_victim = True
        elif self.counting:
            self.count_ticks += 1

        # broken door reflection
        if self.broken_door_hq:
            self.send_message(
                Message(content={
                    'chat_text': "De robot blijft wel erg lang voor deze deur staan. Hoe denk je dat dit komt?"},
                        from_id=self.agent_id,
                        to_id=None))
            self.animation = True
            self.broken_door_hq = False

        # Some code that sets the condition for sending the message and activating the notification
        if open_door: # Deur open maken
            self.send_message(Message(content={'chat_text': "We keken even mee met de missie, en vroegen ons iets af."
                                              " Hoe kan het dat de robot niet deze ruimte in gaat?"},
                                      from_id=self.agent_id, to_id=None))
            self.animation = True
        elif carry_victim: # Gewond slachtoffer dragen
            self.send_message(Message(content={'chat_text': "We zien dat de robot je niet komt helpen. Waarom zou dat zijn?"},
                                      from_id=self.agent_id, to_id=None))
            self.animation = True
        elif mud_path: # Modderpad
            self.send_message(Message(content={'chat_text': "We zien dat de missie wat traag verloopt. "
                                              "Hoe kan het dat de robot langzamer loopt?"},
                                      from_id=self.agent_id, to_id=None))
            self.animation = True
        elif self.earthquake: # Seismograaf
            self.send_message(
                Message(
                    content={
                        'chat_text': "We zien dat de robot stilstaat in die deuropening. Waarom denk je dat dat is?"}
                    , from_id=self.agent_id, to_id=None))
            self.animation = True
            self.earthquake = False
        if state['World']['nr_ticks'] == 100:
            if self.modder_geleerd == False and self.state['setting']['exp_condition'] == 'exp' and \
                                self.state['setting']['scenario'] == 3:
                self.send_message(
                    Message(
                        content={'chat_text': "We zien dat de missie wat traag verloopt. Hoe kan het dat de robot soms langzamer loopt?"}
                        , from_id=self.agent_id, to_id=None))
                self.animation = True
        # Code that changes the image
        if self.animation:
            if self.animation_timer > 0:
                self.animation_timer = self.animation_timer - 1
            else:
                self.animation = False
                self.animation_timer = 30




        # Setting the action to the animation action (make sure this is only done upon a change in variable)
        action = ManageHQImg.__name__
        action_kwargs['animation'] = self.animation

        return action, action_kwargs

    def create_context_menu_for_other(self, agent_id_who_clicked, clicked_object_id, click_location):
        """ Generate options for a context menu for a specific object/location that a user NOT controlling this
        human agent opened.
        """
        print("Context menu other")
        context_menu = []

        return context_menu

    def _set_messages(self, messages=None):
        # make sure we save the entire message and not only the content
        for mssg in messages:
            received_message = mssg
            self.received_messages.append(received_message)

