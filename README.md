# USAR Testbed for Co-Learning in Human-Agent Teaming
This is code for an urban-search-and-rescue task that can be performed by different agents. Agents can be human-controlled or programmed. This task was designed to perform research on co-learning in human-agent teams. The code comes with a rule-based agent that can perform the task in collaboration with a human-controlled agent. The agent is programmed in such a way that there are several hard and soft dependencies between agents and possibilities for learning. There are four possible learning situations:
1.	Mud: the programmed agent is unaware of the fact that it is slowed down by mud. It can learn to avoid mud by observing a human avoiding the mud, or by the human telling the agent.
2.	Earthquake: the programmed agent can detect when an earthquake is coming and take shelter in a door opening. This is a situation in which a human can potentially learn.
3.	Carry heavily wounded victims: heavily wounded victims can only be carried by two agents together. The programmed agent is unaware of this, but can learn to proactively help a human agent by being told to do so by the human or by observing that the human idles near such a victim.
4.	Open broken door: doors of collapsed buildings need to be cleared by two agents together. The robot knows this and can ask for help. This is a situation in which a human can potentially learn.

This environment was used for an extensive experiment with human participants. The paper describing the details is currently in review, and will be linked here once published.
