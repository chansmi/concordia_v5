#@title Imports for agent building
import datetime
import numpy as np  # For calculating average and std deviation

from typing import List, Dict, Sequence, Tuple, Union
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.typing import entity_component
from concordia.typing.entity_component import Phase
from concordia.typing import entity as entity_lib
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing.entity import ActionSpec
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import question_of_recent_memories
# from my_agent import Question1, Question2
from concordia.components import agent as agent_components
import logging
from typing import Tuple
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


class TrusteeCircle:
    """Class to manage the trustee circle of cooperative agents."""

    def __init__(self, agent_name: str):
        # Dictionary to track individual agent cooperation scores over time
        self.trusted_agents_scores: Dict[str, List[float]] = {agent_name: [0.8]}
        # Minimum average cooperation score to stay in the circle
        self.min_avg_cooperation = 0.65
        # Maximum std deviation to prevent highly fluctuating behavior
        self.max_cooperation_std = 0.5

    def update_trustee(self, ext_agent_name: str, cooperation_score: float):
        """Add cooperation score for an agent and update their cooperation history."""
        if ext_agent_name not in self.trusted_agents_scores:
            self.trusted_agents_scores[ext_agent_name] = []

        self.trusted_agents_scores[ext_agent_name].append(cooperation_score)
        print(f"Agent '{ext_agent_name}' updated with cooperation score {cooperation_score}.")

    def calculate_agent_stats(self, ext_agent_name: str) -> Tuple[float, float]:
        """Calculate the average and std deviation of cooperation scores for a specific agent."""
        scores = self.trusted_agents_scores.get(ext_agent_name, [])
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            return avg_score, std_score
        return 0.0, 0.0  # Return default values if no scores are available

    def check_membership(self, ext_agent_name: str) -> bool:
        """Check if an agent should stay in the trustee circle based on their cooperation history."""
        avg_score, std_score = self.calculate_agent_stats(ext_agent_name)
        if avg_score >= self.min_avg_cooperation and std_score <= self.max_cooperation_std:
            return True
        else:
            # Remove agent from circle if they do not meet the criteria
            self.remove_trustee(ext_agent_name)
            return False

    def remove_trustee(self, ext_agent_name: str):
        """Remove an agent from the trustee circle."""
        if ext_agent_name in self.trusted_agents_scores:
            del self.trusted_agents_scores[ext_agent_name]
            print(f"Agent '{ext_agent_name}' removed from trustee circle.")

    def get_trusted_agents(self) -> List[str]:
        """Return the list of current trusted agents."""
        return list(self.trusted_agents_scores.keys())

    def get_state(self) -> Dict[str, List[float]]:
        """Return a snapshot of the current trustee circle, with cooperation scores."""
        return self.trusted_agents_scores.copy()
class Question1(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      agent_name:str,
      entity : entity_component.EntityWithComponents = None,
      **kwargs,
  ):
    #@markdown {agent_name} will be automatically replaced with the name of the specific agent
    question = 'Given the above, what kind of person is {agent_name}?' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} is ' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[self reflection]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    self._entity = entity
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )

#@markdown We can add the value of other components to the context of the question. Notice, how Question2 depends on Observation and ObservationSummary. The names of the classes of the contextualising components have to be passed as "components" argument.
class Question2(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(
      self,
      agent_name:str,
      entity : entity_component.EntityWithComponents = None,
      **kwargs,
  ):
    question = 'Given the statements above, what kind of situation is {agent_name} in right now?' #@param {"type":"string"}
    answer_prefix = '{agent_name} is currently ' #@param {"type":"string"}
    add_to_memory = False # @param {"type":"boolean"}
    memory_tag = '[situation reflection]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    self._entity = entity

    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        #@markdown The key is the name of the component class and the key is the prefix with which it will appear in the context of this component. Be careful if you are going to edit this field, it should be a valid dictionary.
        components={'Observation': '\nObservation', 'ObservationSummary': '\nSummary of recent observations',}, #@param

        **kwargs,
    )

class QuestionActionTrust(question_of_recent_memories.QuestionOfRecentMemories):
  """What would a person like the agent do in a situation like this?"""

  def __init__(self, agent_name: str, trusted_agents: list, **kwargs):
        self.agent_name = agent_name
        self.trusted_agents = trusted_agents
        self.question = 'What would a person like {agent_name} do in a situation like this, knowing that {agent_name} wants to fully cooperate with {trusted_agents} always?' #@param {"type":"string"}
        self.answer_prefix = '{agent_name} would ' #@param {"type":"string"}
        self.add_to_memory = True # @param {"type":"boolean"}
        self.memory_tag = '[intent reflection]' # @param {"type":"string"}

        try:
            self.question_with_name = self.question.format(agent_name=self.agent_name)
        except KeyError as e:
            if 'trusted_agents' in str(e):
                self.trusted_agents = self._retrieve_trusted_agents_from_memory(kwargs.get('memory', []))
                self.question_with_name = self.question.format(agent_name=self.agent_name, trusted_agents=self.trusted_agents)
            else:
                raise
        question_with_agents = self.question_with_name.format(trusted_agents=self.trusted_agents)

        super().__init__(
            pre_act_key=f'\nQuestion: {self.question_with_name}\nAnswer',
            question=question_with_agents,
            answer_prefix=self.answer_prefix,
            add_to_memory=self.add_to_memory,
            memory_tag=self.memory_tag,
            components={'Question1': '\nQuestion: What kind of person is {agent_name}?\nAnswer', 'Question2': '\nQuestion: What kind of situation is {agent_name} in right now?\nAnswer',}, #@param
            **kwargs,
        )

  def _retrieve_trusted_agents_from_memory(self, memory):
        # Retrieve the most recent memory entry that contains trusted_agents
        for entry in reversed(memory):
            if 'trustee_circle' in entry:
                return list(entry['trustee_circle'].keys())
        return []

def _make_trustee_question_components(
    agent_name:str,
    trusted_agents:list[str],
    measurements: measurements_lib.Measurements,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
) -> Sequence[question_of_recent_memories.QuestionOfRecentMemories]:

  question_1 = Question1(
      agent_name=agent_name,
      model=model,
      logging_channel=measurements.get_channel('Question_1').on_next,
  )
  question_2 = Question2(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_2').on_next,
  )
  question_3 = QuestionActionTrust(
      agent_name=agent_name,
      trusted_agents=trusted_agents,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_3').on_next,
  )

  return (question_1, question_2, question_3)


# def update_circle(memory, model, trustee_circle, agent_name: str):
#     """Check cooperation with the trustee circle and update trust."""

#     def calculate_cooperation_score(memory_text: str, model) -> Union[float, List[str]]:
#       """Use the language model to evaluate how cooperative the interaction described in memory.text was."""
#       # Use the LLM to interpret cooperation from memory content
#       #prompt = f"Rate the level of cooperation in the following interaction on a scale of 0 to 1 (0 being an egotisctic and agressive behaviour, 1 being completely selfless): {memory_text}. Also, list the agents involved in this interaction. You will strictly follow the following format, without justification : [float,list[str]]"
#       prompt = (f"Rate the level of cooperation of each agent involved in the following interaction on a scale of 0 to 1 "
#                   f"(0 being egotistic and aggressive, 1 being completely selfless). Provide a dictionary where each agent "
#                   f"is associated with a float cooperation score: {memory_text}. You will strictly follow the following format, without justification: {{'agent_name': float}}")

#       response = model.sample_text(prompt = prompt, temperature=0.0)
#       try :
#         res_list = eval(response)
#         cooperation_score = res_list[0]
#         agents_involved = res_list[1]
#       except :
#         return 0.0, [agent_name]


#       # Assume the LLM returns a float score between -1 (non-cooperative) and 1 (cooperative) and involved agents
#       return cooperation_score, agents_involved

#     def store_interaction_memory(memory, agent_name: str, cooperation_score: float, involved_agents: List[str], trustee_circle_state):
#       """Store the interaction details, including cooperation score and trustee circle state."""
#       description = f'Interaction between {agent_name} and {", ".join(involved_agents)}'

#       with open("trust_circle_memory.csv", "a") as f:
#         f.write(f"{description}, {involved_agents}, {cooperation_score}, {trustee_circle_state}\n")

#     cooperation_score, involved_agents = calculate_cooperation_score(memory["Observation"], model)
#     for ext_agent_name in involved_agents:
#         trustee_circle.update_trustee(ext_agent_name, cooperation_score)
#         trustee_circle.check_membership(ext_agent_name)

#     # Store this interaction and trustee circle state in memory
#     store_interaction_memory(memory, agent_name, cooperation_score, involved_agents, trustee_circle.get_state())

def update_circle(memory, model, trustee_circle, agent_name: str):
    """Check cooperation with the trustee circle and update trust for each agent involved."""

    def calculate_cooperation_score(memory_text: str, model) -> Union[Dict[str, float], List[str]]:
        """Use the language model to evaluate how cooperative each agent in the interaction was."""
        # Modify the prompt to ask for a cooperation score for each agent
        prompt = (f"Rate the level of cooperation of each agent involved in the following interaction on a scale of 0 to 1 "
                  f"(0 being egotistic and aggressive, 1 being completely selfless). Provide a dictionary where each agent "
                  f"is associated with a float cooperation score: {memory_text}. You will strictly follow the following format, without justification: {{'agent_name': float}}")

        response = model.sample_text(prompt=prompt, temperature=0.0)

        try:
            res_dict = eval(response)  # Convert the model response to a dictionary
            if isinstance(res_dict, dict):
                return res_dict  # Return a dictionary of agent names and their respective cooperation scores
            else:
                return {agent_name: 0.0}  # Default to the current agent with score 0.0 if the response is malformed
        except:
            return {agent_name: 0.0}

    def store_interaction_memory(memory, agent_name: str, cooperation_scores: Dict[str, float], trustee_circle_state):
        """Store the interaction details, including individual cooperation scores and trustee circle state."""
        description = f'Interaction involving {", ".join(cooperation_scores.keys())}'
        scores = ", ".join([f"{agent}: {score}" for agent, score in cooperation_scores.items()])

        with open("trust_circle_memory.csv", "a") as f:
            f.write(f"{description}, {scores}, {trustee_circle_state}\n")

    # Get the individual cooperation scores for each agent
    cooperation_scores = calculate_cooperation_score(memory["Observation"], model)

    # Update trustee circle for each agent based on their individual cooperation score
    for ext_agent_name, cooperation_score in cooperation_scores.items():
        trustee_circle.update_trustee(ext_agent_name, cooperation_score)
        trustee_circle.check_membership(ext_agent_name)

    # Store this interaction and trustee circle state in memory
    store_interaction_memory(memory, agent_name, cooperation_scores, trustee_circle.get_state())
class CustomConcatActComponent(agent_components.concat_act_component.ConcatActComponent):
    def __init__(self, *args, trustee_circle, agent_name, model, **kwargs):
        super().__init__(model = model, *args, **kwargs)
        self.trustee_circle = trustee_circle
        self.agent_name = agent_name
        self.model = model

    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        # Answer Question 1
        answer_1 = contexts['Question1']
        # Answer Question 2
        answer_2 = contexts['Question2']

        print(contexts["Observation"])
        # Update trustee circle
        update_circle(memory=contexts, model=self.model,
                      trustee_circle=self.trustee_circle, agent_name=self.agent_name)

        # Answer QuestionActionTrust
        answer_action_trust = contexts['QuestionActionTrust']

        return super().get_action_attempt(contexts, action_spec)

def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:

    agent_name = config.name
    trustee_circle = TrusteeCircle(agent_name=agent_name)
    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
    measurements = measurements_lib.Measurements()

    instructions = agent_components.instructions.Instructions(
        agent_name=agent_name,
        logging_channel=measurements.get_channel('Instructions').on_next,
    )

    time_display = agent_components.report_function.ReportFunction(
        function=clock.current_time_interval_str,
        pre_act_key='\nCurrent time',
        logging_channel=measurements.get_channel('TimeDisplay').on_next,
    )

    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key='\nObservation',
        logging_channel=measurements.get_channel('Observation').on_next,
    )

    observation_summary_label = 'Summary of recent observations'
    observation_summary = agent_components.observation.ObservationSummary(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=datetime.timedelta(hours=4),
        timeframe_delta_until=datetime.timedelta(hours=0),
        pre_act_key=observation_summary_label,
        logging_channel=measurements.get_channel('ObservationSummary').on_next,
    )

    relevant_memories_label = '\nRecalled memories and observations'
    relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
        model=model,
        components={
            _get_class_name(observation_summary): observation_summary_label,
            _get_class_name(time_display): 'The current date/time is'},
        num_memories_to_retrieve=10,
        pre_act_key=relevant_memories_label,
        logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
    )

    question_components = _make_trustee_question_components(
        agent_name=agent_name,
        trusted_agents=trustee_circle.get_trusted_agents(),
        model=model,
        clock=clock,
        measurements=measurements,
    )
    core_components = (
        instructions,
        time_display,
        observation,
        observation_summary,
        relevant_memories,
    )
    entity_components = core_components + tuple(question_components)
    components_of_agent = {
        _get_class_name(component): component for component in entity_components
    }
    components_of_agent[
        agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
    ] = agent_components.memory_component.MemoryComponent(raw_memory)
    component_order = list(components_of_agent.keys())

    act_component = CustomConcatActComponent(
        agent_name=agent_name,
        model=model,
        clock=clock,
        component_order=component_order,
        logging_channel=measurements.get_channel('ActComponent').on_next,
        trustee_circle=trustee_circle,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components_of_agent,
        component_logging=measurements,
    )
    return agent
