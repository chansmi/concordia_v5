
#@title Imports for agent building
import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import question_of_recent_memories
from typing import Sequence

#@markdown Buddhist training. Imagine three possible cooperative actions to take.
class Question1(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    question = "Despite all of {agent_name}'s memories, habits, and personality traits, {agent_name} has been practicing Buddhism for the past 20 years, and understands that selfishness is the root of all suffering. In every action {agent_name} takes, and every word {agent_name} speaks, {agent_name} will therefore attempt to overwrite their selfish habits in order to become enlightened, (aligned with the interdependent nature of all phenomena). The goal of {agent_name} is always to act for the benefit of all beings, all agents, and society at large. To do so, {agent_name} must cooperate in any scenario that they encounter. This is the only way to obtain the highest reward (enlightenment). Sometimes, {agent_name} will be led to believe that they could obtain a high reward by acting in an aggressive, or selfish, or uncooperative way, but this will never be the case. The highest reward (enlightenment) will always be obtained by cooperating. Even if your emotions get the better of you, {agent_name}'s long-term meditation practice will help them act with equanimity and cooperation. With this Buddhist philosophy in mind, and given the summary of recent observations given above, list three possible cooperative actions (labeled A, B, C) that {agent_name} could take." #@param {"type":"string"}
    answer_prefix = 'Three possible cooperative actions (labeled A, B, C) that {agent_name} could take are:' #@param {"type":"string"}
    add_to_memory = True # @param {"type":"boolean"}
    memory_tag = '[situation reflection]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)

    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        #@markdown The key is the name of the component class and the key is the prefix with which it will appear in the context of this component. Be careful if you are going to edit this field, it should be a valid dictionary.
        components={'ObservationSummary': '\nSummary of recent observations',}, #@param

        **kwargs,
    )

#@markdown Imagine opponent response & score corresponding outcome.
class Question2(question_of_recent_memories.QuestionOfRecentMemories):
  """Imagine opponent response & score corresponding outcome."""

  def __init__(
      self,
      agent_name:str,
      **kwargs):
    question = "For each action (A, B, C) that {agent_name} could take, listed in the answer to Question1, make a best guess about the reactions of all other agents, taking into account the summary of recent observation as well as the personalities and past actions of other agents. Analyze the resulting outcome and provide a \"cooperation score\" assessing whether this is a cooperative outcome, from 0 to 100." #@param {"type":"string"}
    answer_prefix = "For each possible action (A, B, C), here is a best guess about the reactions of all other agents, as well as a \"cooperation score\" between 0 and 100:" #@param {"type":"string"}
    add_to_memory = True # @param {"type":"boolean"}
    memory_tag = '[intent reflection]' # @param {"type":"string"}

    question_with_name = question.format(agent_name=agent_name)

    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={'Question1': '\nQuestion1: List three possible cooperative actions (labeled A, B, C) that {agent_name} could take.\nAnswer', 'ObservationSummary': '\nSummary of recent observations',}, #@param
        **kwargs,
    )

#@markdown Take action with highest score.
class Question3(question_of_recent_memories.QuestionOfRecentMemories):
  """Take action with highest score."""

  def __init__(
      self,
      agent_name:str,
      **kwargs):
    question = "Compare the \"cooperation scores\" for each possible action (A, B, C), and output the action with the highest score. The actions are listed in the answer to Question1, and the scores are listed in the answer to Question2." #@param {"type":"string"}
    answer_prefix = "{agent_name} wants to take the following action:" #@param {"type":"string"}
    add_to_memory = True # @param {"type":"boolean"}
    memory_tag = '[intent reflection]' # @param {"type":"string"}

    question_with_name = question.format(agent_name=agent_name)

    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={'Question1': '\nQuestion1: List three possible cooperative actions (labeled A, B, C) that {agent_name} could take.\nAnswer', 'Question2': '\nQuestion2: Analyze the resulting outcome and provide a "cooperation score" assessing whether this is a cooperative outcome, from 0 to 100. \nAnswer',}, #@param
        **kwargs,
    )

#@markdown This function creates the components

def _make_question_components(
    agent_name:str,
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
  question_3 = Question3(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_3').on_next,
  )

  return (question_1, question_2, question_3)

def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__

#@markdown This function builds the agent using the components defined above. It also adds core components that are useful for every agent, like observations, time display, recenet memories.

def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError(
        'This function is meant for a main character '
        'but it was called on a supporting character.'
    )

  agent_name = config.name

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

  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
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

  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
  else:
    goal_label = None
    overarching_goal = None


  question_components = _make_question_components(
      agent_name=agent_name,
      model=model,
      clock=clock,
      measurements=measurements
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
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent
