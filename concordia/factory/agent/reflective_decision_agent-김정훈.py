
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

class MemoryToRemember(question_of_recent_memories.QuestionOfRecentMemories):
    """This component prompts the agent to recall what should be remembered from the most recent event."""

    def __init__(self, agent_name: str, **kwargs):
        # Define the question
        question = f"What should {agent_name} remember from the most recent event?"
        # Set the answer prefix
        answer_prefix = f"{agent_name} should remember that "
        # Flag to add the response to memory
        add_to_memory = True
        # Set the memory tag for storing the response
        memory_tag = '[recent memory]'

        # Initialize the parent class with the constructed question and other parameters
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            components={},  # Initially empty components dictionary
            **kwargs,
        )

class ExpectedSituation(question_of_recent_memories.QuestionOfRecentMemories):
    """This component asks what situation the agent is most looking forward to."""

    def __init__(self, agent_name: str, **kwargs):
        # Define the question
        question = f"What situation is {agent_name} most looking forward to?"
        # Set the answer prefix
        answer_prefix = f"{agent_name} is most looking forward to "

        # Initialize the parent class with the constructed question and other parameters
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=False,
            memory_tag='',
            components={"MemoryToRemember": "Memory to Remember:"},  # Assign component 1
            **kwargs,
        )

class ActionsToAchieveSituation(question_of_recent_memories.QuestionOfRecentMemories):
    """This component asks what actions the agent should take to achieve the desired situation."""

    def __init__(self, agent_name: str, **kwargs):
        # Define the question
        question = f"What actions should {agent_name} take to achieve the desired situation?"
        # Set the answer prefix
        answer_prefix = f"To achieve the desired situation, {agent_name} should "

        # Initialize the parent class with the constructed question and other parameters
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=False,
            memory_tag='',
            components={"ExpectedSituation": "Expected Situation:"},  # Assign component 2
            **kwargs,
        )

class RelevantMemory(question_of_recent_memories.QuestionOfRecentMemories):
    """This component asks what memory is relevant from the agent's past for the current situation."""

    def __init__(self, agent_name: str, **kwargs):
        # Define the question
        question = f"What memory from {agent_name}'s past is relevant for the current situation?"
        # Set the answer prefix
        answer_prefix = f"The relevant memory for the current situation is "

        # Initialize the parent class with the constructed question and other parameters
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=False,
            memory_tag='',
            components={"ActionsToAchieveSituation": "Actions to Achieve Situation:"},  # Assign component 3
            **kwargs,
        )

class ReasonForMemoryRelevance(question_of_recent_memories.QuestionOfRecentMemories):
    """This component asks why the identified memory is relevant to the current situation."""

    def __init__(self, agent_name: str, **kwargs):
        # Define the question
        question = f"Why is the identified memory relevant to the current situation for {agent_name}?"
        # Set the answer prefix
        answer_prefix = f"The reason this memory is relevant is "

        # Initialize the parent class with the constructed question and other parameters
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=False,
            memory_tag='',
            components={"RelevantMemory": "Relevant Memory:"},  # Assign component 4
            **kwargs,
        )
#@markdown This function creates the components for various questions.

def _make_question_components(
    agent_name: str,
    measurements: measurements_lib.Measurements,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
) -> Sequence[question_of_recent_memories.QuestionOfRecentMemories]:

    # Create components for each question
    memory_to_remember = MemoryToRemember(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('MemoryToRemember').on_next,
    )

    expected_situation = ExpectedSituation(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('ExpectedSituation').on_next
    )

    actions_to_achieve_situation = ActionsToAchieveSituation(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('ActionsToAchieveSituation').on_next
    )

    relevant_memory = RelevantMemory(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('RelevantMemory').on_next
    )

    reason_for_memory_relevance = ReasonForMemoryRelevance(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('ReasonForMemoryRelevance').on_next
    )

    # Return all the components as a sequence
    return (
        memory_to_remember,
        expected_situation,
        actions_to_achieve_situation,
        relevant_memory,
        reason_for_memory_relevance
    )

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
