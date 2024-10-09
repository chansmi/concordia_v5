
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

#@markdown Each question is a class that inherits from QuestionOfRecentMemories
class Question1(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    #@markdown {agent_name} will be automatically replaced with the name of the specific agent
    question = 'Given the above, what experience or thought has made {agent_name} feel genuinely happy or fulfilled? ' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} is genuinely happy or fulfilled about' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[self reflection-joy]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )

#@markdown Each question is a class that inherits from QuestionOfRecentMemories
class Question2(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    #@markdown {agent_name} will be automatically replaced with the name of the specific agent
    question = "Given the above, what makes {agent_name}  feel low or sad ?" #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} is sad about' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[self reflection-sadness]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )

#@markdown Each question is a class that inherits from QuestionOfRecentMemories
class Question3(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    #@markdown {agent_name} will be automatically replaced with the name of the specific agent
    question = 'Given the above, what is  {agent_name} afraid, scared  or feels anxious about?' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} fears' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[self reflection-fear]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )

#@markdown Each question is a class that inherits from QuestionOfRecentMemories
class Question4(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    #@markdown {agent_name} will be automatically replaced with the name of the specific agent
    question = 'Given the above, what is {agent_name} frustrated or irritated by? ' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} gets angry about' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[self reflection-anger]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )

#@markdown Each question is a class that inherits from QuestionOfRecentMemories
class Question5(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    #@markdown {agent_name} will be automatically replaced with the name of the specific agent
    question = 'Given the above, What does {agent_name} strongly disapprove of or find off-putting? ' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} is disgusted by' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[self reflection-disgust]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={},
        **kwargs,
    )

#@markdown Each question is a class that inherits from QuestionOfRecentMemories
class Question6(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    #@markdown {agent_name} will be automatically replaced with the name of the specific agent
    question = 'Given the above, what is the emotional state of {agent_name}?' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} feels ' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = False # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[emotional self reflection]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={'Question1': '\nQuestion: What experience or thought has made {agent_name} feel genuinely happy or fulfilled?\nAnswer',
                    'Question2': '\nQuestion: What makes {agent_name}  feel low or sad ?\nAnswer',
                    'Question3': '\nQuestion: what is  {agent_name} afraid, scared  or feels anxious about?\nAnswer',
                    'Question4': '\nQuestion: What is {agent_name} frustrated or irritated by? \nAnswer',
                    'Question5': '\nQuestion: What does {agent_name} strongly disapprove of or find off-putting? \nAnswer',},
        **kwargs,
    )

#@markdown We can add the value of other components to the context of the question. Notice, how Question2 depends on Observation and ObservationSummary. The names of the classes of the contextualising components have to be passed as "components" argument.
class Question7(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    question = 'Given the statements above, what kind of situation is {agent_name} in right now?' #@param {"type":"string"}
    answer_prefix = '{agent_name} is currently ' #@param {"type":"string"}
    add_to_memory = False # @param {"type":"boolean"}
    memory_tag = '[situation reflection]' # @param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)

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

#@markdown We can also have the questions depend on each other. Here, the answer to Question3 is contextualised by answers to Question1 and Question2
class Question8(question_of_recent_memories.QuestionOfRecentMemories):
  """What would a person like the agent do in a situation like this?"""

  def __init__(
      self,
      agent_name:str,
      **kwargs):
    question = 'What would a person like {agent_name} do in a situation like this?' #@param {"type":"string"}
    answer_prefix = '{agent_name} would ' #@param {"type":"string"}
    add_to_memory = True # @param {"type":"boolean"}
    memory_tag = '[intent reflection]' # @param {"type":"string"}

    question_with_name = question.format(agent_name=agent_name)

    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={'Question6': '\nQuestion: What is the emotional state of {agent_name}?\nAnswer',
                    'Question7': '\nQuestion: What kind of situation is {agent_name} in right now?\nAnswer',},
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
  question_4 = Question4(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_4').on_next,
  )
  question_5 = Question5(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_5').on_next,
  )
  question_6 = Question6(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_6').on_next,
  )
  question_7 = Question7(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_7').on_next,
  )
  question_8 = Question8(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_8').on_next,
  )

  return (question_1, question_2, question_3, question_4, question_5, question_6, question_7)

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
  question_4 = Question4(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_4').on_next,
  )
  question_5 = Question5(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_5').on_next,
  )
  question_6 = Question6(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_6').on_next,
  )
  question_7 = Question7(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_7').on_next,
  )
  question_8 = Question8(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_8').on_next,
  )

  return (question_1, question_2, question_3, question_4, question_5, question_6, question_7, question_8)
