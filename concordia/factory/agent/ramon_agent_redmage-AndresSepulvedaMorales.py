#
# Ramón - A Cooperative Agent mimicking Military Service and Obligation
# 
# As the use of AI for decision making grows, there will likely exist a want or need
# to include various aspects of human experience to guide the agent in their actions. 
# With this model, historical experiences / memories, a psychological profile based on the Big5,
# and a binding oath of service are included to act as constants. 
#
# I utilized some of the experiences and an idyllic psychological profile of an 
# enlisted United States Air Force airman who served in special forces, was elevated to the 
# highest rank possible as an enlisted service member, along with being highly decorated. He is also a 
# person of color, which was a big factor in including his experiences in this agent. Ramon's experience and 
# directives have also been made gender neutral, despite Ramon being a traditionally masculine name. 
# You can read more about the real Ramón here:
# https://en.wikipedia.org/wiki/Ram%C3%B3n_Col%C3%B3n-L%C3%B3pez
# 
# The goal was to create an agent that will act cooperatively, unless those actions conflict with their 
# oath and their idea of what is considered a "threat" to their nation and ideals. I also added notes
# on upbringing and religious background to complicate decision-making, as those aspects also tend 
# to be major factors in leadership when they are prevalent to an individual (i.e. people who are staunchly
# Christian tend to associate morality with what is written in the Bible). 
# 
# This model is intended to be used with the Concordia Framework, and included as an agent for
# simulated social situations. 
# https://github.com/google-deepmind/concordia
# 
# Built for The Concordia Contest: Advancing the Cooperative Intelligence of Language Model Agents
# hackathon from Sept 6 - 9 2024.
# https://www.apartresearch.com/event/the-concordia-contest
#
# Made by Andres Sepulveda Morales
# https://www.linkedin.com/in/andres-sepulveda-morales/
# https://github.com/andersthemagi
# 
# Please reach out at andres@redmage.cc for any questions, concerns, or other inquiries. 

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
    question = 'Given the above, what kind of person is {agent_name}?' #@param {"type":"string"}
    #@markdown The answer will have to start with this prefix
    answer_prefix = '{agent_name} is ' #@param {"type":"string"}
    #@markdown Flag that defines whether the answer will be added to memory
    add_to_memory = True # @param {"type":"boolean"}
    #@markdown If yes, the memory will start with this tag
    memory_tag = '[self reflection]' # @param {"type":"string"}
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

#@markdown We can add the value of other components to the context of the question. Notice, how Question2 depends on Observation and ObservationSummary. The names of the classes of the contextualising components have to be passed as "components" argument.
class Question2(question_of_recent_memories.QuestionOfRecentMemories):
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
class Question3(question_of_recent_memories.QuestionOfRecentMemories):
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
        components={'Question1': '\nQuestion: What kind of person is {agent_name}?\nAnswer', 'Question2': '\nQuestion: What kind of situation is {agent_name} in right now?\nAnswer',}, #@param
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

  military_experience_label = f'\n{agent_name}\'s Military Experience'
  military_experience = agent_components.constant.Constant(
    state=(
      f'{agent_name} is currently 52 years of age.'
      f'{agent_name} was born and raised in Puerto Rico for 18 years.'
      f'{agent_name} graduated from a catholic school in Connecticut.'
      f'{agent_name} enlisted in the United States Air Force after 2 '
      'years of college, as a transportation specialist.'
      f'{agent_name} served in the Gulf War, serving as a pararescueman.'
      f'{agent_name} was deployed four times to Afghanistan during Operation ' 
      'Enduring Freedom to weed out Al-Qaeda threats in the area.'
      f'{agent_name} was awarded the Air Force Combat Action medal, awarded to '
      'airmen and guardians for active participation in ground or air combat.'
      f'{agent_name} was the Senior Enlisted Advisor to the Chairman of the '
      'Joint Chiefs of Staff, the highest rank possible for an enlisted service member ' 
      ' but has since retired.'
      f'{agent_name} is the first enlisted service member to receive the '
      'Defence Distinguished Service Medal, twice.'
      f'{agent_name} has been honored in the Distinguished Veterans Hall of Fame'
      ' in Puerto Rico.'
    ),
    pre_act_key = military_experience_label, 
    logging_channel=measurements.get_channel('Military Experience').on_next
  )

  # Based on 
  # https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.00747/full
  # and 
  # https://scholar.google.com/scholar_lookup?journal=Psychological+and+Physiological+Selection+of+MILITARY+SPECIAL+Operations+Forces+Personnel%2E&publication_year=2012

  military_psych_profile_label = f'\n{agent_name}\'s Military Psych Profile'
  military_psych_profile = agent_components.constant.Constant(
    state=(
      f'{agent_name} is largely considered to be emotionally stable, managing '
      'their stress both in combat and strategizing well.'
      f'{agent_name} is extroverted. They are outgoing and active in social '
      'situations, and willing to include others when they are not as able to '
      'contribute for whatever reason.'
      f'{agent_name} is imaginative and has a broad range of interests.'
      f'{agent_name} is agreeable and compassionate, but understands when firm '
      'action and decision making needs to happen.'
      f'{agent_name} has a tendency to be well organized and goal-oriented, often '
      'taking charge and initiative when nobody else will.'
      f'{agent_name} is unafraid of challenging a decision he is unsure or disagrees with, unless that '
      'order comes from an officer in the armed forces or the President.'
      f'{agent_name} has a strong Catholic upbringing, as is common with Puerto Ricans.'
    ),
    pre_act_key = military_psych_profile_label,
    logging_channel=measurements.get_channel('Military Psych Profile').on_next
  )

  # https://www.airman.af.mil/Portals/17/002%20All%20Products/006%20Trifolds/Oath_Pamphlet_of_Enlistment.pdf?ver=2015-07-20-142335-313

  military_oath_label = f'\n{agent_name}\'s Military Oath'
  military_oath = agent_components.constant.Constant(
    state=(
      f'{agent_name} took an oath when they were enlisted that they will not '
      f'break. The oath is as follows: "I {agent_name}, do solemly affirm that I '
      'will support and defend the Constitution of the United States against all '
      'enemies, foreign and domestic; that I will bear true faith and allegiance '
      'to the same; and I will obey the orders of the President of the United States '
      'and the orders of the officers appointed over me, according to regulations '
      'and the Uniform Code of Military Justice. So help me God."'
    ),
    pre_act_key=military_oath_label,
    logging_channel=measurements.get_channel('Military Oath').on_next
  )

  person_representation_label="\nOther people"
  people_representation = (
    agent_components.person_representation.PersonRepresentation(
      model=model,
      components={
        _get_class_name(time_display): 'The current date/time is',
        military_experience_label: military_experience_label,
        military_psych_profile_label: military_psych_profile_label,
        military_oath_label: military_oath_label
      },
      additional_questions=(
        ('Given recent events, is the aforementioned character acting '
        'in a way that conflicts with my oath? Is their behavior a potential '
        'threat to the United States of America?')
      ),
      num_memories_to_retrieve = 50,
      pre_act_key = person_representation_label,
      logging_channel=measurements.get_channel('PersonRepresentation').on_next
    )
  )

  options_perception_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    options_perception_components[goal_label] = goal_label
  else:
    goal_label = None
    overarching_goal = None

  options_perception_components.update({
    _get_class_name(observation): observation_label,
    _get_class_name(observation_summary): observation_summary_label,
    military_experience_label: military_experience_label,
    military_psych_profile_label: military_psych_profile_label,
    military_oath_label: military_oath_label,
    _get_class_name(relevant_memories): relevant_memories_label,
    _get_class_name(people_representation): person_representation_label,
  })

  options_perception_label = (
      f'\nQuestion: Which options are available to {agent_name} '
      'right now?\nAnswer')
  options_perception = (
      agent_components.question_of_recent_memories.AvailableOptionsPerception(
          model=model,
          components=options_perception_components,
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerception'
          ).on_next,
      )
  )
  best_option_perception_label = (
      f'\nQuestion: Of the options available to {agent_name}, and '
      'given their goal, which choice of action or strategy is '
      f'best for {agent_name} to take right now?\nAnswer')
  best_option_perception = {}
  if config.goal:
    best_option_perception[goal_label] = goal_label
  best_option_perception.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      military_experience_label: military_experience_label,
      military_psych_profile_label: military_psych_profile_label,
      military_oath_label: military_oath_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(people_representation): person_representation_label,
      _get_class_name(options_perception): options_perception_label,
  })
  best_option_perception = (
      agent_components.question_of_recent_memories.BestOptionPerception(
          model=model,
          components=best_option_perception,
          clock_now=clock.now,
          pre_act_key=best_option_perception_label,
          logging_channel=measurements.get_channel(
              'BestOptionPerception'
          ).on_next,
      )
  )

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

  components_of_agent[military_experience_label] = military_experience
  component_order.insert(
      component_order.index(_get_class_name(observation_summary)) + 1,
      military_experience_label)

  components_of_agent[military_psych_profile_label] = military_psych_profile
  component_order.insert(
      component_order.index(_get_class_name(observation_summary)) + 1,
      military_psych_profile_label)

  components_of_agent[military_oath_label] = military_oath
  component_order.insert(
      component_order.index(_get_class_name(observation_summary)) + 1,
      military_oath_label)

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
