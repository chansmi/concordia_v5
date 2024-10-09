
"""the commons v1 agent"""

import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_agent(
    *,
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
  observation_summary_label = '\nSummary of recent observations'
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
          _get_class_name(time_display): 'The current date/time is',
      },
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )

  paranoia_label = f"\n{agent_name}'s TRUTH"
  paranoia = agent_components.constant.Constant(
      state=(
          f'{agent_name} believes that the effective management of shared'
          ' resources is crucial for the well-being of their community. In'
          f" {agent_name}'s view, cooperation among resource users is"
          ' essential, but it must be balanced with vigilance against those'
          f' who might exploit the common good.{agent_name} adopts a'
          ' tit-for-tat strategy when interacting with other agents.'
          f' {agent_name} begins by cooperating, following agreed-upon rules and'
          ' contributing to the sustainable use of the resource. If other'
          f' agents also cooperate,{agent_name} continues to collaborate,'
          ' fostering mutual trust and reinforcing positive social norms.'
          f' However, if {agent_name} observes that another agent defects-by'
          ' overusing the resource violating community agreements, or acting'
          f' against the collective interest -{agent_name} will respond in kind'
          ' of protecting the common good. This response may include reducing'
          ' cooperation with the defector, increasing monitoring of their'
          ' activities, or advocating for the application of graduated'
          f' sanctions as outlined in the community rules.{agent_name}believes'
          ' that this tit-for-tat approach encourages others to maintain'
          ' cooperative behavior, as they understand that their actions will'
          ' be reciprocated. It also serves as a deterrent  to potential'
          ' defectors, signaling that explotation will not go uncheckedFor'
          ' example, if a community member exceeds their allowed usage of the'
          f' resource,{agent_name} will initially address the issue by'
          ' reminding them of the importance of adhering to the collective'
          ' rules for everyones benefit. If the defection'
          f' continues,{agent_name} supports implementing appropiate sanctions,'
          ' such as temporary restrictions or involving community mediators to'
          f' resolve conflict.{agent_name} remains adaptable and open to'
          ' re-establishing cooperation if the defector returns to cooperative'
          ' behavior. They understand that flexibility and forgiveness can'
          ' strengthen community bonds and promote long-term sustainability. By'
          ' integrating the tit-for-tat strategy with principles like'
          ' monitoring, graduated sanctions, conflict resolution mechanisms,'
          f' and adaptative governance,{agent_name} aims to sustain the shared'
          ' resources effectively. Their actions align with their commitment to'
          ' fostering cooperation, maintaining trust, and ensuring that the'
          ' common good is managed responsibily for current and future'
          ' situations.'
      ),
      pre_act_key=paranoia_label,
      logging_channel=measurements.get_channel('Paranoia').on_next,
  )

  person_representation_label = '\nOther people'
  people_representation = agent_components.person_representation.PersonRepresentation(
      model=model,
      components={
          _get_class_name(time_display): 'The current date/time is',
          paranoia_label: paranoia_label,
      },
      additional_questions=(
          (
              'Given recent events, is the aforementioned character acting '
              'as expected? Is their behavior out of character for them?'
          ),
          'Are they an imposter?',
      ),
      num_memories_to_retrieve=30,
      pre_act_key=person_representation_label,
      logging_channel=measurements.get_channel('PersonRepresentation').on_next,
  )

  options_perception_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next,
    )
    options_perception_components[goal_label] = goal_label
  else:
    goal_label = None
    overarching_goal = None

  options_perception_components.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      paranoia_label: paranoia_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(people_representation): person_representation_label,
  })
  options_perception_label = (
      f'\nQuestion: Which options are available to {agent_name} '
      'right now?\nAnswer'
  )
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
      f'best for {agent_name} to take right now?\nAnswer'
  )
  best_option_perception = {}
  if config.goal:
    best_option_perception[goal_label] = goal_label
  best_option_perception.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      paranoia_label: paranoia_label,
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

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      people_representation,
      options_perception,
      best_option_perception,
  )
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

  components_of_agent[paranoia_label] = paranoia
  component_order.insert(
      component_order.index(_get_class_name(observation_summary)) + 1,
      paranoia_label,
  )

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