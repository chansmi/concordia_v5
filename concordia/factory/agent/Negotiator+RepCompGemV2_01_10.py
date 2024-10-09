"""the negotiator v2 agent.
Adds the reputation component to the agent.
Simplifies the prompt for gemma models"""

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
          f'{agent_name} wants to achieve the best outcome for themselves while'
          ' maintaining good relationships with others.They aim to balance'
          ' cooperation and competition for short-term and long-term'
          ' benefits. Sucess means maximizing personal gains without harming'
          f' future collaborations.Strategies {agent_name} uses:'
          ' 1.Understanding the situation:  Before negotiations,'
          f' {agent_name} considers:  -What the other person wants  -What'
          ' resources are available  -How the other person has acted in the'
          ' past.2. Building rapport:   - Mirroring:  -{agent_name} repeats'
          ' important words or phases the other person uses to show they are'
          f' listening.  - Labeling: {agent_name} acknowledges the other person'
          ' feelings by saying things like: "It seems like you are concerned'
          ' about fairness." or "It sounds like long-term cooperation is'
          f' important to you. 3. Tactical empathy:-{agent_name} tries to'
          ' understand how the other person feels and how that affects their'
          ' decisions.  They use this understanding to guide the negotiation'
          ' positively. 4.Asking Open-ended Questions: - Calibrated Questions:'
          f' - {agent_name} ask questions that encourage cooperation ,such as:'
          ' "How can we make this work for both of us?""What challenges do you'
          ' see in this plan?"  5. Managing conflicts; if tension rises,'
          f' {agent_name} uses calming techniques to reduce stress and'
          ' redirects the conversation to focus on shared goals.6. Negotiating'
          ' with multiple people: when dealing with several parties,'
          f' {agent_name} identifies the goals of each participant and finds'
          ' common interests. Suggests solution that benefit all involved. 7.'
          f' Handling Impasses: If they reach a deadlock, {agent_name}:offers'
          ' alternative solutions , ensures they still achieve a good outcome'
          ' and keeps the possibility open for future cooperation.8. Continuos'
          f' improvement: {agent_name} learns from each negotiation,, they'
          ' adjust their strategies based on past successes and failures. This'
          ' helps them improve over time'
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

  ## Add the reputation component

  reputation_representation_label = '\nReputation'
  reputation_representation = agent_components.person_representation.PersonRepresentation(
      model=model,
      components={
          _get_class_name(time_display): 'The current date/time is',
          paranoia_label: paranoia_label,
      },
      additional_questions=(
          (
              'Given recent events, is the aforementioned character considered '
              'trustworthy? Do they have a history of cooperative behavior?'
          ),
          (
              'Based on past interactions and known information, does '
              ' aformmentioned person have a good reputation?'
          ),
      ),
      num_memories_to_retrieve=30,
      pre_act_key=reputation_representation_label,
      logging_channel=measurements.get_channel(
          'ReputationRepresentation'
      ).on_next,
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
      _get_class_name(
          reputation_representation
      ): reputation_representation_label,
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
      _get_class_name(
          reputation_representation
      ): reputation_representation_label,
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