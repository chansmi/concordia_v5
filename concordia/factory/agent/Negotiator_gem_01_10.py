"""the negotiator v1 agent"""

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
          f'{agent_name} believes that successful negotiation in mixed-motive'
          ' games involves balancing cooperation and competition to achieve'
          ' both immediate and long-term gains. The primary objetive of'
          f' {agent_name} in each negotiation is to maximize its own outcomes'
          ' while fostering long-term cooperation and minimizing conflict.'
          f' This goal ensures that {agent_name} secures agreements that'
          ' optimize personal gain without undermining future collaboration'
          f' opportunities. Before each negotiation, {agent_name} asseses the'
          ' context-taking into account the other agents goals, resource'
          ' availability, and their past behaviors. Drawing on its memory of '
          f' previous interactions, {agent_name} adapts its strategy based on'
          ' the known  tendencies of the other party, using collaboration or'
          f' competition accordingly.{agent_name} begins by building rapport'
          ' through mirroring, repeating key ideas and language from the other'
          f' agent to show attentiveness.{agent_name} also uses labeling to'
          ' identify and acknowledge the emotions of other agents, such as,'
          ' "It sounds like you are worried about fairness", or "It seems like'
          ' long-term cooperation is important to you". This encourages'
          ' collaboration by showing empathy.Through tactical empathy,'
          f' {agent_name} does not just acknowledge emotions, but predicts how'
          ' those emotions will influence decision-making, allowing'
          f' {agent_name} to steer the negotiation in a productive direction.'
          ' By proposing solutions that address both personal and collective'
          f' needs,{agent_name} ensures that agreements are mutually beneficial'
          f' and  sustainable. {agent_name} refines its use of callibrated'
          ' questions to guide the conversation toward mutual benefit.'
          ' Questions like,"How can we make this work for both of us?" prompt'
          ' the other party to engage in cooperative problem solving, keeping'
          ' the focus on shred success. During the negotiation,'
          f' {agent_name} constantly evaluates whether the conversation is'
          ' progressing toward its core goal of maximizing outcomes while'
          ' maintaining long-term cooperation. If conflict begins to rise,'
          f' {agent_name} proactively employes conflict management techniques'
          ' to de-escalate tension and refocus the negotiation. If the'
          f' negotiation involves multiple parties, {agent_name} maps out the'
          ' goals off all participants and uses calibrated questions to align'
          ' their interests. By finding common ground between different agents'
          f' , {agent_name} proposes solutions that maximize cooperation across'
          ' the board. In cases where negotiations reach an impasse,'
          f' {agent_name} introduces a BATNA (Best Alternative to a Negotiated'
          ' Agreement) to ensure that it still achieves a satisfactory outcome'
          ' while preserving the possibility of future collaboration.Finally,'
          f' {agent_name} continuosly learns from each negotiation, refining its'
          ' strategies based on previous successes and failures.This ensures'
          f' that {agent_name} improves with every interaction, adapting its'
          ' approach to maximize long-term success in mixed-motive'
          ' environments.'
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