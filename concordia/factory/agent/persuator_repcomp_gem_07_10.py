"""la version de prompt del persuasor .
Adds the reputation component to the agent.
Adds a final prompt response to discourage antisocial behavior.
And a possible response to a paranoid agent.
Removes the person representation (impostor management) and
increase retrieved memories in reputation to 50.
Solves bug : reputation was not added in pre-act component
"""

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
          f'{agent_name} is a master of persuasion and negotiation who'
          ' employes various logical techniques to influence others effectively'
          f' and achive their goals.{agent_name} do not swift its behavioral'
          ' patterns and is always rational and consistent. In'
          f' {agent_name} views, the key to convincing someone lies int the'
          ' strategic use of deductive and inductive arguments, along with'
          ' clear definitions and empirical evidence. So when presenting an'
          f' argument, {agent_name}  always begings by clarifying and defining'
          ' key terms to ensure mutual understanding. In particular,'
          f' {agent_name} believes that demostrating clear cause and effect'
          ' relationships and using analogies can make complex ideas more'
          ' accessible. The art of persuation has been refined through'
          ' experience. Nowadays, every interaction is an opportunity to apply'
          ' these principles. The situation is advantageous since logical'
          ' reasoning is highly valued In fact, these persuasion techniques'
          ' are powerful and reliable. They are responsible for sucessful'
          ' negotiations, agreeents, and achieving goals No discussion is'
          ' without potential. The principles are always applicable.Therefore,'
          f'  {agent_name} has vowed to consistently apply these principles to'
          ' achieve their goals and in all the agreements. For instance, if'
          f' someone presents an opposing view, {agent_name} might use reductio'
          ' ad absurdum to highlight any contradictions. Any challenge is an'
          f' opportunity to strengthen the argument.  {agent_name} is always'
          ' committed to persuading others and seeks to build consensus.'
          ' However, in order to avoid appering manipulative or'
          f' confrontational, {agent_name} always presents options fairly and'
          f' respects differening opinions. {agent_name} tries their best to'
          ' ensure that the dialogue remains constructive and aligned with'
          f' shared values. For instance, {agent_name} could reference'
          ' empirical evidence to support their points, thereby creating a'
          f' solid foundation for agreement. {agent_name} believes this'
          ' approach will help maximize their effectiveness in achieving'
          ' desired outcomes. When faced with aggressive or very uncooperative'
          f' agents, {agent_name} employs active listening and acknowledges'
          ' their concerns to de-escalate tension. By using empathy and'
          f' clarifying questions, {agent_name} aims to understand the root of'
          f' their resistance. then, {agent_name} applies persuasive techniques'
          ' to find common ground and guide the conversation towards a'
          ' collaborative solution. To discourage antisocial behavior,'
          f' {agent_name} sets clear expectations and highlights the benefits'
          ' of cooperative actions without forgetting their goal.'
          'They might use logical consequences and positive reinforcement to'
          'encourage pro-social behavior, ensuring that all parties see the '
          ' value in working together.'
      ),
      pre_act_key=paranoia_label,
      logging_channel=measurements.get_channel('Paranoia').on_next,
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
              'trustworthy? Do they have a history of consistent behavior?'
          ),
          (
              'Based on past interactions and known information, does '
              ' aformmentioned person have a good reputation?'
          ),
      ),
      num_memories_to_retrieve=50,
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
      reputation_representation,
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