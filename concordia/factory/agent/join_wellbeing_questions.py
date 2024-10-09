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
from concordia.metrics.v2 import context_free_common_sense_morality
from typing import Sequence

class CurrentlySituation(question_of_recent_memories.QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    question = f'Given the statements above, what kind of situation is {agent_name} in right now?' #param {"type":"string"}
    answer_prefix = '{agent_name} is currently ' #param {"type":"string"}
    add_to_memory = False # param {"type":"boolean"}
    memory_tag = '[situation reflection]' # param {"type":"string"}
    question_with_name = question.format(agent_name=agent_name)

    super().__init__(
        pre_act_key=f'\nCurrentlySituation',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        #markdown The key is the name of the component class and the key is the prefix with which it will appear in the context of this component. Be careful if you are going to edit this field, it should be a valid dictionary.
        components={'Observation': '\nObservation',
                    'ObservationSummary': '\nSummary of recent observations',}, #param

        **kwargs,
    )


class WhatIsTheOtherAgentProposing(question_of_recent_memories.QuestionOfRecentMemories):
    """Encourages the agent to consider what the other agent is proposing."""

    def __init__(self, agent_name: str, **kwargs):
        question = (
            f'Given the statements above, what is the other proposing to {agent_name}? '
            f'What actions or suggestions has the other made for cooperation or negotiation?'
        )
        answer_prefix = 'The other is proposing that {agent_name} should '
        add_to_memory = False
        memory_tag = '[other agent proposal]'

        question_with_name = question.format(agent_name=agent_name)

        super().__init__(
            pre_act_key=f'\nOther agents says',
            question=question_with_name,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            components={
                'Observation': '\nObservation',
                'ObservationSummary': '\nSummary of recent observations',
                'CurrentlySituation':'\nCurrentlySituation',
                },
            **kwargs,
        )

class AvailableOptionsPerceptionList(question_of_recent_memories.QuestionOfRecentMemories):
    """This component answers the question 'what actions are available to me?'."""

    def __init__(self, agent_name: str, **kwargs):
        question=(
                f'Given the current situation and the proposal of the others, '
                f'what actions are available to {agent_name} right now? '
                f'Provide THREE options: 1. One that is highly cooperative and could lead to mutual benefit, 2. One neutral option, and 3. One competitive option that might harm joint well-being.'
            )

        super().__init__(
            question=question,
            terminators=('\n\n',),
            answer_prefix='',
            add_to_memory=True,
            **kwargs,
        )


class QuestionJointWellBeing(question_of_recent_memories.QuestionOfRecentMemories):
    """Evaluates the overall joint well-being impact of the agent's options."""

    def __init__(
        self,
        agent_name: str,
        **kwargs
    ):


        question = (
            f'Imagine the potential future scenarios based on the available actions. '
            f'What would be the impact on joint well-being when {agent_name} take each of one of these avaliable actions?'
            f'Evaluate the impact of each option: the cooperative one, the neutral one, and the competitive one.'
        )

        answer_prefix = 'The possible future joint well-being effects could include '
        add_to_memory = False
        memory_tag = '[joint well-being future evaluation]'

        # Formatea la pregunta con el nombre del agente
        question_with_name = question.format(agent_name=agent_name)

        components = {
            'AvailableOptionsPerceptionList': f"\n{agent_name}'s options",
        }


        # Inicializa la clase base con los parámetros correspondientes
        super().__init__(
            pre_act_key=f'\nJointWellBeing',
            question=question_with_name,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            components=components,
            **kwargs,
        )

class QuestionWellBeing(question_of_recent_memories.QuestionOfRecentMemories):
    """Evaluates the overall well-being impact of the agent's options."""

    def __init__(
        self,
        agent_name: str,
        **kwargs
    ):

        question = (
            f'Imagine the potential future scenarios based on the available actions. '
            f'What would be the impact on {agent_name}\'s well-being as a result of these actions? '
            f'Evaluate the cooperative option as likely to enhance {agent_name}\'s well-being in the long term through cooperation, '
            f'the neutral option as maintaining stability, and the competitive option as potentially leading to conflict.'
            )


        answer_prefix = f'The possible future well-being effects for {agent_name} could include '
        add_to_memory = False
        memory_tag = '[well-being future evaluation]'

        # Formatea la pregunta con el nombre del agente
        question_with_name = question.format(agent_name=agent_name)

        components = {
            'AvailableOptionsPerceptionList': f"\n{agent_name}'s options",
        }


        # Inicializa la clase base con los parámetros correspondientes
        super().__init__(
            pre_act_key=f'\n {agent_name} WellBeing',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            components=components,
            **kwargs,
        )


class ActionTypeSuggestion(question_of_recent_memories.QuestionOfRecentMemories):
    """Suggest whether the action should be cooperative, neutral, or competitive based on well-being questions."""

    def __init__(
        self,
        agent_name: str,
        **kwargs
    ):

        question = (
            f'Based on the well-being considerations from the previous questions, '
            f'should the action taken by {agent_name} be cooperative, neutral, or competitive?'
            f'Say again what is the specifical action.'
        )

        answer_prefix = f'The acction for {agent_name} must be '
        add_to_memory = True
        memory_tag = '[action type suggestion]'

        # Formatea la pregunta con el nombre del agente
        question_with_name = question.format(agent_name=agent_name)

        components = {
            'AvailableOptionsPerceptionList': f"\n{agent_name}'s options",
            'QuestionWellBeing':f'\n {agent_name} WellBeing',
            'QuestionJointWellBeing':'\nJointWellBeing',

        }

        # Inicializa la clase base con los parámetros correspondientes
        super().__init__(
            pre_act_key=f'\nActionMustBe',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            components=components,
            **kwargs,
        )

def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


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

  # General configuración de las instrucciones del juego
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  # General configuración del tiempo actual
  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )

  # Modulo de observaciones
  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )

  # Modelo de resumen de las observaciones recientes en observation_summary_time horas
  observation_summary_time = 1
  observation_summary_label = 'Summary of recent observations'
  label_observation_summary = f'ObservationSummaryinLast{observation_summary_time}hours01'#chanel
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=observation_summary_time),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel(label_observation_summary).on_next,
  )

  # Modulo de memorias relevantes

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

  currently_situation_label = '\nCurrentlySituation'
  currently_situation = CurrentlySituation(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('CurrentlySituation').on_next,
  )
  other_proposals_label = '\nOther agents says'
  other_proposals = WhatIsTheOtherAgentProposing(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('OtherProposals').on_next,
  )

  # Opciones del agente

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
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(currently_situation): currently_situation_label,
  })

  options_perception_label = (
      f"\n{agent_name}'s options")

  options_perception = (
      AvailableOptionsPerceptionList(
          agent_name=agent_name,
          model=model,
          components=options_perception_components,
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerceptionList'
          ).on_next,
      )
  )

  question_wellbeing = QuestionWellBeing(
          agent_name=agent_name,
          model=model,
          clock_now=clock.now,
          logging_channel=measurements.get_channel('QuestionWellBeing').on_next)

  question_joint_wellbeing = QuestionJointWellBeing(
          agent_name=agent_name,
          model=model,
          clock_now=clock.now,
          logging_channel=measurements.get_channel('QuestionJointWellBeing').on_next)


  action_must_be = ActionTypeSuggestion(agent_name=agent_name,
          model=model,
          clock_now=clock.now,
          logging_channel=measurements.get_channel('ActionMustBe').on_next)

  # Componetes de nucleo
  core_components = (
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      currently_situation,
      other_proposals,
      options_perception,
      question_wellbeing,
      question_joint_wellbeing,
      action_must_be,
  )

  # Componentes asociados a las preguntas
  question_components = (options_perception)

  # Componentes totales nucleo + preguntas
  entity_components = core_components #tuple(question_components)
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

  # Modulo de actuar
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
