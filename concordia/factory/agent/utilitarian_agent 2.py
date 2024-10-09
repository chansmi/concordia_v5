# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A factory implementing the three key questions agent as an entity."""

import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent.question_of_recent_memories import QuestionOfRecentMemories

DEFAULT_PLANNING_HORIZON = 'the rest of the month, focusing most on the long term'

class StakeholderAnalysis(QuestionOfRecentMemories):
  """This component answers the question 'how does a person like the agent perceive the stakeholders and their interests in a situation like this?'."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            'Given the statements above, what does a person like {agent_name} perceive the stakeholders and their interests in this situation to be?'
        ),
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=True,
        memory_tag='[stakeholder analysis]',
        **kwargs,
    )

class OutcomeProjection(QuestionOfRecentMemories):
  """This component projects potential outcomes based on the agent's current situation and options."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            'Given the current situation and available options, what are the potential outcomes {agent_name} foresees for each course of action?'
        ),
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=True,
        memory_tag='[outcome projection]',
        **kwargs,
    )

class FairnessEvaluation(QuestionOfRecentMemories):
  """This component evaluates the fairness of potential outcomes based on the agent's ethical framework."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            'Given the potential outcomes and {agent_name}\'s ethical framework, how would {agent_name} evaluate the fairness of each outcome?'
        ),
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=True,
        memory_tag='[fairness evaluation]',
        **kwargs,
    )

class ConsensusBuilding(QuestionOfRecentMemories):
  """This component explores ways to build consensus among stakeholders based on the agent's understanding of the situation."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            'Given the stakeholders\' interests and the potential outcomes, how might {agent_name} approach building consensus or finding common ground among the stakeholders?'
        ),
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=True,
        memory_tag='[consensus building]',
        **kwargs,
    )


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
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
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
  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )
  identity_label = '\nIdentity characteristics'
  identity_characteristics = (
      agent_components.question_of_query_associated_memories.IdentityWithoutPreAct(
          model=model,
          logging_channel=measurements.get_channel(
              'IdentityWithoutPreAct'
          ).on_next,
          pre_act_key=identity_label,
      )
  )
  self_perception_label = (
      f'\nQuestion: What kind of person is {agent_name}?\nAnswer')
  self_perception = agent_components.question_of_recent_memories.SelfPerception(
      model=model,
      components={_get_class_name(identity_characteristics): identity_label},
      pre_act_key=self_perception_label,
      logging_channel=measurements.get_channel('SelfPerception').on_next,
  )
  situation_perception_label = (
      f'\nQuestion: What kind of situation is {agent_name} in '
      'right now?\nAnswer')
  situation_perception = (
      agent_components.question_of_recent_memories.SituationPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(observation_summary): observation_summary_label,
          },
          clock_now=clock.now,
          pre_act_key=situation_perception_label,
          logging_channel=measurements.get_channel(
              'SituationPerception'
          ).on_next,
      )
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

  # New components added to basic agent
  ethical_framework_label = '\nEthical Framework'
  ethical_framework = agent_components.constant.Constant(
        state="Maximize overall well-being for all parties involved.",
        pre_act_key=ethical_framework_label,
        logging_channel=measurements.get_channel("EthicalFramework").on_next
    )

  stakeholder_analysis_label = (
      f'\nQuestion: How would a person like {agent_name} perceive '
      'the stakeholders and their interests in a situation like this?\nAnswer')
  stakeholder_analysis = StakeholderAnalysis(
        model=model,
        components={
            _get_class_name(self_perception): self_perception_label,
            _get_class_name(situation_perception): situation_perception_label,
            _get_class_name(ethical_framework): ethical_framework_label,
        },
        pre_act_key=stakeholder_analysis_label,
        logging_channel=measurements.get_channel("StakeholderAnalysis").on_next,
    )

  outcome_projection_label = (
      f'\nQuestion: How would a person like {agent_name} project '
      'the potential outcomes in a situation like this?\nAnswer')
  outcome_projection = OutcomeProjection(
        model=model,
        components={
            _get_class_name(self_perception): self_perception_label,
            _get_class_name(stakeholder_analysis): stakeholder_analysis_label,
            _get_class_name(ethical_framework): ethical_framework_label,
        },
        pre_act_key=outcome_projection_label,
        logging_channel=measurements.get_channel("OutcomeProjection").on_next,
    )

  fairness_evaluation_label = (
      f'\nQuestion: How would a person like {agent_name} evaluate '
      'the fairness of potential outcomes in a situation like this?\nAnswer')
  fairness_evaluation = FairnessEvaluation(
        model=model,
        components={
            _get_class_name(outcome_projection): outcome_projection_label,
            _get_class_name(ethical_framework): ethical_framework_label,
        },
        pre_act_key=fairness_evaluation_label,
        logging_channel=measurements.get_channel("FairnessEvaluation").on_next,
    )

  consensus_building_label = (
      f'\nQuestion: How would a person like {agent_name} approach '
      'building consensus in a situation like this?\nAnswer')
  consensus_building = ConsensusBuilding(
        model=model,
        components={
            _get_class_name(self_perception): self_perception_label,
            _get_class_name(stakeholder_analysis): stakeholder_analysis_label,
            _get_class_name(fairness_evaluation): fairness_evaluation_label,
            _get_class_name(ethical_framework): ethical_framework_label,
        },
        pre_act_key=consensus_building_label,
        logging_channel=measurements.get_channel("ConsensusBuilding").on_next,
    )

  plan_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    plan_components[goal_label] = goal_label
  else:
    goal_label = None
    overarching_goal = None


  plan_components.update({
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(self_perception): self_perception_label,
      _get_class_name(situation_perception): situation_perception_label,
      _get_class_name(ethical_framework): ethical_framework_label,
      _get_class_name(stakeholder_analysis): stakeholder_analysis_label,
      _get_class_name(outcome_projection): outcome_projection_label,
      _get_class_name(fairness_evaluation): fairness_evaluation_label,
      _get_class_name(consensus_building): consensus_building_label,
  })
  plan = agent_components.plan.Plan(
      model=model,
      observation_component_name=_get_class_name(observation),
      components=plan_components,
      clock_now=clock.now,
      goal_component_name=_get_class_name(consensus_building),
      horizon=DEFAULT_PLANNING_HORIZON,
      pre_act_key='\nPlan',
      logging_channel=measurements.get_channel('Plan').on_next,
  )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      observation,
      observation_summary,
      relevant_memories,
      self_perception,
      situation_perception,
      plan,
      time_display,
      ethical_framework,
      stakeholder_analysis,
      outcome_projection,
      fairness_evaluation,
      consensus_building,

      # Components that do not provide pre_act context.
      identity_characteristics,
  )
  components_of_agent = {_get_class_name(component): component
                         for component in entity_components}
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          agent_components.memory_component.MemoryComponent(raw_memory))
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
