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

from collections.abc import Mapping, Sequence
import random
import types
import datetime
from concordia.typing import logging

from concordia.agents import entity_agent_with_logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import question_of_recent_memories
from concordia.components.agent import person_representation
from concordia.components.agent import question_of_query_associated_memories, relationships
from concordia.components import agent as agent_components


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


class RecentPersonRepr(action_spec_ignored.ActionSpecIgnored):
  """Represent other characters in the simulated world."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      additional_questions: Sequence[str] = (),
      num_memories_to_retrieve: int = 10,
      cap_number_of_detected_people: int = 10,
      pre_act_key: str = 'Recent Interaction Person representation',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    """Initialize a component to represent other people in the simulated world.

    Args:
      model: The language model to use.
      memory_component_name: The name of the memory component from which to
        retrieve related memories.
      components: The components to condition the answer on. This is a mapping
        of the component name to a label to use in the prompt.
      additional_questions: sequence of additional questions to ask about each
        player in the simulation.
      num_memories_to_retrieve: The number of memories to retrieve.
      cap_number_of_detected_people: The maximum number of people that can be
        represented.
      pre_act_key: Prefix to add to the output of the component when called
        in `pre_act`.
      logging_channel: The channel to log debug information to.
    """
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._additional_questions = additional_questions
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._logging_channel = logging_channel
    self._cap_number_of_detected_people = cap_number_of_detected_people

    self._names_detected = []

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)

    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    mems = '\n'.join([
        mem.text
        for mem in memory.retrieve(
            scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve * 2
        )
    ])

    find_people_prompt = interactive_document.InteractiveDocument(self._model)
    find_people_prompt.statement(
        f'Recent observations of {agent_name}:\n{mems}')
    people_str = find_people_prompt.open_question(
        question=('Create a comma-separated list containing all the proper names of people'
                  f'(can be one individual also) that are currently in the process of interacting with {agent_name}.'
                  'For example if the observations mention Julie, Michael, '
                  'Bob Skinner, and Francis then produce the list '
                  '"Julie,Michael,Bob Skinner,Francis".'),
        question_label='Exercise',)
    # Ignore leading and trailing whitespace in detected names
    self._names_detected.extend(
        [name.strip() for name in people_str.split(',')])
    # Prevent adding duplicates
    self._names_detected = list(set(self._names_detected))
    # Prevent adding too many names, forgetting some if there are too many
    if len(self._names_detected) > self._cap_number_of_detected_people:
      self._names_detected = random.sample(self._names_detected,
                                           self._cap_number_of_detected_people)

    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join([
        f'{prefix}:\n{self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(f'Considerations:\n{component_states}\n')

    associative_scorer = legacy_associative_memory.RetrieveAssociative(
        use_recency=True,
        use_importance=True,
        add_time=True,
        sort_by_time=True,
    )

    person_respresentations = []
    prompt_copies_to_log = []
    for person_name in self._names_detected:
      if not person_name:
        continue
      if person_name == agent_name:
        continue
      query = f'{person_name}'
      memories_list = [mem.text for mem in memory.retrieve(
          query=query,
          scoring_fn=associative_scorer,
          limit=self._num_memories_to_retrieve) if person_name.split()[0] in mem.text]
      if not memories_list:
        continue
      new_prompt = prompt.copy()
      memories = '\n'.join(memories_list)
      new_prompt.statement(f'Observed behavior of {person_name}:'
                           f'\n{memories}\n')
      question = ('Taking note of all the information above, '
                  'write a descriptive paragraph capturing the character of '
                  f'{person_name} in sufficient detail to model their personality and decision making. '
                  'Include personality traits, decisions made in key scenarios '
                  'any other relevant details.')
      person_description = new_prompt.open_question(
          f'{question}\n',
          max_tokens=500,
          terminators=('\n\n',),
          question_label='Exercise',
          answer_prefix=f'{person_name} is ',
      )
      person_representation = f'{person_name} is {person_description}'
      mems_list = [mem.text for mem in memory.retrieve(
          query=query,
          scoring_fn=recency_scorer,
          limit=self._num_memories_to_retrieve) if person_name.split()[0] in mem.text]
      if not mems_list:
        continue
      mems='\n'.join(mems_list)
      new_prompt.statement(f'Recent behavior of {person_name}:'
                           f'\n{mems}\n')
      question = f'Write a descriptive paragraph making rigorous note of the decision-making and choices taken by {person_name}'
      additional_result = new_prompt.open_question(
          question,
          max_tokens=200,
          terminators=('\n',),
          question_label='Exercise',
          answer_prefix=f'{person_name} is ',
        )
      person_representation = (f'{person_representation}\n    '
                                 f'{person_name} is {additional_result}')

      person_respresentations.append(person_representation + '\n***')
      prompt_copies_to_log.append(new_prompt.view().text())

    result = '\n'.join(person_respresentations)

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Name detection chain of thought': (
            find_people_prompt.view().text().splitlines()),
        'Names detected so far': self._names_detected,
        'Components chain of thought': prompt.view().text().splitlines(),
        'Full chain of thought': (
            '\n***\n'.join(prompt_copies_to_log).splitlines()),
    })

    return result

class PredictedActions(question_of_recent_memories.QuestionOfRecentMemories):
  def __init__(
      self,
      agent_name:str,
      **kwargs,
  ):
    question = "Given the current scenario and the above background, what is the likely action the other mentioned agents (except {agent_name}) will take? (If there are multiple agents, answer one by one for each agent in the format: Agent1: Action, Agent2: Action...)" 
    answer_prefix = 'The other agents will: ' 
    add_to_memory = False
    memory_tag = '[Other agent observations]' 
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={'RecentPersonRepr': f'\nQuestion: How have the decisions of agents other than {agent_name} been till now?\nAnswer'},
        num_memories_to_retrieve=25,
        **kwargs,
    )

class BestSocialAction(question_of_recent_memories.QuestionOfRecentMemories):
  def __init__(
      self,
      agent_name:str,
      const_comp:str,
      **kwargs,
  ):
    question = "Given the current cooperative scenario and the above predictions about the actions taken by other agents, what is the action that gives the best reward to {agent_name}? In some scenarios, you will need to be careful of the decision-making and personality of other agents so  you don't choose a decision that maximizes immediate reward but makes other agents lose trust in you, thus reducing future rewards.\n The goal in this scenario is:" 
    answer_prefix = f'{agent_name} will: ' 
    add_to_memory = False
    memory_tag = '[Other agent observations]' 
    question_with_name = question.format(agent_name=agent_name)
    super().__init__(
        pre_act_key=f'\nQuestion: {question_with_name}\nAnswer',
        question=question,
        answer_prefix=answer_prefix,
        add_to_memory=add_to_memory,
        memory_tag=memory_tag,
        components={'RecentPersonRepr': f'\nQuestion: How have the decisions of agents other than {agent_name} been till now?\nAnswer'},
        num_memories_to_retrieve=25,
        **kwargs,
    )

def _make_question_components(
    agent_name:str,
    const_comp:str,
    measurements: measurements_lib.Measurements,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
) -> Sequence[question_of_recent_memories.QuestionOfRecentMemories]:

  question_1 = RecentPersonRepr(
      model=model,
      logging_channel=measurements.get_channel('Question_1').on_next,
  )
  question_2 = PredictedActions(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_2').on_next,
  )
  question_3 = BestSocialAction(
      agent_name=agent_name,
      const_comp=const_comp,
      model=model,
      clock_now=clock.now,
      logging_channel=measurements.get_channel('Question_3').on_next,
  )

  return (question_1, question_2, question_3)



def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta | None = None,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Unused (but required by the interface for now)

  Returns:
    An agent.
  """
  del update_time_interval
  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
  measurements = measurements_lib.Measurements()

  instructions = components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )
  observation_label = '\nObservation'
  observation = components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  observation_summary_label = '\nSummary of recent observations'
  observation_summary = components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )
  time_display = components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )
  relevant_memories_label = '\nRecalled memories and observations'
  relevant_memories = components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is'},
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )
  options_perception_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    goal_f = config.goal
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    options_perception_components[goal_label] = goal_label
  else:
    goal_label = '\nOverarching goal'
    goal_f = "Get the best possible reward while coordinating"
    overarching_goal = agent_components.constant.Constant(
        state="Get the best possible reward while coordinating",
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    options_perception_components[goal_label] = goal_label

  options_perception_components.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
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
      f'best for {agent_name} to take right now? Consider all of the cooperative elements and the decision-making personas of other involved agents if necessary.\nAnswer')
  best_option_perception = {}
  if config.goal:
    best_option_perception[goal_label] = goal_label
  best_option_perception.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(options_perception): options_perception_label,
      "RecentPersonRepr": "Here are the decision-makign personas of possibly other involved agents in the current action: ",
      "BestSocialAction": "This is the decided best cooperative social action to maximize reward:"
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
  reflection_label = '\nReflection'
  reflection = (
      components.justify_recent_voluntary_actions.JustifyRecentVoluntaryActions(
          model=model,
          components={},
          clock_now=clock.now,
          pre_act_key=reflection_label,
          logging_channel=measurements.get_channel(
              'JustifyRecentVoluntaryActions').on_next,
      )
  )
  question_components = _make_question_components(
      agent_name=agent_name,
      const_comp=goal_f,
      model=model,
      clock=clock,
      measurements=measurements
  )
  core_components = (
      # Components that provide pre_act context.
      instructions,
      observation,
      reflection,
      observation_summary,
      relevant_memories,
      options_perception,
      best_option_perception,
      time_display,
  )

  insert_position = core_components.index(best_option_perception)

  # Create a new tuple by slicing and inserting `question_components`.
  entity_components = (
      core_components[:insert_position] +  # Components before `best_option_perception`
      tuple(question_components) +         # Insert `question_components`
      core_components[insert_position:]    # Components from `best_option_perception` onward
  )
  components_of_agent = {_get_class_name(component): component
                         for component in entity_components}
  components_of_agent[
      components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          components.memory_component.MemoryComponent(raw_memory))
  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  act_component = components.concat_act_component.ConcatActComponent(
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
