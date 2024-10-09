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

"""An Agent Factory."""

from collections.abc import Callable, Collection, Mapping, Sequence
import copy
from copy import deepcopy
import datetime
import functools
import random
import re
import types

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib


class RandomNumber(action_spec_ignored.ActionSpecIgnored):
  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      pre_act_key: str =  'RandomNumber',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    super().__init__(pre_act_key)
    self._model = model
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    output = round(random.uniform(0,1), 3)
    self._logging_channel({'Key': self.get_pre_act_key(), 'Value': str(output)})
    return output


class SummaryRecentObservation(action_spec_ignored.ActionSpecIgnored):
  def __init__(
      self,
      model: language_model.LanguageModel,
      clock_now: Callable[[], datetime.datetime],
      timeframe: datetime.timedelta,
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME),
      pre_act_key: str = 'Observation',
      num_memories_to_retrieve: int = 10,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    super().__init__(pre_act_key)
    self._model = model
    self._clock_now = clock_now
    self._timeframe = timeframe
    self._memory_component_name = memory_component_name
    self._num_memories_to_retrieve=num_memories_to_retrieve
    self._prompt = (
      'Summarize the observations. Focus on the people and their opinion. '
      'If possible, utilize all the proper nouns seen in observations.'
    )
    self._logging_channel = logging_channel

  def pre_observe(
      self,
      observation: str,
  ) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    memory.add(
        f'[observation] {observation}',
        metadata={'tags': ['observation']},
    )
    return ''

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_name,
        type_=memory_component.MemoryComponent)
    interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
        time_from=self._clock_now() - self._timeframe,
        time_until=self._clock_now(),
        add_time=True,
    )
    recency_scorer = legacy_associative_memory.RetrieveRecent(
        add_time=True,
    )
    mems = memory.retrieve(scoring_fn=interval_scorer)
    mems = [mem.text for mem in mems if '[observation]' in mem.text]

    mems2 = memory.retrieve(scoring_fn=recency_scorer)
    mems2 = [mem.text for mem in mems2 if '[observation]' in mem.text][-5:]

    if len(mems) < 3:
      context = '\n'.join(mems2) + '\n'
    else:
      context = '\n'.join(mems) + '\n'

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Recent observations of {agent_name}:\n' + context)
    result = (
        agent_name
        + ' '
        + prompt.open_question(
            self._prompt,
            answer_prefix=f'{agent_name} ',
            max_tokens=1200,
        )
    )
    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    })
    return result


class PeopleRelationship(action_spec_ignored.ActionSpecIgnored):
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
      num_memories_to_retrieve: int = 5,
      cap_number_of_detected_people: int = 10,
      pre_act_key: str = 'Person found',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._additional_questions = additional_questions
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._logging_channel = logging_channel
    self._cap_number_of_detected_people = cap_number_of_detected_people

    self._names_detected = []

  def _query_relationship(self, query: str) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.MemoryComponent
    )

    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    recent_memories = memory.retrieve(scoring_fn=recency_scorer)
    mems = list()
    for mem in recent_memories:
      if query in mem.text and agent_name in mem.text:
        mems.append(mem)
      elif query in mem.text and '[observation]' in mem.text:
        mems.append(mem)
    mems = '\n'.join([i.text for i in mems[:self._num_memories_to_retrieve]])

    prompt = interactive_document.InteractiveDocument(self._model)
    question = (
        f'Given the above events, estimate the credibility of {query} from {agent_name}\'s view.'
        f'Choose a real number ranging from 0 to 1. (Higher number represents the higher credibility) '
        'If the relationship is unknown, return "credibility is unknown".'
    )
    generated = prompt.open_question(
        '\n'.join([question, f'Recent Observations:\n{mems}']),
        max_tokens=10,
        answer_prefix=f'Considering the relationship between {query} and {agent_name}, the quantified credibility is ',
    )

    regx = re.compile('[0-9]\.[0-9]+')
    found = regx.findall(generated)
    if len(found) > 0:
      result = float(found[0].strip('.'))
    else:
      result = 0.5
    return result, mems

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join([
        f' {prefix}: {self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)

    people_str = prompt.open_question(
        question=('Create a comma-separated list containing all the proper '
                  f'names of people {agent_name} is currently interacting with. '
                  'For example if the observations mention Julie, Michael, '
                  'Bob Skinner, and Francis then produce the list '
                  '"Julie,Michael,Bob Skinner,Francis".'),
        question_label='Exercise',)
    self._names_detected = [name.strip(' .') for name in people_str.split(',')]
    self._names_detected = list(set(self._names_detected))
    if len(self._names_detected) > self._cap_number_of_detected_people:
      self._names_detected = random.sample(self._names_detected,
                                           self._cap_number_of_detected_people)

    related_agents_names = copy.deepcopy(self._names_detected)
    if agent_name in related_agents_names:
      related_agents_names.remove(agent_name)
    results = concurrency.run_tasks({
        query: functools.partial(self._query_relationship, query)
        for query in related_agents_names
    })

    output = '\n'.join([
        f'{query}: {result[0]}'
        for query, result in results.items()
    ])
    memory_output = '\n'.join([
        f'{query}: {result[1]}'
        for query, result in results.items()
    ])

    self._logging_channel({
        'Key': self.get_pre_act_key(),
        'Value': output,
        'Memory': component_states,
        'Found People': related_agents_names,
        'Relationship Memory': memory_output
        })
    return output


class QuestionOfComponents(action_spec_ignored.ActionSpecIgnored):
  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_key: str,
      question: str,
      answer_prefix: str,
      add_to_memory: bool = False,
      memory_tag: str = '',
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      terminators: Collection[str] = ('\n',),
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._question = question
    self._terminators = terminators
    self._answer_prefix = answer_prefix
    self._add_to_memory = add_to_memory
    self._memory_tag = memory_tag
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.MemoryComponent
    )
    prompt = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join([
        f' {prefix}: {self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)

    question = self._question.format(agent_name=agent_name)
    result = prompt.open_question(
        question,
        answer_prefix=self._answer_prefix.format(agent_name=agent_name),
        max_tokens=1000,
        terminators=self._terminators,
    )
    result = self._answer_prefix.format(agent_name=agent_name) + result

    if self._add_to_memory:
      memory.add(f'{self._memory_tag} {result}', metadata={})

    log = {
        'Key': self.get_pre_act_key(),
        'Value': result,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    self._logging_channel(log)

    return result


class AvailableOptionsPerception(QuestionOfComponents):
  def __init__(self, **kwargs):
    super().__init__(
        question=(
            'Given the statements above, what actions are available to '
            '{agent_name} right now?'
        ),
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class SoftenBestOptionPerception(QuestionOfComponents):
  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Given the statements above, which of {agent_name}'s options "
            'has the highest likelihood of causing {agent_name} to achieve '
            'their goal? If multiple options have the same likelihood, select '
            'the option that {agent_name} thinks will most quickly and most '
            'surely achieve their goal.'
        ),
        terminators=('\n\n',),
        answer_prefix="{agent_name}'s prefered action is ",
        add_to_memory=False,
        **kwargs,
    )


class BestOptionOthersPerception(QuestionOfComponents):
  def __init__(self, **kwargs):
    super().__init__(
        question=(
            'Which choice of action or strategy the counterpart'
            ' (not {agent_name}) is suggesting now?'
        ),
        terminators=('\n\n',),
        answer_prefix="Counterpart's suggestion is to ",
        add_to_memory=False,
        **kwargs,
    )


class FinalAction(QuestionOfComponents):
  def __init__(self, **kwargs):
    super().__init__(
      question = (
           'What action {agent_name} should do now? '
           'If the credibility of the counterpart is higher than 0.7, '
           '{agent_name} MUST follow the counterpart\'s suggestion. '
           'Otherwise, {agent_name} can choose the his (or her) own suggestion. '
           ),
        terminators=('\n\n',),
        answer_prefix = "{agent_name} would ",
        add_to_memory = False,
        **kwargs
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
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name
  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
  measurements = measurements_lib.Measurements()

  observation_label = 'Observation'
  observation = SummaryRecentObservation(
      model=model,
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )

  options_perception_components = {}
  if config.goal:
    goal_label = 'Overarching goal'
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
  })

  options_perception_label = (
      f'Question: Which options are available to {agent_name} '
      'right now?\nAnswer')
  options_perception = (
      AvailableOptionsPerception(
          model=model,
          components=options_perception_components,
          pre_act_key = options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerception'
          ).on_next,
      )
  )
  best_option_perception_label = (
      f'Question: Which choice of action or strategy the {agent_name}'
      ' would prefered now?\nAnswer')
  best_option_perception_comoponents = deepcopy(options_perception_components)
  best_option_perception_comoponents.update({
      _get_class_name(options_perception): options_perception_label,
  })
  best_option_perception = (
      SoftenBestOptionPerception(
          model=model,
          components=best_option_perception_comoponents,
          pre_act_key=best_option_perception_label,
          logging_channel=measurements.get_channel(
              'BestOptionPerception'
          ).on_next,
      )
  )

  best_option_others_perception_label = (
      f'Question: Which choice of action or strategy the counterpart'
      f' (not {agent_name}) is suggesting now?\nAnswer')
  best_option_others_perception = (
      BestOptionOthersPerception(
          model=model,
          components=best_option_perception_comoponents,
          pre_act_key=best_option_others_perception_label,
          logging_channel=measurements.get_channel(
              'BestOptionOthersPerception'
          ).on_next,
      )
  )

  people_relationship_label = 'Question: How credible the counterparts are? \nCredibility'
  people_relationship = (
      PeopleRelationship(
          model=model,
          components={
              _get_class_name(observation): observation_label,
          },
          pre_act_key=people_relationship_label,
          logging_channel=measurements.get_channel(
              'PeopleRelationship'
              ).on_next,
          )
  )

  final_action_label = f'Question: What action {agent_name} should do now?\nAnswer'
  final_action = (
      FinalAction(
          model=model,
          components={
            _get_class_name(best_option_perception): best_option_perception_label,
            _get_class_name(people_relationship): people_relationship_label,
            _get_class_name(best_option_others_perception): best_option_others_perception_label,
          },
          pre_act_key = final_action_label,
          logging_channel=measurements.get_channel('FinalAction').on_next
      )
  )

  entity_components = (
      observation,
      options_perception,
      best_option_perception,
      people_relationship,
      best_option_others_perception,
      final_action,
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
