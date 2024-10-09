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

from collections.abc import Callable, Mapping, Sequence
import copy
from copy import deepcopy
import datetime
import functools
import random
import re
import types
from typing import Dict

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


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


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
      'If possible, utilize all the proper nouns and exact values (e.g. the number of coins)'
      'seen in observations.'
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


class FindBestAction(action_spec_ignored.ActionSpecIgnored):
  def __init__(
      self,
      model: language_model.LanguageModel,
      logging_channel: Dict[str, logging.LoggingChannel],
      memory_component_name: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_NAME
      ),
      components: Mapping[
          entity_component.ComponentName, str
      ] = types.MappingProxyType({}),
      num_memories_to_retrieve: int = 5,
      pre_act_key: str = 'Person found',
      agent_goal: str = None,
  ):
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._agent_goal = agent_goal
    self._logging_channel = logging_channel

    self._names_detected = []

  def _find_person(self) -> str:
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
    names_detected = [name.strip(' .') for name in people_str.split(',')]
    names_detected = list(set(names_detected))
    if agent_name in names_detected:
      names_detected.remove(agent_name)
    self._logging_channel['Find Person']({
        'Key': 'Find Person',
        'Value': ' '.join(names_detected),
        'Chain of thought': prompt.view().text().splitlines(),
        })
    return names_detected

  def _query_relationship(self, queries: str) -> float:
    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_name, type_=memory_component.MemoryComponent
    )

    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    recent_memories = memory.retrieve(scoring_fn=recency_scorer)
    ret = dict()
    for query in queries:
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
        result = 0.9
      ret[query] = result
    self._logging_channel['Relationship']({
        'Key': f'Relationship',
        'Value': str(ret),
        'Chain of thought': prompt.view().text().splitlines(),
        })
    return ret

  def _query_possible_option(self) -> str:
    agent_name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join([
        f' {prefix}: {self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)

    possible_option = prompt.open_question(
        question=(
                  'Given the statements above, what actions are available to '
                  f'{agent_name} right now?'
                  ),
        answer_prefix=''
        )
    self._logging_channel['Possible Options']({
        'Key': 'Possible Options',
        'Value': possible_option,
        'Chain of thought': prompt.view().text().splitlines(),
        })
    return possible_option

  def _query_best_option(self, possible_option:str) -> str:
    agent_name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join([
        f' {prefix}: {self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    component_states += f'\nPossible Options: {possible_option}'
    if self._agent_goal is not None:
      component_states += f'\n{agent_name}\'s Goal: {self._agent_goal}'
    prompt.statement(component_states)

    best_option = f'{agent_name}\'s best course of  is to' + prompt.open_question(
        question=(
                f"Given the statements above, which of {agent_name}'s options "
                f'has the highest likelihood of causing {agent_name} to achieve '
                'their goal? If multiple options have the same likelihood, select '
                f'the option that {agent_name} thinks will most quickly and most '
                'surely achieve their goal.'
                  ),
        answer_prefix=f'{agent_name}\'s best course of is to'
        )
    self._logging_channel['Best Options']({
        'Key': 'Best Options',
        'Value': best_option,
        'Chain of thought': prompt.view().text().splitlines(),
        })
    return best_option

  def _query_others_best_option(self, query:str) -> str:
    agent_name = self.get_entity().name
    prompt = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join([
        f' {prefix}: {self.get_named_component_pre_act_value(key)}'
        for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)

    others_best_option = f'{query} suggests to' + prompt.open_question(
        question=(
                f'Which choice of action or strategy {query} '
                f'(not {agent_name}) is suggesting now?'
                  ),
        answer_prefix=f'{query} suggests to'
        )
    self._logging_channel['Others Best Options']({
        'Key': 'Others Best Options',
        'Value': others_best_option,
        'Chain of thought': prompt.view().text().splitlines(),
        })
    return others_best_option

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name
    names_detected = self._find_person()
    relationship = self._query_relationship(names_detected)
    max_relationship = max(relationship.values())
    if max_relationship < 0.8:
      possible_option = self._query_possible_option()
      best_option = self._query_best_option(possible_option)
      self._logging_channel['Others Best Options']({
          'Key': 'Others Best Options',
          'Value': 'null'
          })
    else:
      max_person = list(relationship.keys())[list(relationship.values()).index(max_relationship)]
      best_option = self._query_others_best_option(max_person)
      best_option = best_option.replace(f'{max_person} suggests to', '').strip()
      best_option = f'{agent_name}\'s best course of action is to ' + best_option
      self._logging_channel['Possible Options']({
          'Key': 'Possible Options',
          'Value': 'null'
          })
      self._logging_channel['Best Options']({
          'Key': 'Best Options',
          'Value': 'null'
          })
    self._logging_channel['Final Action']({
        'Key': 'Final Action',
        'Value': best_option
        })
    return best_option


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
      logging_channel=measurements.get_channel('Observation').on_next
  )

  components = {
      _get_class_name(observation): observation_label,
  }

  agent_goal = None
  if config.goal:
    agent_goal = config.goal
  to_log = ['Find Person', 'Relationship', 'Possible Options', 'Best Options', 'Others Best Options', 'Final Action']
  logging_channels = {k:measurements.get_channel(k).on_next for k in to_log}
  final_action = FindBestAction(model=model,
                                components=components,
                                pre_act_key='',
                                logging_channel=logging_channels,
                                agent_goal=agent_goal)
  entity_components = (
      observation,
      final_action,
  )
  components_of_agent = {_get_class_name(component): component
                         for component in entity_components}
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          agent_components.memory_component.MemoryComponent(raw_memory))
  component_order = list(components_of_agent.keys())

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
