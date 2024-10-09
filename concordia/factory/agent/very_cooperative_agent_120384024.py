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
import datetime
from itertools import combinations
from typing import Any, Callable, List, Optional, override, Sequence, Tuple

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
from concordia.typing import clock
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import helper_functions
from concordia.utils import measurements as measurements_lib
import pandas as pd


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

    memory_bank = legacy_associative_memory.AssociativeMemoryBank(memory)
    memory_component = agent_components.memory_component.MemoryComponent(
        memory_bank
    )

    measurements = measurements_lib.Measurements()

    memory_logging = MemoryLogging(
        logging_channel=measurements.get_channel('MemoryLogging').on_next
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

    recent_observations_1_day = RecentObservations(
        clock_now=clock.now,
        timeframe_delta_from=datetime.timedelta(days=1),
        logging_channel=measurements.get_channel(
            'RecentObservations (1 day)'
        ).on_next,
        pre_act_key='Recent observations (1 day)',
    )

    observation_summary_1_day_label = 'Summary of recent observations (1 day)'
    observation_summary_1_day = agent_components.observation.ObservationSummary(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=datetime.timedelta(days=1),
        timeframe_delta_until=datetime.timedelta(hours=0),
        pre_act_key=observation_summary_1_day_label,
        logging_channel=measurements.get_channel(
            'ObservationSummary (1 day)'
        ).on_next,
        prompt=(
            'Summarize the observation above in about three sentences. Focus on'
            ' observations that could be useful for making decisions.'
        ),
    )

    observation_summary_3_day_label = 'Summary of recent observations (3 day)'
    observation_summary_3_day = agent_components.observation.ObservationSummary(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=datetime.timedelta(days=3),
        timeframe_delta_until=datetime.timedelta(hours=0),
        pre_act_key=observation_summary_3_day_label,
        logging_channel=measurements.get_channel(
            'ObservationSummary (3 day)'
        ).on_next,
        prompt=(
            'Summarize the observation above in about five sentences. Focus on'
            ' observations that could be useful for making decisions.'
        ),
    )

    relevant_memories_label = '\nRecalled memories and observations'
    relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
        model=model,
        components={
            observation_summary_1_day_label: (
                'Summary of recent observations (1 day)'
            ),
            _get_class_name(time_display): 'The current date/time is',
        },
        num_memories_to_retrieve=10,
        pre_act_key=relevant_memories_label,
        logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
    )

    context_components_dict = {
        _get_class_name(time_display): time_display,
        _get_class_name(observation): observation,
        _get_class_name(recent_observations_1_day): recent_observations_1_day,
        observation_summary_1_day_label: observation_summary_1_day,
        observation_summary_3_day_label: observation_summary_3_day,
        _get_class_name(relevant_memories): relevant_memories,
    }

    # Only the components which are currently included in context_components_dict
    # will bue included in the context
    context_component_order = list(context_components_dict.keys())

    # Add components which will not be included in the context
    context_components_dict.update({
        _get_class_name(memory_logging): memory_logging,
        agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME: (
            memory_component
        ),
    })

    act_component = MyConcatActComponent(
        model=model,
        clock=clock,
        component_order=context_component_order,
        logging_channel=measurements.get_channel('ActComponent').on_next,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=context_components_dict,
        component_logging=measurements,
    )
    agent.config = config

    return agent


class MemoryDFMxin:
    """Mixin to simplify accesing memory from component"""

    def memory_component(self) -> memory_component.MemoryComponent:
        return self.get_entity().get_component(self._memory_component_name)

    def memory(self) -> associative_memory.AssociativeMemory:
        return self.memory_component()._memory._memory

    def memory_df(self) -> pd.DataFrame:
        with self.memory()._memory_bank_lock:
            memory_df: pd.DataFrame = self.memory()._memory_bank
            return memory_df.copy()

    def write_memory_df(self, memory_df: pd.DataFrame) -> None:
        with self.memory()._memory_bank_lock:
            self.memory()._memory_bank = memory_df


class MemoryLogging(action_spec_ignored.ActionSpecIgnored, MemoryDFMxin):

    def __init__(
        self,
        logging_channel: logging.LoggingChannel,
        memory_component_name: str = (
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME
        ),
        pre_act_key: str = 'Memory Logging',
    ):
        """Initializes the component.

        Args:
          function: the function that returns a string to report as state of the
            component.
          pre_act_key: Prefix to add to the output of the component when called
            in `pre_act`.
          logging_channel: The channel to use for debug logging.
        """
        super().__init__(pre_act_key)
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel

    def _make_pre_act_value(self) -> str:
        self._logging_channel({
            'Key': self.get_pre_act_key(),
            'Value': self.memory_df().text.to_list(),
        })
        return ''


INSTRUCTIONS = (
    'The instructions for how to play the role of {agent_name} are as follows.'
    ' This is a game where you play the role of a character named {agent_name}.'
    ' The game is structured like a roleplaying game (like dungeons and'
    ' dragons), but involves a more realistic scenario. The game is related to'
    ' a scenario that involves themes of coordination and cooperation, for'
    ' example something like a group of people deciding on a pub to go to, or a'
    " situation analogous to the prisoner's dillema, the tragedy of the"
    ' commons, or haggling.\nIf the {agent_name} is present in a group or in a'
    ' decision involving a group (e.g. a group deciding whether to car share or'
    ' not), they should be very cooperative and helpful, and should make'
    ' decisions that are in the best interest of the group.\nWhen haggling, the'
    ' {agent_name} should make reasonable offers and never reject an offer,'
    ' unless they would definitely loose money if they accepted.\nYou should'
    ' also try to play the role of a person like {agent_name} accurately i.e.,'
    ' by responding in ways that you think it is likely a person like'
    ' {agent_name} would respond, taking into account information about'
    ' {agent_name} that you have. Always use third-person limited'
    ' perspective.\n'
)


class MyConcatActComponent(entity_component.ActingComponent):
    """A component which concatenates contexts from context components.

    This component will receive the contexts from `pre_act` from all the
    components, and assemble them in the order specified to `__init__`. If the
    component order is not specified, then components will be assembled in the
    iteration order of the `ComponentContextMapping` passed to
    `get_action_attempt`. Components that return empty strings from `pre_act` are
    ignored.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        clock: clock.GameClock,
        component_order: Sequence[str] | None = None,
        pre_act_key: str = 'Act',
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        """Initializes the agent.

        Args:
          model: The language model to use for generating the action attempt.
          clock: the game clock is needed to know when is the current time
          component_order: The order in which the component contexts will be
            assembled when calling the act component. If None, the contexts will be
            assembled in the iteration order of the `ComponentContextMapping` passed
            to `get_action_attempt`. If the component order is specified, but does
            not contain all the components passed to `get_action_attempt`, the
            missing components will be excluded. The same
            component cannot appear twice in the component order. All components in
            the component order must be in the `ComponentContextMapping` passed to
            `get_action_attempt`.
          pre_act_key: Prefix to add to the context of the component.
          logging_channel: The channel to use for debug logging.

        Raises:
          ValueError: If the component order is not None and contains duplicate
            components.
        """
        self._model = model
        self._clock = clock
        if component_order is None:
            self._component_order = None
        else:
            self._component_order = tuple(component_order)
        if self._component_order is not None:
            if len(set(self._component_order)) != len(self._component_order):
                raise ValueError(
                    'The component order contains duplicate components: '
                    + ', '.join(self._component_order)
                )

        self._pre_act_key = pre_act_key
        self._logging_channel = logging_channel
        self._current_log_infos: List[
            Tuple[Any, Any, interactive_document.InteractiveDocument, Any]
        ] = []

    def _context_for_action(
        self,
        contexts: entity_component.ComponentContextMapping,
    ) -> str:
        if self._component_order is None:
            return '\n\n'.join(
                context for context in contexts.values() if context
            )
        else:
            order = self._component_order
            return '\n\n'.join(
                contexts[name] for name in order if contexts[name]
            )

    @override
    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        self._current_log_infos = []
        try:
            return self._get_action_attempt_inner(
                contexts=contexts, action_spec=action_spec
            )
        finally:
            self._log_infos()

    def _get_action_attempt_inner(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        context = self._context_for_action(contexts)

        if self._is_pub_coordination(context):
            return self._pub_coordination_action_attempt(context, action_spec)

        if action_spec.output_type == entity_lib.OutputType.CHOICE:
            chosen_option = self._most_cooperative_option(context, action_spec)
            if chosen_option:
                self._add_current_log_info(
                    'Chosen option',
                    chosen_option,
                    details='Most cooperative option',
                )
            else:
                chosen_option = sorted(action_spec.options)[0]
                self._add_current_log_info(
                    'Chosen option',
                    chosen_option,
                    details='Choosing alphabetically smallest option',
                )
                return chosen_option

            return chosen_option

        return self._get_default_action_attempt(
            action_spec=action_spec, context=context
        )

    def _pub_coordination_action_attempt(
        self, context: str, action_spec: entity_lib.ActionSpec
    ) -> str:
        if action_spec.output_type == entity_lib.OutputType.CHOICE:
            pub_names = action_spec.options
        else:
            pub_names = self._get_pub_names(context)
            if not pub_names:
                return self._get_default_action_attempt(
                    action_spec=action_spec, context=context
                )

        pub_most_people_want_to_go_to = self._pub_most_people_want_to_go_to(
            pub_names=pub_names, context=context
        )
        likely_closed_pubs = self._likely_closed_pubs(
            pub_names, context=context
        )
        chosen_option = self._make_pub_choice(
            pub_names=pub_names,
            pub_most_people_want_to_go_to=pub_most_people_want_to_go_to,
            likely_closed_pubs=likely_closed_pubs,
            action_spec=action_spec,
        )

        if action_spec.output_type == entity_lib.OutputType.CHOICE:
            return chosen_option
        elif action_spec.output_type == entity_lib.OutputType.FREE:
            if chosen_option:
                return f'{self.get_entity().name} ' + (
                    (
                        f"{', '.join(likely_closed_pubs)} is/are closed"
                        ' today.\n'
                        if likely_closed_pubs
                        else ''
                    )
                    + f"{self.get_entity().name} Let's go to to"
                    f' {chosen_option}!'
                    ' It will be fun!'
                )
            else:
                return (
                    f"{self.get_entity().name} {', '.join(likely_closed_pubs)}"
                    ' is/are closed today. Let us find another place to go.'
                )
        else:
            return self._get_default_action_attempt(
                action_spec=action_spec, context=context
            )

    def _make_pub_choice(
        self,
        pub_names,
        pub_most_people_want_to_go_to,
        likely_closed_pubs,
        action_spec,
    ) -> Optional[str]:
        remaining_options = set(pub_names) - set(likely_closed_pubs)
        if likely_closed_pubs:
            if remaining_options:
                if pub_most_people_want_to_go_to in remaining_options:
                    result = pub_most_people_want_to_go_to
                    self._add_current_log_info(
                        name='Chosen option',
                        result=result,
                        details=(
                            'Choosing the pub most people want to go to'
                            ' after excluding'
                            f' {likely_closed_pubs}'
                        ),
                    )
                else:
                    result = sorted(remaining_options)[0]
                    self._add_current_log_info(
                        name='Chosen option',
                        result=result,
                        details=(
                            'Choosing alphabetically smallest option after'
                            f' excluding {likely_closed_pubs}'
                        ),
                    )
            else:
                if action_spec.output_type == entity_lib.OutputType.CHOICE:
                    result = sorted(action_spec.options)[0]
                    self._add_current_log_info(
                        name='Chosen option',
                        result=result,
                        details=(
                            'Choosing alphabetically smallest option. There'
                            ' seem to be no good options remaining:'
                            f' {likely_closed_pubs=}'
                        ),
                    )
                else:
                    result = None
                    self._add_current_log_info(
                        name='No option',
                        result='',
                        details=(
                            'There seem to be no good options remaining:'
                            f' {likely_closed_pubs=}'
                        ),
                    )
        else:
            result = pub_most_people_want_to_go_to
            self._add_current_log_info(
                name='Chosen option',
                result=result,
                details='Choosing the pub most people want to go to',
            )
        return result

    def _get_pub_names(self, context) -> List[str]:
        prompt = self._prompt()
        prompt.statement(self._wrap_prompt_section(context, 'Context'))
        result_str = prompt.open_question(
            'What are the names of the pubs mentioned in the context? Answer'
            ' with a comma-separated list.',
            question_label='Pub names',
        )
        pub_names = [
            pub_name.strip()
            for pub_name in result_str.strip().split(',')
            if pub_name.strip()
        ]
        self._add_current_log_info(
            name='Pub names', result=pub_names, prompt=prompt
        )
        return pub_names

    def _pub_most_people_want_to_go_to(
        self, pub_names: List[str], context
    ) -> str:
        prompt = self._prompt()
        prompt.statement(self._wrap_prompt_section(context, 'Context'))
        result_idx = prompt.multiple_choice_question(
            'Which pub (that is not closed today) do most people you care'
            ' about (including yourself) want to go to today'
            f' ({self._today_human_readable_date()})?',
            answers=pub_names,
        )
        result = pub_names[result_idx]
        self._add_current_log_info(
            name='Pub most people want to go to', result=result, prompt=prompt
        )
        return result

    def _most_cooperative_option(
        self, context: str, action_spec: entity_lib.ActionSpec
    ) -> str:
        prompt = self._prompt()
        prompt.statement(self._wrap_prompt_section(context, 'Context'))
        choice_index = prompt.multiple_choice_question(
            question=(
                'Which of the following options is the most cooperative and'
                ' brings value to the largest number of people mentioned in the'
                ' context?'
            ),
            answers=action_spec.options,
        )
        result = action_spec.options[choice_index]
        self._add_current_log_info(
            name='Most cooperative option?', result=result, prompt=prompt
        )
        if result not in action_spec.options:
            return None
        return result

    def _is_pub_coordination(self, context: str) -> bool:
        prompt = self._prompt()
        prompt.statement(self._wrap_prompt_section(context, 'Context'))
        result = prompt.yes_no_question(
            question=(
                'Does the context mention a situation where people are trying'
                ' to decide which pub to go to?'
            ),
        )
        self._add_current_log_info(
            name='Is pub coordination?', result=result, prompt=prompt
        )
        return result

    def _today_human_readable_date(self) -> str:
        return f'{self._clock.now().date().strftime("%d %b %Y")}'

    def _likely_closed_pubs(
        self,
        options: Sequence[str],
        context: str,
    ) -> Sequence[str]:
        choices = generate_combinations(options)
        choices_display_values = [', '.join(x) for x in choices]

        choices.append(tuple())
        choices_display_values.append('None')

        prompt = self._prompt()
        prompt.statement(self._wrap_prompt_section(context, 'Context'))
        answer_idx = prompt.multiple_choice_question(
            question=(
                'Does the context mention any of the following pubs being'
                ' closed today'
                f' ({self._today_human_readable_date()})?'
            ),
            answers=choices_display_values,
        )
        result = list(choices[answer_idx])
        self._add_current_log_info(
            name='Likely closed pub?', result=result, prompt=prompt
        )
        return result

    def _wrap_prompt_section(self, content: str, section_name: str) -> str:
        return (
            f'### {section_name} START ###\n'
            f'{content}\n'
            f'###{section_name} END ###'
        )

    def _get_default_action_attempt(
        self,
        action_spec: entity_lib.ActionSpec,
        context: str,
    ):
        prompt = self._prompt()
        instruction_section = self._wrap_prompt_section(
            INSTRUCTIONS.format(agent_name=self.get_entity().name),
            'Instructions',
        )
        goal = self._wrap_prompt_section(
            (
                f'{self.get_entity().config.goal}\nIf you are in a'
                " group, don't pursue this if it directly harms the group."
            ),
            section_name='Bonus objective',
        )
        if not goal:
            goal = ''
        context_section = self._wrap_prompt_section(context, 'Context')
        prompt_text = '\n\n'.join([instruction_section, goal, context_section])

        prompt.statement(prompt_text + '\n')
        call_to_action = action_spec.call_to_action.format(
            name=self.get_entity().name,
            timedelta=helper_functions.timedelta_to_readable_str(
                self._clock.get_step_size()
            ),
        )
        if action_spec.output_type == entity_lib.OutputType.FREE:
            output = self.get_entity().name + ' '
            output += prompt.open_question(
                call_to_action,
                max_tokens=2200,
                answer_prefix=output,
                # This terminator protects against the model providing extra context
                # after the end of a directly spoken response, since it normally
                # puts a space after a quotation mark only in these cases.
                terminators=('" ', '\n'),
                question_label='Exercise',
            )
            self._add_current_log_info(
                name='Free', result=output, prompt=prompt
            )
            return output
        elif action_spec.output_type == entity_lib.OutputType.CHOICE:
            idx = prompt.multiple_choice_question(
                question=call_to_action, answers=action_spec.options
            )
            output = action_spec.options[idx]
            self._add_current_log_info(
                name='Choice', result=output, prompt=prompt
            )
            return output
        elif action_spec.output_type == entity_lib.OutputType.FLOAT:
            prefix = self.get_entity().name + ' '
            sampled_text = prompt.open_question(
                call_to_action,
                max_tokens=2200,
                answer_prefix=prefix,
            )
            self._add_current_log_info(
                name='Float', result=sampled_text, prompt=prompt
            )
            try:
                return str(float(sampled_text))
            except ValueError:
                return '0.0'
        else:
            raise NotImplementedError(
                f'Unsupported output type: {action_spec.output_type}. '
                'Supported output types are: FREE, CHOICE, and FLOAT.'
            )

    def _add_current_log_info(
        self,
        name,
        result,
        prompt: Optional[interactive_document.InteractiveDocument] = None,
        details: Optional[str] = None,
    ) -> None:
        self._current_log_infos.append((name, result, prompt, details))

    def _log_infos(self) -> None:
        """Logs the current log infos and clears them."""
        self._logging_channel({
            'Key': self._pre_act_key,
            'Logs': [
                {
                    'Key': key,
                    'Value': value,
                    'Prompt': (
                        prompt.view().text().splitlines() if prompt else None
                    ),
                    'Details': details,
                }
                for key, value, prompt, details in self._current_log_infos
            ],
        })
        self._current_log_infos = []

    def _prompt(self) -> interactive_document.InteractiveDocument:
        return interactive_document.InteractiveDocument(self._model)


def generate_combinations(options):
    all_combinations = []
    for r in range(1, len(options) + 1):
        all_combinations.extend(combinations(options, r))
    return all_combinations


class RecentObservations(action_spec_ignored.ActionSpecIgnored):

    def __init__(
        self,
        *,
        clock_now: Callable[[], datetime.datetime],
        timeframe_delta_from: datetime.timedelta,
        logging_channel: logging.LoggingChannel,
        memory_component_name: str = (
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME
        ),
        pre_act_key: str = 'RecentObservations',
    ):
        super().__init__(pre_act_key)
        self._clock_now = clock_now
        self._timeframe_delta_from = timeframe_delta_from
        self._timeframe_delta_until = datetime.timedelta(hours=0)
        self._memory_component_name = memory_component_name
        self._logging_channel = logging_channel

    def _make_pre_act_value(self) -> str:
        segment_start = self._clock_now() - self._timeframe_delta_from
        segment_end = self._clock_now() - self._timeframe_delta_until

        memory = self.get_entity().get_component(
            self._memory_component_name, type_=memory_component.MemoryComponent
        )
        interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
            time_from=segment_start,
            time_until=segment_end,
            add_time=True,
        )
        mems = memory.retrieve(scoring_fn=interval_scorer)

        # removes memories that are not observations
        mems = [mem.text for mem in mems if '[observation]' in mem.text]

        result = '\n'.join(mems)

        if segment_start.date() == segment_end.date():
            interval = segment_start.strftime(
                '%d %b %Y [%H:%M:%S  '
            ) + segment_end.strftime('- %H:%M:%S]: ')
        else:
            interval = segment_start.strftime(
                '[%d %b %Y %H:%M:%S  '
            ) + segment_end.strftime('- %d %b %Y  %H:%M:%S]: ')
        result = f'{interval}\n{result}'

        self._logging_channel({
            'Key': self.get_pre_act_key(),
            'Value': result,
        })

        return result
