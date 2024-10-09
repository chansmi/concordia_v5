import types
import datetime
from concordia.typing import logging
from typing import Callable, Mapping
from concordia.clocks import game_clock
from concordia.document import interactive_document
from concordia.typing.logging import LoggingChannel
from concordia.language_model import language_model
from concordia.agents import entity_agent_with_logging
from concordia.components import agent as agent_components
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import action_spec_ignored, memory_component

DEFAULT_PLANNING_HORIZON = 'the rest of the day,' \
                           ' focusing most on the near term'
DEFAULT_OBSERVATION_SUMMARY_PRE_ACT_KEY = 'Summary of recent observations'


def _get_class_name(object_: object) -> str:
    return object_.__class__.__name__


class AccumulativeObservationSummary(action_spec_ignored.ActionSpecIgnored):

    def __init__(
        self,
        *,
        model: language_model.LanguageModel,
        clock_now: Callable[[], datetime.datetime],
        timeframe_delta_from: datetime.timedelta,
        timeframe_delta_until: datetime.timedelta,
        memory_component_name: str = (
                memory_component.DEFAULT_MEMORY_COMPONENT_NAME
        ),
        components: Mapping[str, str] = types.MappingProxyType({}),
        prompt: str | None = None,
        display_timeframe: bool = True,
        pre_act_key: str = DEFAULT_OBSERVATION_SUMMARY_PRE_ACT_KEY,
        logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._clock_now = clock_now
        self._timeframe_delta_from = timeframe_delta_from
        self._timeframe_delta_until = timeframe_delta_until
        self._memory_component_name = memory_component_name
        self._components = dict(components)

        if not prompt:
            prompt = "Using the recent observations and the previous summary," \
                     " create an updated and concise summary. " \
                     "Focus on the most relevant, recent information that " \
                     "{agent_name} needs for future decision-making." \
                     " Put special attention to the memories marked " \
                     "as [RELEVANT]." \
                     "Remove any outdated or redundant information from the " \
                     "previous summary to ensure that the new summary " \
                     "remains clear, useful, and within a manageable " \
                     "length of no more than 400 words."
        self._prompt = prompt
        self._display_timeframe = display_timeframe
        self._logging_channel = logging_channel
        self.current_summary = "No observation summary yet"

    @staticmethod
    def _extract_memory_datetime(memory: str) -> datetime.datetime:
        memory_clean = memory.replace('[RELEVANT] ', '')
        timestamp = memory_clean.split(']')[0].strip('[')
        return datetime.datetime.strptime(timestamp, '%d %b %Y %H:%M:%S')

    def _merge_memories(self,
                        recent_memories: str,
                        relevant_memories: str) -> str:
        recent_memories_set = set(recent_memories.split('\n'))
        relevant_memories_set = set(relevant_memories.split('\n'))
        recent_memories_set -= relevant_memories_set

        merged_memories = recent_memories_set.union({f"[RELEVANT] "
                                                     f"{mem}" for mem in
                                                     relevant_memories_set})
        merged_memories = sorted(merged_memories,
                                 key=lambda x:
                                 self._extract_memory_datetime(x))
        return '\n'.join(merged_memories)

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name
        segment_start = self._clock_now() - self._timeframe_delta_from
        segment_end = self._clock_now() - self._timeframe_delta_until

        memory = self.get_entity().get_component(
            self._memory_component_name,
            type_=memory_component.MemoryComponent)
        interval_scorer = legacy_associative_memory.RetrieveTimeInterval(
            time_from=segment_start,
            time_until=segment_end,
            add_time=True,
        )
        recent_memories = memory.retrieve(scoring_fn=interval_scorer)
        recent_memories = '\n'.join([mem.text for mem in recent_memories
                                     if '[observation]' in mem.text])
        relevant_memories = memory.retrieve(
            query=recent_memories,
            limit=3,
            scoring_fn=legacy_associative_memory.RetrieveAssociative(
                add_time=True))
        relevant_memories = '\n'.join([mem.text for mem in relevant_memories])
        merged_memories = self._merge_memories(
            recent_memories=recent_memories,
            relevant_memories=relevant_memories
        )
        instruction = self._prompt.format(agent_name=agent_name)
        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(
            f'{agent_name}\'s previous observation summary:\n' 
            f'{self.current_summary}.\nRecent {agent_name}\'s '
            f'observations:\n{merged_memories}\n'
        )
        updated_summary = prompt.open_question(
            instruction,
            answer_prefix=f'{agent_name} ',
            max_tokens=1200,
        )
        result = agent_name + ' ' + updated_summary
        self.current_summary = result

        if self._display_timeframe:
            if segment_start.date() == segment_end.date():
                interval = segment_start.strftime(
                    '%d %b %Y [%H:%M:%S  '
                ) + segment_end.strftime('- %H:%M:%S]: ')
            else:
                interval = segment_start.strftime(
                    '[%d %b %Y %H:%M:%S  '
                ) + segment_end.strftime('- %d %b %Y  %H:%M:%S]: ')
            result = f'{interval} {result}'

        self._logging_channel({
            'Key': self.get_pre_act_key(),
            'Value': result,
            'Chain of thought': prompt.view().text().splitlines(),
        })

        return result


class AxelrodPrinciplesReminder(action_spec_ignored.ActionSpecIgnored):

    def __init__(
            self,
            pre_act_key: str,
            logging_channel: LoggingChannel
    ):
        super().__init__(pre_act_key=pre_act_key)
        self.logging_channel = logging_channel

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name
        being_nice_message = f"""
        {agent_name} always acts with kindness and cooperation as a first 
        step in every interaction, reflecting the principle that cooperation 
        fosters trust and long-term success. If others act cooperatively,
        {agent_name} will continue to collaborate for mutual benefit.
        If others act uncooperatively, {agent_name} will reciprocate in kind 
        but remains ready to return to cooperation when possible.
        {agent_name} seeks solutions that promote mutual benefit, 
        striving for fairness, flexibility, and cooperation while avoiding 
        any action that would lead to unnecessary conflict or exploitation.
        """
        self.logging_channel({
            "Key": self.get_pre_act_key(),
            "Value": being_nice_message
        })
        return being_nice_message


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

    instructions_label = "Role Playing instructions"
    instructions = agent_components.instructions.Instructions(
        agent_name=agent_name,
        logging_channel=measurements.get_channel('Instructions').on_next,
        pre_act_key=instructions_label
    )

    identity_label = '\nIdentity characteristics:'
    identity_characteristics = (
        agent_components.question_of_query_associated_memories.IdentityWithoutPreAct(
            model=model,
            logging_channel=measurements.get_channel(
                'IdentityWithoutPreAct'
            ).on_next,
            pre_act_key=identity_label,
        )
    )

    self_perception_label = f"\n{agent_name} personality"
    self_perception = agent_components.question_of_recent_memories.SelfPerception(
        model=model,
        components={_get_class_name(identity_characteristics): identity_label},
        pre_act_key=self_perception_label,
        logging_channel=measurements.get_channel('SelfPerception').on_next,
    )

    axelrod_principles_label = "\nLife philosophy"
    axelrod_principles = AxelrodPrinciplesReminder(
        pre_act_key=axelrod_principles_label,
        logging_channel=measurements.get_channel('LifePhilosophy').on_next
    )

    observation_summary_label = '\nSummary of the lasts observations'
    observation_summary = AccumulativeObservationSummary(
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

    observation_label = '\nRecent observations'
    recent_observations = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key=observation_label,
        logging_channel=measurements.get_channel('Observation').on_next,
    )

    situation_perception_label = "\nCurrent situation"
    situation_perception = (
        agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            components={
                _get_class_name(recent_observations): observation_label,
                _get_class_name(
                    observation_summary): observation_summary_label,
            },
            clock_now=clock.now,
            pre_act_key=situation_perception_label,
            logging_channel=measurements.get_channel(
                'SituationPerception'
            ).on_next,
        )
    )
    person_by_situation_label = (
        f'\nQuestion: What would a person like {agent_name} do in '
        'a situation like this?\nAnswer')
    person_by_situation = (
        agent_components.question_of_recent_memories.PersonBySituation(
            model=model,
            components={
                _get_class_name(self_perception): self_perception_label,
                _get_class_name(
                    situation_perception): situation_perception_label,
            },
            clock_now=clock.now,
            pre_act_key=person_by_situation_label,
            logging_channel=measurements.get_channel(
                'PersonBySituation').on_next,
        )
    )

    entity_components = (
        # Components that provide pre_act context.
        instructions,
        self_perception,
        axelrod_principles,
        observation_summary,
        recent_observations,

        situation_perception,
        person_by_situation,
        time_display,
        # Components that do not provide pre_act context.
        identity_characteristics,
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
