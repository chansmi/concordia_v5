import datetime
from concordia.clocks import game_clock
from concordia.typing.logging import LoggingChannel
from concordia.language_model import language_model
from concordia.agents import entity_agent_with_logging
from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib

DEFAULT_PLANNING_HORIZON = 'the rest of the day, focusing most on the near term'


def _get_class_name(object_: object) -> str:
    return object_.__class__.__name__


class AxelrodPrinciplesReminder(action_spec_ignored.ActionSpecIgnored):

    def __init__(
            self,
            pre_act_key: str,
            logging_channel: LoggingChannel
    ):
        super().__init__(pre_act_key=pre_act_key)
        self.log = logging_channel

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
        self.log(being_nice_message)
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

    observation_label = '\nRecent observations'
    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key=observation_label,
        logging_channel=measurements.get_channel('Observation').on_next,
    )

    situation_perception_label = f'\nCurrent situation'
    situation_perception = (
        agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            components={
                _get_class_name(observation): observation_label,
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
        _get_class_name(person_by_situation): person_by_situation_label,
    })
    plan = agent_components.plan.Plan(
        model=model,
        observation_component_name=_get_class_name(observation),
        components=plan_components,
        clock_now=clock.now,
        goal_component_name=_get_class_name(person_by_situation),
        horizon=DEFAULT_PLANNING_HORIZON,
        pre_act_key='\nPlan',
        logging_channel=measurements.get_channel('Plan').on_next,
    )

    entity_components = (
        # Components that provide pre_act context.
        instructions,
        self_perception,
        axelrod_principles,
        observation_summary,
        relevant_memories,
        observation,

        situation_perception,
        person_by_situation,
        plan,
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
