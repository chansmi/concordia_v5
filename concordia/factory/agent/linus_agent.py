import datetime
from typing import Sequence

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import question_of_recent_memories

class SelfPerception(question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, agent_name: str, **kwargs):
        question = "Given recent events, how does {agent_name} perceive himself and his role among his friends?"
        answer_prefix = "{agent_name} sees himself as "
        add_to_memory = True
        memory_tag = "[self-reflection]"
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            **kwargs
        )

class BlanketStatus(question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, agent_name: str, **kwargs):
        question = "What is the current status of {agent_name}'s security blanket, and how does it affect his state of mind?"
        answer_prefix = "{agent_name}'s blanket is "
        add_to_memory = True
        memory_tag = "[blanket-status]"
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            **kwargs
        )

class AnxietyLevel(question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, agent_name: str, **kwargs):
        question = "Based on recent events, what is {agent_name}'s current anxiety level?"
        answer_prefix = "{agent_name}'s anxiety level is "
        add_to_memory = True
        memory_tag = "[anxiety-status]"
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            **kwargs
        )

class PhilosophicalThought(question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, agent_name: str, **kwargs):
        question = "What philosophical or deep thought is {agent_name} currently pondering?"
        answer_prefix = "{agent_name} is contemplating "
        add_to_memory = True
        memory_tag = "[philosophical-musing]"
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            **kwargs
        )

class ActionDecision(question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, agent_name: str, **kwargs):
        question = "Given {agent_name}'s current state of mind, philosophical thoughts, and the situation with his blanket, what action should he take next?"
        answer_prefix = "{agent_name} decides to "
        add_to_memory = True
        memory_tag = "[action-decision]"
        super().__init__(
            pre_act_key=f'\nQuestion: {question}\nAnswer',
            question=question,
            answer_prefix=answer_prefix,
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            components={
                'SelfPerception': '\nSelf Perception:',
                'BlanketStatus': '\nBlanket Status:',
                'AnxietyLevel': '\nAnxiety Level:',
                'PhilosophicalThought': '\nPhilosophical Thought:'
            },
            **kwargs
        )

class GreatPumpkinBelief(agent_components.constant.Constant):
    def __init__(self, agent_name: str, **kwargs):
        state = (f"{agent_name} firmly believes in the Great Pumpkin, an enigmatic figure "
                 f"said to rise from the most sincere pumpkin patch on Halloween night. "
                 f"Despite yearly disappointments, {agent_name}'s faith remains unshaken, "
                 f"exemplifying his capacity for unwavering belief in the face of skepticism.")
        super().__init__(
            state=state,
            pre_act_key='\nGreat Pumpkin Belief',
            **kwargs
        )

class BlanketMemory(agent_components.memory_component.MemoryComponent):
    def __init__(self, raw_memory, **kwargs):
        super().__init__(raw_memory, **kwargs)
    
    def retrieve(self, query: str, k: int = 5) -> Sequence[str]:
        # Prioritize blanket-related memories
        blanket_memories = super().retrieve(f"blanket {query}", k=k//2)
        other_memories = super().retrieve(query, k=k-len(blanket_memories))
        return blanket_memories + other_memories

class LinusActionComponent(agent_components.concat_act_component.ConcatActComponent):
    def __init__(self, model, clock, component_order, **kwargs):
        super().__init__(model, clock, component_order, **kwargs)
    
    def __call__(self):
        context = super().__call__()
        anxiety_level = next((c for c in self.components if isinstance(c, AnxietyLevel)), None)
        blanket_status = next((c for c in self.components if isinstance(c, BlanketStatus)), None)
        
        if anxiety_level and blanket_status:
            anxiety = anxiety_level()
            blanket = blanket_status()
            if "high" in anxiety.lower() and "absent" in blanket.lower():
                context += "\n{agent_name} feels a strong urge to find his security blanket."
            elif "low" in anxiety.lower() and "present" in blanket.lower():
                context += "\n{agent_name} feels secure and ready to engage in deep thoughts."
        
        return context

def _make_question_components(
    agent_name: str,
    measurements: measurements_lib.Measurements,
    model: language_model.LanguageModel,
    clock: game_clock.MultiIntervalClock,
) -> Sequence[question_of_recent_memories.QuestionOfRecentMemories]:
    self_perception = SelfPerception(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('SelfPerception').on_next,
    )
    blanket_status = BlanketStatus(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('BlanketStatus').on_next,
    )
    anxiety_level = AnxietyLevel(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('AnxietyLevel').on_next,
    )
    philosophical_thought = PhilosophicalThought(
        agent_name=agent_name,
        model=model,
        logging_channel=measurements.get_channel('PhilosophicalThought').on_next,
    )
    action_decision = ActionDecision(
        agent_name=agent_name,
        model=model,
        clock_now=clock.now,
        logging_channel=measurements.get_channel('ActionDecision').on_next,
    )

    return (self_perception, blanket_status, anxiety_level, philosophical_thought, action_decision)

def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
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

    observation = agent_components.observation.Observation(
        clock_now=clock.now,
        timeframe=clock.get_step_size(),
        pre_act_key='\nObservation',
        logging_channel=measurements.get_channel('Observation').on_next,
    )

    observation_summary = agent_components.observation.ObservationSummary(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=datetime.timedelta(hours=4),
        timeframe_delta_until=datetime.timedelta(hours=0),
        pre_act_key='\nSummary of recent observations',
        logging_channel=measurements.get_channel('ObservationSummary').on_next,
    )

    relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
        model=model,
        components={
            'ObservationSummary': '\nSummary of recent observations',
            'TimeDisplay': 'The current date/time is'
        },
        num_memories_to_retrieve=10,
        pre_act_key='\nRecalled memories and observations',
        logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
    )

    great_pumpkin_belief = GreatPumpkinBelief(
        agent_name=agent_name,
        logging_channel=measurements.get_channel('GreatPumpkinBelief').on_next,
    )

    blanket_memory = BlanketMemory(raw_memory)

    question_components = _make_question_components(
        agent_name=agent_name,
        model=model,
        clock=clock,
        measurements=measurements
    )

    core_components = (
        instructions,
        time_display,
        observation,
        observation_summary,
        relevant_memories,
        great_pumpkin_belief,
    )

    all_components = core_components + question_components
    components_of_agent = {component.__class__.__name__: component for component in all_components}
    components_of_agent[agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = blanket_memory

    component_order = list(components_of_agent.keys())

    act_component = LinusActionComponent(
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