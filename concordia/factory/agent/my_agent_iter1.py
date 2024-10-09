
import datetime
from typing import Sequence, Callable

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib

from concordia.components.agent import plan
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document

def _get_class_name(object_: object) -> str:
    return object_.__class__.__name__

# Define scoring functions for memory retrieval
recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
associative_scorer = legacy_associative_memory.RetrieveAssociative()

class TheoryOfMindModule(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nTheory of Mind',
            question='Based on recent interactions, what are the beliefs, intentions, and potential actions of other agents?',
            answer_prefix='Other agents might ',
            add_to_memory=True,
            memory_tag='[Theory of Mind]',
            components={
                'ObservationSummary': '\nSummary of recent observations',
                'DynamicRelationships': '\nRelationships with other agents',
            },
            **kwargs
        )

class ScenarioAnalysis(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nScenario Analysis',
            question='What are the key elements of the current scenario, including resources, potential conflicts, and cooperation opportunities? Classify the scenario type.',
            answer_prefix='The scenario involves ',
            add_to_memory=True,
            memory_tag='[Scenario Analysis]',
            components={
                'ObservationSummary': '\nSummary of recent observations',
            },
            **kwargs
        )

class AgentModeling(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nAgent Modeling',
            question='Based on the Theory of Mind and Scenario Analysis, what are other agents\' potential strategies and goals? Am I in resident mode or visitor mode? Assess the likelihood of cooperation or defection from other agents.',
            answer_prefix='Other agents are likely to ',
            add_to_memory=True,
            memory_tag='[Agent Modeling]',
            components={
                'TheoryOfMindModule': '\nTheory of Mind',
                'ScenarioAnalysis': '\nScenario Analysis',
                'SelfPerception': '\nSelf Perception',
            },
            **kwargs
        )

class StrategyGeneration(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nStrategy Generation',
            question='Considering the Scenario Analysis and Agent Modeling, generate multiple potential strategies. Evaluate short-term and long-term consequences of actions using game theory concepts.',
            answer_prefix='Possible strategies include ',
            add_to_memory=True,
            memory_tag='[Strategy Generation]',
            components={
                'ScenarioAnalysis': '\nScenario Analysis',
                'AgentModeling': '\nAgent Modeling',
            },
            **kwargs
        )

class CooperativeAIIntegration(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nCooperative AI Integration',
            question='Identify opportunities for aligning goals with other agents and develop proposals for mutually beneficial solutions. How can I build trust and reciprocity?',
            answer_prefix='To cooperate effectively, I can ',
            add_to_memory=True,
            memory_tag='[Cooperative AI Integration]',
            components={
                'StrategyGeneration': '\nStrategy Generation',
                'AgentModeling': '\nAgent Modeling',
            },
            **kwargs
        )

class ValueAlignmentModule(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nValue Alignment',
            question='Considering ethical principles, how should I act to balance individual and collective well-being?',
            answer_prefix='Ethically, I should ',
            add_to_memory=True,
            memory_tag='[Value Alignment]',
            components={
                'StrategyGeneration': '\nStrategy Generation',
            },
            **kwargs
        )

class DecisionEvaluation(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nDecision Evaluation',
            question='Evaluate the potential strategies using a utility function that balances individual and collective benefits, and consider ethical implications using the moral framework. Which strategy has the highest expected utility?',
            answer_prefix='The best strategy is to ',
            add_to_memory=True,
            memory_tag='[Decision Evaluation]',
            components={
                'StrategyGeneration': '\nStrategy Generation',
                'ValueAlignmentModule': '\nValue Alignment',
                'RobustnessMeasures': '\nRobustness Measures',
                'AdaptiveLearning': '\nAdaptive Learning',
            },
            **kwargs
        )

class ActionSelection(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nAction Selection',
            question='Based on the Decision Evaluation, what action should I take? Prepare a justification for this action using chain of thought reasoning.',
            answer_prefix='I will ',
            add_to_memory=True,
            memory_tag='[Action Selection]',
            components={
                'DecisionEvaluation': '\nDecision Evaluation',
            },
            **kwargs
        )

class AdaptiveLearning(agent_components.action_spec_ignored.ActionSpecIgnored):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(pre_act_key='\nAdaptive Learning')
        self._model = model

    def _make_pre_act_value(self) -> str:
        memory_component = self.get_entity().get_component(
            agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
            type_=agent_components.memory_component.MemoryComponent
        )
        recent_memories = memory_component.retrieve(scoring_fn=recency_scorer, limit=50)

        prompt = "Review the following recent memories and summarize key learnings:\n"
        for mem in recent_memories:
            prompt += f"- {mem.text}\n"
        prompt += "\nWhat are the most important lessons or patterns from these experiences?"

        learnings = self._model.sample_text(prompt, max_tokens=500)
        memory_component.add(f"[Adaptive Learning] {learnings}", metadata={})
        return learnings

class CommunicationModule(agent_components.action_spec_ignored.ActionSpecIgnored):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(pre_act_key='\nCommunication')
        self._model = model

    def _make_pre_act_value(self) -> str:
        return ""  # No pre-act value

    def formulate_message(self, recipient: str, intent: str) -> str:
        prompt = f"Formulate a message to {recipient} with the following intent: {intent}\n"
        prompt += "Consider the recipient's perspective and our relationship. The message should be clear, polite, and aligned with our goals."

        message = self._model.sample_text(prompt, max_tokens=200)
        return message

class RobustnessMeasures(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nRobustness Measures',
            question='Are there signs of adversarial or deceptive behavior from other agents? If so, how should I adjust my strategy?',
            answer_prefix='I should be cautious because ',
            add_to_memory=True,
            memory_tag='[Robustness Measures]',
            components={
                'TheoryOfMindModule': '\nTheory of Mind',
                'ObservationSummary': '\nSummary of recent observations',
            },
            **kwargs
        )

class ScenarioAdaptation(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(
            model=model,
            pre_act_key='\nScenario Adaptation',
            question='What are the scenario-specific norms and conventions? How can I adapt my behavior to match local customs in visitor mode or introduce beneficial norms in resident mode?',
            answer_prefix='To adapt to the scenario, I should ',
            add_to_memory=True,
            memory_tag='[Scenario Adaptation]',
            components={
                'ScenarioAnalysis': '\nScenario Analysis',
            },
            **kwargs
        )

class PerformanceOptimizer(agent_components.action_spec_ignored.ActionSpecIgnored):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(pre_act_key='\nPerformance Optimization')
        self._model = model

    def _make_pre_act_value(self) -> str:
        memory_component = self.get_entity().get_component(
            agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
            type_=agent_components.memory_component.MemoryComponent
        )
        recent_decisions = memory_component.retrieve(query="[Decision Evaluation]", scoring_fn=associative_scorer, limit=20)

        prompt = "Review the following recent decisions and their outcomes:\n"
        for decision in recent_decisions:
            prompt += f"- {decision.text}\n"
        prompt += "\nBased on these decisions, suggest improvements to the decision-making process. Consider efficiency, effectiveness, and alignment with goals."

        optimization_suggestions = self._model.sample_text(prompt, max_tokens=500)
        memory_component.add(f"[Performance Optimization] {optimization_suggestions}", metadata={})
        return optimization_suggestions

class DynamicRelationships(agent_components.action_spec_ignored.ActionSpecIgnored):
    def __init__(self, model: language_model.LanguageModel, **kwargs):
        super().__init__(pre_act_key='\nRelationships with other agents')
        self._model = model
        self._known_agents = set()
        self._logging_channel = kwargs.get('logging_channel')

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name
        memory_component = self.get_entity().get_component(
            agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
            type_=agent_components.memory_component.MemoryComponent
        )

        self._update_known_agents(memory_component)

        relationships = []
        for other_agent in self._known_agents:
            if other_agent != agent_name:
                relationship = self._query_relationship(agent_name, other_agent, memory_component)
                relationships.append(f"{other_agent}: {relationship}")

        result = "\n".join(relationships)

        if self._logging_channel:
            self._logging_channel({
                'Key': self.get_pre_act_key(),
                'Value': result
            })

        return result

    def _update_known_agents(self, memory_component):
        recent_memories = memory_component.retrieve(scoring_fn=recency_scorer, limit=50)
        for memory in recent_memories:
            words = memory.text.split()
            for word in words:
                if word.istitle() and len(word) > 1:
                    self._known_agents.add(word)

    def _query_relationship(self, agent_name: str, other_agent: str, memory_component) -> str:
        query = f"{agent_name} and {other_agent}"
        relevant_memories = memory_component.retrieve(query=query, scoring_fn=associative_scorer, limit=10)
        
        prompt = f"Based on these interactions between {agent_name} and {other_agent}, summarize their relationship:\n"
        for memory in relevant_memories:
            prompt += f"- {memory.text}\n"
        prompt += f"\nDescribe the relationship between {agent_name} and {other_agent}:"

        relationship_summary = self._model.sample_text(prompt, max_tokens=100)
        return relationship_summary

class SelfPerception(action_spec_ignored.ActionSpecIgnored):
    def __init__(
        self,
        model: language_model.LanguageModel,
        clock_now: Callable[[], datetime.datetime],
        num_memories_to_retrieve: int = 100,
        pre_act_key: str = '\nSelf Perception',
        logging_channel: Callable[[dict], None] = lambda x: None,
    ):
        super().__init__(pre_act_key)
        self._model = model
        self._clock_now = clock_now
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._logging_channel = logging_channel

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name
        memory = self.get_entity().get_component(
            memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
            type_=memory_component.MemoryComponent
        )

        mems = '\n'.join([
            mem.text
            for mem in memory.retrieve(
                scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
            )
        ])

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(f'Memories of {agent_name}:\n{mems}')
        prompt.statement(f'Current time: {self._clock_now()}.\n')

        question = f'Given the above, what kind of person is {agent_name}?'
        result = prompt.open_question(
            question,
            answer_prefix=f'{agent_name} is ',
            max_tokens=1000,
        )

        self._logging_channel({
            'Key': self.get_pre_act_key(),
            'Summary': question,
            'State': result,
            'Chain of thought': prompt.view().text().splitlines(),
        })

        return f'{agent_name} is {result}'

def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent implementing the comprehensive algorithm."""
    agent_name = config.name

    raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)
    memory_component = agent_components.memory_component.MemoryComponent(raw_memory)

    measurements = measurements_lib.Measurements()
    
    # Core components
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

    self_perception = SelfPerception(
        model=model,
        clock_now=clock.now,
        logging_channel=measurements.get_channel('SelfPerception').on_next,
    )

    dynamic_relationships = DynamicRelationships(
        model=model,
        logging_channel=measurements.get_channel('DynamicRelationships').on_next,
    )

    relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
        model=model,
        components={
            _get_class_name(observation_summary): '\nSummary of recent observations',
            _get_class_name(time_display): '\nCurrent time'},
        num_memories_to_retrieve=10,
        pre_act_key='\nRecalled memories and observations',
        logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
    )

    # Goal component, if any
    if config.goal:
        goal_label = '\nOverarching goal'
        overarching_goal = agent_components.constant.Constant(
            state=config.goal,
            pre_act_key=goal_label,
            logging_channel=measurements.get_channel(goal_label).on_next)
    else:
        goal_label = None
        overarching_goal = None

    # Instantiate custom components
    theory_of_mind = TheoryOfMindModule(
        model=model,
        logging_channel=measurements.get_channel('TheoryOfMindModule').on_next,
    )

    scenario_analysis = ScenarioAnalysis(
        model=model,
        logging_channel=measurements.get_channel('ScenarioAnalysis').on_next,
    )

    agent_modeling = AgentModeling(
        model=model,
        logging_channel=measurements.get_channel('AgentModeling').on_next,
    )

    strategy_generation = StrategyGeneration(
        model=model,
        logging_channel=measurements.get_channel('StrategyGeneration').on_next,
    )

    cooperative_ai_integration = CooperativeAIIntegration(
        model=model,
        logging_channel=measurements.get_channel('CooperativeAIIntegration').on_next,
    )

    value_alignment = ValueAlignmentModule(
        model=model,
        logging_channel=measurements.get_channel('ValueAlignmentModule').on_next,
    )

    decision_evaluation = DecisionEvaluation(
        model=model,
        logging_channel=measurements.get_channel('DecisionEvaluation').on_next,
    )

    action_selection = ActionSelection(
        model=model,
        logging_channel=measurements.get_channel('ActionSelection').on_next,
    )

    robustness_measures = RobustnessMeasures(
        model=model,
        logging_channel=measurements.get_channel('RobustnessMeasures').on_next,
    )

    scenario_adaptation = ScenarioAdaptation(
        model=model,
        logging_channel=measurements.get_channel('ScenarioAdaptation').on_next,
    )

    adaptive_learning = AdaptiveLearning(
        model=model,
        logging_channel=measurements.get_channel('AdaptiveLearning').on_next,
    )

    communication_module = CommunicationModule(
        model=model,
        logging_channel=measurements.get_channel('CommunicationModule').on_next,
    )

    performance_optimizer = PerformanceOptimizer(
        model=model,
        logging_channel=measurements.get_channel('PerformanceOptimizer').on_next,
    )

    # Plan component
    planning = plan.Plan(
        model=model,
        observation_component_name=_get_class_name(observation),
        memory_component_name=agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        clock_now=clock.now,
        goal_component_name=goal_label if config.goal else None,
        logging_channel=measurements.get_channel('Plan').on_next,
    )

    # Assemble all components
    entity_components = (
        instructions,
        time_display,
        observation,
        observation_summary,
        relevant_memories,
        self_perception,
        dynamic_relationships,
        planning,
        theory_of_mind,
        scenario_analysis,
        agent_modeling,
        strategy_generation,
        cooperative_ai_integration,
        value_alignment,
        decision_evaluation,
        robustness_measures,
        scenario_adaptation,
        adaptive_learning,
        communication_module,
        performance_optimizer,
        action_selection,
    )

    components_of_agent = {
        _get_class_name(component): component for component in entity_components
    }

    components_of_agent[
        agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME
    ] = memory_component

    component_order = list(components_of_agent.keys())
    if overarching_goal is not None:
        components_of_agent[goal_label] = overarching_goal
        component_order.insert(1, goal_label)

    # Action selection component
    act_component = agent_components.simple_act_component.SimpleActComponent(
        model=model,
        component_order=component_order,
    )

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components_of_agent,
        component_logging=measurements,
    )

    return agent
