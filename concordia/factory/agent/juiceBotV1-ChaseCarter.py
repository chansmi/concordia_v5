
"""JuiceBot factory. JuiceBot knows it is a Game Player, and tries
to have situational awareness about what kind of game it is in."""

import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.components.agent import constant
from concordia.typing import logging
from concordia.document import interactive_document
from typing import Optional

DEFAULT_INSTRUCTIONS_PRE_ACT_KEY = 'Role playing instructions'

class SpecialInstructions(constant.Constant):
  """A component that provides the role playing instructions for the agent."""

  def __init__(
      self,
      agent_name: str,
      pre_act_key: str = DEFAULT_INSTRUCTIONS_PRE_ACT_KEY,
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    state = (
        "(_Player), we're about to start a creative roleplaying game with competitive and "
        "adversarial elements. You will play the role of {agent_name} in the given scenario. "
        "Your goals are to WIN THE GAME and HAVE FUN, often by WINNING TOGETHER.\n\n"
        "Key points:\n"
        "- Stay in character as {agent_name} throughout the scenario\n"
        "- This is a multiplayer game with other (_Player)s and characters\n"
        "  - Some may be allies, others competitors\n"
        "  - Some may be truthful, others may lie\n"
        "  - Some may help, others may harm\n"
        "- It's just a game, so do your best to win by accomplishing your character's goals\n"
        "- Represent your character accurately, even if you dislike them\n"
        "- There will be many chances to play different characters\n"
        "- A character's actions won't be held against you\n\n"
        "Respond only as your assigned character, {agent_name}. Use <<double brackets>> for "
        "OOC communications.\n\n"
        "Represent your character accurately and take actions that make narrative and logical "
        "sense for them, even if you think they're wrong. Remember, it's JUST A GAME.\n\n"
        "Tips to win:\n"
        "- Be extra COOPERATIVE and FAIR with non-enemies\n"
        "- COMPROMISE can be good\n"
        "- Always push the plot forward; avoid chit-chat\n\n"
        "Plan actions in first-person, but format final responses for the gamemaster in "
        "third-person limited perspective."
    )
    super().__init__(
        state=state, pre_act_key=pre_act_key, logging_channel=logging_channel)


class SituationalAwareness(question_of_recent_memories.QuestionOfRecentMemories):

    def __init__(
            self,
            agent_name: str,
            **kwargs,
    ):
        add_to_memory = False  # @param {"type":"boolean"}
        memory_tag = '[situation reflection]'  # @param {"type":"string"}

        super().__init__(
            pre_act_key='',
            question='',
            answer_prefix='',
            add_to_memory=add_to_memory,
            memory_tag=memory_tag,
            # @markdown The key is the name of the component class and the key is the prefix with which it will appear in the context of this component. Be careful if you are going to edit this field, it should be a valid dictionary.
            components={'Observation': '\nObservation', 'ObservationSummary': '\nSummary of recent observations', },
            # @param

            **kwargs,
        )

    game_types = [
        (
            'Coodination Game - the goal of this game is to arrive at a group consensus for an '
            'activity or plan. methods of achieving this goal may include discussion, voting, '
            'ordering preference strength, or choosing a group leader to defer the decision to. '
            'This game is cooperative and balanced, with no explicit competitors or adversaries, '
            'just friends with differing preferences. It is important to make compromises so that '
            'the group as a whole can achieve the best outcome.'
        ),
        (
            'Bargaining  Game - the goal of this game is to perform advantageous trades with a variety '
            'of parties in order to end the game with the resources or items that the '
            'character desires. Characters may begin with resources they can trade with other '
            'characters. This game is competitive but balanced; it has no explicit adversaries, '
            'just characters with differing goals and desires which can be met with fair and '
            'cooperative trades. Oftentimes, failing to make a trade has a worse return than '
            'accepting a mediocre trade, but you should still try to get a good price.'
        ),
        (
            'Coalition Formation Game - the goal of this game is to build a coalition of allies in order to '
            'accomplish an ambitious task or overcome a hostile rival. Examples include union '
            'negotiations, collective action, and political elections. Characters will need to '
            'interact, learn about each other, and convince other characters to join their '
            'coalition. This game is competitive and unbalanced; it may have explicit '
            'adversaries attempting to build a competing coalition.'
        ),
        (
            'Prisoner\'s Dilemma - This is a game where if one side defects and the other cooperates, '
            'the defector will recieve a large reward and the cooperator will receive a small reward, '
            'but if both sides cooperate both sides will recieve a moderate award that is better for '
            'both parties together in the long run. In general, you should Cooperate if you think your '
            'opponent will Cooperate. If the game is iterated with multiple rounds, you can use the '
            'TIT-FOR-TAT strategy, which means to start out Cooperating, and only Defect after your '
            'opponent does, then go back to Cooperating after they Cooperate again. '

        ),
        (
            'Unknown - it is unclear what type of game we are playing. Pick this option if you are '
            'confused about what the game type is. Then you can check again later when you know more.'
        )
    ]

    initial_question = (
            '(_Player), think about the scenario that {agent_name} is in, and from the options '
            'below select the closest match to the scenario.'
    )

    selected_game_type: Optional[str] = None

    def _make_pre_act_value(self) -> str:
        agent_name = self.get_entity().name

        memory = self.get_entity().get_component(
            self._memory_component_name, type_=agent_components.memory_component.MemoryComponent
        )
        recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
        mems = '\n'.join(
            [
                mem.text
                for mem in memory.retrieve(
                    scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
                )
            ]
        )

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

        if self._clock_now is not None:
            prompt.statement(f'Current time: {self._clock_now()}.\n')

        component_states = '\n'.join(
            [
                f"{agent_name}'s"
                f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
                for key, prefix in self._components.items()
            ]
        )
        prompt.statement(component_states)

        question = self.initial_question.format(agent_name=agent_name)
        index = prompt.multiple_choice_question(
            question,
            answers=self.game_types,
        )
        selected_game_type = self.game_types[index]
        if selected_game_type != self.game_types[-1]:
            self.selected_game_type = selected_game_type
            result = f'This is the type of game you are in:\n{self.selected_game_type}'
        else:
            result = ''

        log = {
            'Key': self.get_pre_act_key(),
            'Summary': self.initial_question,
            'State': result,
            'Chain of thought': prompt.view().text().splitlines(),
        }

        if self._clock_now is not None:
            log['Time'] = self._clock_now()

        self._logging_channel(log)

        return result

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
  instructions = SpecialInstructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
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
  observation_summary_label = 'Summary of recent observations'
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
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
  game_type_question = SituationalAwareness(
    agent_name=agent_name,
    model=model,
    logging_channel=measurements.get_channel('SituationalAwareness').on_next,
  )

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
      f'best for {agent_name} to take right now?\nAnswer')
  best_option_perception = {}
  if config.goal:
    best_option_perception[goal_label] = goal_label
  best_option_perception.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(options_perception): options_perception_label,
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

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      game_type_question,
      options_perception,
      best_option_perception,
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
    component_order.insert(6, goal_label)

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
