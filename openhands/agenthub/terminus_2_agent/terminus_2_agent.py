"""Terminus-2 Agent for OpenHands.

A keystroke-based terminal agent that sends raw keystrokes to a terminal session
and receives screen capture output. Uses JSON-formatted LLM responses with
analysis, plan, and commands fields.
"""

import os
from collections import deque
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from openhands.events.action import Action

from openhands.agenthub.terminus_2_agent.terminus_json_plain_parser import (
    ParsedCommand,
    TerminusJSONPlainParser,
)
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import AgentFinishAction, AgentThinkAction, MessageAction
from openhands.events.action.terminus_2 import Terminus2CmdRunAction
from openhands.events.event import Event, EventSource
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.terminus_2 import Terminus2CmdOutputObservation
from openhands.events.tool import ToolCallMetadata
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.runtime.plugins import PluginRequirement
from openhands.utils.prompt import PromptManager


MAX_OUTPUT_BYTES = 10000
MAX_LLM_RETRY = 3
# After this many CONSECUTIVE steps whose reply could not be parsed, end the
# episode cleanly rather than emitting a constant think action that the stuck
# detector kills. Each failed step costs MAX_LLM_RETRY model calls and runs nothing.
MAX_CONSECUTIVE_PARSE_FAILURES = 3
COMMAND_EXEC_TIMEOUT = int(os.getenv('COMMAND_EXEC_TIMEOUT', '300'))

TIMEOUT_TEMPLATE = (
    'Previous command:\n{command}\n\n'
    'The previous command timed out after {timeout_sec} seconds\n\n'
    'It is possible that the command is not yet finished executing. '
    'If that is the case, then do nothing. It is also possible that you '
    'have entered an interactive shell and should continue sending '
    'keystrokes as normal.\n\n'
    'Here is the current state of the terminal:\n\n{terminal_state}'
)

COMPLETION_CONFIRMATION = (
    'Current terminal state:\n{terminal_state}\n\n'
    'Are you sure you want to mark the task as complete? '
    'This will trigger your solution to be graded and you won\'t be able to '
    'make any further corrections. If so, include "task_complete": true '
    'in your JSON response again.'
)


class Terminus2Agent(Agent):
    VERSION = '1.0'
    """
    The Terminus-2 Agent sends raw keystrokes to a terminal session and receives
    screen capture output, using JSON-formatted LLM responses.

    Unlike function-calling agents (CodeAct, OpenCode), Terminus-2 parses the
    LLM's raw text response as JSON with fields: analysis, plan, commands, and
    optionally task_complete.

    Key features:
    - Keystroke-based terminal interaction (tmux-style)
    - JSON response parsing with auto-correction
    - Double confirmation for task completion
    - Output truncation to 10KB
    - Duration-based command timeouts
    """

    sandbox_plugins: list[PluginRequirement] = []

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.parser = TerminusJSONPlainParser()
        self._pending_completion = False
        self._conversation_messages: list[dict[str, str]] = []
        self._needs_llm_call = True
        # Carried ACROSS controller iterations. _build_messages rebuilds the message
        # list from the event history every step, and AgentThinkAction is skipped
        # there, so a parse failure used to leave no trace at all: the next step
        # rebuilt byte-identical context, the model repeated the identical malformed
        # reply, and the episode died in the stuck detector having run nothing. See
        # _note_parse_failure.
        self._last_parse_error: str | None = None
        self._consecutive_parse_failures = 0

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        self.llm = self.llm_registry.get_router(self.config)

        from openhands.agenthub.nemo_gym_client import NemoGymClient

        self.nemo_gym_client = NemoGymClient(self.llm)

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            prompt_dir = (
                self.config.custom_prompt_dir
                if self.config.custom_prompt_dir
                else os.path.join(os.path.dirname(__file__), 'prompts')
            )

            template_overrides = {}
            if self.config.system_prompt_path:
                template_overrides['system_prompt.j2'] = self.config.system_prompt_path
            if self.config.system_prompt_long_horizon_path:
                template_overrides['system_prompt_long_horizon.j2'] = (
                    self.config.system_prompt_long_horizon_path
                )

            self._prompt_manager = PromptManager(
                prompt_dir=prompt_dir,
                system_prompt_filename=self.config.resolved_system_prompt_filename,
                template_overrides=template_overrides if template_overrides else None,
            )

        return self._prompt_manager

    def reset(self) -> None:
        super().reset()
        self.pending_actions.clear()
        self._pending_completion = False
        self._conversation_messages = []
        self._needs_llm_call = True
        self._last_parse_error = None
        self._consecutive_parse_failures = 0

    def _has_terminal_observation(self, events: list[Event]) -> bool:
        """Check if any Terminus2CmdOutputObservation exists in the event history."""
        return any(isinstance(e, Terminus2CmdOutputObservation) for e in events)

    async def step(self, state: State) -> 'Action':
        """Performs one step of the Terminus-2 agent.

        On the very first step (before any terminal observations exist), sends a
        no-op action to capture the initial terminal screen. This mirrors the
        original Terminus-2 behavior of capturing tmux state before the first
        LLM call.

        Returns pending actions from the queue, or calls the LLM to get
        new commands when the queue is empty.
        """
        if self.pending_actions:
            return self.pending_actions.popleft()

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        if not self._has_terminal_observation(condensed_history):
            return Terminus2CmdRunAction(keystrokes='', duration=0.5)

        messages = self._build_messages(condensed_history, state)

        commands, is_task_complete, response_text, model_response = (
            await self._call_llm_and_parse(messages)
        )

        if is_task_complete:
            if self._pending_completion:
                finish = AgentFinishAction(thought='Task completed (confirmed)')
                self._attach_response_metadata(finish, model_response, total_calls=0)
                return finish
            else:
                self._pending_completion = True
        else:
            self._pending_completion = False

        total_calls = len(commands)
        for i, cmd in enumerate(commands):
            action = Terminus2CmdRunAction(
                keystrokes=cmd.keystrokes,
                duration=min(cmd.duration, COMMAND_EXEC_TIMEOUT),
                thought=response_text if i == 0 else '',
            )
            self._attach_response_metadata(action, model_response, total_calls=total_calls)
            self.pending_actions.append(action)

        if not self.pending_actions:
            if self._pending_completion:
                noop = Terminus2CmdRunAction(
                    keystrokes='', duration=0.5, thought=response_text
                )
                self._attach_response_metadata(noop, model_response, total_calls=0)
                return noop
            if self._consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                # Stop cleanly instead of emitting a constant think action that the
                # stuck detector eventually kills. Both the old think text and its
                # observation were fixed strings, so the loop was byte-identical and
                # the agent could never recover from it. Ending here keeps whatever
                # patch exists and reports an honest reason.
                logger.error(
                    'Terminus-2: %d consecutive parse failures, ending episode',
                    self._consecutive_parse_failures,
                )
                finish = AgentFinishAction(
                    thought=(
                        f'Stopping after {self._consecutive_parse_failures} consecutive '
                        f'unparseable responses. Last error: {self._last_parse_error}'
                    )
                )
                self._attach_response_metadata(finish, model_response, total_calls=0)
                return finish
            # Carry the real error in the thought so it is not a constant string.
            think = AgentThinkAction(
                thought=(
                    'No commands to execute, waiting for next input'
                    if not self._last_parse_error
                    else f'Response could not be parsed: {self._last_parse_error}'
                )
            )
            self._attach_response_metadata(think, model_response, total_calls=0)
            return think

        return self.pending_actions.popleft()

    @staticmethod
    def _assistant_message_from_event(event: Event, text: str) -> Message:
        """Rebuild an assistant Message from a stored event, preserving the
        prompt/generation token ids and logprobs that NemoGymClient stashes on
        the response's `_provider_specific_fields`. Without this, the assistant
        turn echoed back to the next LLM call would lose the per-turn token
        metadata, even when the action itself carries `tool_call_metadata`.
        """
        provider_fields: dict = {}
        metadata = getattr(event, 'tool_call_metadata', None)
        if metadata is not None and metadata.model_response is not None:
            provider_fields = getattr(
                metadata.model_response, '_provider_specific_fields', None
            ) or {}
        return Message(
            role='assistant',
            content=[TextContent(text=text)],
            prompt_token_ids=provider_fields.get('prompt_token_ids'),
            generation_token_ids=provider_fields.get('generation_token_ids'),
            generation_log_probs=provider_fields.get('generation_log_probs'),
        )

    @staticmethod
    def _attach_response_metadata(
        action: 'Action', model_response: object | None, total_calls: int
    ) -> None:
        """Attach the LLM response to the action so prompt/generation token ids
        and logprobs flow into the trajectory, mirroring the CodeAct/OpenCode
        pattern (function_calling.py:349-373). Without this, NemoGymClient's
        `_provider_specific_fields` is dropped after each LLM call.
        """
        if model_response is None:
            return
        action.tool_call_metadata = ToolCallMetadata(
            model_response=model_response,
            total_calls_in_response=total_calls,
        )
        response_id = getattr(model_response, 'id', None)
        if response_id is not None:
            action.response_id = response_id

    def _build_messages(
        self, condensed_history: list[Event], state: State
    ) -> list[Message]:
        """Build the conversation messages from event history.

        Converts the event stream into a user/assistant message sequence:
        - System message: JSON format instructions from the prompt template
        - First user message: the instruction content (from get_instruction /
          INSTRUCTION_TEMPLATE_PATH) used directly, with initial terminal state
          appended if available. This keeps the agent consistent with CodeAct/
          Codex/OpenCode which pass the MessageAction.content through unchanged,
          so that INSTRUCTION_TEMPLATE_PATH overrides work without double-wrapping.
        - Subsequent turns: assistant = LLM JSON response, user = terminal output
        """
        messages: list[Message] = []

        system_prompt = self.prompt_manager.get_system_message()
        messages.append(Message(role='system', content=[TextContent(text=system_prompt)]))

        initial_user_msg = self._find_initial_user_message(condensed_history)
        initial_terminal_event = self._find_initial_terminal_event(condensed_history)

        if initial_user_msg:
            if initial_terminal_event is not None:
                terminal_text = initial_terminal_event.terminal_state
                first_user_text = f'{initial_user_msg}\n\n{terminal_text}'
            else:
                first_user_text = initial_user_msg
            messages.append(
                Message(role='user', content=[TextContent(text=first_user_text)])
            )

        batch_observations: list[str] = []
        last_timed_out = False
        last_keystrokes = ''

        for event in condensed_history:
            if isinstance(event, MessageAction):
                if event.source == EventSource.USER:
                    continue
                elif event.source == EventSource.AGENT:
                    if batch_observations:
                        terminal_output = self._combine_observations(
                            batch_observations
                        )
                        user_text = self._format_terminal_output(
                            terminal_output, last_timed_out, last_keystrokes
                        )
                        messages.append(
                            Message(role='user', content=[TextContent(text=user_text)])
                        )
                        batch_observations = []
                        last_timed_out = False

                    messages.append(
                        self._assistant_message_from_event(event, event.content)
                    )

            elif isinstance(event, Terminus2CmdRunAction):
                if event.thought:
                    if batch_observations:
                        terminal_output = self._combine_observations(
                            batch_observations
                        )
                        user_text = self._format_terminal_output(
                            terminal_output, last_timed_out, last_keystrokes
                        )
                        messages.append(
                            Message(role='user', content=[TextContent(text=user_text)])
                        )
                        batch_observations = []
                        last_timed_out = False

                    messages.append(
                        self._assistant_message_from_event(event, event.thought)
                    )
                last_keystrokes = event.keystrokes

            elif isinstance(event, Terminus2CmdOutputObservation):
                if event is initial_terminal_event:
                    continue
                batch_observations.append(event.terminal_state)
                last_timed_out = event.timed_out

            elif isinstance(event, ErrorObservation):
                batch_observations.append(f'ERROR: {event.content}')

            elif isinstance(event, AgentThinkAction):
                pass

        if batch_observations:
            terminal_output = self._combine_observations(batch_observations)
            if self._pending_completion:
                confirmation = COMPLETION_CONFIRMATION.format(
                    terminal_state=terminal_output
                )
                messages.append(
                    Message(role='user', content=[TextContent(text=confirmation)])
                )
            else:
                user_text = self._format_terminal_output(
                    terminal_output, last_timed_out, last_keystrokes
                )
                messages.append(
                    Message(role='user', content=[TextContent(text=user_text)])
                )
        elif self._pending_completion:
            confirmation = COMPLETION_CONFIRMATION.format(terminal_state='')
            messages.append(
                Message(role='user', content=[TextContent(text=confirmation)])
            )

        # Surface a parse failure from the PREVIOUS step. Without this the model is
        # handed byte-identical context after a malformed reply (AgentThinkAction is
        # skipped above, and the in-call retry feedback died with step()'s local
        # list), so it reproduces the same malformed reply until the stuck detector
        # kills the episode. Measured on 40 stuck Terminus2 rollouts: 100% showed
        # this signature, 45% died in under 300s having run a single command.
        if self._last_parse_error:
            messages.append(
                Message(
                    role='user',
                    content=[
                        TextContent(
                            text=(
                                'Your previous reply could not be parsed and NO '
                                'command was executed.\n'
                                f'ERROR: {self._last_parse_error}\n\n'
                                'Reply with ONLY a single valid JSON object with the '
                                'keys "analysis", "plan" and "commands". Do not wrap '
                                'it in markdown fences or add any text around it.'
                            )
                        )
                    ],
                )
            )

        return messages

    def _find_initial_user_message(self, events: list[Event]) -> str | None:
        """Find the initial user message (task instruction) from the event history."""
        for event in events:
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                return event.content
        return None

    def _find_initial_terminal_event(
        self, events: list[Event]
    ) -> Terminus2CmdOutputObservation | None:
        """Find the first Terminus2CmdOutputObservation in the event history.

        Returns the event object itself (not just the string) so that
        _build_messages can skip it in the loop via identity comparison,
        avoiding duplication with the first user message.
        """
        for event in events:
            if isinstance(event, Terminus2CmdOutputObservation):
                return event
        return None

    _NEW_OUTPUT_PREFIX = 'New Terminal Output:\n'
    _SCREEN_PREFIX = 'Current Terminal Screen:\n'

    @staticmethod
    def _combine_observations(observations: list[str]) -> str:
        """Combine multiple terminal observations into a single output.

        In the original Terminus-2, all commands in a batch execute in tmux and
        then a single get_incremental_output() captures the cumulative output.
        This method replicates that by stripping the per-observation prefix,
        joining the raw screen content, and re-adding a single prefix.
        """
        if not observations:
            return ''
        if len(observations) == 1:
            return observations[0]

        new_pfx = Terminus2Agent._NEW_OUTPUT_PREFIX
        scr_pfx = Terminus2Agent._SCREEN_PREFIX

        screens: list[str] = []
        last_prefix = new_pfx
        for obs in observations:
            if obs.startswith(new_pfx):
                screens.append(obs[len(new_pfx):])
                last_prefix = new_pfx
            elif obs.startswith(scr_pfx):
                screens.append(obs[len(scr_pfx):])
                last_prefix = scr_pfx
            else:
                screens.append(obs)

        return last_prefix + '\n'.join(screens)

    def _format_terminal_output(
        self, terminal_output: str, timed_out: bool, keystrokes: str
    ) -> str:
        """Format terminal output for the next user message."""
        if timed_out:
            return TIMEOUT_TEMPLATE.format(
                command=keystrokes,
                timeout_sec=COMMAND_EXEC_TIMEOUT,
                terminal_state=self._limit_output_length(terminal_output),
            )
        return self._limit_output_length(terminal_output)

    async def _call_llm_and_parse(
        self, messages: list[Message]
    ) -> tuple[list[ParsedCommand], bool, str, object | None]:
        """Call the LLM and parse the JSON response, with retry on parse errors.

        Returns (commands, is_task_complete, response_text, model_response).
        The model_response is the raw litellm ModelResponse — callers attach it
        to actions via ToolCallMetadata so prompt/generation token ids and
        logprobs are persisted on the trajectory, matching CodeAct/OpenCode.
        """
        last_response: object | None = None
        last_error: str | None = None
        for attempt in range(MAX_LLM_RETRY):
            response = await self.nemo_gym_client.model_call(messages)
            last_response = response

            response_text = response.choices[0].message.content or ''
            logger.debug(f'Terminus-2 LLM response (attempt {attempt + 1}): {response_text[:200]}...')

            messages.append(
                Message(role='assistant', content=[TextContent(text=response_text)])
            )

            result = self.parser.parse_response(response_text)

            if result.error:
                last_error = result.error
                feedback = f'Previous response had parsing errors:\nERROR: {result.error}'
                if result.warning:
                    feedback += f'\nWARNINGS: {result.warning}'
                feedback += '\n\nPlease fix these issues and provide a proper JSON response.'
                logger.warning(f'Terminus-2 parse error (attempt {attempt + 1}): {result.error}')

                messages.append(
                    Message(role='user', content=[TextContent(text=feedback)])
                )
                continue

            if result.warning:
                logger.info(f'Terminus-2 parse warnings: {result.warning}')

            commands = [
                ParsedCommand(keystrokes=cmd.keystrokes, duration=min(cmd.duration, COMMAND_EXEC_TIMEOUT))
                for cmd in result.commands
            ]
            # A clean parse clears the carried-over failure state.
            self._last_parse_error = None
            self._consecutive_parse_failures = 0
            return commands, result.is_task_complete, response_text, response

        logger.error('Terminus-2: exhausted LLM retries due to parse errors')
        # Carry the failure to the NEXT step. The retry feedback appended above
        # only ever lived in this local `messages` list, which step() discards, so
        # without this the model is handed identical context next iteration and
        # reproduces the identical malformed reply forever.
        self._last_parse_error = str(last_error or 'response was not valid JSON')[:800]
        self._consecutive_parse_failures += 1
        return [], False, '', last_response

    @staticmethod
    def _limit_output_length(output: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
        """Limit output to specified byte length, keeping first and last portions."""
        if len(output.encode('utf-8')) <= max_bytes:
            return output

        portion_size = max_bytes // 2
        output_bytes = output.encode('utf-8')
        first_portion = output_bytes[:portion_size].decode('utf-8', errors='ignore')
        last_portion = output_bytes[-portion_size:].decode('utf-8', errors='ignore')
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode('utf-8'))
            - len(last_portion.encode('utf-8'))
        )

        return (
            f'{first_portion}\n[... output limited to {max_bytes} bytes; '
            f'{omitted_bytes} interior bytes omitted ...]\n{last_portion}'
        )
