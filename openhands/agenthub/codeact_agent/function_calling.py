"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json
import re
from os import getenv

from litellm import (
    ModelResponse,
)

from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    CondensationRequestTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    ThinkTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.security_utils import RISK_LEVELS
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMContextWindowExceedError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    ActionSecurityRisk,
    AgentDelegateAction,
    AgentFinishAction,
    AgentThinkAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
    TaskTrackingAction,
    ValidationFailureAction,
    FunctionCallNotExistsAction
)
from openhands.events.action.agent import CondensationRequestAction
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.tool import ToolCallMetadata
from openhands.llm.tool_names import TASK_TRACKER_TOOL_NAME


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, 'thought'):
        return action
    if thought and action.thought:
        action.thought = f'{thought}\n{action.thought}'
    elif thought:
        action.thought = thought
    return action


def set_security_risk(action: Action, arguments: dict) -> None:
    """Set the security risk level for the action."""

    # Set security_risk attribute if provided
    if 'security_risk' in arguments:
        if arguments['security_risk'] in RISK_LEVELS:
            if hasattr(action, 'security_risk'):
                action.security_risk = getattr(
                    ActionSecurityRisk, arguments['security_risk']
                )
        else:
            logger.warning(f'Invalid security_risk value: {arguments["security_risk"]}')


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    choice = response.choices[0]
    assistant_msg = choice.message

    # NeMo lenient tool-call recovery (2026-09-13, gated; extended 2026-10-13): the
    # policy drifts into tool syntax the server-side hermes parser rejects
    # ('<tool_call>' JSON inside <think>, bare JSON, <function=..>, <execute_bash>,
    # <finish>, ...), leaving tool_calls empty while the content clearly contains
    # an intended call. Synthesize real tool calls from it (see
    # _nemo_recover_tool_calls for the recognised shapes).
    _msg0 = getattr(response.choices[0], 'message')
    if (
        (not (hasattr(_msg0, 'tool_calls') and _msg0.tool_calls))
        and isinstance(getattr(_msg0, 'content', None), str)
        and _msg0.content.strip()
        and _nemo_lenient_enabled()
    ):
        _recovered = _nemo_recover_tool_calls(_msg0.content)
        if _recovered:
            _msg0.tool_calls = _recovered

    # Check if both content and tool_calls are None - this indicates context length has been hit
    has_content = assistant_msg.content is not None
    has_tool_calls = hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls

    if not has_content and not has_tool_calls:
        raise LLMContextWindowExceedError(
            'LLM returned empty response with no content and no tool calls. This indicates the context length limit has been exceeded.'
        )

    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        # Check if there's assistant_msg.content. If so, add it to the thought
        thought = ''
        if isinstance(assistant_msg.content, str):
            thought = assistant_msg.content
        elif isinstance(assistant_msg.content, list):
            for msg in assistant_msg.content:
                if msg['type'] == 'text':
                    thought += msg['text']

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            logger.debug(f'Tool call in function_calling.py: {tool_call}')

            try:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.decoder.JSONDecodeError as e:
                    raise FunctionCallValidationError(
                        f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                    ) from e

                # ================================================
                # CmdRunTool (Bash)
                # ================================================

                if tool_call.function.name == create_cmd_run_tool()['function']['name']:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {tool_call.function.name}'
                        )
                    # convert is_input to boolean
                    is_input = arguments.get('is_input', 'false') == 'true'
                    action = CmdRunAction(command=arguments['command'], is_input=is_input)

                    # Set hard timeout if provided (capped at 600 seconds max)
                    if 'timeout' in arguments:
                        try:
                            command_execution_timeout = int(getenv("COMMAND_EXEC_TIMEOUT", "300"))
                            action.set_hard_timeout(min(float(arguments['timeout']), command_execution_timeout))
                        except ValueError as e:
                            raise FunctionCallValidationError(
                                f"Invalid float passed to 'timeout' argument: {arguments['timeout']}"
                            ) from e
                    set_security_risk(action, arguments)

                # ================================================
                # IPythonTool (Jupyter)
                # ================================================
                elif tool_call.function.name == IPythonTool['function']['name']:
                    if 'code' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "code" in tool call {tool_call.function.name}'
                        )
                    action = IPythonRunCellAction(code=arguments['code'])
                    set_security_risk(action, arguments)

                # ================================================
                # AgentDelegateAction (Delegation to another agent)
                # ================================================
                elif tool_call.function.name == 'delegate_to_browsing_agent':
                    action = AgentDelegateAction(
                        agent='BrowsingAgent',
                        inputs=arguments,
                    )

                # ================================================
                # AgentFinishAction
                # ================================================
                elif tool_call.function.name == FinishTool['function']['name']:
                    action = AgentFinishAction(
                        final_thought=arguments.get('message', ''),
                    )

                # ================================================
                # LLMBasedFileEditTool (LLM-based file editor, deprecated)
                # ================================================
                elif tool_call.function.name == LLMBasedFileEditTool['function']['name']:
                    if 'path' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "path" in tool call {tool_call.function.name}'
                        )
                    if 'content' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "content" in tool call {tool_call.function.name}'
                        )
                    action = FileEditAction(
                        path=arguments['path'],
                        content=arguments['content'],
                        start=arguments.get('start', 1),
                        end=arguments.get('end', -1),
                        impl_source=arguments.get(
                            'impl_source', FileEditSource.LLM_BASED_EDIT
                        ),
                    )
                elif (
                    tool_call.function.name
                    == create_str_replace_editor_tool()['function']['name']
                ):
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {tool_call.function.name}'
                        )
                    if 'path' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "path" in tool call {tool_call.function.name}'
                        )
                    path = arguments['path']
                    command = arguments['command']
                    other_kwargs = {
                        k: v for k, v in arguments.items() if k not in ['command', 'path']
                    }

                    if command == 'view':
                        action = FileReadAction(
                            path=path,
                            impl_source=FileReadSource.OH_ACI,
                            view_range=other_kwargs.get('view_range', None),
                        )
                    else:
                        if 'view_range' in other_kwargs:
                            # Remove view_range from other_kwargs since it is not needed for FileEditAction
                            other_kwargs.pop('view_range')

                        # Filter out unexpected arguments
                        valid_kwargs_for_editor = {}
                        # Get valid parameters from the str_replace_editor tool definition
                        str_replace_editor_tool = create_str_replace_editor_tool()
                        valid_params = set(
                            str_replace_editor_tool['function']['parameters'][
                                'properties'
                            ].keys()
                        )

                        for key, value in other_kwargs.items():
                            if key in valid_params:
                                # security_risk is valid but should NOT be part of editor kwargs
                                if key != 'security_risk':
                                    valid_kwargs_for_editor[key] = value
                            else:
                                raise FunctionCallValidationError(
                                    f'Unexpected argument {key} in tool call {tool_call.function.name}. Allowed arguments are: {valid_params}'
                                )

                        action = FileEditAction(
                            path=path,
                            command=command,
                            impl_source=FileEditSource.OH_ACI,
                            **valid_kwargs_for_editor,
                        )

                    set_security_risk(action, arguments)
                # ================================================
                # AgentThinkAction
                # ================================================
                elif tool_call.function.name == ThinkTool['function']['name']:
                    action = AgentThinkAction(thought=arguments.get('thought', ''))

                # ================================================
                # CondensationRequestAction
                # ================================================
                elif tool_call.function.name == CondensationRequestTool['function']['name']:
                    action = CondensationRequestAction()

                # ================================================
                # BrowserTool
                # ================================================
                elif tool_call.function.name == BrowserTool['function']['name']:
                    if 'code' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "code" in tool call {tool_call.function.name}'
                        )
                    action = BrowseInteractiveAction(browser_actions=arguments['code'])
                    set_security_risk(action, arguments)

                # ================================================
                # TaskTrackingAction
                # ================================================
                elif tool_call.function.name == TASK_TRACKER_TOOL_NAME:
                    if 'command' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "command" in tool call {tool_call.function.name}'
                        )
                    if arguments['command'] == 'plan' and 'task_list' not in arguments:
                        raise FunctionCallValidationError(
                            f'Missing required argument "task_list" for "plan" command in tool call {tool_call.function.name}'
                        )

                    raw_task_list = arguments.get('task_list', [])
                    if not isinstance(raw_task_list, list):
                        raise FunctionCallValidationError(
                            f'Invalid format for "task_list". Expected a list but got {type(raw_task_list)}.'
                        )

                    # Normalize task_list to ensure it's always a list of dictionaries
                    normalized_task_list = []
                    for task_idx, task in enumerate(raw_task_list):
                        if isinstance(task, dict):
                            # Task is already in correct format, ensure required fields exist
                            normalized_task = {
                                'id': task.get('id', f'task-{task_idx + 1}'),
                                'title': task.get('title', 'Untitled task'),
                                'status': task.get('status', 'todo'),
                                'notes': task.get('notes', ''),
                            }
                        else:
                            # Unexpected format, raise validation error
                            logger.warning(
                                f'Unexpected task format in task_list: {type(task)} - {task}'
                            )
                            raise FunctionCallValidationError(
                                f'Unexpected task format in task_list: {type(task)}. Each task should be a dictionary.'
                            )
                        normalized_task_list.append(normalized_task)

                    action = TaskTrackingAction(
                        command=arguments['command'],
                        task_list=normalized_task_list,
                    )

                # ================================================
                # MCPAction (MCP)
                # ================================================
                elif mcp_tool_names and tool_call.function.name in mcp_tool_names:
                    action = MCPAction(
                        name=tool_call.function.name,
                        arguments=arguments,
                    )
                else:
                    raise FunctionCallNotExistsError(
                        f'Tool {tool_call.function.name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.'
                    )

            except FunctionCallValidationError as e:
                # Convert validation errors to ValidationFailureAction instead of raising
                action = ValidationFailureAction(
                    function_name=tool_call.function.name,
                    error_message=str(e),
                    thought=thought if i == 0 else '',
                )

            except FunctionCallNotExistsError as e:
                # Send error as a user message while preserving the assistant message
                action = FunctionCallNotExistsAction(
                    function_name=tool_call.function.name,
                    error_message=str(e),
                    thought=thought if i == 0 else "",
                )

            # Add thought to first action
            if i == 0 and not isinstance(action, (ValidationFailureAction, FunctionCallNotExistsAction)):
                action = combine_thought(action, thought)

            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
            )
            actions.append(action)
    else:
        message_action = MessageAction(
            content=str(assistant_msg.content) if assistant_msg.content else '',
            wait_for_response=True,
        )
        # Add metadata for non-tool-call messages to preserve token IDs and logprobs
        message_action.tool_call_metadata = ToolCallMetadata(
            model_response=response,
            total_calls_in_response=0,
        )
        actions.append(message_action)

    # Add response id to actions
    # This will ensure we can match both actions without tool calls (e.g. MessageAction)
    # and actions with tool calls (e.g. CmdRunAction, IPythonRunCellAction, etc.)
    # with the token usage data
    for action in actions:
        action.response_id = response.id

    assert len(actions) >= 1
    return actions


def _nemo_lenient_enabled() -> bool:
    import os as _os

    _flag = _os.path.join(
        _os.path.dirname(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        ),
        'LENIENT_ON',
    )
    return _os.environ.get('OH_LENIENT_TOOLCALLS', '') in ('1', 'true') or _os.path.exists(
        _flag
    )


# ---------------------------------------------------------------------------
# NeMo lenient tool-call recovery (2026-09-13, extended 2026-10-13).
#
# A policy SFT'd on another harness's syntax (Terminus2) emits CodeAct tool calls
# in a dozen near-miss shapes that vLLM's hermes parser rejects. Each rejected turn
# costs a full generation and the SWE-bench fake-user "Please continue" reply, and
# the rollout then loops to max_iteration. Measured on exp_080 (2026-10-13): 43% of
# rollouts ended in that loop; the shapes below cover every variant seen in 1,655
# finished rollouts. Every detector is gated by LENIENT_ON, capped at
# _NEMO_MAX_CALLS calls per message, and the whole recovery is wrapped so that a
# recovery bug can never kill a rollout.
# ---------------------------------------------------------------------------
_NEMO_MAX_CALLS = 2
_NEMO_TOOL_ALIASES = {
    'execute_bash': 'execute_bash',
    'bash': 'execute_bash',
    'execute_command': 'execute_bash',
    'run_command': 'execute_bash',
    'cmd_run': 'execute_bash',
    'tool_execute': 'execute_bash',
    'function_to_execute': 'execute_bash',
    'execute_ipython_cell': 'execute_ipython_cell',
    'execute_ipython': 'execute_ipython_cell',
    'str_replace_editor': 'str_replace_editor',
    'str_replace_based_edit_tool': 'str_replace_editor',
    'file_editor': 'str_replace_editor',
    'finish': 'finish',
    'think': 'think',
    'task_tracker': 'task_tracker',
    'edit_file': 'edit_file',
    'browser': 'browser',
    'web_read': 'web_read',
}
# Pseudo-XML blocks the policy writes instead of a tool call. '<think>' is the
# model's reasoning wrapper and is deliberately NOT a tool tag.
_NEMO_TAG_TOOLS = (
    'execute_bash',
    'tool_execute',
    'function_to_execute',
    'finish',
    'str_replace_editor',
    'execute_ipython',
)
# Typed arguments the policy writes as JSON text inside tags/attributes.
_NEMO_JSON_ARG_KEYS = ('view_range', 'insert_line', 'timeout')
_NEMO_RE_TAG_OPEN = re.compile(
    r'<(' + '|'.join(_NEMO_TAG_TOOLS) + r')(?:\s[^<>]*)?>', re.S
)
_NEMO_RE_FUNCTION = re.compile(
    r'<function=([\w.-]+)>(.*?)(?:</function>|(?=<function=)|\Z)', re.S
)
_NEMO_RE_PARAM_EQ = re.compile(r'<parameter=([\w.-]+)>(.*?)</parameter>', re.S)
_NEMO_RE_INVOKE = re.compile(r'<invoke(?:\s+name="([\w.-]+)")?\s*>(.*?)(?:</invoke>|\Z)', re.S)
_NEMO_RE_PARAM_NAME = re.compile(
    r'<parameter\s+name="([\w.-]+)"\s*>(.*?)</parameter>', re.S
)
_NEMO_RE_ATTR_CALL = re.compile(r'<tool_call\s+name="([\w.-]+)"([^<>]*?)/?>', re.S)
_NEMO_RE_ATTR = re.compile(
    r'([\w.-]+)=("(?:[^"\\]|\\.)*"|\[[^\]]*\]|-?\d+|true|false)', re.S
)
_NEMO_RE_JSON_NAME = re.compile(r'\{\s*"name"\s*:')
_NEMO_RE_FINISH_MSG = re.compile(r'"message"\s*:\s*"(.*)', re.S)
_NEMO_RE_ANY_TAG = re.compile(r'<[^<>]*>')


def _nemo_canonical_tool(name) -> str | None:
    if not isinstance(name, str):
        return None
    return _NEMO_TOOL_ALIASES.get(name.strip().strip('"\'').lower())


def _nemo_quoted_marker(content: str, pos: int) -> bool:
    """True when the tag at `pos` is quoted in prose (e.g. "use the `<finish>` tag")."""
    return pos > 0 and content[pos - 1] == '`'


def _nemo_child(body: str, tag: str) -> str | None:
    m = re.search(r'<' + tag + r'>(.*?)</' + tag + r'>', body, re.S)
    return m.group(1).strip() if m else None


def _nemo_strip_tags(text: str) -> str:
    return _NEMO_RE_ANY_TAG.sub('', text).strip()


_NEMO_RE_PARAM_ANY = re.compile(
    r'<parameter(?:=|\s+name=")([\w.-]+)"?\s*>(.*?)</parameter>', re.S
)
_NEMO_EDITOR_COMMANDS = ('view', 'create', 'str_replace', 'insert', 'undo_edit')
_NEMO_SHELL_HINTS = (
    '/', '|', '&&', ';', '--', '=', '.py', 'python', 'pytest', 'git ', 'grep', 'sed ',
    'cat ', 'ls', 'cd ', 'echo', 'find', 'pip', 'rg ', 'head', 'tail', 'awk', 'make',
)
_NEMO_OTHER_MARKUP = ('<parameter', '<command>', '<invoke', '<execute_bash', '<tool_call', '<function')
_NEMO_FOREIGN_MARKUP_BASH = ('<parameter', '<invoke', '<tool_call', '<function', '<str_replace_editor', '<finish')
_NEMO_FOREIGN_MARKUP_FN = ('<invoke', '<tool_call', '<execute_bash', '<str_replace_editor', '<finish')
_NEMO_STRONG_SHELL = (
    '/', '|', '&&', '--', '=', '$', 'python', 'pytest', 'git ', 'grep', 'sed ', 'cat ',
    'ls', 'cd ', 'echo', 'find', 'pip', 'rg ', 'head', 'tail', 'awk', 'make', 'printf',
)
_NEMO_ARG_KEYS = {
    'execute_bash': ('command', 'is_input', 'timeout', 'security_risk'),
    'finish': ('message', 'task_completed'),
    'str_replace_editor': ('command', 'path', 'file_text', 'old_str', 'new_str', 'insert_line', 'view_range', 'security_risk'),
    'think': ('thought',),
    'task_tracker': ('command', 'task_list'),
    'execute_ipython_cell': ('code', 'security_risk'),
}


def _nemo_param_children(body: str) -> dict:
    """<parameter=K>V</parameter> and <parameter name="K">V</parameter> children."""
    return {k: v.strip() for k, v in _NEMO_RE_PARAM_ANY.findall(body)}


def _nemo_looks_like_command(text: str) -> bool:
    """Reject prose, placeholders and tag debris masquerading as a shell command."""
    t = text.strip()
    if not t or t in ('...', '..') or t.startswith('<') or len(t) > 4000:
        return False
    if t.lower() in _NEMO_TOOL_ALIASES or '**' in t:
        return False
    words = len(t.split())
    sentences = t.count('. ') + (1 if t.endswith('.') else 0)
    if words > 25 and sentences >= 2:
        return False
    if any(h in t for h in _NEMO_STRONG_SHELL):
        return True
    return words <= 8


def _nemo_clean_message(message) -> str:
    message = (message or '').strip() if isinstance(message, str) else ''
    return '' if message in ('...', '..') else message


def _nemo_coerce_args(args: dict) -> dict | None:
    out: dict = {}
    for key, value in args.items():
        if not isinstance(key, str):
            continue
        if key == 'security_risk':
            if isinstance(value, str) and value.strip().upper() in RISK_LEVELS:
                out[key] = value.strip().upper()
            continue
        if key in _NEMO_JSON_ARG_KEYS and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            out[key] = value
    return out


def _nemo_args_from_tag_body(tool: str, body: str) -> dict | None:
    """Map a pseudo-XML block body to the tool's argument dict."""
    stripped = body.strip()
    if stripped.startswith('{'):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    params = _nemo_param_children(body)
    if tool == 'execute_bash':
        command = (
            params.get('command')
            or _nemo_child(body, 'command')
            or _nemo_child(body, 'execute_command')
            or _nemo_child(body, 'cmd')
        )
        if command is None and not any(m in body for m in _NEMO_FOREIGN_MARKUP_BASH):
            command = _nemo_strip_tags(body)
        if not command or not _nemo_looks_like_command(command):
            return None
        args = {'command': command}
        risk = params.get('security_risk') or _nemo_child(body, 'security_risk')
        if risk:
            args['security_risk'] = risk
        return args
    if tool == 'finish':
        message = params.get('message')
        if message is None:
            message = _nemo_child(body, 'message')
        if message is None:
            if any(m in body for m in _NEMO_OTHER_MARKUP):
                return None  # another call's markup, not a final message
            message = _nemo_strip_tags(body)
        args = {'message': _nemo_clean_message(message)}
        task_completed = params.get('task_completed') or _nemo_child(body, 'task_completed')
        if task_completed:
            args['task_completed'] = task_completed
        return args
    if tool == 'str_replace_editor':
        args = {}
        for key in (
            'command',
            'path',
            'view_range',
            'old_str',
            'new_str',
            'file_text',
            'insert_line',
            'security_risk',
        ):
            value = params.get(key)
            if value is None:
                value = _nemo_child(body, key)
            if value is not None:
                args[key] = value
        return args if _nemo_editor_args_ok(args) else None
    if tool == 'execute_ipython_cell':
        code = params.get('code') or _nemo_child(body, 'code') or _nemo_strip_tags(body)
        return {'code': code} if code else None
    return None


def _nemo_editor_args_ok(args) -> bool:
    return (
        isinstance(args, dict)
        and args.get('command') in _NEMO_EDITOR_COMMANDS
        and isinstance(args.get('path'), str)
        and args['path'].startswith('/')
    )


def _nemo_call_from_json_obj(obj) -> tuple[str, dict] | None:
    """Normalise {"name":..,"arguments":..} / OpenAI / top-level-arg objects."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get('function'), dict):
        inner = obj['function']
        name = inner.get('name')
        args = inner.get('arguments', inner.get('parameters'))
    else:
        name = obj.get('name') or obj.get('tool') or obj.get('tool_name')
        args = None
        for key in ('arguments', 'parameters', 'input', 'args'):
            if key in obj:
                args = obj[key]
                break
        if args is None:
            args = {
                k: v
                for k, v in obj.items()
                if k not in ('name', 'tool', 'tool_name', 'type', 'id', 'function')
            }
    tool = _nemo_canonical_tool(name)
    if tool is None:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return tool, args


def _nemo_unterminated_finish(text: str) -> tuple[str, dict] | None:
    """A `finish` JSON the model never closed: salvage its message string."""
    if not re.match(r'\{\s*"name"\s*:\s*"finish"', text):
        return None
    m = _NEMO_RE_FINISH_MSG.search(text)
    if not m:
        return 'finish', {'message': ''}
    raw = m.group(1)
    for tail in ('</tool_call>', '</think>', '</function_call>'):
        idx = raw.find(tail)
        if idx != -1:
            raw = raw[:idx]
    raw = raw.rstrip().rstrip('}').rstrip()
    if raw.endswith('"'):
        raw = raw[:-1]
    try:
        message = json.loads('"' + raw + '"')
    except ValueError:
        message = raw.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    return 'finish', {'message': message}


def _nemo_detect_json(content: str) -> list:
    """<tool_call>{json}</tool_call> markers and bare {"name": ..} objects."""
    found = []
    decoder = json.JSONDecoder()
    starts = []
    for m in re.finditer(r'<tool_call>', content):
        brace = content.find('{', m.end())
        if brace != -1:
            starts.append(brace)
    for m in _NEMO_RE_JSON_NAME.finditer(content):
        starts.append(m.start())
    consumed_until = -1
    for start in sorted(set(starts)):
        if start < consumed_until:
            continue
        try:
            obj, end = decoder.raw_decode(content[start:])
        except ValueError:
            salvaged = _nemo_unterminated_finish(content[start:])
            if salvaged:
                found.append((start, salvaged[0], salvaged[1]))
                consumed_until = len(content)
            continue
        consumed_until = start + end
        call = _nemo_call_from_json_obj(obj)
        if call:
            found.append((start, call[0], call[1]))
    return found


def _nemo_detect_function_tags(content: str) -> list:
    """OpenHands' own non-native form: <function=NAME><parameter=K>V</parameter>."""
    found = []
    for m in _NEMO_RE_FUNCTION.finditer(content):
        tool = _nemo_canonical_tool(m.group(1))
        if tool is None or any(mk in m.group(2) for mk in _NEMO_FOREIGN_MARKUP_FN):
            continue
        args = _nemo_param_children(m.group(2))
        if not args and tool != 'finish':
            continue
        if tool == 'finish' and not args:
            if any(mk in m.group(2) for mk in _NEMO_OTHER_MARKUP):
                continue
            args = {'message': _nemo_clean_message(_nemo_strip_tags(m.group(2)))}
        found.append((m.start(), tool, args))
    return found


def _nemo_detect_invoke(content: str) -> list:
    """Anthropic-style <invoke name="NAME"><parameter name="K">V</parameter> (name optional)."""
    found = []
    for m in _NEMO_RE_INVOKE.finditer(content):
        args = _nemo_param_children(m.group(2))
        tool = _nemo_canonical_tool(m.group(1)) if m.group(1) else _nemo_tool_from_params(args)
        if tool is None:
            continue
        if tool == 'finish':
            args = {'message': _nemo_clean_message(args.get('message', ''))}
        if args or tool == 'finish':
            found.append((m.start(), tool, args))
    return found


def _nemo_tool_from_params(args: dict) -> str | None:
    if 'path' in args and 'command' in args:
        return 'str_replace_editor'
    if 'command' in args:
        return 'execute_bash'
    if 'message' in args:
        return 'finish'
    return None


def _nemo_detect_attr_calls(content: str) -> list:
    """Attribute form: <tool_call name="str_replace_editor" command="view" path=".." />."""
    found = []
    for m in _NEMO_RE_ATTR_CALL.finditer(content):
        tool = _nemo_canonical_tool(m.group(1))
        if tool is None:
            continue
        args = {}
        for key, raw in _NEMO_RE_ATTR.findall(m.group(2)):
            if raw.startswith('"'):
                try:
                    args[key] = json.loads(raw)
                except ValueError:
                    args[key] = raw[1:-1]
            else:
                try:
                    args[key] = json.loads(raw)
                except ValueError:
                    args[key] = raw
        if args:
            found.append((m.start(), tool, args))
    return found


def _nemo_detect_pseudo_xml(content: str) -> list:
    """<execute_bash>, <tool_execute>, <function_to_execute>, <finish>, <str_replace_editor>."""
    found = []
    opens = list(_NEMO_RE_TAG_OPEN.finditer(content))
    for idx, m in enumerate(opens):
        if _nemo_quoted_marker(content, m.start()):
            continue
        raw_tag = m.group(1)
        tool = _nemo_canonical_tool(raw_tag)
        if tool is None:
            continue
        close = content.find('</' + raw_tag + '>', m.end())
        next_open = opens[idx + 1].start() if idx + 1 < len(opens) else len(content)
        if close != -1 and close < next_open:
            body = content[m.end() : close]
        else:
            # Unterminated block: runs until the next tool tag or the end of the
            # reasoning/tools wrapper.
            end = next_open
            for stop in ('</think>', '</tools>', '</tool_calls>'):
                j = content.find(stop, m.end())
                if j != -1 and j < end:
                    end = j
            body = content[m.end() : end]
        args = _nemo_args_from_tag_body(tool, body)
        if args is None:
            continue
        if tool == 'finish' and close == -1 and not args.get('message'):
            continue  # a bare '<finish>' mention with nothing behind it
        found.append((m.start(), tool, args))
    return found


def _nemo_args_sane(tool: str, args: dict) -> bool:
    """Last gate before a recovered call is handed to response_to_actions."""
    allowed = _NEMO_ARG_KEYS.get(tool)
    if allowed is not None:
        for key in [k for k in args if k not in allowed]:
            del args[key]
    if tool == 'execute_bash':
        return isinstance(args.get('command'), str) and _nemo_looks_like_command(args['command'])
    if tool == 'str_replace_editor':
        return _nemo_editor_args_ok(args)
    if tool == 'finish':
        args['message'] = _nemo_clean_message(args.get('message'))
        return True
    return True


def _nemo_recover_tool_calls_impl(content: str):
    from litellm import ChatCompletionMessageToolCall

    candidates = []
    for detector in (
        _nemo_detect_json,
        _nemo_detect_function_tags,
        _nemo_detect_invoke,
        _nemo_detect_attr_calls,
        _nemo_detect_pseudo_xml,
    ):
        candidates.extend(detector(content))
    candidates.sort(key=lambda c: c[0])
    calls = []
    seen = set()
    for _pos, tool, args in candidates:
        args = _nemo_coerce_args(args)
        if args is None or not _nemo_args_sane(tool, args):
            continue
        key = (tool, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        calls.append(
            ChatCompletionMessageToolCall(
                id=f'nemo_lenient_{len(calls)}',
                type='function',
                function={'name': tool, 'arguments': json.dumps(args)},
            )
        )
        if len(calls) >= _NEMO_MAX_CALLS:
            break
    return calls or None


def _nemo_recover_tool_calls(content: str):
    """Recover tool calls the server-side parser rejected, from the raw assistant text.

    Returns a litellm ChatCompletionMessageToolCall list or None. Detectors are
    merged by position in the text and capped at _NEMO_MAX_CALLS:

    1. ``<tool_call>{json}</tool_call>`` and bare ``{"name": ..}`` objects (also
       ``parameters`` / top-level arguments; an unterminated ``finish`` JSON still
       yields its message).
    2. OpenHands' own text form ``<function=NAME><parameter=K>V</parameter>``.
    3. ``<invoke name="NAME"><parameter name="K">V</parameter>``.
    4. ``<tool_call name="NAME" k="v" .../>``.
    5. Pseudo-XML blocks: ``<execute_bash>``, ``<tool_execute>``,
       ``<function_to_execute>``, ``<finish>``, ``<str_replace_editor>``.
    """
    try:
        return _nemo_recover_tool_calls_impl(content)
    except Exception as exc:  # noqa: BLE001 -- recovery is best-effort by design
        logger.warning(f'[nemo_lenient] recovery skipped ({type(exc).__name__}: {exc})')
        return None
