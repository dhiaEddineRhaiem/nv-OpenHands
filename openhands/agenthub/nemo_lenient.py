"""NeMo lenient tool-call recovery, shared by the CodeAct and OpenCode agents.

A policy SFT'd on another harness's syntax emits tool calls in a dozen near-miss shapes that
vLLM's hermes parser rejects. Each rejected turn costs a full generation plus the SWE-bench
fake-user "Please continue" reply, and the rollout then loops to max_iteration. Measured on
exp_080 (2026-10-13): 43% of rollouts ended in that loop; the shapes recovered here cover every
variant seen in 1,655 finished rollouts (92.4% of 1,391 rejected completions became tool calls,
0 exceptions, 0 recoveries on marker-free prose).

The detectors are format-level and agent-independent. What differs per agent is the tool table:
which names (and aliases) are legal, which arguments each tool accepts and requires, which
argument a bare tag body maps to, and which sanity rule applies (shell command, absolute path,
final message). Tables are built from ``openhands.llm.tool_names`` at call time, so they follow
diversified tool names too.

Everything is gated by the LENIENT_ON flag file (or OH_LENIENT_TOOLCALLS=1), capped at
NEMO_LENIENT_MAX_CALLS calls per message, and wrapped so that a recovery bug can never kill a
rollout.

Visibility rule (2026-10-13): recovery only looks at the text AFTER the first ``</think>``, i.e.
exactly the region the server-side pipeline hands to its tool parser (vLLM's deepseek_r1 reasoning
parser splits on ``</think>`` and hermes only sees the remainder). Text inside the think block is
reasoning, never an action: recovering calls from there taught exp_080's policy within ~10 steps
to put the whole call inside ``<think>...</think>`` and stop, a shape no stock reasoning+tool
pipeline can execute. An unclosed think block is likewise not recoverable. When calls are
recovered, the content is cut before the first recovered call (what hermes does natively) so the
history never shows the call text twice.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from openhands.core.logger import openhands_logger as logger

NEMO_LENIENT_MAX_CALLS = 2
RISK_LEVELS = ('LOW', 'MEDIUM', 'HIGH')
THINK_OPEN = '<think>'
THINK_CLOSE = '</think>'
_RE_DANGLING_OPENER = re.compile(r'(?:(?:<tool_calls?>|<function=[\w.-]+>|<invoke(?:\s+name="[\w.-]+")?\s*>)\s*)+\Z')


def post_think_region(content: str) -> Optional[tuple[int, str]]:
    """Return ``(offset, text)`` of the part of ``content`` a tool parser is allowed to see.

    Mirrors vLLM's deepseek_r1 reasoning parser: everything up to the first ``</think>`` is
    reasoning, the remainder is content. ``None`` when the think block is never closed (the whole
    output is reasoning), the full text when it carries no think markup at all.
    """
    idx = content.find(THINK_CLOSE)
    if idx != -1:
        offset = idx + len(THINK_CLOSE)
        return offset, content[offset:]
    if THINK_OPEN in content:
        return None
    return 0, content


def _content_before_call(content: str, cut: int) -> str:
    """The assistant text kept once calls are recovered: what precedes the first call, minus a
    dangling ``<tool_call>``/``<tool_calls>``/``<function=..>``/``<invoke>`` opener (hermes also
    returns the text before the tag)."""
    return _RE_DANGLING_OPENER.sub('', content[:cut])


@dataclass(frozen=True)
class LenientTool:
    """How the recovery treats one tool of an agent."""

    name: str
    aliases: tuple[str, ...] = ()
    allowed: tuple[str, ...] = ()  # argument whitelist; empty = keep everything
    required: tuple[str, ...] = ()  # must be present after mapping, else the call is dropped
    raw_body_arg: Optional[str] = None  # a bare pseudo-XML body maps to this argument
    body_children: tuple[str, ...] = ()  # child tags accepted inside a pseudo-XML block
    kind: str = 'plain'  # 'command' | 'message' | 'editor' | 'plain'
    path_args: tuple[str, ...] = ()  # must be absolute when present
    command_values: tuple[str, ...] = ()  # editor: legal values of 'command'
    json_args: tuple[str, ...] = ()  # arguments the policy writes as JSON text
    defaults: tuple[tuple[str, str], ...] = ()  # filled in when missing

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name,) + self.aliases

    @property
    def accepts_xml(self) -> bool:
        return bool(self.raw_body_arg or self.body_children)


def lenient_enabled() -> bool:
    flag = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'LENIENT_ON',
    )
    return os.environ.get('OH_LENIENT_TOOLCALLS', '') in ('1', 'true') or os.path.exists(flag)


def codeact_tools() -> tuple[LenientTool, ...]:
    try:
        from openhands.llm.tool_names import (
            BROWSER_TOOL_NAME,
            EXECUTE_BASH_TOOL_NAME,
            FINISH_TOOL_NAME,
            LLM_BASED_EDIT_TOOL_NAME,
            STR_REPLACE_EDITOR_TOOL_NAME,
            TASK_TRACKER_TOOL_NAME,
        )
    except Exception:  # noqa: BLE001 -- standalone use (tests) without the package
        EXECUTE_BASH_TOOL_NAME, STR_REPLACE_EDITOR_TOOL_NAME = 'execute_bash', 'str_replace_editor'
        FINISH_TOOL_NAME, TASK_TRACKER_TOOL_NAME = 'finish', 'task_tracker'
        LLM_BASED_EDIT_TOOL_NAME, BROWSER_TOOL_NAME = 'edit_file', 'browser'
    return (
        LenientTool(
            EXECUTE_BASH_TOOL_NAME,
            aliases=('execute_bash', 'bash', 'execute_command', 'run_command', 'cmd_run',
                     'tool_execute', 'function_to_execute', 'run_bash', 'bash_exec', 'shell_run'),
            allowed=('command', 'is_input', 'timeout', 'security_risk'),
            required=('command',),
            raw_body_arg='command',
            body_children=('command', 'execute_command', 'bash-command', 'bash_command', 'cmd'),
            kind='command',
            json_args=('timeout',),
        ),
        LenientTool(
            STR_REPLACE_EDITOR_TOOL_NAME,
            aliases=('str_replace_editor', 'str_replace_based_edit_tool', 'file_editor',
                     'text_editor', 'code_editor'),
            allowed=('command', 'path', 'file_text', 'old_str', 'new_str', 'insert_line',
                     'view_range', 'security_risk'),
            required=('command', 'path'),
            body_children=('command', 'path', 'file_text', 'old_str', 'new_str', 'insert_line',
                           'view_range', 'security_risk'),
            kind='editor',
            path_args=('path',),
            command_values=('view', 'create', 'str_replace', 'insert', 'undo_edit'),
            json_args=('view_range', 'insert_line'),
        ),
        LenientTool(
            FINISH_TOOL_NAME,
            aliases=('finish',),
            allowed=('message', 'task_completed'),
            raw_body_arg='message',
            body_children=('message', 'task_completed'),
            kind='message',
        ),
        LenientTool('think', allowed=('thought',), required=('thought',)),
        LenientTool(
            TASK_TRACKER_TOOL_NAME,
            aliases=('task_tracker', 'todo_tracker', 'plan_tracker'),
            allowed=('command', 'task_list'),
            required=('command',),
            json_args=('task_list',),
        ),
        LenientTool(
            'execute_ipython_cell',
            aliases=('execute_ipython', 'ipython'),
            allowed=('code', 'security_risk'),
            required=('code',),
            raw_body_arg='code',
            body_children=('code',),
        ),
        LenientTool(
            LLM_BASED_EDIT_TOOL_NAME,
            aliases=('edit_file', 'modify_file', 'update_file', 'file_edit'),
            allowed=('path', 'content', 'start', 'end'),
            required=('path', 'content'),
            path_args=('path',),
        ),
        LenientTool(BROWSER_TOOL_NAME, aliases=('browser',), allowed=('code',), required=('code',)),
        LenientTool('web_read', allowed=('url',), required=('url',)),
    )


def opencode_tools() -> tuple[LenientTool, ...]:
    try:
        from openhands.llm.tool_names import (
            BASH_TOOL_NAME,
            EDIT_TOOL_NAME,
            FINISH_TOOL_NAME,
            GLOB_TOOL_NAME,
            GREP_TOOL_NAME,
            LIST_DIR_TOOL_NAME,
            OPENCODE_APPLY_PATCH_TOOL_NAME,
            QUESTION_TOOL_NAME,
            READ_TOOL_NAME,
            TODO_READ_TOOL_NAME,
            TODO_WRITE_TOOL_NAME,
            WRITE_TOOL_NAME,
        )
    except Exception:  # noqa: BLE001 -- standalone use (tests) without the package
        BASH_TOOL_NAME, EDIT_TOOL_NAME, READ_TOOL_NAME, WRITE_TOOL_NAME = 'bash', 'edit', 'read', 'write'
        GLOB_TOOL_NAME, GREP_TOOL_NAME, LIST_DIR_TOOL_NAME, FINISH_TOOL_NAME = 'glob', 'grep', 'list_dir', 'finish'
        OPENCODE_APPLY_PATCH_TOOL_NAME, QUESTION_TOOL_NAME = 'apply_patch', 'question'
        TODO_READ_TOOL_NAME, TODO_WRITE_TOOL_NAME = 'todo_read', 'todo_write'
    return (
        LenientTool(
            BASH_TOOL_NAME,
            aliases=('bash', 'execute_bash', 'shell', 'execute_command', 'run_command', 'cmd_run',
                     'tool_execute', 'function_to_execute'),
            allowed=('command', 'is_input', 'timeout', 'security_risk'),
            required=('command',),
            raw_body_arg='command',
            body_children=('command', 'execute_command', 'bash-command', 'bash_command', 'cmd'),
            kind='command',
            json_args=('timeout',),
            defaults=(('security_risk', 'LOW'),),
        ),
        LenientTool(
            EDIT_TOOL_NAME,
            aliases=('edit',),
            allowed=('file_path', 'old_string', 'new_string', 'replace_all'),
            required=('file_path', 'old_string', 'new_string'),
            body_children=('file_path', 'old_string', 'new_string', 'replace_all'),
            kind='editor',
            path_args=('file_path',),
            json_args=('replace_all',),
        ),
        LenientTool(
            READ_TOOL_NAME,
            aliases=('read', 'read_file'),
            allowed=('file_path', 'offset', 'limit'),
            required=('file_path',),
            raw_body_arg='file_path',
            body_children=('file_path', 'offset', 'limit'),
            path_args=('file_path',),
            json_args=('offset', 'limit'),
        ),
        LenientTool(
            WRITE_TOOL_NAME,
            aliases=('write', 'write_file'),
            allowed=('file_path', 'content'),
            required=('file_path', 'content'),
            body_children=('file_path', 'content'),
            path_args=('file_path',),
        ),
        LenientTool(GLOB_TOOL_NAME, aliases=('glob',), allowed=('pattern', 'path'), required=('pattern',),
                    raw_body_arg='pattern', body_children=('pattern', 'path')),
        LenientTool(GREP_TOOL_NAME, aliases=('grep',), allowed=('pattern', 'path', 'include'),
                    required=('pattern',), raw_body_arg='pattern', body_children=('pattern', 'path', 'include')),
        LenientTool(LIST_DIR_TOOL_NAME, aliases=('list_dir', 'list', 'ls'), allowed=('path', 'ignore'),
                    raw_body_arg='path', body_children=('path', 'ignore'), json_args=('ignore',)),
        LenientTool(
            FINISH_TOOL_NAME,
            aliases=('finish',),
            allowed=('message',),
            raw_body_arg='message',
            body_children=('message',),
            kind='message',
        ),
        LenientTool('think', allowed=('thought',), required=('thought',)),
        LenientTool(OPENCODE_APPLY_PATCH_TOOL_NAME, aliases=('apply_patch',), allowed=('patchText',),
                    required=('patchText',), raw_body_arg='patchText', body_children=('patchText', 'patch')),
        LenientTool(TODO_WRITE_TOOL_NAME, aliases=('todo_write', 'todowrite'), allowed=('todos',),
                    required=('todos',), json_args=('todos',)),
        LenientTool(TODO_READ_TOOL_NAME, aliases=('todo_read', 'todoread')),
        LenientTool(QUESTION_TOOL_NAME, aliases=('question',), allowed=('questions',),
                    required=('questions',), json_args=('questions',)),
    )


# ---------------------------------------------------------------------------
# Detectors (format-level, agent-independent)
# ---------------------------------------------------------------------------
_RE_FUNCTION = re.compile(r'<function=([\w.-]+)>(.*?)(?:</function>|(?=<function=)|\Z)', re.S)
_RE_INVOKE = re.compile(r'<invoke(?:\s+name="([\w.-]+)")?\s*>(.*?)(?:</invoke>|\Z)', re.S)
_RE_PARAM_ANY = re.compile(
    r'<(?:parameter|arg)(?:=|\s+name=")([\w.-]+)"?\s*>(.*?)</(?:parameter|arg)>', re.S
)
_RE_ATTR_CALL = re.compile(r'<tool_call\s+name="([\w.-]+)"([^<>]*?)/?>', re.S)
_RE_ATTR = re.compile(r'([\w.-]+)=("(?:[^"\\]|\\.)*"|\[[^\]]*\]|-?\d+|true|false)', re.S)
_RE_JSON_NAME = re.compile(r'\{\s*"name"\s*:')
_RE_FINISH_MSG = re.compile(r'"message"\s*:\s*"(.*)', re.S)
_RE_ANY_TAG = re.compile(r'<[^<>]*>')
_RE_SECURITY_CHILD = re.compile(r'<security[-_]risk>\s*(\w+)\s*</security[-_]risk>', re.S)
_FOREIGN_MARKUP = ('<parameter', '<invoke', '<tool_call', '<function')
_STRONG_SHELL = (
    '/', '|', '&&', '--', '=', '$', 'python', 'pytest', 'git ', 'grep', 'sed ', 'cat ',
    'ls', 'cd ', 'echo', 'find', 'pip', 'rg ', 'head', 'tail', 'awk', 'make', 'printf',
)


class _Table:
    def __init__(self, tools: Sequence[LenientTool]):
        self.tools = tuple(tools)
        self.by_name: dict[str, LenientTool] = {}
        for t in self.tools:
            for n in t.names:
                self.by_name.setdefault(n.lower(), t)
        xml = sorted({n for t in self.tools if t.accepts_xml for n in t.names}, key=len, reverse=True)
        self.re_tag_open = re.compile(r'<(' + '|'.join(re.escape(n) for n in xml) + r')(?:\s[^<>]*)?>', re.S) if xml else None
        cmd = sorted({n for t in self.tools if t.kind == 'command' for n in t.names}, key=len, reverse=True)
        self.re_cmd_attr = re.compile(
            r'<(?:' + '|'.join(re.escape(n) for n in cmd) + r')\s+command="(.{1,2000}?)"\s*(?:security_risk="\w+"\s*)?/?>', re.S
        ) if cmd else None
        self.finish_names = tuple(n for t in self.tools if t.kind == 'message' for n in t.names)

    def canonical(self, name) -> Optional[LenientTool]:
        if not isinstance(name, str):
            return None
        return self.by_name.get(name.strip().strip('"\'').lower())


def _quoted_marker(content: str, pos: int) -> bool:
    return pos > 0 and content[pos - 1] == '`'


def _child(body: str, tag: str) -> Optional[str]:
    m = re.search(r'<' + re.escape(tag) + r'>(.*?)</' + re.escape(tag) + r'>', body, re.S)
    return m.group(1).strip() if m else None


def _strip_tags(text: str) -> str:
    return _RE_ANY_TAG.sub('', text).strip()


def _param_children(body: str) -> dict:
    return {k: v.strip() for k, v in _RE_PARAM_ANY.findall(body)}


def _looks_like_command(text: str, table: _Table) -> bool:
    t = text.strip()
    if not t or t in ('...', '..') or t.startswith('<') or len(t) > 4000:
        return False
    if t.lower() in table.by_name or '**' in t or '">' in t or t.endswith('>'):
        return False
    if re.fullmatch(r'[\w.-]+\s+\.{2,}', t):
        return False
    words = len(t.split())
    sentences = t.count('. ') + (1 if t.endswith('.') else 0)
    if words > 25 and sentences >= 2:
        return False
    if any(h in t for h in _STRONG_SHELL):
        return True
    first = t.split()[0]
    return words <= 8 and (first[0].islower() or first[0] in './$~(')


def _clean_message(message) -> str:
    message = (message or '').strip() if isinstance(message, str) else ''
    return '' if message in ('...', '..') else message


def _args_from_tag_body(tool: LenientTool, body: str, table: _Table) -> Optional[dict]:
    stripped = body.strip()
    if stripped.startswith('{'):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    args: dict = dict(_param_children(body))
    risk_match = _RE_SECURITY_CHILD.search(body)
    body_wo_risk = _RE_SECURITY_CHILD.sub('', body)
    for child in tool.body_children:
        if child in args:
            continue
        value = _child(body_wo_risk, child)
        if value is None:
            continue
        # synonyms of the raw-body argument (command / execute_command / cmd ...) map onto it
        key = tool.raw_body_arg if (tool.raw_body_arg and child not in tool.allowed) else child
        args.setdefault(key, value)
    if tool.raw_body_arg and tool.raw_body_arg not in args:
        foreign = _FOREIGN_MARKUP + tuple('<' + n for n in table.by_name if n != tool.name.lower())
        if not any(m in body for m in foreign):
            raw = _strip_tags(body_wo_risk)
            if raw:
                args[tool.raw_body_arg] = raw
    if risk_match and 'security_risk' in tool.allowed:
        args.setdefault('security_risk', risk_match.group(1))
    return args or None


def _call_from_json_obj(obj, table: _Table):
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
            args = {k: v for k, v in obj.items() if k not in ('name', 'tool', 'tool_name', 'type', 'id', 'function')}
    tool = table.canonical(name)
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


def _unterminated_finish(text: str, table: _Table):
    m = re.match(r'\{\s*"name"\s*:\s*"([\w.-]+)"', text)
    if not m:
        return None
    tool = table.canonical(m.group(1))
    if tool is None or tool.kind != 'message':
        return None
    mm = _RE_FINISH_MSG.search(text)
    if not mm:
        return tool, {'message': ''}
    raw = mm.group(1)
    cut = len(raw)
    for tail in ('</tool_call>', '</think>', '</function_call>', '"}}', '"}\n', '"} '):
        idx = raw.find(tail)
        if idx != -1 and idx < cut:
            cut = idx
    raw = raw[:cut].rstrip().rstrip('}').rstrip()
    if raw.endswith('"'):
        raw = raw[:-1]
    try:
        message = json.loads('"' + raw + '"')
    except ValueError:
        message = raw.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
    return tool, {'message': message}


def _detect_json(content: str, table: _Table) -> list:
    found = []
    decoder = json.JSONDecoder()
    starts = []
    for m in re.finditer(r'<tool_call>', content):
        brace = content.find('{', m.end())
        if brace != -1:
            starts.append(brace)
    for m in _RE_JSON_NAME.finditer(content):
        starts.append(m.start())
    consumed_until = -1
    for start in sorted(set(starts)):
        if start < consumed_until:
            continue
        try:
            obj, end = decoder.raw_decode(content[start:])
        except ValueError:
            salvaged = _unterminated_finish(content[start:], table)
            if salvaged:
                found.append((start, salvaged[0], salvaged[1]))
                consumed_until = len(content)
            continue
        consumed_until = start + end
        call = _call_from_json_obj(obj, table)
        if call:
            found.append((start, call[0], call[1]))
    return found


def _tool_from_params(args: dict, table: _Table) -> Optional[LenientTool]:
    best = None
    for tool in table.tools:
        if tool.required and all(k in args for k in tool.required):
            if best is None or len(tool.required) > len(best.required):
                best = tool
    if best is None and 'message' in args:
        best = next((t for t in table.tools if t.kind == 'message'), None)
    return best


def _detect_function_tags(content: str, table: _Table) -> list:
    found = []
    foreign = ('<invoke', '<tool_call') + tuple('<' + n for n in table.by_name)
    for m in _RE_FUNCTION.finditer(content):
        tool = table.canonical(m.group(1))
        if tool is None or any(mk in m.group(2) for mk in foreign):
            continue
        args = _param_children(m.group(2))
        if not args and tool.kind != 'message':
            continue
        if tool.kind == 'message' and not args:
            if any(mk in m.group(2) for mk in _FOREIGN_MARKUP):
                continue
            args = {'message': _clean_message(_strip_tags(m.group(2)))}
        found.append((m.start(), tool, args))
    return found


def _detect_invoke(content: str, table: _Table) -> list:
    found = []
    for m in _RE_INVOKE.finditer(content):
        args = _param_children(m.group(2))
        tool = table.canonical(m.group(1)) if m.group(1) else _tool_from_params(args, table)
        if tool is None:
            continue
        if tool.kind == 'message':
            args = {'message': _clean_message(args.get('message', ''))}
        if args or tool.kind == 'message':
            found.append((m.start(), tool, args))
    return found


def _detect_attr_calls(content: str, table: _Table) -> list:
    found = []
    for m in _RE_ATTR_CALL.finditer(content):
        tool = table.canonical(m.group(1))
        if tool is None:
            continue
        args = {}
        for key, raw in _RE_ATTR.findall(m.group(2)):
            try:
                args[key] = json.loads(raw)
            except ValueError:
                args[key] = raw[1:-1] if raw.startswith('"') else raw
        if args:
            found.append((m.start(), tool, args))
    return found


def _detect_cmd_attr(content: str, table: _Table) -> list:
    found = []
    if table.re_cmd_attr is None:
        return found
    for m in table.re_cmd_attr.finditer(content):
        tool = table.canonical(m.group(0)[1:].split()[0])
        if tool is not None and _looks_like_command(m.group(1), table):
            found.append((m.start(), tool, {tool.raw_body_arg or 'command': m.group(1)}))
    return found


def _detect_pseudo_xml(content: str, table: _Table) -> list:
    found = []
    if table.re_tag_open is None:
        return found
    opens = list(table.re_tag_open.finditer(content))
    for idx, m in enumerate(opens):
        if _quoted_marker(content, m.start()) or 'command="' in m.group(0):
            continue
        raw_tag = m.group(1)
        tool = table.canonical(raw_tag)
        if tool is None:
            continue
        close = content.find('</' + raw_tag + '>', m.end())
        next_open = opens[idx + 1].start() if idx + 1 < len(opens) else len(content)
        if close != -1 and close < next_open:
            body = content[m.end():close]
        else:
            end = next_open
            for stop in ('</think>', '</tools>', '</tool_calls>'):
                j = content.find(stop, m.end())
                if j != -1 and j < end:
                    end = j
            body = content[m.end():end]
        args = _args_from_tag_body(tool, body, table)
        if args is None:
            continue
        if tool.kind == 'message':
            if 'message' not in args:
                if any(mk in body for mk in _FOREIGN_MARKUP):
                    continue
                args['message'] = _strip_tags(body)
            args['message'] = _clean_message(args.get('message'))
            if close == -1 and not args['message']:
                continue  # a bare '<finish>' mention with nothing behind it
        found.append((m.start(), tool, args))
    return found


def _coerce_args(tool: LenientTool, args: dict, table: _Table) -> Optional[dict]:
    out: dict = {}
    for key, value in args.items():
        if not isinstance(key, str):
            continue
        if tool.allowed and key not in tool.allowed:
            continue
        if key == 'security_risk':
            if isinstance(value, str) and value.strip().upper() in RISK_LEVELS:
                out[key] = value.strip().upper()
            continue
        if key in tool.json_args and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
            out[key] = value
    for key, value in tool.defaults:
        out.setdefault(key, value)
    if tool.kind == 'message':
        out['message'] = _clean_message(out.get('message'))
    for key in tool.required:
        if key not in out:
            return None
    if tool.kind == 'command':
        cmd = out.get(tool.raw_body_arg or 'command')
        if not isinstance(cmd, str) or not _looks_like_command(cmd, table):
            return None
    if tool.command_values and out.get('command') not in tool.command_values:
        return None
    for key in tool.path_args:
        if key in out and not (isinstance(out[key], str) and out[key].startswith('/')):
            return None
    return out


def _recover_impl(content: str, table: _Table, max_calls: int) -> tuple[Optional[list], str]:
    from litellm import ChatCompletionMessageToolCall

    region = post_think_region(content)
    if region is None:
        return None, content
    offset, text = region
    candidates = []
    for detector in (_detect_json, _detect_function_tags, _detect_invoke, _detect_attr_calls,
                     _detect_cmd_attr, _detect_pseudo_xml):
        candidates.extend(detector(text, table))
    candidates.sort(key=lambda c: c[0])
    calls = []
    seen = set()
    first_pos = None
    for pos, tool, args in candidates:
        args = _coerce_args(tool, args, table)
        if args is None:
            continue
        key = (tool.name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        if first_pos is None:
            first_pos = pos
        calls.append(
            ChatCompletionMessageToolCall(
                id=f'nemo_lenient_{len(calls)}',
                type='function',
                function={'name': tool.name, 'arguments': json.dumps(args)},
            )
        )
        if len(calls) >= max_calls:
            break
    if not calls:
        return None, content
    return calls, _content_before_call(content, offset + first_pos)


def recover_tool_calls_and_content(
    content: str, tools: Sequence[LenientTool], max_calls: int = NEMO_LENIENT_MAX_CALLS
) -> tuple[Optional[list], str]:
    """Recover tool calls the server-side parser rejected, from the raw assistant text.

    Returns ``(calls, content)``: a litellm ChatCompletionMessageToolCall list or None, and the
    assistant content to keep -- unchanged when nothing is recovered, otherwise cut before the
    first recovered call exactly as hermes does for a native ``<tool_call>``. Only the text after
    the first ``</think>`` is searched (see the module docstring). Shapes, merged by position in
    that text: ``<tool_call>{json}</tool_call>`` and bare ``{"name": ..}`` objects (also
    ``parameters`` / top-level arguments; an unterminated final-message JSON still yields its
    message), ``<function=NAME><parameter=K>V</parameter>``, ``<invoke name="NAME">`` with
    ``<parameter name>``/``<arg name>`` children (name optional), ``<tool_call name=.. k="v"/>``,
    ``<bash command="..">``, and pseudo-XML blocks named after any tool of the table.
    """
    try:
        return _recover_impl(content, _Table(tools), max_calls)
    except Exception as exc:  # noqa: BLE001 -- recovery is best-effort by design
        logger.warning(f'[nemo_lenient] recovery skipped ({type(exc).__name__}: {exc})')
        return None, content


def recover_tool_calls(content: str, tools: Sequence[LenientTool], max_calls: int = NEMO_LENIENT_MAX_CALLS):
    """Calls-only form of :func:`recover_tool_calls_and_content`."""
    return recover_tool_calls_and_content(content, tools, max_calls)[0]
