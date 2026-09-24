"""Post-hoc extraction sensitivity analysis, not a semantic correctness oracle.

Extraction cannot see the reference, method, or frozen label. No external calls.
The frozen OpenAct evaluator and generation results are never modified.
"""
import re
import signal
from contextlib import contextmanager

VERSION = 'projection-answer-sensitivity-v1'
NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?'
MARKER = re.compile(r'\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\b\s*[:=]?|[:=])\s*', re.I)
CONCLUSION = re.compile(r'\b(?:therefore|thus|hence|so)\b\s*[,;:]?\s*', re.I)
MATH = re.compile(r'\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|(?<!\\)\$(.*?)(?<!\\)\$', re.S)


def braced(text, start):
    if start >= len(text) or text[start] != '{':
        raise ValueError('expected brace')
    depth = 1
    for i in range(start + 1, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        if depth == 0: return text[start + 1:i], i + 1
    raise ValueError('unclosed brace')


def boxes(text):
    found = []
    for m in re.finditer(r'\\(?:boxed|fbox)\s*\{', text):
        try:
            value, end = braced(text, m.end() - 1)
            if value.strip(): found.append((value.strip(), m.start(), end))
        except ValueError:
            continue
    return found


def top_level_rhs(text):
    """Keep the result of an equality chain, not an inequality or system."""
    depth = 0; equal = []
    for i, c in enumerate(text):
        if c in '{[(': depth += 1
        elif c in '}])': depth -= 1
        elif c == '=' and depth == 0: equal.append(i)
    if equal and not re.search(r'[,;<>]|\\(?:le|ge|ne|approx)', text):
        return text[equal[-1] + 1:].strip()
    return text.strip()


def terminal_segment(text):
    """First sentence/newline outside balanced display/inline math."""
    text = text.lstrip()
    spans = [(m.start(), m.end()) for m in MATH.finditer(text)]
    for i, c in enumerate(text):
        if any(a <= i < b for a, b in spans): continue
        if c == '\n' or (c == '.' and not (i and i + 1 < len(text) and text[i-1].isdigit() and text[i+1].isdigit())):
            if text[:i].strip(): return text[:i].strip()
    return text.strip()


def from_segment(segment, task):
    segment = terminal_segment(segment).replace('**', '').strip()
    spans = list(MATH.finditer(segment))
    flags = []
    if spans:
        # Choose the expression after a final equality when it is written in a
        # separate inline span; otherwise the first answer-bearing math span.
        start = 0
        is_matches = list(re.finditer(r'\b(?:is|equals?)\b\s*[:=]?\s*', segment, re.I))
        if is_matches:
            tail = [s for s in spans if s.start() >= is_matches[-1].end()]
            if tail: spans = tail
        if len(spans) > 1: flags.append('multiple_math_spans_in_conclusion')
        raw = next(g for g in spans[start].groups() if g is not None)
        return top_level_rhs(raw), flags
    is_matches = list(re.finditer(r'\b(?:is|equals?)\b\s*[:=]?\s*', segment, re.I))
    if is_matches: segment = segment[is_matches[-1].end():]
    if '=' in segment: segment = top_level_rhs(segment)
    # Preserve an unwrapped TeX expression instead of extracting its last digit.
    if re.match(r'^[+-]?\s*\\(?:frac|dfrac|tfrac|sqrt|begin)', segment):
        return segment.strip(), flags
    nums = list(re.finditer(NUMBER, segment))
    if nums:
        if len(nums) > 1: flags.append('multiple_numbers_in_conclusion')
        n = nums[0]
        suffix = segment[n.end():]
        # Keep base notation and arithmetic, rather than accepting a prefix.
        if re.match(r'\s*[_^+*/−-]', suffix):
            return segment[n.start():].strip(), flags
        return n.group(), flags
    if segment:
        if len(segment.split()) > 4: flags.append('prose_candidate')
        return segment.strip(), flags
    return None, ['empty_conclusion']


def extract(response, task, legacy_candidate=None):
    """Deterministic rule; no ground truth or experimental condition inputs."""
    found = boxes(response)
    if found:
        raw, start, end = found[-1]
        flags = []
        if response[end:].strip() and len(response[end:].strip()) > 180:
            flags.append('substantial_text_after_box')
        return dict(candidate=raw, rule='last_balanced_box', flags=flags)
    hash_answers = list(re.finditer(r'####\s*([^\n]+)', response))
    if task == 'gsm8k' and hash_answers:
        raw, flags = from_segment(hash_answers[-1].group(1), task)
        return dict(candidate=raw, rule='last_hash_answer', flags=flags)
    for rule, pattern in [('last_answer_marker', MARKER), ('last_conclusion', CONCLUSION)]:
        matches = list(pattern.finditer(response))
        if matches:
            raw, flags = from_segment(response[matches[-1].end():], task)
            if raw:
                return dict(candidate=raw, rule=rule, flags=flags)
    # A final displayed equation is useful but less reliable than an explicit answer.
    spans = list(MATH.finditer(response))
    if spans and len(response[spans[-1].end():].strip()) <= 180:
        raw = next(g for g in spans[-1].groups() if g is not None)
        return dict(candidate=top_level_rhs(raw), rule='last_math_fallback', flags=['fallback_extraction'])
    return dict(candidate=legacy_candidate, rule='frozen_parser_fallback', flags=['fallback_extraction'])


def canonical(value, task):
    if value is None: return ''
    text = str(value).strip().replace('−', '-').replace('**', '')
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    text = text.replace('\\left', '').replace('\\right', '')
    for delimiter in ('\\[', '\\]', '\\(', '\\)', '$', '\\,', '\\!', '\\;', '\\quad', '\\qquad'):
        text = text.replace(delimiter, '')
    for cmd in ('text', 'mathrm', 'mathbf'):
        while re.search(r'\\' + cmd + r'\{([^{}]*)\}', text):
            text = re.sub(r'\\' + cmd + r'\{([^{}]*)\}', r'\1', text)
    text = text.strip().rstrip('.,;:!').strip()
    # Only a single-variable assignment; not a multi-variable solution or inequality.
    if re.match(r'^[A-Za-z]\s*=', text) and text.count('=') == 1 and not re.search(r'[,;<>]', text):
        text = text.split('=', 1)[1].strip()
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    if task == 'gsm8k':
        text = re.sub(r'\s*(?:\\?%|dollars?|minutes?|hours?|days?|beads?|miles?|cups?)\s*$', '', text, flags=re.I)
    for style in ('pmatrix', 'bmatrix', 'vmatrix'):
        # Determinant delimiters have a different meaning and must remain distinct.
        if style != 'vmatrix': text = text.replace('{'+style+'}', '{matrix}')
    return text.strip()


def tex_expression(text):
    """Recursive fractions/roots; rejects unsupported commands for review."""
    out = ''; i = 0
    while i < len(text):
        if text.startswith('\\frac', i):
            j = i + 5
            while j < len(text) and text[j].isspace(): j += 1
            a, j = braced(text, j)
            while j < len(text) and text[j].isspace(): j += 1
            b, i = braced(text, j)
            out += '((' + tex_expression(a) + ')/(' + tex_expression(b) + '))'
        elif text.startswith('\\sqrt', i):
            j = i + 5
            while j < len(text) and text[j].isspace(): j += 1
            a, i = braced(text, j)
            out += 'sqrt(' + tex_expression(a) + ')'
        else:
            out += text[i]; i += 1
    out = out.replace('\\cdot', '*').replace('\\times', '*').replace('\\pi', 'pi')
    out = out.replace('{', '(').replace('}', ')')
    if len(out) > 1600 or re.search(r'[^A-Za-z0-9\s()+*/^.,-]', out):
        raise ValueError('unsupported symbolic syntax')
    if re.search(r'\.\s*[A-Za-z]|[A-Za-z]\s*\.', out):
        raise ValueError('not a mathematical decimal')
    allowed = {'sqrt', 'pi', 'e', 'sin', 'cos', 'tan', 'log', 'exp'}
    if any(len(x) > 1 and x not in allowed for x in re.findall(r'[A-Za-z]+', out)):
        raise ValueError('unsupported symbolic identifier')
    return out


@contextmanager
def time_limit(seconds=1.):
    def expired(*args): raise TimeoutError('symbolic equality timeout')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def match(candidate, reference, task):
    """Same numeric tolerances as the frozen evaluator; symbolic failures flagged."""
    import math
    from fractions import Fraction
    a, b = canonical(candidate, task), canonical(reference, task)
    if not a or not b: return False, ['missing_candidate']
    if re.sub(r'\s+', '', a).lower() == re.sub(r'\s+', '', b).lower():
        return True, []
    try:
        aa, bb = float(Fraction(a)), float(Fraction(b))
        return math.isclose(aa, bb, rel_tol=0.001, abs_tol=1e-6), []
    except (ValueError, ZeroDivisionError): pass
    try:
        a, b = tex_expression(a), tex_expression(b)
        with time_limit():
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
            ts = standard_transformations + (implicit_multiplication_application, convert_xor)
            local = {'sqrt':sp.sqrt, 'pi':sp.pi, 'e':sp.E}
            aa = parse_expr(a, local_dict=local, transformations=ts)
            bb = parse_expr(b, local_dict=local, transformations=ts)
            return bool(sp.simplify(aa-bb) == 0), []
    except Exception as e:
        return False, ['symbolic_' + type(e).__name__]
