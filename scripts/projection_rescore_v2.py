"""Reference-free terminal-answer extraction with exact integer comparison.

The immutable v1 module supplies brace parsing and bounded symbolic conversion.
This is still a post-hoc automatic scorer, not a proof/semantic judge.
"""
import math
import re
from fractions import Fraction

import projection_rescore_v1 as v1

VERSION = 'projection-answer-sensitivity-v2'
HEADING = re.compile(r'(?im)^\s*(?:#{1,6}\s*|\*\*)?(?:final\s+answer|conclusion|answer)\s*:?\s*(?:\*\*)?\s*$')
PREDICATE = re.compile(r'\b(?:is|are|has|have|equals?|takes?|needs?|requires?|simplifies?\s+to|evaluates?\s+to|total\s+of|result\s+of|size)\b\s*[:=]?\s*', re.I)
NUMBER = re.compile(v1.NUMBER)


def paragraph_spans(text):
    return [(m.start(), m.group().strip()) for m in re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)', text) if m.group().strip()]


def leading_atom(text):
    """Return an expression only when it starts this phrase, never any matching reference."""
    text=text.lstrip().removeprefix('**').lstrip()
    text=re.sub(r'^(?:a|an)\s+(?=[+-]?\d)', '', text, flags=re.I)
    m=v1.MATH.match(text)
    if m:
        return v1.top_level_rhs(next(x for x in m.groups() if x is not None)),m.end(),[]
    n=NUMBER.match(text)
    if n:
        suffix=text[n.end():]
        # More than one proposed value is a list, not one selected answer.
        if re.match(r'\s*,\s*(?:and\s+)?[+-]?\d',suffix) or re.match(r'\s+(?:and|or)\s+[+-]?\d',suffix,re.I):
            return text.strip(),len(text),['multiple_final_values']
        if re.match(r'\s*[_^+*/]|\s*[-−]\s*(?:\d|[({\\])',suffix):
            return text.strip(),len(text),['unwrapped_expression']
        return n.group(),n.end(),[]
    if re.match(r'^[+-]?\s*\\(?:frac|dfrac|tfrac|sqrt|begin)',text):
        return text.strip(),len(text),['unwrapped_expression']
    return None,0,[]


def candidate_from_paragraph(text):
    """Prefer the last answer-bearing predicate over subject names or boilerplate."""
    plain=text.replace('**','').strip()
    math_spans=[(m.start(),m.end()) for m in v1.MATH.finditer(plain)]
    claims=[]
    for p in PREDICATE.finditer(plain):
        if any(a <= p.start() < b for a,b in math_spans):continue
        value,end,flags=leading_atom(plain[p.end():])
        if value is not None:claims.append((p.start(),value,flags))
    if claims:
        _,value,flags=claims[-1]
        return value,flags
    # An answer marker itself provides a boundary; no period splitting of Mr./Dr.
    markers=list(v1.MARKER.finditer(plain))
    if markers:
        value,_,flags=leading_atom(plain[markers[-1].end():])
        if value is not None:return value,flags
    prefix=re.sub(r'^\s*(?:therefore|thus|hence|so)\b\s*[,;:]?\s*','',plain,flags=re.I)
    value,_,flags=leading_atom(prefix)
    if value is not None:return value,flags
    spans=list(v1.MATH.finditer(plain))
    if spans:
        value=v1.top_level_rhs(next(x for x in spans[-1].groups() if x is not None))
        return value,['terminal_math_fallback']
    return None,[]


def extract(response, task, legacy_candidate=None):
    found=v1.boxes(response)
    if found:
        value,start,end=found[-1]
        return dict(candidate=value,rule='last_balanced_box',flags=['substantial_text_after_box'] if len(response[end:].strip())>180 else [])
    hashes=list(re.finditer(r'####\s*([^\n]+)',response))
    if task=='gsm8k' and hashes:
        value,_,flags=leading_atom(hashes[-1].group(1))
        if value is not None:return dict(candidate=value,rule='last_hash_answer',flags=flags)
    paragraphs=paragraph_spans(response)
    headings=list(HEADING.finditer(response))
    if headings:
        after=[p for p in paragraphs if p[0]>=headings[-1].end()]
        # A heading and its paragraph can share a single-newline block.
        tail=response[headings[-1].end():].strip()
        if tail:
            after=paragraph_spans(tail)
            if after:
                value,flags=candidate_from_paragraph(after[-1][1])
                if value is not None:return dict(candidate=value,rule='final_answer_section',flags=flags)
    if paragraphs:
        value,flags=candidate_from_paragraph(paragraphs[-1][1])
        if value is not None:return dict(candidate=value,rule='terminal_paragraph',flags=flags)
    # Explicit final answer takes precedence over a later prose-only caveat.
    markers=list(v1.MARKER.finditer(response))
    if markers:
        tail=response[markers[-1].end():]
        value,_,flags=leading_atom(tail)
        if value is not None:return dict(candidate=value,rule='last_answer_marker',flags=flags+['later_prose_caveat'])
    return dict(candidate=legacy_candidate,rule='frozen_parser_fallback',flags=['fallback_extraction'])


def canonical(value,task):
    text=v1.canonical(value,task)
    text=re.sub(r'_\{([0-9]+)\}',r'_\1',text)
    text=re.sub(r'\\sqrt\s*([A-Za-z0-9])',r'\\sqrt{\1}',text)
    text=re.sub(r'\\frac\s*([0-9])\s*([0-9])',r'\\frac{\1}{\2}',text)
    # An explicit approximate assignment is a scalar claim, with the usual
    # numeric tolerance applied below only when at least one number is noninteger.
    text=re.sub(r'^[A-Za-z]\s*\\approx\s*','',text)
    return text


def match(candidate,reference,task):
    a,b=canonical(candidate,task),canonical(reference,task)
    if not a or not b:return False,['missing_candidate']
    if re.sub(r'\s+','',a).lower()==re.sub(r'\s+','',b).lower():return True,[]
    try:
        x,y=Fraction(a),Fraction(b)
        if x.denominator==y.denominator==1:return x==y,[]
        return math.isclose(float(x),float(y),rel_tol=.001,abs_tol=1e-6),[]
    except (ValueError,ZeroDivisionError):pass
    return v1.match(a,b,task)
