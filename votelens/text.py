"""Extract readable comment text from wikitext for toxicity scoring."""

import html
import re
from typing import NamedTuple
import unicodedata

import mwparserfromhell
from mwparserfromhell.nodes import Comment, ExternalLink, HTMLEntity, Tag, Template, Text, Wikilink

TEXT_VERSION = "wikitext-v1"
TEXT_STATUSES = ("ready", "empty", "symbols_only", "needs_review")
FORMATTING_TAGS = {
    "a", "b", "big", "em", "font", "i", "small", "span", "strong", "sub", "sup", "u",
}
VOTE_LABEL = re.compile(
    r"(?:(?:strong|weak|mild|qualified|conditional|provisional|tentative|strongly|weakly|very)\s+)*"
    r"(support|oppose|opposition|neutral|abstain)\b", re.I,
)
VOTE_DIRECTIONS = {"support": 1, "oppose": -1, "opposition": -1, "neutral": 0, "abstain": 0}
VOTE_SEPARATOR = re.compile(r"^\s*(?:[.:,;!]+|--+|[—–])(?=\s|$)\s*")
USER_TITLE = re.compile(r"^(?:user(?:[ _]talk)?\s*:|special\s*:\s*contributions\s*/)(.+)$", re.I)
SIGNATURE_DELIMITER = re.compile(r"(?:--+|[—–]|~{2,})\s*$")
SIGNATURE_DATE = re.compile(r"\s*\d{1,2}:\d{2},?\s+\d{1,2}\s+[a-z]+\s+\d{4}\s*\(UTC\)\s*$", re.I)
SIGNATURE_LABELS = {"talk", "contribs", "contributions"}
USER_STUB = re.compile(r"\[\[\s*(?:user(?:[ _]talk)?\s*:|special:contributions(?:/|$))[^|\]\n]*(?:\|\s*)?$", re.I)
MARKUP = re.compile(r"\[\[|\]\]|\{\{|\}\}|'''|</?[a-z][^>]*>", re.I)
FORMAT_TAG = re.compile(
    rf"""</?(?:{'|'.join(sorted(FORMATTING_TAGS))})(?=\s|/?>)(?:[^<>"']|"[^"]*"|'[^']*')*>""",
    re.I,
)
ORPHAN_QUOTES = re.compile(r"(?<!')'{3,}\s*$")
ORPHAN_FORMAT_TAIL = re.compile(r"</?(?:span|font|small|big|sup|sub|b|i|strong|em)\b[^>]*$", re.I)
BARE_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\[\]]+", re.I)
EMBEDDED_URL = re.compile(r"(?<=\w)https?://", re.I)
LEADING_EMOTICON = re.compile(r"^(\s*)([:;])(?=[-^']?[()DPp/\\])")
VISIBLE_TEMPLATE_PARAMETER = {
    "small": 1, "smaller": 1, "big": 1, "larger": 1, "nowrap": 1, "nobr": 1,
    "em": 1, "strong": 1, "color": 2, "lang": 2, "u": 1, "user": 1, "user link": 1,
}
SIGNATURE_TEMPLATES = {"unsigned", "unsigned2", "unsigned ip"}
REVIEW_FLAGS = {"unknown_template", "unparsed_markup", "possible_signature"}


class CleanedComment(NamedTuple):
    txt_clean: str
    text_status: str
    text_flags: str


def _account(text):
    normalized = unicodedata.normalize("NFC", text).replace("_", " ")
    return " ".join(normalized.split()).casefold()


def _is_vote_label(text, vote):
    label = VOTE_LABEL.fullmatch(text.strip().rstrip("!.,;:").strip())
    return bool(label and VOTE_DIRECTIONS[label[1].lower()] == vote)


def _strip_vote_label(nodes, vote, flags):
    for position, node in enumerate(nodes):
        if isinstance(node, Comment) or (isinstance(node, Text) and not str(node).strip()):
            continue
        remaining = nodes[position + 1:]
        if isinstance(node, Tag) and str(node.tag).casefold() in {"b", "i", "strong", "em"}:
            inner_flags = set()
            label_text = _render_nodes(node.contents.nodes, inner_flags)
            if not (inner_flags & REVIEW_FLAGS) and _is_vote_label(label_text, vote):
                flags.add("vote_label_removed")
                if remaining and isinstance(remaining[0], Text):
                    return [Text(VOTE_SEPARATOR.sub("", str(remaining[0]), count=1)), *remaining[1:]]
                return remaining
        if isinstance(node, Text):
            first_text = str(node).lstrip().lstrip(":*#;").lstrip()
            if not first_text:
                continue
            label = VOTE_LABEL.match(first_text)
            if label and VOTE_DIRECTIONS[label[1].lower()] == vote:
                rest = first_text[label.end():]
                separator = VOTE_SEPARATOR.match(rest)
                if separator or (not rest.strip() and not remaining):
                    flags.add("vote_label_removed")
                    tail = rest[separator.end():] if separator else ""
                    return [Text(tail), *remaining]
        break
    return nodes


def _signature_parts(nodes):
    for node in nodes:
        if isinstance(node, Tag) and str(node.tag).casefold() in FORMATTING_TAGS:
            yield from _signature_parts(node.contents.nodes)
        elif not isinstance(node, Comment):
            yield node


def _signature_kind(nodes, voter):
    """Classify a possible tail using links to the voter and decoration."""
    has_identity = False
    has_user_link = False
    decoration_only = True
    for node in _signature_parts(nodes):
        if isinstance(node, Wikilink):
            title = str(node.title).strip()
            user = USER_TITLE.fullmatch(title)
            label_flags = set()
            display = _render_nodes(node.text.nodes, label_flags) if node.text is not None else ""
            clean_label = not (label_flags & REVIEW_FLAGS) and not MARKUP.search(display)
            if user:
                has_user_link = True
                target = _account(user[1].split("#", 1)[0])
                label = _account(display if node.text is not None else user[1])
                same_account = bool(voter) and (target == voter or target.startswith(voter + "/"))
                short_alias = not label or voter.startswith(label) or label == target or label in SIGNATURE_LABELS
                matched = same_account and short_alias and clean_label
                has_identity |= matched
                decoration_only &= matched
            else:
                decoration_only &= (
                    clean_label and _account(title) == "wikipedia:esperanza" and 0 < len(display.strip()) <= 3
                )
        elif isinstance(node, (Text, HTMLEntity)):
            text = html.unescape(str(node))
            footer = SIGNATURE_DATE.sub("", text)
            decoration_only &= not any(character.isalnum() for character in footer)
        else:
            decoration_only = False
    return has_identity and decoration_only, has_user_link


def _strip_signature(nodes, voter, flags):
    voter = _account(voter)
    if _signature_kind(nodes, voter)[0]:
        flags.add("signature_removed")
        return []

    for position, node in reversed(list(enumerate(nodes[:-1]))):
        if not isinstance(node, Text):
            continue
        marker = SIGNATURE_DELIMITER.search(str(node))
        if not marker:
            continue
        confirmed, possible = _signature_kind(nodes[position + 1:], voter)
        if confirmed:
            flags.add("signature_removed")
            return [*nodes[:position], Text(str(node)[:marker.start()])]
        if possible:
            flags.add("possible_signature")
            break
    return nodes


def _render_nodes(nodes, flags):
    return "".join(_render_node(node, flags) for node in nodes)


def _render_node(node, flags):
    if isinstance(node, Text):
        text, format_tags = FORMAT_TAG.subn("", str(node))
        text, user_stub = USER_STUB.subn(" ", text)
        text, tag_stub = ORPHAN_FORMAT_TAIL.subn(" ", text)
        text, quotes = ORPHAN_QUOTES.subn("", text)
        if format_tags or user_stub or tag_stub or quotes:
            flags.add("markup_repaired")
        return html.unescape(text)
    if isinstance(node, HTMLEntity):
        return node.normalize()
    if isinstance(node, Comment):
        return " "
    if isinstance(node, Tag):
        tag = str(node.tag).casefold()
        if tag in {"s", "strike", "del"}:
            flags.add("struck_text_removed")
            return " "
        if tag in {"ref", "references", "script", "style", "math", "br", "hr"}:
            return " "
        if tag in {"nowiki", "pre", "code"}:
            return html.unescape(str(node.contents))
        return _render_nodes(node.contents.nodes, flags)
    if isinstance(node, Wikilink):
        title = str(node.title).strip()
        if title.casefold().startswith(("file:", "image:", "category:")):
            return " "
        if node.text is None:
            return title.lstrip(":").replace("_", " ")
        if not str(node.text).strip():
            flags.add("unparsed_markup")
            return str(node)
        return _render_nodes(node.text.nodes, flags)
    if isinstance(node, ExternalLink):
        flags.add("url_removed")
        return _render_nodes(node.title.nodes, flags) if node.title is not None else " "
    if isinstance(node, Template):
        name = _account(str(node.name)).removeprefix("template:").strip()
        parameter = VISIBLE_TEMPLATE_PARAMETER.get(name)
        if parameter is not None and node.has(parameter):
            return _render_nodes(node.get(parameter).value.nodes, flags)
        if name in SIGNATURE_TEMPLATES:
            flags.add("signature_removed")
            return " "
        flags.add("unknown_template")
        return str(node)
    flags.add("unparsed_markup")
    return str(node)


def _remove_url(match):
    url = match.group()
    return " " + url[len(url.rstrip(".,!?;)")):]


def clean_comment(raw, vote, voter):
    flags = set()
    # A colon or semicolon at the start otherwise becomes list markup.
    parse_input = LEADING_EMOTICON.sub(lambda match: f"{match[1]}&#{ord(match[2])};", raw)
    nodes = _strip_vote_label(mwparserfromhell.parse(parse_input).nodes, vote, flags)
    nodes = _strip_signature(nodes, voter, flags)
    text = _render_nodes(nodes, flags)
    text, removed_urls = BARE_URL.subn(_remove_url, text)
    if removed_urls:
        flags.add("url_removed")
    text = unicodedata.normalize("NFC", " ".join(text.split()))
    text = re.sub(r"(?<=\w) ([.,!?])", r"\1", text)
    text = re.sub(r"(?<=[.!?])\s*--+$", "", text)
    if MARKUP.search(text) or EMBEDDED_URL.search(text):
        flags.add("unparsed_markup")

    if flags & REVIEW_FLAGS:
        status = "needs_review"
    elif not text:
        status = "empty"
    elif any(character.isalnum() for character in text):
        status = "ready"
    else:
        status = "symbols_only"
    return CleanedComment(text, status, "|".join(sorted(flags)))
