"""Conservative local comparison of regulations with traceable assignments.

Structure and role headings define owners. Text similarity is a retrieval
heuristic, not a calibrated probability or proof of organizational succession.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .models import AnalysisResult, Document, Finding, Fragment, Function, FunctionMatch, MatrixRow, UnitChange
from .confidence import assess_result

UNKNOWN_OWNER = "Владелец не установлен"
UNIT_WORDS = r"(?:департамент|отдел|управление|служба|сектор|центр|бюро|блок|комитет)"
UNIT_START = re.compile(rf"^({UNIT_WORDS})\s+", re.I)
NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.|\))?\s*")
LETTER = re.compile(r"^\s*([а-яa-z])[.)]\s+", re.I)
ACTION_ROOTS = (
    "осуществл", "обеспеч", "провод", "провед", "контрол", "провер", "разрабатыв", "разработ",
    "согласов", "согласу", "утвержд", "организ", "отвеча", "формир", "оценив", "оценк",
    "подгот", "готов", "выполн", "координир", "расслед", "консульт", "взаимодейств",
    "анализир", "представл", "запрашив", "вынос", "внос", "инициир", "актуализир",
    "консолидир", "участв", "участи", "содейств", "использ", "выявл", "определ",
    "предлаг", "получ", "присутств", "довод", "вести", "ведет", "ведут", "ведение",
    "копир", "опечат", "изуч", "пользов", "расшир", "требов", "информир", "руковод",
    "соблюд", "придержив", "оказыв", "приним", "подписыв", "раскрыв", "внедр",
    "создан", "созда", "хран", "обработ", "веден", "мониторинг", "аудит",
)
ACTION = re.compile(r"\b(?:" + "|".join(ACTION_ROOTS) + r")[а-яё]*\b", re.I)
FINITE = re.compile(
    r"\b(?:осуществля(?:ет|ют)|обеспечива(?:ет|ют)|провод(?:ит|ят)|контролиру(?:ет|ют)|"
    r"проверя(?:ет|ют)|разрабатыва(?:ет|ют)|согласовыва(?:ет|ют)|утвержда(?:ет|ют)|"
    r"вед[её]т|ведут|организу(?:ет|ют)|отвеча(?:ет|ют)|формиру(?:ет|ют)|оценива(?:ет|ют)|"
    r"подготавлива(?:ет|ют)|готов(?:ит|ят)|выполня(?:ет|ют)|координиру(?:ет|ют)|"
    r"взаимодейству(?:ет|ют)|анализиру(?:ет|ют)|представля(?:ет|ют)|запрашива(?:ет|ют)|"
    r"вынос(?:ит|ят)|участву(?:ет|ют)|консолидиру(?:ет|ют)|актуализиру(?:ет|ют)|"
    r"консультиру(?:ет|ют)|иницииру(?:ет|ют)|соблюда(?:ет|ют)|хран(?:ит|ят))\b", re.I,
)
MODAL = re.compile(r"\b(?:не\s+име(?:ет|ют)\s+прав[ао]|име(?:ет|ют)\s+право|обязан[аы]?|долж(?:ен|ны|на)|вправе|запрещено)\b", re.I)
STRUCTURE_INTRO = re.compile(r"состоит\s+из|следующ\w*\s+структурн\w*\s+подразделени|в\s+структуру.+вход|структура.+включа", re.I)
STOP = set("и в во на по с со для от до из за при а или его ее их это том числе общества компании организации соответствии настоящего положения функций функции работы деятельность деятельности рамках части внутреннего внутренних аудита бва вопросам отношении который которые также осуществляет обеспечивает проводит выполняет".split())


def _body(text: str) -> str:
    return LETTER.sub("", NUMBER.sub("", text, count=1), count=1).strip(" •–—\t")


def _clause(text: str) -> str:
    match = NUMBER.match(text)
    return match.group(1) if match else ""


def _norm(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"проект(?:ов|ы)? документации,? регламентирующей работу бва", "внд бва", text)
    text = re.sub(r"внутренних нормативных документов", "внд", text)
    return " ".join(re.findall(r"[а-яa-z0-9]+", text))


@lru_cache(maxsize=30000)
def _stem(word: str) -> str:
    word = word.lower().replace("ё", "е")
    for root, canonical in (("готов", "подгот"), ("подгот", "подгот"), ("провед", "провод"), ("согласу", "соглас"), ("согласов", "соглас")):
        if word.startswith(root):
            return canonical
    for root in ACTION_ROOTS:
        if len(root) >= 5 and word.startswith(root):
            return root
    for suffix in ("иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "иях", "ия", "ие", "ии", "ий", "ый", "ая", "ое", "ые", "ой", "ей", "ых", "их", "ов", "ев", "ам", "ям", "ах", "ях", "ом", "ем", "ую", "юю", "а", "я", "ы", "и", "у", "ю", "е"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


@lru_cache(maxsize=12000)
def _tokens(text: str) -> set[str]:
    return {_stem(w) for w in _norm(text).split() if len(w) > 2 and w not in STOP and not w.isdigit()}


@lru_cache(maxsize=40000)
def _similarity(a: str, b: str) -> float:
    a, b = _body(a), _body(b)
    if _norm(a) == _norm(b):
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    return (0.55 * overlap / len(ta | tb) + 0.25 * overlap / min(len(ta), len(tb))
            + 0.20 * SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio())


def _clean_unit(value: str) -> str:
    value = re.split(r"\(|[;:.]|\s+[—–]\s", value)[0]
    match = FINITE.search(value) or MODAL.search(value)
    if match:
        value = value[:match.start()]
    value = re.sub(r"\s+не\s*$", "", value, flags=re.I)
    return re.sub(r"\s+", " ", value.strip(" ,.-–—"))


@dataclass
class _Catalog:
    units: dict[str, list[Fragment]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, source: Fragment) -> None:
        name = _clean_unit(name)
        if not UNIT_START.match(name) or len(name) > 160 or len(name.split()) < 2:
            return
        existing = next((n for n in self.units if _norm(n) == _norm(name)), name)
        self.units.setdefault(existing, []).append(source)
        for parenthesis in re.findall(r"\(([^)]*)\)", source.text):
            for alias in re.findall(r"\b[А-ЯЁA-Z][А-ЯЁA-Z0-9-]{1,15}\b", parenthesis):
                self.aliases[alias] = existing

    def mentions(self, text: str) -> list[str]:
        words = " " + " ".join(_stem(w) for w in _norm(text).split()) + " "
        result = []
        for unit in self.units:
            needle = " " + " ".join(_stem(w) for w in _norm(unit).split()) + " "
            if needle in words:
                result.append(unit)
        for alias, unit in self.aliases.items():
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text) and unit not in result:
                result.append(unit)
        return result


def _catalog(documents: list[Document]) -> _Catalog:
    catalog = _Catalog()
    for doc in documents:
        in_structure = False
        for f in doc.fragments:
            body = _body(f.text)
            if STRUCTURE_INTRO.search(body):
                in_structure = True
                continue
            if in_structure and UNIT_START.match(body):
                catalog.add(body, f)
            elif in_structure and body:
                in_structure = False
    if catalog.units:
        return catalog
    for doc in documents:
        for f in doc.fragments:
            body = _body(f.text)
            if UNIT_START.match(body) and (FINITE.search(body) or MODAL.search(body) or
                    (len(body) < 140 and not re.search(r"[.!?].+", body))):
                catalog.add(body, f)
    return catalog


def _subject_owners(body: str, catalog: _Catalog, *, heading: bool = False) -> list[str]:
    predicate = FINITE.search(body) or MODAL.search(body)
    prefix = body[:predicate.start()] if predicate else body
    owners = catalog.mentions(prefix)
    if owners:
        return owners
    if re.match(r"^руководство\s+(?:бва|блока внутреннего аудита)\s+осуществля\w*\s+главный\s+аудитор\b", body, re.I):
        return ["Главный аудитор"]
    if heading and re.search(r"^директоры\s+департаментов\b", body, re.I):
        # An unresolved explicit abbreviation narrows the group; it must not
        # silently expand to every known department.
        if re.search(r"\b[А-ЯЁA-Z]{2,15}\b", prefix):
            return [UNKNOWN_OWNER]
        return [u for u in catalog.units if u.lower().startswith("департамент")]
    if re.match(r"^(?:главный\s+аудитор\s+и\s+)?работники\s+([А-ЯЁA-Z]{2,12})\b", prefix, re.I):
        match = re.search(r"работники\s+([А-ЯЁA-Z]{2,12})\b", prefix, re.I)
        return [f"{match.group(1).upper()} (общие функции)"]
    if re.match(r"^главный\s+аудитор\b", prefix, re.I):
        return ["Главный аудитор"]
    if heading and re.match(r"^(?:директор|руководитель|менеджер)\s+", prefix, re.I):
        return ["Роль: " + re.split(r"[:;(]", prefix)[0].strip().capitalize()]
    if re.match(r"^[А-ЯЁA-Z]{2,12}\b", prefix) and predicate:
        return [prefix.split()[0] + " (общие функции)"]
    if re.search(r"внутренний\s+аудит\s*$", prefix, re.I):
        return ["БВА (общие функции)"]
    return []


def _norm_type(text: str, inherited: str) -> str:
    if re.search(r"не\s+име(?:ет|ют)\s+прав|запрещ|\bне\s+(?:" + "|".join(ACTION_ROOTS) + r")", text, re.I):
        return "prohibition"
    if re.search(r"име(?:ет|ют)\s+право|\bвправе\b", text, re.I):
        return "right"
    if re.search(r"\bобязан[аы]?\b|\bдолж(?:ен|на|ны)\b", text[:(ACTION.search(text).start() if ACTION.search(text) else len(text))], re.I):
        return "duty"
    return inherited


def _role(text: str) -> str:
    text = _norm(_body(text))
    if re.match(r"обсуждени\w* и согласовани", text):
        return "agree"
    if re.match(r"(?:провер|контрол|оценив|оценк|аудит|мониторинг)|(?:провод|провед)\w*\s+(?:аудит|провер|оценк)", text):
        return "control"
    if re.match(r"(?:утвержд|утверждение)", text):
        return "approve"
    if re.match(r"(?:соглас|согласование)", text):
        return "agree"
    if re.match(r"(?:участ|участи|содейств)", text):
        return "participate"
    if re.match(r"(?:организ|координир|руковод)", text):
        return "coordinate"
    return "execute"


def _unique_sources(sources: list[Fragment]) -> tuple[Fragment, ...]:
    return tuple({s.id: s for s in sources}.values())


def _split_actions(text: str) -> list[str]:
    """Split explicit coordinated predicates, not arbitrary list commas."""
    result = []
    for part in re.split(r";\s*", text):
        starts = [m.start() for m in FINITE.finditer(part)]
        cuts = []
        for pos in starts[1:]:
            connector = re.search(r"(?:,\s*|\s+и\s+)$", part[:pos])
            if connector:
                cuts.append((connector.start(), pos))
        start = 0
        for end, next_start in cuts:
            chunk = part[start:end].strip(" ,;")
            if chunk:
                result.append(chunk)
            start = next_start
        chunk = part[start:].strip(" ,;")
        if chunk:
            result.append(chunk)
    return result


def extract_units_and_functions(documents: list[Document]) -> tuple[dict[str, list[Fragment]], list[Function]]:
    catalog = _catalog(documents)
    functions: list[Function] = []
    seen: set[tuple] = set()
    for document in documents:
        in_definitions = False
        owners: list[str] = []
        contexts: list[Fragment] = []
        owner_prefix = ""
        inherited = "duty"
        intro: Fragment | None = None
        intro_text = ""
        scopes: dict[str, tuple[str, Fragment]] = {}
        for index, fragment in enumerate(document.fragments):
            text = fragment.text.strip()
            body, clause = _body(text), _clause(text)
            if re.match(r"^термины\s+и\s+определения\b", body, re.I):
                in_definitions = True
                owners, contexts, scopes = [], [], {}
                continue
            if in_definitions:
                # A later numbered section may resume substantive assignments.
                if re.match(r"^\s*\d{1,2}\.\s*[А-Яа-яA-Za-z]", text):
                    in_definitions = False
                else:
                    continue
            is_letter = bool(LETTER.match(text))
            if re.match(r"^УТВЕРЖДЕНО\b", body, re.I):
                continue
            if not body or body.strip(".;:") == "":
                continue
            if (is_letter or body.endswith(".")) and re.match(r"^(?:директор|руководитель|менеджер|аудитор)\s+", body, re.I) and not FINITE.search(body) and not MODAL.search(body):
                # A position in a staffing list is an entity, not an action
                # named "руководитель" merely sharing a verb stem.
                continue
            if clause:
                intro, intro_text = None, ""
                if owner_prefix and not (clause == owner_prefix or clause.startswith(owner_prefix + ".")):
                    owners, contexts, scopes = [], [], {}
                    inherited, owner_prefix = "duty", ""
                if "." not in clause:
                    owners, contexts, scopes = [], [], {}
                    inherited, owner_prefix = "duty", ""
            is_heading = body.endswith(":") or (not body.endswith(".") and not FINITE.search(body) and bool(re.match(r"^(?:директор|главный аудитор|работники|руководитель)", body, re.I)))
            explicit = _subject_owners(body, catalog, heading=is_heading) if not is_letter and (is_heading or FINITE.search(body)) else []
            if STRUCTURE_INTRO.search(body):
                owners, contexts, scopes = [], [], {}
                inherited, owner_prefix = "duty", ""
                continue
            if UNIT_START.match(body) and not FINITE.search(body) and not MODAL.search(body):
                mention = catalog.mentions(body)
                if mention:
                    owners, contexts, scopes = mention, [fragment], {}
                    owner_prefix = clause
                    inherited = "duty"
                    continue
            heading_only = is_heading and explicit and (not FINITE.search(body) or MODAL.search(body))
            if explicit:
                if explicit != owners:
                    scopes = {}
                    contexts = [fragment]
                    inherited = "duty"
                    if not heading_only:
                        owner_prefix = clause
                owners = explicit
                contexts = [fragment] if heading_only else contexts
                if heading_only:
                    owner_prefix = clause
            if heading_only:
                contexts = [fragment]
                inherited = _norm_type(body, inherited)
                continue
            if owners and contexts and not owner_prefix and clause:
                owner_prefix = clause.rsplit(".", 1)[0] if clause.count(".") >= 2 else clause
            next_is_letter = index + 1 < len(document.fragments) and bool(LETTER.match(document.fragments[index + 1].text))
            active_context = list(contexts)
            assignment_owners = owners[:]
            if is_letter and intro:
                active_context.append(intro)
                named = catalog.mentions(body)
                if named and set(named) <= set(owners):
                    assignment_owners = named
                    if len(owners) > 1:
                        for owner in named:
                            scopes[owner] = (body, fragment)
                chunks = [body]
            else:
                if not ACTION.search(body):
                    continue
                if not owners and not FINITE.search(body):
                    # Definitions, titles and references merely mentioning an
                    # activity aren't assignments.
                    continue
                function_text = body
                if explicit:
                    first = FINITE.search(body)
                    if first:
                        start = first.start()
                        negation = re.search(r"\bне\s+$", body[:start], re.I)
                        function_text = body[negation.start() if negation else start:]
                chunks = _split_actions(function_text)
                if next_is_letter:
                    intro = fragment
                    intro_text = chunks[-1] if chunks else body
                    # A numbered introduction may be an actual duty as well as
                    # the context for its lettered subitems (e.g. 5.3.4).
                    if not FINITE.search(intro_text) or len(_tokens(intro_text)) < 3:
                        chunks = chunks[:-1]
            if not assignment_owners:
                assignment_owners = [UNKNOWN_OWNER]
            for chunk in chunks:
                chunk = chunk.strip(" ;")
                if len(_tokens(chunk)) < 2:
                    continue
                # A prohibition in one clause never changes adjacent positive
                # actions; an explicit subitem verb overrides its introduction.
                chunk_type = _norm_type(chunk, inherited)
                explicit_action = bool(FINITE.match(chunk) or re.match(r"^(?:не\s+)?(?:провер|контрол|оценк|аудит|соглас|утвержд)", chunk, re.I))
                own_role = bool(re.match(r"^обсуждени\w*\s+и\s+согласовани", _norm(chunk)))
                action_role = _role(intro_text if is_letter and intro_text and not (explicit_action or own_role) else chunk)
                for owner in assignment_owners:
                    context = active_context[:]
                    if owner in catalog.units:
                        # Preserve the evidence resolving an acronym/role to
                        # its canonical department, not just the duty clause.
                        context.append(catalog.units[owner][0])
                    scope = ""
                    if owner in scopes:
                        scope, scope_source = scopes[owner]
                        if scope_source.id != fragment.id:
                            context.append(scope_source)
                    key = (fragment.id, owner, _norm(chunk), chunk_type, action_role)
                    if key in seen:
                        continue
                    seen.add(key)
                    digest = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:14]
                    functions.append(Function(
                        f"{document.period}:fn:{digest}", owner, chunk, fragment,
                        chunk_type, action_role, scope, _unique_sources(context), owner != UNKNOWN_OWNER,
                    ))
    return catalog.units, functions


def _match_units_with_profiles(
    before: dict[str, list[Fragment]], after: dict[str, list[Fragment]],
    before_functions: list[Function], after_functions: list[Function],
) -> list[UnitChange]:
    """Use corroborating duties when a department was renamed completely."""
    old_profiles: dict[str, list[Function]] = defaultdict(list)
    new_profiles: dict[str, list[Function]] = defaultdict(list)
    for function in before_functions:
        old_profiles[function.unit].append(function)
    for function in after_functions:
        new_profiles[function.unit].append(function)

    candidates: list[tuple[float, str, str]] = []
    for old in before:
        for new in after:
            name_score = _similarity(old, new)
            old_duties, new_duties = old_profiles[old], new_profiles[new]
            profile_score = 0.0
            if old_duties and new_duties:
                old_coverage = sum(max(_function_similarity(a, b) for b in new_duties) for a in old_duties) / len(old_duties)
                new_coverage = sum(max(_function_similarity(a, b) for a in old_duties) for b in new_duties) / len(new_duties)
                profile_score = (old_coverage + new_coverage) / 2
            score = 0.4 * name_score + 0.6 * profile_score if profile_score else name_score
            enough_evidence = name_score >= 0.70 or (
                min(len(old_duties), len(new_duties)) >= 2 and profile_score >= 0.75
            )
            if score >= 0.48 and enough_evidence:
                candidates.append((score, old, new))

    changes: list[UnitChange] = []
    used_old: set[str] = set()
    used_new: set[str] = set()
    for score, old, new in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
        if old in used_old or new in used_new:
            continue
        status = "preserved" if _norm(old) == _norm(new) else "transformed"
        changes.append(UnitChange(status, old, new, round(score, 2), [before[old][0], after[new][0]]))
        used_old.add(old)
        used_new.add(new)
    for old in before:
        if old not in used_old:
            changes.append(UnitChange("removed", old, None, 0.5, [before[old][0]]))
    for new in sorted(set(after) - used_new):
        changes.append(UnitChange("created", None, new, 0.5, [after[new][0]]))
    return changes


def _same_action(left: Function, right: Function) -> bool:
    if left.norm_type != right.norm_type:
        return False
    if left.role != right.role:
        return False
    la, ra = ACTION.search(left.text), ACTION.search(right.text)
    if la and ra and la.start() < 12 and ra.start() < 12:
        if _stem(la.group()) == _stem(ra.group()):
            return True
        # A combined old assignment can be split into separate new clauses.
        # Require substantial text overlap as well as a shared action stem.
        old_actions = {_stem(match.group()) for match in ACTION.finditer(left.text)}
        new_actions = {_stem(match.group()) for match in ACTION.finditer(right.text)}
        return bool(old_actions & new_actions) and _similarity(left.text, right.text) >= 0.70
    return True


def _function_similarity(left: Function, right: Function) -> float:
    if not _same_action(left, right):
        return 0.0
    score = _similarity(left.text, right.text)
    a, b = _tokens(left.text), _tokens(right.text)
    if a and b and min(len(a), len(b)) >= 5 and min(len(a), len(b)) / max(len(a), len(b)) >= 0.45:
        if len(a & b) / min(len(a), len(b)) >= 0.96:
            # E.g. unchanged report duty with a removed periodicity qualifier.
            # Keep it as a changed formulation instead of a spurious loss.
            score = max(score, 0.82)
    return score


def _action_stems(text: str) -> set[str]:
    return {_stem(match.group()) for match in ACTION.finditer(text)}


def _merge_split_actions(rows: list[MatrixRow]) -> None:
    """Link a compound old duty to separately documented new sub-duties."""
    consumed: set[int] = set()
    for old_row in rows:
        if not old_row.before or old_row.after:
            continue
        old_text = " ".join(function.text for function in old_row.before)
        actions = _action_stems(old_text)
        if len(actions) < 2:
            continue
        old_objects = _tokens(old_text) - actions
        matches: dict[str, MatrixRow] = {}
        for candidate in rows:
            if candidate is old_row or id(candidate) in consumed or candidate.before or not candidate.after or candidate.norm_type != old_row.norm_type:
                continue
            candidate_actions = _action_stems(candidate.label) & actions
            shared_objects = old_objects & (_tokens(candidate.label) - _action_stems(candidate.label))
            if len(shared_objects) < 2:
                continue
            for action in candidate_actions:
                if action not in matches or len(shared_objects) > len(old_objects & _tokens(matches[action].label)):
                    matches[action] = candidate
        selected = {id(candidate): candidate for candidate in matches.values()}
        if len(matches) < 2 or len(selected) < 2:
            continue
        for candidate in selected.values():
            old_row.after.extend(candidate.after)
            consumed.add(id(candidate))
        old_row.notes.append("Составная прежняя функция сопоставлена с несколькими новыми пунктами; проверьте оба назначения.")
    rows[:] = [row for row in rows if id(row) not in consumed]


def _possible_abbreviation(old: Function, new: Function) -> bool:
    if old.norm_type != new.norm_type or old.role != new.role:
        return False
    old_first, new_first = ACTION.search(old.text), ACTION.search(new.text)
    if not old_first or not new_first or _stem(old_first.group()) != _stem(new_first.group()):
        return False
    short, long = _tokens(new.text), _tokens(old.text)
    return 2 <= len(short) <= 4 and len(long) > len(short) and len(short & long) / len(short) >= 0.75


def _matrix_rows(before: list[Function], after: list[Function], incomplete_after: bool = False) -> list[MatrixRow]:
    rows: list[MatrixRow] = []
    for side, functions in (("before", before), ("after", after)):
        for function in functions:
            scored = []
            for row in rows:
                representatives = row.before + row.after
                score = max(_function_similarity(function, candidate) for candidate in representatives)
                scored.append((score, any(f.unit == function.unit for f in representatives), row))
            best = max(scored, key=lambda x: (x[0], x[1]), default=(0.0, False, None))
            threshold = 0.88 if side == "before" or (best[2] and not best[2].before) else 0.78
            if best[0] >= threshold:
                getattr(best[2], side).append(function)
            else:
                row_id = "row:" + hashlib.sha256(f"{function.norm_type}|{function.role}|{_norm(function.text)}".encode()).hexdigest()[:14]
                rows.append(MatrixRow(row_id, function.text, [function] if side == "before" else [], [function] if side == "after" else [], "", function.norm_type))
    _merge_split_actions(rows)
    for row in rows:
        old = {f.unit for f in row.before if f.owner_known}
        new = {f.unit for f in row.after if f.owner_known}
        if any(not f.owner_known for f in row.before + row.after):
            row.status = "unknown"
            row.notes.append("Владелец части назначений не установлен; отсутствие отметки не доказывает потерю.")
        elif not row.after:
            candidates = [f for f in after if any(
                _function_similarity(old_f, f) >= 0.56 or _possible_abbreviation(old_f, f)
                for old_f in row.before
            )]
            if incomplete_after or candidates:
                row.status = "unknown"
                row.notes.append("Нужно проверить возможную переформулировку или неполноту извлечения; потеря не подтверждена.")
            else:
                row.status = "lost"
                row.notes.append("Прямое закрепление не найдено в обработанных документах после; это кандидат для проверки, не факт прекращения работы.")
        elif not row.before:
            row.status = "new"
        elif old != new:
            row.status = "moved"
            row.notes.append("Изменился набор владельцев; сравните области ответственности и контекст назначений.")
        else:
            row.status = "preserved" if {_norm(f.text) for f in row.before} == {_norm(f.text) for f in row.after} else "changed"
            if row.status == "changed":
                row.notes.append("Владелец сохранён; формулировка изменилась. Проверьте объём, условия и периодичность.")
        dept_assignments = [f for f in row.after if f.owner_known and not ("(общие функции)" in f.unit or f.unit.startswith("Роль:") or f.unit == "Главный аудитор")]
        row.candidate_overlap = row.norm_type == "duty" and len({f.unit for f in dept_assignments}) > 1
        if row.candidate_overlap:
            if len({f.scope for f in dept_assignments if f.scope}) > 1:
                row.notes.append("Несколько владельцев с разными описаниями области; две отметки сами по себе не доказывают дубль.")
            else:
                row.notes.append("Несколько владельцев: требуется проверка пересечения ответственности, а не автоматическое признание дубля.")
        if row.norm_type == "prohibition":
            row.notes.append("Запрет; не является выполняемой обязанностью и исключён из индикаторов потери и конфликта.")
    return rows


def _row_sources(row: MatrixRow) -> list[Fragment]:
    return list(_unique_sources([s for f in row.before + row.after for s in (f.source, *f.context_sources)]))


def _is_generic_duty(text: str) -> bool:
    normalized = _norm(text)
    return any(re.search(pattern, normalized) for pattern in (
        r"прочих поручени",
        r"организу\w* работу (?:днм|дккм|департамент|отдел|центр)",
        r"повышени\w* профессионального уровня",
        r"по всему кругу вопросов",
        r"участву\w* в разработке (?:внд|проектов документации)",
    ))


def _find_duplicates(rows: list[MatrixRow]) -> list[Finding]:
    findings = []
    for row in rows:
        if (not row.candidate_overlap or len({f.scope for f in row.after if f.scope}) > 1
                or _is_generic_duty(row.label)):
            continue
        findings.append(Finding(
            "duplicate", "Проверить пересечение: " + row.label[:110],
            "Одна сопоставленная обязанность указана у нескольких владельцев. Совместное выполнение или разные области могут объяснять пересечение; факт дублирования не установлен.",
            0.65, _row_sources(row), "Сверить роль и область каждого владельца; уточнить, кто отвечает за конечный результат.", row.id,
        ))
    return findings


def _find_conflicts(functions: list[Function]) -> list[Finding]:
    by_unit: dict[str, list[Function]] = defaultdict(list)
    for f in functions:
        if f.norm_type == "duty" and f.owner_known:
            by_unit[f.unit].append(f)
    findings = []
    for unit, items in by_unit.items():
        # A corporate-wide charter and the chief auditor's governance duties
        # do not describe a single operational team executing its own control.
        if unit == "Главный аудитор" or "(общие функции)" in unit or unit.startswith("Роль:"):
            continue
        audit = [f for f in items if f.role == "control"]
        execution = [f for f in items if f.role == "execute" and not re.search(r"аудит|провер|оценк|мониторинг|контрол", f.text, re.I)]
        seen = set()
        for check in audit:
            for execute in execution:
                objects = (_tokens(check.text) & _tokens(execute.text)) - {"систем", "работ", "процесс", "план", "результат", "информац", "организ"}
                if check.id == execute.id or len(objects) < 2 or (check.scope and execute.scope and check.scope != execute.scope):
                    continue
                key = (unit, tuple(sorted(objects)))
                if key in seen:
                    continue
                seen.add(key)
                findings.append(Finding(
                    "conflict", f"Проверить совмещение исполнения и контроля: {unit}",
                    "В обязанностях одного владельца найдены выполнение и контроль сходного процесса. Требуется проверить объект, область и независимость проверки; нарушение не утверждается.",
                    0.65, list(_unique_sources([execute.source, check.source, *execute.context_sources, *check.context_sources])),
                    "Проверить независимость контроля; при подтверждении совмещения разделить роли или зафиксировать компенсирующий контроль.",
                    function_ids=[execute.id, check.id],
                ))
    return findings


def analyze_documents(before_docs: list[Document], after_docs: list[Document], *, after_complete: bool = False) -> AnalysisResult:
    before_units, before_functions = extract_units_and_functions(before_docs)
    after_units, after_functions = extract_units_and_functions(after_docs)
    warnings = [w for d in before_docs + after_docs for w in d.warnings]
    if not before_docs or not after_docs:
        warnings.append("Для сравнения необходимы оба комплекта документов.")
    if not before_units or not after_units:
        warnings.append("Не во всех комплектах найден достоверный перечень подразделений; проверьте структуру и заголовки.")
    if not before_functions or not after_functions:
        warnings.append("Не во всех комплектах выделены обязанности; отсутствие найденных рисков не означает отсутствие проблем.")
    unknown_before = sum(not f.owner_known for f in before_functions)
    unknown_after = sum(not f.owner_known for f in after_functions)
    if unknown_before or unknown_after:
        warnings.append(f"Владелец не установлен для {unknown_before} назначений до и {unknown_after} после. Они показаны отдельно и требуют проверки.")
    rows = _matrix_rows(before_functions, after_functions, bool(any(d.warnings for d in after_docs) or not after_functions))
    findings = []
    for row in rows:
        if row.status == "lost" and row.norm_type != "prohibition":
            title = "Прямое право не найдено после" if row.norm_type == "right" else "Ответственный за обязанность не найден после"
            findings.append(Finding(
                "loss", title, row.label + ". Поиск по доступным материалам не дал достаточно близкого соответствия. Общая функция могла сохраниться в другом пункте или документе.",
                0.5, _row_sources(row), "Сопоставить с общими обязанностями и приложениями; подтвердить сохранение, передачу или обоснованное исключение.", row.id,
            ))
    findings.extend(_find_duplicates(rows))
    findings.extend(_find_conflicts(after_functions))
    matches = []
    for row in rows:
        status = row.status if row.status in {"preserved", "moved", "changed", "lost", "new"} else "changed"
        if row.before:
            for old in row.before:
                new = max(row.after, key=lambda f: (_function_similarity(old, f), old.unit == f.unit), default=None)
                matches.append(FunctionMatch(old, new, _function_similarity(old, new) if new else 0.0, status))
        else:
            matches.extend(FunctionMatch(None, new, 0.0, status) for new in row.after)
    result = AnalysisResult(
        _match_units_with_profiles(before_units, after_units, before_functions, after_functions), matches, findings, warnings,
        rows, list(_unique_sources([f for d in before_docs + after_docs for f in d.fragments])),
        list(dict.fromkeys([*before_units, *(f.unit for f in before_functions if f.owner_known)])),
        list(dict.fromkeys([*after_units, *(f.unit for f in after_functions if f.owner_known)])),
        {"documents_before": len(before_docs), "documents_after": len(after_docs),
         "fragments_before": sum(len(d.fragments) for d in before_docs), "fragments_after": sum(len(d.fragments) for d in after_docs),
         "functions_before": len(before_functions), "functions_after": len(after_functions),
         "unknown_owners_before": unknown_before, "unknown_owners_after": unknown_after, "matrix_rows": len(rows)},
    )
    assess_result(result, before_docs, after_docs, after_complete=after_complete,
                  similarity=_similarity, tokens=_tokens, actions=_action_stems)
    return result
