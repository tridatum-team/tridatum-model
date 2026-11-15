"""Narrow lm-format-enforcer 0.11.3 token-trie integration corrections.

The upstream free-text cache excludes incomplete escape prefixes and mixed
text/JSON-boundary tokens. Its shortcut traverses only the quote-start branch,
so valid Qwen paths such as ``\\u`` and period/quote/comma/quote are omitted.
Reuse the parser and ordinary cache; traverse a once-built residual-token trie.
This module is imported only for the explicit JSON-schema diagnostic profile.

LMFE's isolated-token alphabet also omits Unicode scalars completed across
multiple byte tokens. Synchronize unrestricted-string admission with those
actual decoded scalars before committing them; never let an invalid character
silently discard an unfinished parser stack and manufacture EOS.
"""
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn as library_prefix_function,
)
from lmformatenforcer.tokenenforcer import TokenEnforcer
from lmformatenforcer.jsonschemaparser import JsonSchemaParser, StringParsingState
from lmformatenforcer.tokenizerprefixtree import TokenizerPrefixTreeNode


def residual_freetext_tree(tokenizer_data):
    """Index only omitted escape/boundary tokens, once per shared tokenizer data."""
    existing = getattr(tokenizer_data, '_trid_residual_freetext_tree', None)
    if existing is not None:
        return existing
    cache = tokenizer_data.tokenizer_tree.json_freetext_tokens
    cached = set(cache.regular_tokens_length_cache.tokens)
    cached.update(cache.quote_tokens_length_cache.tokens)
    root = TokenizerPrefixTreeNode()
    for token, text, _ in tokenizer_data.regular_tokens:
        if token in cached or not text or text.startswith('"'):
            # Quote-start paths already traverse the full upstream trie.
            continue
        escaped = False
        boundary = False
        invalid_control = False
        for character in text:
            if ord(character) < 32:
                invalid_control = True
                break
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                boundary = True
                break
        # Reject raw controls inside string content. Controls after its closing
        # quote are handled as JSON syntax/whitespace by the unchanged parser.
        if invalid_control or not (boundary or '\\' in text):
            continue
        node = root
        for character in text:
            if character not in node.children:
                node.children[character] = TokenizerPrefixTreeNode()
            node = node.children[character]
        node.tokens.append(token)
    tokenizer_data._trid_residual_freetext_tree = root
    return root


class EscapeAwareTokenEnforcer(TokenEnforcer):
    def __init__(self, tokenizer_data, parser):
        super().__init__(tokenizer_data, parser)
        self.residual_tree = residual_freetext_tree(tokenizer_data)

    def _apply_new_characters(self, state, token_sequence):
        # Preserve upstream contextual decoding, but do not inherit its
        # catch-and-ForceStop behavior or unchecked stack-dropping transitions.
        new_state = TokenEnforcer.OutputTensorState(parser=state.parser)
        token = token_sequence[-1]
        if token in self.tokenizer_tree.new_word_tokens:
            new_state.current_word_tokens = [token]
            continuation = self.tokenizer_tree.tokens_to_strs[token]
        else:
            new_state.current_word_tokens = state.current_word_tokens + [token]
            previous = self.decoder(state.current_word_tokens)
            decoded = self.decoder(new_state.current_word_tokens)
            if not decoded.startswith(previous):
                raise ValueError('LMFE contextual token decode changed an already committed prefix')
            continuation = decoded[len(previous):]
        for character in continuation:
            parser = new_state.parser
            top = parser.object_stack[-1] if isinstance(parser, JsonSchemaParser) and parser.object_stack else None
            free_string = (isinstance(top, StringParsingState) and not top.allowed_strings
                           and not top.regex_parser and top.seen_opening_quote and not top.seen_closing_quote)
            if free_string:
                if ord(character) < 32 or 0xD800 <= ord(character) <= 0xDFFF or character == '\ufffd':
                    raise ValueError('Invalid decoded Unicode/control character inside JSON string')
                if character not in parser.config.alphabet:
                    # Per-answer parser config/context only; shared tokenizer data,
                    # original citation enums, prompt bytes and token IDs stay fixed.
                    # This also covers the first character under minLength=1.
                    parser.config.alphabet += character
                    parser.context.alphabet_without_quotes = parser.config.alphabet.replace('"', '')
            if character not in parser.get_allowed_characters():
                raise ValueError(f'Invalid committed JSON transition for U+{ord(character):04X}; '
                                 'refusing to discard an unfinished parser stack')
            new_state.parser = parser.add_character(character)
        return new_state

    def _collect_allowed_tokens(self, parser, tree_node, allowed_tokens, shortcut_key):
        super()._collect_allowed_tokens(parser, tree_node, allowed_tokens, shortcut_key)
        if isinstance(shortcut_key, tuple) and shortcut_key[0] == 'json_freetext':
            # No shortcut on the small residual trie: the maintained parser
            # handles complete syntax transitions across each candidate token.
            super()._collect_allowed_tokens(parser, self.residual_tree, allowed_tokens, None)


def build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser):
    prefix = library_prefix_function(tokenizer_data, parser)
    # Reuse the single tokenizer trie. Only per-answer prefix/parser state changes.
    prefix.token_enforcer = EscapeAwareTokenEnforcer(tokenizer_data, parser)
    return prefix
